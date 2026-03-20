"""
Prometheus telemetry collector for Ella Core.

Continuously polls the ``/metrics`` endpoint and stores the resulting
time-series feature vectors as a structured CSV or in-memory DataFrame.

Target metrics (examples from a 5G core's Prometheus exporter):
  - app_ngap_messages_total          (NGAP message counter)
  - app_nas_messages_total           (NAS message counter)
  - app_active_sessions              (concurrent PDU sessions)
  - app_registration_requests_total  (registration attempts)
  - app_auth_failures_total          (authentication failures)
  - app_request_latency_seconds      (AMF/SMF processing latency)

The collector extracts *counter deltas* per poll interval so that the AI
model sees rates (messages/sec) rather than ever-growing counters.
"""

from __future__ import annotations

import csv
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from core.ella_config import EllaConfig, get_config


# ---- Metrics we care about ------------------------------------------------
DEFAULT_METRIC_NAMES: List[str] = [
    "app_ngap_messages_total",
    "app_nas_messages_total",
    "app_active_sessions",
    "app_registration_requests_total",
    "app_auth_failures_total",
    "app_request_latency_seconds",
]


# ---------------------------------------------------------------------------
# Prometheus text format parser (simple, no external lib needed)
# ---------------------------------------------------------------------------

def _parse_prometheus_text(text: str) -> Dict[str, float]:
    """
    Parse Prometheus exposition format into ``{metric_name: value}`` dict.

    Handles lines like:
        app_ngap_messages_total 42
        app_active_sessions{slice="1"} 7
    """
    result: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Split on whitespace; last token is the value
        parts = line.split()
        if len(parts) >= 2:
            metric_name = parts[0].split("{")[0]  # strip labels
            try:
                result[metric_name] = float(parts[-1])
            except ValueError:
                continue
    return result


# ---------------------------------------------------------------------------
# Single snapshot
# ---------------------------------------------------------------------------

def fetch_metrics(
    config: Optional[EllaConfig] = None,
    metric_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Fetch one snapshot of Prometheus metrics from Ella Core.

    Returns a dict mapping each metric name to its current value.
    Metrics not found in the response default to ``0.0``.
    """
    cfg = config or get_config()
    url = cfg.metrics_url
    metric_names = metric_names or DEFAULT_METRIC_NAMES

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[collector] WARNING: could not reach {url}: {exc}")
        return {m: 0.0 for m in metric_names}

    all_metrics = _parse_prometheus_text(resp.text)
    return {m: all_metrics.get(m, 0.0) for m in metric_names}


# ---------------------------------------------------------------------------
# Continuous collection
# ---------------------------------------------------------------------------

def collect_telemetry(
    duration_seconds: float = 300.0,
    poll_interval: float = 1.0,
    config: Optional[EllaConfig] = None,
    metric_names: Optional[List[str]] = None,
    output_csv: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Poll Ella Core's ``/metrics`` endpoint at *poll_interval* for
    *duration_seconds* and return the collected data as a DataFrame.

    Each row contains:
      - ``timestamp`` (ISO 8601)
      - ``elapsed``   (seconds since start)
      - one column per metric (the *delta* since the previous poll for
        counters, or the raw gauge value)

    Parameters
    ----------
    duration_seconds : float
        Total collection time.
    poll_interval : float
        Seconds between polls (default 1 s → 1 Hz sampling).
    config : EllaConfig, optional
        Ella Core configuration.
    metric_names : list[str], optional
        Which Prometheus metrics to track.
    output_csv : Path, optional
        If provided, write results to this CSV after collection.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    pd.DataFrame
        Collected time-series telemetry.
    """
    cfg = config or get_config()
    names = metric_names or DEFAULT_METRIC_NAMES

    rows: List[Dict[str, Any]] = []
    prev_snapshot: Optional[Dict[str, float]] = None
    start = time.time()

    if verbose:
        print(f"[collector] Collecting telemetry for {duration_seconds}s "
              f"at {1/poll_interval:.0f} Hz from {cfg.metrics_url}")

    while (time.time() - start) < duration_seconds:
        tick = time.time()
        snapshot = fetch_metrics(config=cfg, metric_names=names)

        row: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed": round(time.time() - start, 3),
        }

        if prev_snapshot is None:
            # First sample — store raw values (deltas will be 0)
            for m in names:
                row[m] = 0.0
        else:
            for m in names:
                delta = snapshot.get(m, 0.0) - prev_snapshot.get(m, 0.0)
                row[m] = max(delta, 0.0)  # monotonic counters
                # For gauges like active_sessions, use raw value
                if "active" in m or "latency" in m:
                    row[m] = snapshot.get(m, 0.0)

        rows.append(row)
        prev_snapshot = snapshot

        if verbose and len(rows) % 30 == 0:
            print(f"[collector]   … {len(rows)} samples collected")

        # Sleep for the remainder of the interval
        elapsed_tick = time.time() - tick
        sleep_time = max(0, poll_interval - elapsed_tick)
        time.sleep(sleep_time)

    df = pd.DataFrame(rows)

    if verbose:
        print(f"[collector] Collection complete — {len(df)} samples")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"[collector] Saved to {output_csv}")

    return df


# ---------------------------------------------------------------------------
# Simulated fallback (when Ella Core is NOT running)
# ---------------------------------------------------------------------------

def generate_simulated_telemetry(
    duration_seconds: float = 300.0,
    poll_interval: float = 1.0,
    attack_start_frac: float = 0.7,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic *simulated* telemetry for offline development and
    demo purposes.  The first ``attack_start_frac`` of the data is normal,
    then anomalous traffic is injected.

    Returns a DataFrame with the same schema as :func:`collect_telemetry`.
    """
    rng = np.random.default_rng(seed)
    num_samples = int(duration_seconds / poll_interval)
    attack_start = int(num_samples * attack_start_frac)

    rows: List[Dict[str, Any]] = []
    for i in range(num_samples):
        elapsed = round(i * poll_interval, 3)
        is_attack = i >= attack_start

        # Normal baseline rates
        ngap = rng.poisson(5)
        nas = rng.poisson(4)
        sessions = max(1, int(rng.normal(10, 2)))
        registrations = rng.poisson(2)
        auth_fail = rng.poisson(0.3)
        latency = max(0.001, rng.normal(0.015, 0.003))

        if is_attack:
            # Signaling storm: massive spikes in registrations and NGAP
            ngap = rng.poisson(150)
            nas = rng.poisson(120)
            registrations = rng.poisson(80)
            auth_fail = rng.poisson(40)
            latency = max(0.001, rng.normal(0.25, 0.08))

        rows.append({
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed": elapsed,
            "app_ngap_messages_total": float(ngap),
            "app_nas_messages_total": float(nas),
            "app_active_sessions": float(sessions),
            "app_registration_requests_total": float(registrations),
            "app_auth_failures_total": float(auth_fail),
            "app_request_latency_seconds": float(latency),
        })

    df = pd.DataFrame(rows)
    print(f"[collector] Generated {len(df)} simulated samples "
          f"(attack starts at sample {attack_start})")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect Ella Core telemetry")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="data/telemetry.csv")
    parser.add_argument("--simulated", action="store_true",
                        help="Generate simulated data instead of polling Ella Core")
    args = parser.parse_args()

    if args.simulated:
        df = generate_simulated_telemetry(
            duration_seconds=args.duration,
            poll_interval=args.interval,
        )
    else:
        df = collect_telemetry(
            duration_seconds=args.duration,
            poll_interval=args.interval,
            output_csv=Path(args.output),
        )

    print(df.head(10))
    print(f"\nShape: {df.shape}")
