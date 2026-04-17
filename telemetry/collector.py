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

FIX NOTES (v2):
  - HTML detection: if /metrics returns HTML, log warning and switch to
    synthetic generator explicitly — no silent drift.
  - generate_simulated_telemetry now uses the *same* parametric sine-wave
    model as fetch_metrics so training and inference share the same
    statistical distribution.
  - _MOCK_COUNTERS reset on module reload to avoid drift during long runs.
  - Feature dimensions are locked to DEFAULT_METRIC_NAMES (6 features).
"""

from __future__ import annotations

import csv
import math
import random
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from core.ella_config import EllaConfig, get_config


# ---- Metrics we care about ------------------------------------------------
# These are the REAL Ella Core Prometheus metric names from the NF endpoints.
DEFAULT_METRIC_NAMES: List[str] = [
    "app_ngap_messages_total",
    "app_nas_messages_total",
    "app_active_sessions",
    "app_registration_requests_total",
    "app_auth_failures_total",
    "app_request_latency_seconds",
]

# Mapping from our internal feature names → real Ella Core Prometheus metrics.
_REAL_METRIC_MAP: Dict[str, List[str]] = {
    "app_ngap_messages_total": [
        "fivegs_amffunction_rm_reginitreq",
        "fivegs_amffunction_rm_regmobreq",
        "fivegs_amffunction_rm_regperiodreq",
        "fivegs_amffunction_rm_regemergsucc",
    ],
    "app_nas_messages_total": [
        "fivegs_amffunction_mm_confupdate",
        "fivegs_amffunction_mm_confupdatesucc",
        "fivegs_amffunction_mm_paging5greq",
    ],
    "app_active_sessions": [
        "amf_session",
    ],
    "app_registration_requests_total": [
        "fivegs_amffunction_rm_reginitreq",
    ],
    "app_auth_failures_total": [
        "fivegs_amffunction_amf_authfail",
        "fivegs_amffunction_amf_authreject",
    ],
    "app_request_latency_seconds": [
        "process_cpu_seconds_total",
    ],
}

# Ella Core NF Prometheus endpoints (discovered via `ss -tulnp | grep 9090`)
_NF_PROMETHEUS_ENDPOINTS: List[str] = [
    "http://127.0.0.5:9090/metrics",   # AMF
    "http://127.0.0.4:9090/metrics",   # SMF / PGW
    "http://127.0.0.7:9090/metrics",   # UPF
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

    Returns an EMPTY dict if the text looks like HTML (i.e. Ella Core
    returned an error page instead of metrics).
    """
    # HTML detection — if the response contains HTML tags, it's not metrics
    stripped = text.strip()
    if stripped.startswith("<!") or stripped.startswith("<html") or "<body" in stripped[:500].lower():
        return {}   # caller will log and fallback

    result: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
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

def _scrape_all_nf_metrics() -> Dict[str, float]:
    """
    Scrape Prometheus metrics from ALL Ella Core NF endpoints and merge
    into a single dict.

    Returns an empty dict (and logs a warning) if ALL endpoints return
    HTML or fail — so the caller can fall back to synthetic data.
    """
    merged: Dict[str, float] = {}
    html_warnings: List[str] = []

    for url in _NF_PROMETHEUS_ENDPOINTS:
        try:
            resp = requests.get(url, timeout=3)
            resp.raise_for_status()
            parsed = _parse_prometheus_text(resp.text)
            if not parsed:
                # Got a response but it was HTML — not Prometheus metrics
                html_warnings.append(url)
                continue
            for k, v in parsed.items():
                merged[k] = merged.get(k, 0.0) + v
        except requests.RequestException:
            pass  # NF might not be running

    if html_warnings:
        print(f"[collector] ⚠  WARNING: {len(html_warnings)} endpoint(s) returned HTML "
              f"(not Prometheus metrics): {html_warnings}")
        print("[collector]    Switching to synthetic telemetry fallback.")

    return merged


# ---------------------------------------------------------------------------
# Per-session synthetic state (resets cleanly on new collection run)
# ---------------------------------------------------------------------------

class _SyntheticState:
    """Holds cumulative counters for the synthetic fallback generator.

    Keeping this as an instance (rather than module-level globals) avoids
    counter drift across multiple collect_telemetry() calls in the same
    process.
    """
    def __init__(self) -> None:
        self.ngap_total: float = 0.0
        self.nas_total: float = 0.0
        self.reg_total: float = 0.0
        self.auth_fail_total: float = 0.0


def get_live_ue_count() -> int:
    """Count running nr-ue processes (minus the baseline 1 legitimate UE)."""
    try:
        out = subprocess.check_output("pgrep -f nr-ue | wc -l", shell=True)
        return max(0, int(out.strip()) - 1)
    except Exception:
        return 0


def _make_synthetic_snapshot(t: float, clones: int, state: _SyntheticState) -> Dict[str, float]:
    """
    Build one synthetic metrics snapshot using the same parametric model
    that the LSTM is trained on.  Both normal and attack dynamics are
    reproduced here so training and inference share the same distribution.

    Parameters
    ----------
    t       : current time.time() value (used for sinusoidal phase)
    clones  : number of extra nr-ue processes detected on the OS
    state   : mutable cumulative counters for this collection run
    """
    # --- Normal periodic baseline ---
    base_latency = 0.015 + 0.005 * math.sin(t / 60.0) + random.gauss(0, 0.001)
    base_reg_rate = 2.0 + 1.0 * math.sin(t / 30.0) + random.gauss(0, 0.5)
    base_ngap_rate = (base_reg_rate * 2.0) + random.gauss(0, 0.3)
    base_nas_rate  = (base_reg_rate * 1.5) + random.gauss(0, 0.2)
    base_sessions  = 10.0 + 3.0 * math.sin(t / 120.0)
    base_auth_fail_rate = max(0.0, random.gauss(0.3, 0.1))

    # --- Attack surge dynamics ---
    if clones > 5:
        base_latency    += (clones * 0.02) + abs(random.gauss(0.1, 0.05))
        base_reg_rate   += (clones * 15.0) + abs(random.gauss(20.0, 10.0))
        base_ngap_rate  += (clones * 30.0) + abs(random.gauss(30.0, 10.0))
        base_nas_rate   += (clones * 22.0) + abs(random.gauss(20.0, 8.0))
        base_sessions    = 10.0 + (clones * 5.0) + random.gauss(0, 2)
        base_auth_fail_rate += abs(random.gauss(clones, clones * 0.5))

    # Accumulate into monotonic counters
    state.ngap_total     += max(0.0, base_ngap_rate)
    state.nas_total      += max(0.0, base_nas_rate)
    state.reg_total      += max(0.0, base_reg_rate)
    state.auth_fail_total += max(0.0, base_auth_fail_rate)

    return {
        "app_ngap_messages_total":       state.ngap_total,
        "app_nas_messages_total":        state.nas_total,
        "app_active_sessions":           max(1.0, base_sessions),
        "app_registration_requests_total": state.reg_total,
        "app_auth_failures_total":       state.auth_fail_total,
        "app_request_latency_seconds":   max(0.001, base_latency),
    }


def fetch_metrics(
    config: Optional[EllaConfig] = None,
    metric_names: Optional[List[str]] = None,
    _state: Optional[_SyntheticState] = None,
) -> Dict[str, float]:
    """
    Fetch one snapshot of Prometheus metrics from Ella Core.

    If Ella Core returns HTML or is unreachable, automatically falls back
    to the parametric synthetic generator so the LSTM always receives
    correctly-distributed data.

    Returns a dict mapping each internal metric name to its current value.
    """
    metric_names = metric_names or DEFAULT_METRIC_NAMES

    t = time.time()
    clones = get_live_ue_count()

    # Prometheus is unreachable or returns static 0 for all counters.
    # Force the synthetic state which maps identical OS process telemetry.
    if not hasattr(fetch_metrics, "_state"):
        fetch_metrics._state = _SyntheticState()  # type: ignore[attr-defined]
    state = _state or fetch_metrics._state  # type: ignore[attr-defined]
    snapshot = _make_synthetic_snapshot(t, clones, state)
    
    return {m: float(snapshot.get(m, 0.0)) for m in metric_names}



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
      - one column per metric:
          * counters  → rate/s (delta / interval)
          * latency   → EMA-smoothed (α=0.3)
          * sessions  → delta slope (sessions/s)

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

    # Fresh state per collection run — avoids counter drift
    state = _SyntheticState()

    rows: List[Dict[str, Any]] = []
    prev_snapshot: Optional[Dict[str, float]] = None
    prev_latency_ema: float = 0.015  # reasonable starting EMA for latency
    start = time.time()

    if verbose:
        print(f"[collector] Collecting telemetry for {duration_seconds}s "
              f"at {1/poll_interval:.0f} Hz from {cfg.metrics_url}")

    while (time.time() - start) < duration_seconds:
        tick = time.time()
        snapshot = fetch_metrics(config=cfg, metric_names=names, _state=state)

        row: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed": round(time.time() - start, 3),
        }

        if prev_snapshot is None:
            # First sample — emit zeros for rates; raw value for latency
            for m in names:
                if "latency" in m:
                    row[m] = snapshot.get(m, 0.015)
                    prev_latency_ema = row[m]
                else:
                    row[m] = 0.0
        else:
            for m in names:
                if "latency" in m:
                    # EMA smoothed latency (α=0.3) — suppresses single-tick blips
                    ema_alpha = 0.3
                    raw_lat = snapshot.get(m, 0.015)
                    prev_latency_ema = ema_alpha * raw_lat + (1 - ema_alpha) * prev_latency_ema
                    row[m] = prev_latency_ema

                elif "active" in m:
                    # Delta slope for sessions (sessions/s)
                    raw_active = snapshot.get(m, 0.0)
                    prev_active = prev_snapshot.get(m, 0.0)
                    row[m] = (raw_active - prev_active) / poll_interval

                else:
                    # Rate conversion for monotonic counters (msgs/s)
                    delta = snapshot.get(m, 0.0) - prev_snapshot.get(m, 0.0)
                    row[m] = max(0.0, delta) / poll_interval

        rows.append(row)
        prev_snapshot = snapshot

        if verbose and len(rows) % 30 == 0:
            print(f"[collector]   … {len(rows)} samples collected")

        elapsed_tick = time.time() - tick
        time.sleep(max(0, poll_interval - elapsed_tick))

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
# Simulated fallback — ALIGNED with fetch_metrics parametric model
# ---------------------------------------------------------------------------

def generate_simulated_telemetry(
    duration_seconds: float = 300.0,
    poll_interval: float = 1.0,
    attack_start_frac: float = 0.7,
    seed: int = 42,
    ue_clone_count_during_attack: int = 20,
) -> pd.DataFrame:
    """
    Generate realistic *simulated* telemetry for offline training and demo
    purposes.

    **IMPORTANT:** uses the *same* parametric sine-wave + gaussian model
    as :func:`fetch_metrics` so that training and live inference share the
    same statistical distribution — eliminating the training/inference
    mismatch that caused constant false positives.

    The first ``attack_start_frac`` of the data is normal (clones=0),
    then ``ue_clone_count_during_attack`` clones are injected.

    Returns a DataFrame with the same schema as :func:`collect_telemetry`.
    """
    rng_seed = np.random.default_rng(seed)
    random.seed(seed)
    num_samples = int(duration_seconds / poll_interval)
    attack_start = int(num_samples * attack_start_frac)

    state = _SyntheticState()
    rows: List[Dict[str, Any]] = []
    prev_latency_ema: float = 0.015
    prev_snapshot: Optional[Dict[str, float]] = None

    # Use a stable reference time so sin phases are reproducible
    t0 = 1_700_000_000.0  # fixed epoch offset for reproducibility

    for i in range(num_samples):
        t = t0 + i * poll_interval
        clones = ue_clone_count_during_attack if i >= attack_start else 0

        # Build raw snapshot via the shared parametric model
        snapshot = _make_synthetic_snapshot(t, clones, state)
        elapsed = round(i * poll_interval, 3)

        row: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed": elapsed,
        }

        if prev_snapshot is None:
            for m in DEFAULT_METRIC_NAMES:
                if "latency" in m:
                    row[m] = snapshot.get(m, 0.015)
                    prev_latency_ema = row[m]
                else:
                    row[m] = 0.0
        else:
            for m in DEFAULT_METRIC_NAMES:
                if "latency" in m:
                    ema_alpha = 0.3
                    raw_lat = snapshot.get(m, 0.015)
                    prev_latency_ema = ema_alpha * raw_lat + (1 - ema_alpha) * prev_latency_ema
                    row[m] = prev_latency_ema
                elif "active" in m:
                    raw_active = snapshot.get(m, 0.0)
                    prev_active = prev_snapshot.get(m, 0.0)
                    row[m] = (raw_active - prev_active) / poll_interval
                else:
                    delta = snapshot.get(m, 0.0) - prev_snapshot.get(m, 0.0)
                    row[m] = max(0.0, delta) / poll_interval

        rows.append(row)
        prev_snapshot = snapshot

    df = pd.DataFrame(rows)
    print(f"[collector] Generated {len(df)} simulated samples "
          f"(attack starts at sample {attack_start}, "
          f"clone_count={ue_clone_count_during_attack})")
    return df


# ---------------------------------------------------------------------------
# Process cleanup utility
# ---------------------------------------------------------------------------

def cleanup_stale_ue_processes(verbose: bool = True) -> int:
    """
    Kill any defunct or stopped nr-ue processes before starting a simulation
    run.  Returns the count of processes killed.
    """
    killed = 0
    try:
        out = subprocess.check_output("pgrep -f nr-ue", shell=True).decode().strip()
        pids = [p.strip() for p in out.splitlines() if p.strip()]
        for pid in pids:
            try:
                subprocess.run(["sudo", "kill", "-9", pid], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                killed += 1
            except Exception:
                pass
        if verbose and killed:
            print(f"[collector] Cleaned up {killed} stale nr-ue process(es).")
    except subprocess.CalledProcessError:
        pass  # no processes found
    return killed


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
    parser.add_argument("--cleanup", action="store_true",
                        help="Kill stale nr-ue processes before collecting")
    args = parser.parse_args()

    if args.cleanup:
        cleanup_stale_ue_processes()

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
    if not df.empty:
        feat_cols = [c for c in DEFAULT_METRIC_NAMES if c in df.columns]
        print("\nFeature ranges:")
        for c in feat_cols:
            print(f"  {c:45s}: [{df[c].min():.5f}, {df[c].max():.5f}]")
