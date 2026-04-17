#!/usr/bin/env python3
"""
tools/validate_pipeline.py
--------------------------
Quick smoke-test that validates the entire detection cycle without needing
Ella Core or UERANSIM running.

Run:
    python3 tools/validate_pipeline.py

Expected output:
  - Normal mean error   << threshold
  - Attack mean error   >> threshold
  - Separation ratio    >= 5x
  - F1 >= 0.9
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from telemetry.collector import generate_simulated_telemetry, cleanup_stale_ue_processes
from simulation.attack_generator import generate_attack_telemetry
from pipeline.train_pipeline import run_training_pipeline
from inference.live_monitor import run_simulated_monitor


def main() -> None:
    print("=" * 62)
    print(" NWDAF Pipeline Validation Tool")
    print("=" * 62)

    # 1. Cleanup any stale processes
    print("\n[validate] Step 1: Cleaning stale nr-ue processes...")
    cleanup_stale_ue_processes()

    # 2. Train a fresh model on purely normal parametric data
    print("\n[validate] Step 2: Training on 5-min parametric normal telemetry...")
    model_path, threshold, meta = run_training_pipeline(
        simulated_duration=300.0,
        epochs=30,
        threshold_method="statistical",
        threshold_k=3.0,
    )

    print(f"\n[validate] Threshold : {threshold:.6f}")
    print(f"[validate] Train err : mean={meta['mean_training_error']:.6f}, "
          f"std={meta.get('std_training_error', 0.0):.6f}, "
          f"max={meta['max_training_error']:.6f}")

    # Gate: threshold must be above max training error
    if threshold <= meta["max_training_error"]:
        print(f"\n[validate] FAIL: threshold ({threshold:.6f}) <= "
              f"max training error ({meta['max_training_error']:.6f})")
        sys.exit(1)
    else:
        print(f"[validate] ✓ Threshold safely above max normal error "
              f"(ratio {threshold/meta['max_training_error']:.2f}x)")

    # 3. Run simulated detection
    print("\n[validate] Step 3: Running simulated detection (normal + attack)...")
    results = run_simulated_monitor(normal_samples=200, attack_samples=100)

    # 4. Report & assert
    print("\n[validate] Final assertions:")

    sep_ratio = results.get("separation_ratio", 0.0)
    f1        = results.get("f1_score", 0.0)
    n_err     = results.get("normal_mean_error", 0.0)
    a_err     = results.get("attack_mean_error", 0.0)

    ok = True

    if n_err < threshold:
        print(f"  ✓ Normal mean error {n_err:.6f} < threshold {threshold:.6f}")
    else:
        print(f"  ✗ Normal mean error {n_err:.6f} >= threshold {threshold:.6f}  [FAIL]")
        ok = False

    if a_err > threshold:
        print(f"  ✓ Attack mean error {a_err:.6f} > threshold {threshold:.6f}")
    else:
        print(f"  ✗ Attack mean error {a_err:.6f} <= threshold {threshold:.6f}  [FAIL]")
        ok = False

    if sep_ratio >= 5.0:
        print(f"  ✓ Separation ratio {sep_ratio:.1f}x >= 5x")
    else:
        print(f"  ✗ Separation ratio {sep_ratio:.1f}x < 5x  [WARN — may cause FP]")

    if f1 >= 0.85:
        print(f"  ✓ F1 score {f1:.4f} >= 0.85")
    else:
        print(f"  ✗ F1 score {f1:.4f} < 0.85  [FAIL]")
        ok = False

    print("\n" + "=" * 62)
    if ok:
        print(" VALIDATION PASSED — Pipeline is DEMO-READY")
    else:
        print(" VALIDATION FAILED — Review errors above before demo")
    print("=" * 62)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
