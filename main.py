"""
NWDAF Master Orchestrator — ``main.py``

Ties all four phases together into a single entry point:

  Phase 1: Environment verification
  Phase 2: Data generation (ABMM + telemetry collection)
  Phase 3: LSTM-Autoencoder training
  Phase 4: Real-time detection & zero-touch mitigation

Run modes:
  python main.py --full          Full pipeline (requires Ella Core + UERANSIM)
  python main.py --train         Train model only (simulated or CSV data)
  python main.py --detect        Run live detection (requires trained model)
  python main.py --demo          Full demo on simulated data (no external deps)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_demo(args: argparse.Namespace) -> None:
    """Run the complete pipeline on simulated data — no Ella Core needed."""
    from pipeline.train_pipeline import run_training_pipeline
    from inference.live_monitor import run_simulated_monitor
    from visualization.error_plot import plot_reconstruction_error_distribution
    from visualization.roc_curve import plot_roc_curve
    from evaluation.reconstruction_error import compute_reconstruction_errors, load_trained_model
    from telemetry.collector import generate_simulated_telemetry
    from simulation.attack_generator import generate_attack_telemetry
    from telemetry.preprocessor import preprocess_telemetry
    from evaluation.threshold import select_threshold
    import numpy as np
    import json

    print("\n" + "=" * 60)
    print(" NWDAF — Zero‑Touch Anomaly Detection Demo")
    print("=" * 60)

    # --- Phase 3: Train ---
    print("\n" + "─" * 60)
    print(" PHASE 3: Training LSTM‑Autoencoder")
    print("─" * 60)

    model_path, threshold, metadata = run_training_pipeline(
        sequence_length=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        simulated_duration=args.train_duration,
    )

    # --- Phase 4: Detect ---
    print("\n" + "─" * 60)
    print(" PHASE 4: Anomaly Detection on Simulated Attack")
    print("─" * 60)

    results = run_simulated_monitor(model_path=model_path)

    print("\n" + "=" * 60)
    print(" Demo Complete — Results Summary")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:25s} : {v:.4f}")
        else:
            print(f"  {k:25s} : {v}")
    print("=" * 60)


def cmd_train(args: argparse.Namespace) -> None:
    """Train the LSTM-AE model."""
    from pipeline.train_pipeline import run_training_pipeline

    csv_path = Path(args.csv) if args.csv else None
    run_training_pipeline(
        normal_telemetry_csv=csv_path,
        sequence_length=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        simulated_duration=args.train_duration,
    )


def cmd_detect(args: argparse.Namespace) -> None:
    """Run live detection against Ella Core."""
    from inference.live_monitor import run_live_monitor, run_simulated_monitor

    if args.simulated:
        run_simulated_monitor()
    else:
        run_live_monitor(
            poll_interval=args.interval,
            max_iterations=args.iterations,
            auto_mitigate=not args.no_mitigate,
        )


def cmd_full(args: argparse.Namespace) -> None:
    """Full pipeline with Ella Core."""
    from core.verify_connectivity import run_full_check
    from simulation.abmm import run_abmm
    from telemetry.collector import collect_telemetry
    from pipeline.train_pipeline import run_training_pipeline
    from inference.live_monitor import run_live_monitor

    # Phase 1: Verify
    print("\n" + "─" * 60)
    print(" PHASE 1: Environment Verification")
    print("─" * 60)
    if not run_full_check():
        print("\n[main] Environment check failed. Fix issues before proceeding.")
        sys.exit(1)

    # Phase 2: Generate baseline data
    print("\n" + "─" * 60)
    print(" PHASE 2: Baseline Data Generation (ABMM)")
    print("─" * 60)
    run_abmm(duration_hours=args.abmm_hours, time_compression=args.time_compression)

    data_path = Path("data/normal_telemetry.csv")
    collect_telemetry(
        duration_seconds=args.collect_duration,
        poll_interval=1.0,
        output_csv=data_path,
    )

    # Phase 3: Train
    print("\n" + "─" * 60)
    print(" PHASE 3: Model Training")
    print("─" * 60)
    run_training_pipeline(
        normal_telemetry_csv=data_path,
        sequence_length=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Phase 4: Monitor
    print("\n" + "─" * 60)
    print(" PHASE 4: Live Monitoring")
    print("─" * 60)
    run_live_monitor(
        poll_interval=args.interval,
        auto_mitigate=not args.no_mitigate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NWDAF — Zero-Touch 5G Signaling Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py --demo                    # Full demo (no external deps)
  python main.py --demo --epochs 20        # Quick demo with fewer epochs
  python main.py --train                   # Train model only
  python main.py --detect --simulated      # Detect on simulated data
  python main.py --full                    # Full pipeline (needs Ella Core)
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true",
                      help="Full demo on simulated data")
    mode.add_argument("--train", action="store_true",
                      help="Train model only")
    mode.add_argument("--detect", action="store_true",
                      help="Run anomaly detection")
    mode.add_argument("--full", action="store_true",
                      help="Full pipeline with Ella Core")

    # Common args
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--seq-len", type=int, default=10,
                        help="Sequence length (default: 10)")
    parser.add_argument("--train-duration", type=float, default=300.0,
                        help="Simulated training data duration in seconds")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to telemetry CSV for training")

    # Detection args
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--no-mitigate", action="store_true")
    parser.add_argument("--simulated", action="store_true")

    # Full pipeline args
    parser.add_argument("--abmm-hours", type=int, default=4)
    parser.add_argument("--time-compression", type=float, default=120.0)
    parser.add_argument("--collect-duration", type=float, default=300.0)

    args = parser.parse_args()

    if args.demo:
        cmd_demo(args)
    elif args.train:
        cmd_train(args)
    elif args.detect:
        cmd_detect(args)
    elif args.full:
        cmd_full(args)


if __name__ == "__main__":
    main()
