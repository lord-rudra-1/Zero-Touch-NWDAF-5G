"""
Live anomaly monitor — Phase 4 real‑time detection & zero‑touch mitigation.

Runs an infinite loop that:
  1. Polls Ella Core's ``/metrics`` every second
  2. Maintains a sliding window of the last ``T`` samples
  3. Feeds each window into the pre-trained LSTM‑Autoencoder
  4. Computes reconstruction error and compares to threshold
  5. If anomaly detected → triggers mitigation via REST API

Can also run in **simulated** mode for demos without Ella Core.
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from core.ella_config import EllaConfig, get_config
from evaluation.reconstruction_error import compute_reconstruction_errors, load_trained_model
from inference.mitigation import block_subscriber, save_mitigation_log
from telemetry.collector import fetch_metrics, DEFAULT_METRIC_NAMES
from telemetry.preprocessor import DEFAULT_FEATURE_COLUMNS


def _load_scaler_params(scaler_path: Path) -> dict:
    """Load scaler parameters saved during training."""
    with scaler_path.open() as f:
        return json.load(f)


def _normalise_with_params(
    raw: np.ndarray,
    scaler_params: dict,
) -> np.ndarray:
    """
    Normalise a 2-D array using saved MinMaxScaler parameters.
    Avoids needing the sklearn scaler object at inference time.
    """
    data_min = np.array(scaler_params["min"], dtype=np.float32)
    data_max = np.array(scaler_params["max"], dtype=np.float32)
    scale = np.array(scaler_params["scale"], dtype=np.float32)

    # MinMax formula: (X - min) / (max - min) = (X - min) * scale
    result = (raw - data_min) * scale
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def run_live_monitor(
    config: Optional[EllaConfig] = None,
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
    poll_interval: float = 1.0,
    max_iterations: Optional[int] = None,
    auto_mitigate: bool = True,
    verbose: bool = True,
) -> None:
    """
    Start the real-time anomaly detection loop.

    Parameters
    ----------
    config : EllaConfig, optional
        Ella Core configuration.
    model_path : Path, optional
        Path to trained model (default: models/lstm_autoencoder.pth).
    metadata_path : Path, optional
        Path to training metadata JSON (contains threshold).
    scaler_path : Path, optional
        Path to scaler parameters JSON.
    poll_interval : float
        Seconds between metric polls.
    max_iterations : int, optional
        Stop after this many iterations (None = infinite).
    auto_mitigate : bool
        Whether to automatically block offending subscribers.
    verbose : bool
        Print each detection result.
    """
    cfg = config or get_config()
    project_root = Path(__file__).resolve().parents[1]

    # Resolve paths
    if model_path is None:
        model_path = project_root / "models" / "lstm_autoencoder.pth"
    if metadata_path is None:
        metadata_path = project_root / "models" / "training_metadata.json"
    if scaler_path is None:
        scaler_path = project_root / "models" / "scaler_params.json"

    # Load model
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    with metadata_path.open() as f:
        metadata = json.load(f)

    threshold = metadata["threshold"]
    sequence_length = metadata["sequence_length"]
    num_features = metadata["num_features"]

    scaler_params = _load_scaler_params(scaler_path)

    model = load_trained_model(
        model_path=model_path,
        num_features=num_features,
    )

    print(f"\n{'='*60}")
    print(f" NWDAF Live Monitor — Zero‑Touch Detection")
    print(f"{'='*60}")
    print(f" Model       : {model_path}")
    print(f" Threshold   : {threshold:.6f}")
    print(f" Seq length  : {sequence_length}")
    print(f" Num features: {num_features}")
    print(f" Polling     : {poll_interval}s")
    print(f" Mitigation  : {'AUTO' if auto_mitigate else 'DISABLED'}")
    print(f"{'='*60}\n")
    print(f"{'Time':>12}  |  {'Error':>10}  |  {'Status':>10}  |  {'Action'}")
    print(f"{'-'*12}--+--{'-'*10}--+--{'-'*10}--+--{'-'*20}")

    # Sliding window buffer
    window: deque = deque(maxlen=sequence_length)
    feature_names = DEFAULT_FEATURE_COLUMNS[:num_features]

    iteration = 0
    anomaly_count = 0
    total_count = 0

    try:
        while max_iterations is None or iteration < max_iterations:
            tick = time.time()

            # 1. Fetch latest metrics
            snapshot = fetch_metrics(config=cfg, metric_names=feature_names)
            feature_vec = np.array(
                [snapshot.get(m, 0.0) for m in feature_names],
                dtype=np.float32,
            )

            window.append(feature_vec)

            if len(window) < sequence_length:
                if verbose:
                    now = datetime.now().strftime("%H:%M:%S")
                    print(f"{now:>12}  |  {'buffering':>10}  |  "
                          f"{'wait':>10}  |  "
                          f"({len(window)}/{sequence_length} samples)")
                iteration += 1
                time.sleep(max(0, poll_interval - (time.time() - tick)))
                continue

            # 2. Build sequence and normalise
            raw_seq = np.array(list(window), dtype=np.float32)  # (T, F)
            norm_seq = _normalise_with_params(
                raw_seq, scaler_params
            )  # (T, F)
            batch = norm_seq[np.newaxis, ...]  # (1, T, F)

            # 3. Compute reconstruction error
            error = compute_reconstruction_errors(model, batch)[0]
            total_count += 1

            # 4. Detect anomaly
            is_anomaly = error > threshold
            status = "⚠ ANOMALY" if is_anomaly else "  Normal"

            action = ""
            if is_anomaly:
                anomaly_count += 1
                if auto_mitigate:
                    # Extract the likely offending IMSI
                    imsi = cfg.subscriber_imsi  # In production, extract from telemetry
                    block_subscriber(imsi, config=cfg)
                    action = f"BLOCKED {imsi}"

            if verbose:
                now = datetime.now().strftime("%H:%M:%S")
                print(f"{now:>12}  |  {error:>10.6f}  |  {status:>10}  |  {action}")

            iteration += 1

            # Sleep for remainder of interval
            elapsed = time.time() - tick
            time.sleep(max(0, poll_interval - elapsed))

    except KeyboardInterrupt:
        print("\n[monitor] Interrupted by user")

    # Summary
    print(f"\n{'='*60}")
    print(f" Detection Summary")
    print(f"{'='*60}")
    print(f" Total windows analysed : {total_count}")
    print(f" Anomalies detected     : {anomaly_count}")
    print(f" Anomaly rate           : "
          f"{(anomaly_count/max(total_count,1))*100:.1f}%")

    if auto_mitigate:
        save_mitigation_log()

    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Simulated mode for demos
# ---------------------------------------------------------------------------

def run_simulated_monitor(
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
    normal_samples: int = 200,
    attack_samples: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Run anomaly detection on simulated telemetry (no Ella Core needed).

    Generates normal + attack data, feeds it through the model, and
    reports detection performance.

    Returns a summary dict.
    """
    from telemetry.collector import generate_simulated_telemetry
    from simulation.attack_generator import generate_attack_telemetry
    from telemetry.preprocessor import preprocess_telemetry, normalise_features, extract_features
    from evaluation.metrics import compute_metrics
    from evaluation.threshold import select_threshold

    project_root = Path(__file__).resolve().parents[1]
    if model_path is None:
        model_path = project_root / "models" / "lstm_autoencoder.pth"
    if metadata_path is None:
        metadata_path = project_root / "models" / "training_metadata.json"
    if scaler_path is None:
        scaler_path = project_root / "models" / "scaler_params.json"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")

    with metadata_path.open() as f:
        metadata = json.load(f)

    threshold = metadata["threshold"]
    sequence_length = metadata["sequence_length"]
    num_features = metadata["num_features"]

    # Load model
    model = load_trained_model(model_path=model_path, num_features=num_features)
    scaler_params = _load_scaler_params(scaler_path)

    # Generate test data
    print("\n[sim-monitor] Generating normal test data...")
    df_normal = generate_simulated_telemetry(
        duration_seconds=normal_samples, poll_interval=1.0,
        attack_start_frac=1.0,  # all normal
    )
    print("[sim-monitor] Generating attack test data...")
    df_attack = generate_attack_telemetry(
        duration_seconds=attack_samples, poll_interval=1.0,
    )

    # Preprocess with saved scaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.data_min_ = np.array(scaler_params["min"])
    scaler.data_max_ = np.array(scaler_params["max"])
    scaler.scale_ = np.array(scaler_params["scale"])
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.n_features_in_ = num_features
    scaler.feature_range = (0, 1)
    scaler.min_ = -scaler.data_min_ * scaler.scale_
    scaler.n_samples_seen_ = 1

    X_normal, _ = preprocess_telemetry(df_normal, sequence_length=sequence_length, scaler=scaler)
    X_attack, _ = preprocess_telemetry(df_attack, sequence_length=sequence_length, scaler=scaler)

    # Compute errors
    errors_normal = compute_reconstruction_errors(model, X_normal)
    errors_attack = compute_reconstruction_errors(model, X_attack)

    errors_all = np.concatenate([errors_normal, errors_attack])
    y_true = np.concatenate([
        np.zeros(len(errors_normal), dtype=int),
        np.ones(len(errors_attack), dtype=int),
    ])
    y_pred = (errors_all > threshold).astype(int)

    metrics = compute_metrics(y_true, y_pred)

    # Print results table
    print(f"\n{'='*60}")
    print(f" Simulated Detection Results")
    print(f"{'='*60}")
    print(f" Normal sequences  : {len(errors_normal)}")
    print(f" Attack sequences  : {len(errors_attack)}")
    print(f" Threshold         : {threshold:.6f}")
    print(f" Normal mean error : {errors_normal.mean():.6f}")
    print(f" Attack mean error : {errors_attack.mean():.6f}")
    print(f" Precision         : {metrics['precision']:.4f}")
    print(f" Recall            : {metrics['recall']:.4f}")
    print(f" F1 Score          : {metrics['f1_score']:.4f}")
    print(f"{'='*60}")

    return {
        "threshold": threshold,
        "normal_mean_error": float(errors_normal.mean()),
        "attack_mean_error": float(errors_attack.mean()),
        **metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NWDAF Live Anomaly Monitor"
    )
    parser.add_argument("--simulated", action="store_true",
                        help="Run on simulated data instead of live Ella Core")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--no-mitigate", action="store_true")
    args = parser.parse_args()

    if args.simulated:
        run_simulated_monitor()
    else:
        run_live_monitor(
            poll_interval=args.interval,
            max_iterations=args.iterations,
            auto_mitigate=not args.no_mitigate,
        )
