"""
End-to-end training pipeline for the LSTM‑Autoencoder.

Ties together telemetry collection → preprocessing → model training →
threshold calculation → model persistence.

Can run in two modes:
  1. **Live** — collects real telemetry from Ella Core during ABMM
  2. **Simulated** — uses generated synthetic normal data (default)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from evaluation.reconstruction_error import compute_reconstruction_errors
from evaluation.threshold import select_threshold
from telemetry.collector import generate_simulated_telemetry
from telemetry.preprocessor import preprocess_telemetry
from training.train_model import save_model, train_lstm_autoencoder


def run_training_pipeline(
    normal_telemetry_csv: Optional[Path] = None,
    sequence_length: int = 10,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    threshold_percentile: float = 95.0,
    simulated_duration: float = 300.0,
    model_filename: str = "lstm_autoencoder.pth",
) -> Tuple[Path, float, dict]:
    """
    Full training pipeline.

    Steps:
      1. Load or generate normal telemetry data
      2. Preprocess and create LSTM sequences
      3. Train the LSTM‑Autoencoder on normal sequences only
      4. Compute reconstruction errors on the training set
      5. Compute the anomaly threshold
      6. Save model + metadata

    Parameters
    ----------
    normal_telemetry_csv : Path, optional
        Pre-collected normal telemetry CSV.  If ``None``, simulated data
        is generated instead.
    sequence_length, epochs, batch_size, learning_rate :
        Model hyper-parameters.
    threshold_percentile : float
        Percentile of normal errors to set as threshold.
    simulated_duration : float
        Duration for simulated telemetry (seconds), used when no CSV.
    model_filename : str
        Filename for the saved model weights.

    Returns
    -------
    model_path : Path
        Path to the saved ``.pth`` model weights.
    threshold : float
        Computed anomaly threshold.
    metadata : dict
        Summary of training stats.
    """
    import pandas as pd

    # ------------------------------------------------------------------
    # 1. Load or generate normal telemetry
    # ------------------------------------------------------------------
    if normal_telemetry_csv and normal_telemetry_csv.is_file():
        print(f"\n[pipeline] Loading normal telemetry from {normal_telemetry_csv}")
        df_normal = pd.read_csv(normal_telemetry_csv)
    else:
        print(f"\n[pipeline] Generating simulated normal telemetry "
              f"({simulated_duration}s)")
        df_normal = generate_simulated_telemetry(
            duration_seconds=simulated_duration,
            poll_interval=1.0,
            attack_start_frac=1.0,  # 100% normal data
        )

    print(f"[pipeline] Normal telemetry samples: {len(df_normal)}")

    # ------------------------------------------------------------------
    # 2. Preprocess → LSTM sequences
    # ------------------------------------------------------------------
    print(f"\n[pipeline] Preprocessing (seq_len={sequence_length})...")
    X_train, scaler = preprocess_telemetry(
        df_normal,
        sequence_length=sequence_length,
    )
    print(f"[pipeline] Training sequences: {X_train.shape}")

    # ------------------------------------------------------------------
    # 3. Train LSTM‑Autoencoder
    # ------------------------------------------------------------------
    print(f"\n[pipeline] Training LSTM‑Autoencoder "
          f"(epochs={epochs}, bs={batch_size}, lr={learning_rate})")
    model = train_lstm_autoencoder(
        X_train,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    # ------------------------------------------------------------------
    # 4. Compute reconstruction errors on training set
    # ------------------------------------------------------------------
    print("\n[pipeline] Computing reconstruction errors on training data...")
    errors = compute_reconstruction_errors(model, X_train)
    print(f"[pipeline] Errors — mean: {errors.mean():.6f}, "
          f"std: {errors.std():.6f}, max: {errors.max():.6f}")

    # ------------------------------------------------------------------
    # 5. Compute threshold
    # ------------------------------------------------------------------
    threshold = select_threshold(
        errors,
        method="percentile",
        percentile=threshold_percentile,
    )

    # ------------------------------------------------------------------
    # 6. Save model + metadata
    # ------------------------------------------------------------------
    model_path = save_model(model, model_filename=model_filename)

    # Save threshold & scaler info alongside the model
    project_root = Path(__file__).resolve().parents[1]
    meta_path = project_root / "models" / "training_metadata.json"
    metadata = {
        "num_training_samples": int(len(df_normal)),
        "num_training_sequences": int(X_train.shape[0]),
        "sequence_length": sequence_length,
        "num_features": int(X_train.shape[2]),
        "epochs": epochs,
        "threshold": float(threshold),
        "threshold_method": f"percentile_{threshold_percentile}",
        "mean_training_error": float(errors.mean()),
        "max_training_error": float(errors.max()),
    }
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[pipeline] Saved training metadata to {meta_path}")

    # Save scaler parameters for inference
    scaler_path = project_root / "models" / "scaler_params.json"
    scaler_info = {
        "min": scaler.data_min_.tolist(),
        "max": scaler.data_max_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with scaler_path.open("w") as f:
        json.dump(scaler_info, f, indent=2)
    print(f"[pipeline] Saved scaler params to {scaler_path}")

    print(f"\n[pipeline] ✓ Training complete")
    print(f"[pipeline]   Model     : {model_path}")
    print(f"[pipeline]   Threshold : {threshold:.6f}")

    return model_path, threshold, metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the NWDAF LSTM-Autoencoder training pipeline"
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to normal telemetry CSV (or use simulated)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--threshold-pct", type=float, default=95.0)
    parser.add_argument("--duration", type=float, default=300.0,
                        help="Simulated data duration in seconds")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else None

    run_training_pipeline(
        normal_telemetry_csv=csv_path,
        sequence_length=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threshold_percentile=args.threshold_pct,
        simulated_duration=args.duration,
    )
