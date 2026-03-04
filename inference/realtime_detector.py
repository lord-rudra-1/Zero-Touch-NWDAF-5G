from pathlib import Path
from time import sleep
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from dataset.synthetic_signaling import generate_synthetic_dataset
from evaluation.reconstruction_error import compute_reconstruction_errors, load_trained_model
from evaluation.threshold import select_threshold
from preprocessing.data_split import split_train_test_sequences


def simulate_realtime_detection(
    model,
    sequences: np.ndarray,
    threshold: float,
    timestamps: Optional[Sequence[Union[str, float]]] = None,
    delay_seconds: float = 0.5,
    device: Optional[torch.device] = None,
) -> None:
    """
    Simulate real-time anomaly detection by streaming sequences one by one.

    For each sequence:
      - compute reconstruction error
      - compare with threshold
      - print time | error | status
      - wait a short delay to mimic streaming
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if sequences.ndim != 3:
        raise ValueError(
            f"`sequences` must be a 3D array of shape "
            f"(num_sequences, sequence_length, num_features), got {sequences.shape}."
        )

    num_sequences = sequences.shape[0]

    if timestamps is not None and len(timestamps) != num_sequences:
        raise ValueError(
            f"Length of timestamps ({len(timestamps)}) does not match number of sequences ({num_sequences})."
        )

    print("Time | Error | Status")
    print("----------------------")

    for i in range(num_sequences):
        seq = sequences[i : i + 1]  # shape (1, seq_len, num_features)

        # Compute reconstruction error for this single sequence.
        error_arr = compute_reconstruction_errors(model, seq, device=device)
        error = float(error_arr[0])

        status = "Anomaly" if error > threshold else "Normal"

        if timestamps is not None:
            t = timestamps[i]
        else:
            # Fallback to simple index-based "time" string.
            t = f"{i:06d}"

        print(f"{t} | {error:.4f} | {status}")

        # Short delay to simulate real-time streaming.
        sleep(delay_seconds)


if __name__ == "__main__":
    # CLI: simulate real-time detection from either a CSV dataset or synthetic data.
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Simulate real-time anomaly detection using a trained LSTM autoencoder.\n"
            "Use --synthetic to stream fully synthetic signaling sequences."
        ),
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=str,
        help="Path to the network traffic CSV file (ignored if --synthetic is set).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use internally generated synthetic signaling data instead of a CSV dataset.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/lstm_autoencoder.pth",
        help="Path to the trained model file (default: models/lstm_autoencoder.pth).",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="Label",
        help="Name of the label column indicating normal vs attack (CSV mode only).",
    )
    parser.add_argument(
        "--normal-label",
        type=str,
        default="Normal",
        help="Value in the label column that indicates normal traffic (CSV mode only).",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="Sequence length for the LSTM autoencoder (default: 10).",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile for threshold selection based on normal errors (default: 95).",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.5,
        help="Delay between sequence predictions to simulate real time (default: 0.5).",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path) if args.csv_path else None
    model_path = Path(args.model_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Data source: CSV or synthetic
    # ------------------------------------------------------------------
    if args.synthetic:
        print("Using synthetic signaling data for real-time simulation...")
        X_train, X_test, y_test = generate_synthetic_dataset(
            num_train_normal=200,
            num_test_attack=100,
            sequence_length=args.sequence_length,
            num_features=4,
        )
        sequences_all = np.concatenate([X_train, X_test], axis=0)
    else:
        if csv_path is None or not csv_path.is_file():
            raise FileNotFoundError(
                "CSV file not found. Either provide a valid path or use --synthetic to run without a dataset."
            )

        # Load and clean dataset.
        df_raw = pd.read_csv(csv_path)
        df_clean = df_raw.dropna().drop_duplicates()

        if args.label_column not in df_clean.columns:
            raise ValueError(
                f"Label column '{args.label_column}' not found in dataset. "
                "Make sure the CSV includes this column."
            )

        # Generate sequences (normal + attack).
        X_train, X_test, y_test = split_train_test_sequences(
            df_clean,
            label_column=args.label_column,
            normal_label=args.normal_label,
            sequence_length=args.sequence_length,
        )
        sequences_all = np.concatenate([X_train, X_test], axis=0)

    # Load trained model.
    num_features = sequences_all.shape[-1]
    model = load_trained_model(
        model_path=model_path,
        num_features=num_features,
        device=device,
    )

    # Compute threshold based on normal errors (use X_train as normal reference).
    normal_errors = compute_reconstruction_errors(model, X_train, device=device)
    threshold = select_threshold(
        normal_errors,
        method="percentile",
        percentile=args.threshold_percentile,
    )

    # Simple synthetic timestamps based on index (can be replaced with real times).
    timestamps = [f"{i:02d}:{(i % 60):02d}:{(i * 2) % 60:02d}" for i in range(sequences_all.shape[0])]

    simulate_realtime_detection(
        model=model,
        sequences=sequences_all,
        threshold=threshold,
        timestamps=timestamps,
        delay_seconds=args.delay_seconds,
        device=device,
    )


