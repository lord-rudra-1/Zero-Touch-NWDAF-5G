from pathlib import Path

import argparse
import numpy as np
import pandas as pd
import torch

from evaluation.attack_simulator import simulate_attacks
from evaluation.reconstruction_error import (
    compute_reconstruction_errors,
    load_trained_model,
)
from evaluation.threshold import select_threshold
from preprocessing.data_split import split_train_test_sequences


def evaluate_on_simulated_attacks(
    csv_path: Path,
    model_path: Path,
    label_column: str = "Label",
    normal_label: str = "Normal",
    sequence_length: int = 10,
    attack_mode: str = "packet_flood",
    anomaly_ratio: float = 0.5,
    threshold_method: str = "percentile",
    threshold_percentile: float = 95.0,
    threshold_k: float = 3.0,
) -> None:
    """
    Evaluate the trained LSTM autoencoder on simulated attack sequences.

    Steps:
    1. Load trained model
    2. Generate attack sequences from normal traffic
    3. Run anomaly detection
    4. Compute reconstruction errors
    5. Count detected anomalies and print statistics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load and clean dataset
    # ------------------------------------------------------------------
    print("\n[1/5] Loading dataset and extracting normal sequences...")
    df_raw = pd.read_csv(csv_path)
    df_clean = df_raw.dropna().drop_duplicates()

    if label_column not in df_clean.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in dataset. "
            "Make sure the CSV includes this column."
        )

    # Use preprocessing pipeline to obtain normal (training) sequences.
    X_train, _, _ = split_train_test_sequences(
        df_clean,
        label_column=label_column,
        normal_label=normal_label,
        sequence_length=sequence_length,
    )

    print(f"Normal training sequences shape (X_train): {X_train.shape}")

    num_sequences, seq_len, num_features = X_train.shape

    # ------------------------------------------------------------------
    # Load trained model
    # ------------------------------------------------------------------
    print("\n[2/5] Loading trained LSTM autoencoder...")
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found at: {model_path}")

    model = load_trained_model(
        model_path=model_path,
        num_features=num_features,
        device=device,
    )

    # ------------------------------------------------------------------
    # Compute threshold from normal reconstruction errors
    # ------------------------------------------------------------------
    print("\n[3/5] Computing threshold from normal reconstruction errors...")
    normal_errors = compute_reconstruction_errors(model, X_train, device=device)
    threshold = select_threshold(
        normal_errors,
        method=threshold_method,
        percentile=threshold_percentile,
        k=threshold_k,
    )

    # ------------------------------------------------------------------
    # Generate simulated attack sequences
    # ------------------------------------------------------------------
    print("\n[4/5] Generating simulated attack sequences...")
    attack_sequences = simulate_attacks(
        X_train,
        mode=attack_mode,  # 'packet_flood', 'abnormal_timing', or 'abnormal_size'
        anomaly_ratio=anomaly_ratio,
    )

    print(f"Simulated attack sequences shape: {attack_sequences.shape}")

    # ------------------------------------------------------------------
    # Run anomaly detection and compute statistics
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluating detection performance on simulated attacks...")
    attack_errors = compute_reconstruction_errors(model, attack_sequences, device=device)

    # All simulated sequences are treated as anomalies here.
    anomaly_flags = (attack_errors > threshold).astype(int)

    total_sequences = attack_sequences.shape[0]
    detected_anomalies = int(anomaly_flags.sum())
    detection_rate = (detected_anomalies / total_sequences) * 100.0 if total_sequences > 0 else 0.0

    print("\nDetection statistics on simulated attacks:")
    print(f"Total sequences: {total_sequences}")
    print(f"Detected anomalies: {detected_anomalies}")
    print(f"Detection rate: {detection_rate:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained LSTM autoencoder on simulated attack sequences.",
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the network traffic CSV file containing a label column.",
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
        help="Name of the label column indicating normal vs attack (default: 'Label').",
    )
    parser.add_argument(
        "--normal-label",
        type=str,
        default="Normal",
        help="Value in the label column that indicates normal traffic (default: 'Normal').",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="Sequence length for the LSTM autoencoder (default: 10).",
    )
    parser.add_argument(
        "--attack-mode",
        type=str,
        default="packet_flood",
        choices=["packet_flood", "abnormal_timing", "abnormal_size"],
        help="Type of simulated attack to generate.",
    )
    parser.add_argument(
        "--anomaly-ratio",
        type=float,
        default=0.5,
        help="Fraction of normal sequences to convert into anomalous sequences (default: 0.5).",
    )
    parser.add_argument(
        "--threshold-method",
        type=str,
        default="percentile",
        choices=["percentile", "statistical"],
        help="Method used to compute anomaly threshold.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile for the percentile-based threshold (default: 95).",
    )
    parser.add_argument(
        "--threshold-k",
        type=float,
        default=3.0,
        help="k value for statistical threshold: mean + k * std (default: 3.0).",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    model_path = Path(args.model_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    evaluate_on_simulated_attacks(
        csv_path=csv_path,
        model_path=model_path,
        label_column=args.label_column,
        normal_label=args.normal_label,
        sequence_length=args.sequence_length,
        attack_mode=args.attack_mode,
        anomaly_ratio=args.anomaly_ratio,
        threshold_method=args.threshold_method,
        threshold_percentile=args.threshold_percentile,
        threshold_k=args.threshold_k,
    )

