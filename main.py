from pathlib import Path

import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import compute_metrics
from evaluation.reconstruction_error import compute_reconstruction_errors, load_trained_model
from evaluation.threshold import select_threshold
from models.lstm_autoencoder import LSTMAutoencoder
from preprocessing.data_split import split_train_test_sequences
from training.train_model import save_model
from visualization.anomaly_timeline import plot_anomaly_timeline
from visualization.error_plot import (
    plot_reconstruction_error_distribution,
    plot_reconstruction_error_over_time,
)
from visualization.error_timeseries import plot_error_timeseries
from visualization.roc_curve import plot_roc_curve


def run_pipeline(
    csv_path: Path,
    label_column: str = "Label",
    normal_label: str = "Normal",
    sequence_length: int = 10,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    threshold_percentile: float = 95.0,
) -> None:
    """
    End-to-end anomaly detection pipeline using an LSTM autoencoder.

    Steps:
    1. Load dataset
    2. Preprocess / clean data
    3. Perform feature engineering and generate LSTM sequences
    4. Train LSTM autoencoder on normal traffic (or load existing model)
    5. Compute reconstruction errors on normal and attack traffic
    6. Compute anomaly threshold from normal errors
    7. Detect anomalies
    8. Evaluate results (metrics, ROC)
    9. Generate visualization plots
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("\n[1/9] Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"Raw dataset shape: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Preprocess / clean data
    # ------------------------------------------------------------------
    print("\n[2/9] Cleaning dataset (drop NaN + duplicates)...")
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"After cleaning: {df.shape}")

    # ------------------------------------------------------------------
    # 3. Feature engineering + sequence generation
    #    (handled inside split_train_test_sequences which calls
    #     prepare_features and generate_lstm_sequences)
    # ------------------------------------------------------------------
    print("\n[3/9] Performing feature engineering and generating LSTM sequences...")

    X_train, X_test, y_test = split_train_test_sequences(
        df,
        label_column=label_column,
        normal_label=normal_label,
        sequence_length=sequence_length,
    )

    print(f"Training sequences shape (X_train): {X_train.shape}")
    print(f"Test sequences shape (X_test): {X_test.shape}")
    print(f"Test labels shape (y_test): {y_test.shape}")

    # ------------------------------------------------------------------
    # 4. Train LSTM autoencoder (or load existing model)
    # ------------------------------------------------------------------
    num_features = X_train.shape[-1]
    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "models"
    model_path = models_dir / "lstm_autoencoder.pth"

    if model_path.is_file():
        print(f"\n[4/9] Found existing model at {model_path}. Loading and skipping training...")
        model = load_trained_model(model_path=model_path, num_features=num_features, device=device)
    else:
        print("\n[4/9] Training LSTM autoencoder on normal traffic...")

        X_train_tensor = torch.from_numpy(X_train).float().to(device)
        train_dataset = TensorDataset(X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = LSTMAutoencoder(num_features=num_features).to(device)
        criterion = nn.MSELoss(reduction="mean")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            num_samples = 0

            for (batch_x,) in train_loader:
                batch_x = batch_x.to(device)
                batch_size_actual = batch_x.size(0)

                # Forward pass
                recon, _ = model(batch_x)

                # Reconstruction loss (MSE)
                loss = criterion(recon, batch_x)

                # Backward pass + optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_size_actual
                num_samples += batch_size_actual

            avg_loss = epoch_loss / max(num_samples, 1)
            print(f"Epoch {epoch}/{epochs} - Training Loss: {avg_loss:.6f}")

        # Save trained model for later reuse.
        model_path = save_model(model, model_filename="lstm_autoencoder.pth")

    # ------------------------------------------------------------------
    # 5. Compute reconstruction errors on normal and attack sequences
    # ------------------------------------------------------------------
    print("\n[5/9] Computing reconstruction errors on normal and attack sequences...")
    errors_normal = compute_reconstruction_errors(model, X_train, device=device)
    errors_attack = compute_reconstruction_errors(model, X_test, device=device)

    print(f"Normal sequences: {X_train.shape[0]} -> mean error = {errors_normal.mean():.6f}, "
          f"std = {errors_normal.std():.6f}")
    print(f"Attack  sequences: {X_test.shape[0]} -> mean error = {errors_attack.mean():.6f}, "
          f"std = {errors_attack.std():.6f}")

    # ------------------------------------------------------------------
    # 6. Compute anomaly threshold from normal errors
    # ------------------------------------------------------------------
    print("\n[6/9] Computing anomaly threshold from normal reconstruction errors...")
    threshold = select_threshold(
        errors_normal,
        method="percentile",
        percentile=threshold_percentile,
    )

    # ------------------------------------------------------------------
    # 7. Detect anomalies on all sequences
    # ------------------------------------------------------------------
    print("\n[7/9] Detecting anomalies on normal and attack sequences...")
    errors_all = np.concatenate([errors_normal, errors_attack], axis=0)
    y_true = np.concatenate(
        [
            np.zeros_like(errors_normal, dtype=int),  # normal sequences
            np.ones_like(errors_attack, dtype=int),   # attack sequences
        ],
        axis=0,
    )
    y_pred = (errors_all > threshold).astype(int)

    # Compute basic detection statistics
    n_normal = errors_normal.shape[0]
    n_attack = errors_attack.shape[0]
    attack_preds = y_pred[n_normal:]
    detected_attacks = int(attack_preds.sum())
    detection_rate = (detected_attacks / n_attack) * 100.0 if n_attack > 0 else 0.0
    false_positives = int(y_pred[:n_normal].sum())

    # ------------------------------------------------------------------
    # 8. Evaluate results (metrics, ROC)
    # ------------------------------------------------------------------
    print("\n[8/9] Evaluating detection performance...")
    metrics = compute_metrics(y_true, y_pred)

    auc_value = plot_roc_curve(
        y_true,
        errors_all,
        title="ROC Curve for LSTM Autoencoder Anomaly Detection",
    )

    # ------------------------------------------------------------------
    # 9. Generate visualization plots
    # ------------------------------------------------------------------
    print("\n[9/9] Generating visualization plots...")

    plot_reconstruction_error_distribution(
        errors_all,
        threshold=threshold,
        title="Reconstruction Error Distribution (Normal + Attack Sequences)",
    )

    plot_error_timeseries(
        errors_all,
        threshold=threshold,
        timestamps=None,
        title="Reconstruction Error Time Series (Normal + Attack)",
        xlabel="Sequence Index",
    )

    plot_anomaly_timeline(
        errors_all,
        y_pred,
        timestamps=None,
        threshold=threshold,
        title="Anomaly Detection Timeline (Normal + Attack)",
        xlabel="Sequence Index",
    )

    plot_reconstruction_error_over_time(
        errors_attack,
        threshold=threshold,
        title="Reconstruction Error vs Time on Attack Sequences",
        time_label="Attack Sequence Index",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\nSummary statistics:")
    print(
        f"Total sequences: {errors_all.shape[0]} "
        f"(normal={n_normal}, attack={n_attack})"
    )
    print(
        f"Threshold (percentile {threshold_percentile:.1f}): {threshold:.6f}"
    )
    print(f"Detected anomalies (overall): {int(y_pred.sum())}")
    print(
        f"Detected attacks: {detected_attacks} / {n_attack} "
        f"({detection_rate:.2f}%)"
    )
    print(f"False positives on normal data: {false_positives}")
    print(f"ROC AUC: {auc_value:.4f}")
    print("Metrics dict:", metrics)
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end LSTM autoencoder pipeline for network anomaly detection.",
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the network traffic CSV file containing a label column.",
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
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer (default: 1e-3).",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile of reconstruction errors used to set anomaly threshold (default: 95).",
    )

    args = parser.parse_args()
    csv_path = Path(args.csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    run_pipeline(
        csv_path=csv_path,
        label_column=args.label_column,
        normal_label=args.normal_label,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        threshold_percentile=args.threshold_percentile,
    )

