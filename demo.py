from pathlib import Path

import argparse
import numpy as np
import pandas as pd
import torch

from dataset.synthetic_signaling import generate_synthetic_dataset
from evaluation.reconstruction_error import compute_reconstruction_errors, load_trained_model
from evaluation.results_summary import summarize_results
from evaluation.threshold import select_threshold
from preprocessing.data_split import split_train_test_sequences
from training.train_model import train_lstm_autoencoder
from visualization.anomaly_timeline import plot_anomaly_timeline
from visualization.error_plot import plot_reconstruction_error_distribution
from visualization.error_timeseries import plot_error_timeseries
from visualization.roc_curve import plot_roc_curve
from visualization.final_plots import plot_confusion_matrix_heatmap


def run_demo(
    csv_path: Path | None = None,
    label_column: str = "Label",
    normal_label: str = "Normal",
    sequence_length: int = 10,
    batch_size: int = 32,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    threshold_percentile: float = 95.0,
    use_synthetic: bool = False,
) -> None:
    """
    Demo script that runs the complete anomaly detection pipeline.

    Steps:
    1. Load dataset
    2. Preprocess features & generate sequences
    3. Train or load LSTM autoencoder
    4. Compute reconstruction errors
    5. Detect anomalies
    6. Display summary metrics
    7. Launch visualization plots
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1–2. Data source: CSV or fully synthetic
    # ------------------------------------------------------------------
    if use_synthetic:
        print("\n[1/7] Generating synthetic signaling dataset (no CSV)...")
        X_train, X_test, y_test = generate_synthetic_dataset(
            num_train_normal=500,
            num_test_attack=200,
            sequence_length=sequence_length,
            num_features=4,
        )
        print(f"Synthetic normal training sequences: {X_train.shape}")
        print(f"Synthetic attack test sequences   : {X_test.shape}")
    else:
        if csv_path is None:
            raise ValueError("csv_path must be provided when not using synthetic data.")

        print("\n[1/7] Loading dataset...")
        df = pd.read_csv(csv_path)
        print(f"Raw dataset shape: {df.shape}")

        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset.")

        # ------------------------------------------------------------------
        # 2. Preprocess features & generate sequences
        # ------------------------------------------------------------------
        print("\n[2/7] Cleaning data and generating LSTM sequences...")
        df_clean = df.dropna().drop_duplicates()
        print(f"After cleaning: {df_clean.shape}")

        X_train, X_test, y_test = split_train_test_sequences(
            df_clean,
            label_column=label_column,
            normal_label=normal_label,
            sequence_length=sequence_length,
        )

    print(f"Normal training sequences (X_train): {X_train.shape}")
    print(f"Attack test sequences (X_test): {X_test.shape}")

    # ------------------------------------------------------------------
    # 3. Train or load LSTM autoencoder
    # ------------------------------------------------------------------
    print("\n[3/7] Preparing LSTM autoencoder model...")
    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "models"
    model_path = models_dir / "lstm_autoencoder.pth"

    if model_path.is_file():
        print(f"Found trained model at {model_path}. Loading...")
        num_features = X_train.shape[-1]
        model = load_trained_model(
            model_path=model_path,
            num_features=num_features,
            device=device,
        )
    else:
        if use_synthetic:
            print("No trained model found. Training a new one on synthetic data for the demo...")
            # Train directly on X_train without any CSV.
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset

            num_features = X_train.shape[-1]
            model = torch.nn.Sequential()  # placeholder for type; will be overwritten
            from models.lstm_autoencoder import LSTMAutoencoder

            model = LSTMAutoencoder(num_features=num_features).to(device)
            criterion = nn.MSELoss(reduction="mean")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            X_train_tensor = torch.from_numpy(X_train).float().to(device)
            train_dataset = TensorDataset(X_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            model.train()
            for epoch in range(1, epochs + 1):
                epoch_loss = 0.0
                num_samples = 0

                for (batch_x,) in train_loader:
                    batch_x = batch_x.to(device)
                    batch_size_actual = batch_x.size(0)

                    recon, _ = model(batch_x)
                    loss = criterion(recon, batch_x)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * batch_size_actual
                    num_samples += batch_size_actual

                avg_loss = epoch_loss / max(num_samples, 1)
                print(f"[Synthetic] Epoch {epoch}/{epochs} - Training Loss: {avg_loss:.6f}")

            models_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
        else:
            print("No trained model found. Training a new one for the demo...")
            model = train_lstm_autoencoder(
                csv_path=csv_path,
                label_column=label_column,
                normal_label=normal_label,
                sequence_length=sequence_length,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
            )

    # ------------------------------------------------------------------
    # 4. Compute reconstruction errors
    # ------------------------------------------------------------------
    print("\n[4/7] Computing reconstruction errors...")
    errors_normal = compute_reconstruction_errors(model, X_train, device=device)
    errors_attack = compute_reconstruction_errors(model, X_test, device=device)

    print(
        f"Normal sequences: {len(errors_normal)} | "
        f"mean error = {errors_normal.mean():.6f}, std = {errors_normal.std():.6f}"
    )
    print(
        f"Attack sequences: {len(errors_attack)} | "
        f"mean error = {errors_attack.mean():.6f}, std = {errors_attack.std():.6f}"
    )

    # ------------------------------------------------------------------
    # 5. Detect anomalies
    # ------------------------------------------------------------------
    print("\n[5/7] Selecting threshold and detecting anomalies...")
    threshold = select_threshold(
        errors_normal,
        method="percentile",
        percentile=threshold_percentile,
    )

    errors_all = np.concatenate([errors_normal, errors_attack], axis=0)
    y_true = np.concatenate(
        [
            np.zeros_like(errors_normal, dtype=int),
            np.ones_like(errors_attack, dtype=int),
        ],
        axis=0,
    )
    y_pred = (errors_all > threshold).astype(int)

    # ------------------------------------------------------------------
    # 6. Display summary
    # ------------------------------------------------------------------
    print("\n[6/7] Summary of detection results:")
    summary = summarize_results(y_true, y_pred)

    # ------------------------------------------------------------------
    # 7. Visualization plots
    # ------------------------------------------------------------------
    print("\n[7/7] Generating visualization plots (figures will open)...")

    # Reconstruction error distribution (normal + attack)
    plot_reconstruction_error_distribution(
        errors_all,
        threshold=threshold,
        title="Reconstruction Error Distribution (Demo)",
    )

    # Error vs time
    plot_error_timeseries(
        errors_all,
        threshold=threshold,
        timestamps=None,
        title="Reconstruction Error vs Time (Demo)",
        xlabel="Sequence Index",
    )

    # Anomaly timeline
    plot_anomaly_timeline(
        errors_all,
        y_pred,
        timestamps=None,
        threshold=threshold,
        title="Anomaly Detection Timeline (Demo)",
        xlabel="Sequence Index",
    )

    # ROC curve
    auc_value = plot_roc_curve(
        y_true,
        errors_all,
        title="ROC Curve for LSTM Autoencoder (Demo)",
    )

    # Confusion matrix heatmap
    plot_confusion_matrix_heatmap(
        y_true,
        y_pred,
        title="Confusion Matrix (Demo)",
        labels=["Normal (0)", "Anomaly (1)"],
    )

    print("\nDemo completed.")
    print(f"ROC AUC (demo): {auc_value:.4f}")
    print("Metrics summary dict:", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Demo script for LSTM autoencoder-based network anomaly detection.\n"
            "Use --synthetic to run without any external dataset."
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
        default=20,
        help="Number of training epochs if model needs training (default: 20).",
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
        help="Percentile of normal reconstruction errors used to set anomaly threshold (default: 95).",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path) if args.csv_path else None
    if not args.synthetic:
        if csv_path is None or not csv_path.is_file():
            raise FileNotFoundError(
                "CSV file not found. Either provide a valid path or use --synthetic to run without a dataset."
            )

    run_demo(
        csv_path=csv_path,
        label_column=args.label_column,
        normal_label=args.normal_label,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        threshold_percentile=args.threshold_percentile,
        use_synthetic=args.synthetic,
    )

