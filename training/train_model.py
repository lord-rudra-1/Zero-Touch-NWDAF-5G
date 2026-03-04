from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.lstm_autoencoder import LSTMAutoencoder
from preprocessing.data_split import split_train_test_sequences


def load_sequences_from_csv(
    csv_path: Path,
    label_column: str = "Label",
    normal_label: str = "Normal",
    sequence_length: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load network traffic data from CSV and convert to LSTM-ready sequences.

    Returns:
        X_train_tensor: normal traffic sequences for training
        X_test_tensor: attack sequences for evaluation
        y_test_tensor: labels for attack sequences
    """
    df = pd.read_csv(csv_path)

    X_train, X_test, y_test = split_train_test_sequences(
        df,
        label_column=label_column,
        normal_label=normal_label,
        sequence_length=sequence_length,
    )

    X_train_tensor = torch.from_numpy(X_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test)

    return X_train_tensor, X_test_tensor, y_test_tensor


def train_lstm_autoencoder(
    csv_path: Path,
    label_column: str = "Label",
    normal_label: str = "Normal",
    sequence_length: int = 10,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 1e-3,
) -> LSTMAutoencoder:
    """
    Train an LSTM autoencoder on normal network traffic sequences.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load preprocessed sequences from the preprocessing pipeline.
    X_train, X_test, y_test = load_sequences_from_csv(
        csv_path,
        label_column=label_column,
        normal_label=normal_label,
        sequence_length=sequence_length,
    )

    print(f"Training sequences shape: {X_train.shape}")
    print(f"Test sequences shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    num_features = X_train.shape[-1]

    # Initialize model, loss function (MSE), and optimizer (Adam).
    model = LSTMAutoencoder(num_features=num_features).to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader for batching.
    train_dataset = TensorDataset(X_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_samples = 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            batch_size_actual = batch_x.size(0)

            # Forward pass through the autoencoder.
            recon, _ = model(batch_x)

            # Compute reconstruction loss.
            loss = criterion(recon, batch_x)

            # Backward pass and optimizer step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size_actual
            num_samples += batch_size_actual

        avg_loss = epoch_loss / max(num_samples, 1)
        print(f"Epoch {epoch}/{epochs} - Training Loss: {avg_loss:.6f}")

    return model


def save_model(model: LSTMAutoencoder, model_filename: str = "lstm_autoencoder.pth") -> Path:
    """
    Save the trained model's state_dict under the project-level models directory.
    """
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / model_filename
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to: {model_path}")
    return model_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train an LSTM autoencoder for network anomaly detection.",
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the network traffic CSV file.",
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

    args = parser.parse_args()
    csv_path = Path(args.csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    model = train_lstm_autoencoder(
        csv_path=csv_path,
        label_column=args.label_column,
        normal_label=args.normal_label,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    # Save the trained model to models/lstm_autoencoder.pth
    save_model(model, model_filename="lstm_autoencoder.pth")

