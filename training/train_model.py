from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.lstm_autoencoder import LSTMAutoencoder


def train_lstm_autoencoder(
    X_train: np.ndarray,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
) -> LSTMAutoencoder:
    """
    Train an LSTM autoencoder on *normal* traffic sequences.

    Parameters
    ----------
    X_train : np.ndarray
        3-D array of shape ``(num_sequences, sequence_length, num_features)``
        containing **only** healthy / normal traffic data.
    batch_size : int
        Mini-batch size for training.
    epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate for the Adam optimiser.
    device : torch.device, optional
        Compute device.  Auto-detected if omitted.

    Returns
    -------
    LSTMAutoencoder
        The trained model (in eval mode).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[train] Using device: {device}")
    print(f"[train] Training sequences shape: {X_train.shape}")

    num_features = X_train.shape[-1]

    # Initialise model, loss, and optimiser.
    model = LSTMAutoencoder(num_features=num_features).to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare DataLoader.
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

            # Forward pass through the autoencoder.
            recon, _ = model(batch_x)

            # Compute reconstruction loss.
            loss = criterion(recon, batch_x)

            # Backward pass and optimiser step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size_actual
            num_samples += batch_size_actual

        avg_loss = epoch_loss / max(num_samples, 1)
        print(f"[train] Epoch {epoch}/{epochs} — Loss: {avg_loss:.6f}")

    model.eval()
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
    print(f"[train] Saved trained model to: {model_path}")
    return model_path


if __name__ == "__main__":
    # Quick smoke-test with random data.
    rng = np.random.default_rng(seed=42)
    dummy_data = rng.random((100, 10, 4), dtype=np.float32)

    model = train_lstm_autoencoder(dummy_data, epochs=5)
    path = save_model(model)
    print(f"Smoke-test complete.  Model saved to {path}")

