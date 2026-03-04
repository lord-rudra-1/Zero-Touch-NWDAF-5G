from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.lstm_autoencoder import LSTMAutoencoder


def load_trained_model(
    model_path: Path,
    num_features: int,
    device: Optional[torch.device] = None,
) -> LSTMAutoencoder:
    """
    Load a trained LSTM autoencoder from disk.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMAutoencoder(num_features=num_features)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def compute_reconstruction_errors(
    model: LSTMAutoencoder,
    sequences: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Compute reconstruction error (MSE) for each sequence.

    Parameters
    ----------
    model : LSTMAutoencoder
        Trained LSTM autoencoder.
    sequences : np.ndarray
        Array of shape (num_sequences, sequence_length, num_features).

    Returns
    -------
    np.ndarray
        1D array of reconstruction errors (MSE per sequence),
        shape (num_sequences,).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(sequences).float().to(device)
        recon, _ = model(x)

        # Mean squared error per sequence:
        # average over (sequence_length, num_features) for each item in batch.
        errors = torch.mean((recon - x) ** 2, dim=(1, 2))

    return errors.cpu().numpy()


def plot_error_histogram(
    errors: np.ndarray,
    bins: int = 50,
    title: str = "Reconstruction Error Histogram",
    threshold: Optional[float] = None,
) -> None:
    """
    Plot a histogram of reconstruction errors.

    Parameters
    ----------
    errors : np.ndarray
        1D array of reconstruction errors.
    bins : int, optional
        Number of histogram bins (default: 50).
    title : str, optional
        Plot title.
    threshold : float, optional
        Optional vertical line indicating an anomaly threshold.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=bins, alpha=0.7, color="steelblue", edgecolor="black")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Frequency")
    plt.title(title)

    if threshold is not None:
        plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.4f}")
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage with dummy data.
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute reconstruction errors using a trained LSTM autoencoder.",
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the trained model file (e.g., models/lstm_autoencoder.pth).",
    )
    parser.add_argument(
        "--num-features",
        type=int,
        required=True,
        help="Number of features in each timestep (must match training).",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=100,
        help="Number of dummy sequences to generate for the example.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="Sequence length (must match training).",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(
        model_path=model_path,
        num_features=args.num_features,
        device=device,
    )

    # Create dummy sequences just to demonstrate error computation.
    dummy_sequences = np.random.rand(args.num_sequences, args.sequence_length, args.num_features).astype(
        np.float32
    )

    errors = compute_reconstruction_errors(model, dummy_sequences, device=device)
    print(f"Computed reconstruction errors for {len(errors)} sequences.")
    print(f"Mean error: {errors.mean():.6f}, Std: {errors.std():.6f}")

    # Plot histogram of reconstruction errors.
    plot_error_histogram(errors, title="Dummy Reconstruction Error Histogram")

