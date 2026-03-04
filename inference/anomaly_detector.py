from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from evaluation.reconstruction_error import compute_reconstruction_errors, load_trained_model
from models.lstm_autoencoder import LSTMAutoencoder


def detect_anomalies(
    model_path: Union[str, Path],
    sequences: np.ndarray,
    timestamps: Sequence[object],
    threshold: float,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Perform anomaly detection using a trained LSTM autoencoder.

    Steps:
    1. Load trained model
    2. Input sequences
    3. Reconstruct sequences
    4. Compute reconstruction error
    5. Compare error with threshold
    6. Mark anomaly if error > threshold

    Parameters
    ----------
    model_path : str or Path
        Path to the trained model file (e.g., 'models/lstm_autoencoder.pth').
    sequences : np.ndarray
        Array of shape (num_sequences, sequence_length, num_features).
    timestamps : Sequence
        Iterable of timestamps (strings, datetime objects, etc.) aligned with sequences.
    threshold : float
        Anomaly threshold on reconstruction error.
    device : torch.device, optional
        Device to run inference on (CPU or CUDA). If None, selected automatically.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - 'timestamp'
            - 'reconstruction_error'
            - 'anomaly_flag' (0 for normal, 1 for anomaly)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if sequences.ndim != 3:
        raise ValueError(
            f"`sequences` must be a 3D array of shape "
            f"(num_sequences, sequence_length, num_features), got {sequences.shape}."
        )

    num_sequences = sequences.shape[0]
    if len(timestamps) != num_sequences:
        raise ValueError(
            f"Length of timestamps ({len(timestamps)}) does not match "
            f"number of sequences ({num_sequences})."
        )

    num_features = sequences.shape[-1]
    print(f"[AnomalyDetector] Loading model from {model_path} (num_features={num_features})")

    # 1. Load trained model
    model: LSTMAutoencoder = load_trained_model(
        model_path=model_path,
        num_features=num_features,
        device=device,
    )

    # 2–4. Input sequences -> reconstruct -> compute reconstruction error
    print("[AnomalyDetector] Computing reconstruction errors for input sequences...")
    errors = compute_reconstruction_errors(model, sequences, device=device)

    # 5–6. Compare with threshold and mark anomalies
    anomaly_flags = (errors > threshold).astype(int)
    num_anomalies = int(anomaly_flags.sum())

    print(
        f"[AnomalyDetector] Applied threshold = {threshold:.6f}. "
        f"Detected {num_anomalies} anomalies out of {len(errors)} sequences."
    )

    # Build output DataFrame
    result_df = pd.DataFrame(
        {
            "timestamp": list(timestamps),
            "reconstruction_error": errors,
            "anomaly_flag": anomaly_flags,
        }
    )

    return result_df


if __name__ == "__main__":
    # Example usage with synthetic data to demonstrate output format.
    rng = np.random.default_rng(seed=42)

    num_sequences = 5
    sequence_length = 10
    num_features = 12

    dummy_sequences = rng.random((num_sequences, sequence_length, num_features), dtype=np.float32)
    dummy_timestamps = [f"10:01:0{i+1}" for i in range(num_sequences)]

    # NOTE: For a real run, provide a valid trained model path.
    example_model_path = Path("models/lstm_autoencoder.pth")

    if not example_model_path.is_file():
        print(
            f"Example model path '{example_model_path}' does not exist. "
            "Run training first to generate a trained model."
        )
    else:
        example_threshold = 0.1
        df_results = detect_anomalies(
            model_path=example_model_path,
            sequences=dummy_sequences,
            timestamps=dummy_timestamps,
            threshold=example_threshold,
        )
        print(df_results)

