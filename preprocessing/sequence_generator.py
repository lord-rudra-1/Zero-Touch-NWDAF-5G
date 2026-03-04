from typing import Optional

import numpy as np


def generate_lstm_sequences(
    data: np.ndarray,
    sequence_length: int = 10,
) -> np.ndarray:
    """
    Convert tabular network traffic data into time sequences for LSTM training.

    Parameters
    ----------
    data : np.ndarray
        Normalized 2D array of shape (num_samples, num_features).
    sequence_length : int, optional
        Length of the time sequence window for LSTM, by default 10.

    Returns
    -------
    np.ndarray
        3D array of shape (num_sequences, sequence_length, num_features)
        created using a sliding window over the input data.
    """
    if data.ndim != 2:
        raise ValueError(
            f"`data` must be a 2D array of shape (num_samples, num_features), "
            f"got shape {data.shape} with {data.ndim} dimensions."
        )

    num_samples, num_features = data.shape

    if num_samples < sequence_length:
        raise ValueError(
            f"Not enough samples ({num_samples}) to create sequences of "
            f"length {sequence_length}."
        )

    # Number of sequences produced by a sliding window of size `sequence_length`
    # moving one step at a time over `num_samples`.
    num_sequences = num_samples - sequence_length + 1

    # Pre-allocate the 3D array for efficiency.
    sequences = np.empty((num_sequences, sequence_length, num_features), dtype=data.dtype)

    # Sliding window: for each starting index i, take data[i : i + sequence_length]
    for i in range(num_sequences):
        sequences[i] = data[i : i + sequence_length]

    return sequences


if __name__ == "__main__":
    # Example usage
    #
    # Suppose we have normalized network traffic data with 5,009 samples
    # and 12 numerical features. We want to convert it into sequences of
    # length 10 for an LSTM autoencoder.

    num_samples = 5010
    num_features = 12
    sequence_length = 10

    # Create dummy normalized data in [0, 1] for demonstration purposes.
    example_data = np.random.rand(num_samples, num_features)

    # Generate LSTM-ready sequences.
    sequences = generate_lstm_sequences(example_data, sequence_length=sequence_length)

    print(f"Input shape: {example_data.shape}")
    print(f"Output shape (num_sequences, sequence_length, num_features): {sequences.shape}")

