from typing import Tuple

import numpy as np

from evaluation.attack_simulator import simulate_attacks


def generate_normal_sequences(
    num_sequences: int = 500,
    sequence_length: int = 10,
    num_features: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic "normal" signaling sequences.

    Each feature is a smooth, low-variance time series normalized to [0, 1].
    This avoids any external dataset and provides training data for the
    unsupervised LSTM autoencoder.
    """
    rng = np.random.default_rng(seed)

    # Base gamma-distributed positive values with mild temporal correlation.
    base = rng.gamma(
        shape=2.0,
        scale=0.2,
        size=(num_sequences, sequence_length, num_features),
    ).astype(np.float32)

    # Add a small smooth temporal component to mimic evolving traffic.
    time_trend = np.linspace(0.9, 1.1, num=sequence_length, dtype=np.float32)
    base *= time_trend[None, :, None]

    # Normalize per-feature to [0, 1].
    max_per_feature = base.max(axis=(0, 1), keepdims=True) + 1e-8
    normalized = base / max_per_feature

    return normalized


def generate_synthetic_dataset(
    num_train_normal: int = 500,
    num_test_attack: int = 200,
    sequence_length: int = 10,
    num_features: int = 4,
    attack_mode: str = "packet_flood",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset of normal training sequences and
    attack sequences, fully in-memory and without any CSV file.

    Returns
    -------
    X_train : np.ndarray
        Normal sequences for training, shape (num_train_normal, T, F).
    X_attack : np.ndarray
        Simulated attack sequences for testing, shape (num_test_attack, T, F).
    y_attack : np.ndarray
        Labels for attack sequences (all ones), shape (num_test_attack,).
    """
    # Normal training sequences.
    X_train = generate_normal_sequences(
        num_sequences=num_train_normal,
        sequence_length=sequence_length,
        num_features=num_features,
        seed=seed,
    )

    # Start from normal-like sequences and inject anomalies.
    X_attack_base = generate_normal_sequences(
        num_sequences=num_test_attack,
        sequence_length=sequence_length,
        num_features=num_features,
        seed=seed + 1,
    )

    X_attack = simulate_attacks(
        X_attack_base,
        mode=attack_mode,
        anomaly_ratio=1.0,  # make all test sequences anomalous
    )
    y_attack = np.ones(num_test_attack, dtype=int)

    return X_train, X_attack, y_attack


if __name__ == "__main__":
    # Quick sanity check.
    X_train_demo, X_attack_demo, y_demo = generate_synthetic_dataset()
    print("X_train shape :", X_train_demo.shape)
    print("X_attack shape:", X_attack_demo.shape)
    print("y_attack shape:", y_demo.shape)

