from typing import Literal, Optional

import numpy as np


def simulate_packet_flood(
    normal_sequences: np.ndarray,
    magnitude: float = 3.0,
    packet_count_feature_index: Optional[int] = None,
    anomaly_ratio: float = 0.2,
) -> np.ndarray:
    """
    Simulate packet-flood anomalies by artificially increasing packet count.

    Parameters
    ----------
    normal_sequences : np.ndarray
        Base sequences, shape (num_sequences, sequence_length, num_features),
        assumed to be normalized to [0, 1].
    magnitude : float, optional
        Multiplicative factor to amplify the selected feature (default: 3.0).
    packet_count_feature_index : int, optional
        Index of the feature representing packet count. If None, uses the first feature (0).
    anomaly_ratio : float, optional
        Fraction of sequences to convert into anomalous packet-flood patterns.

    Returns
    -------
    np.ndarray
        Array of anomalous sequences, same shape as input.
    """
    if normal_sequences.ndim != 3:
        raise ValueError("normal_sequences must have shape (num_sequences, sequence_length, num_features).")

    num_sequences, sequence_length, num_features = normal_sequences.shape
    if packet_count_feature_index is None:
        packet_count_feature_index = 0

    if not (0 <= packet_count_feature_index < num_features):
        raise ValueError("packet_count_feature_index is out of bounds.")

    anomalous = normal_sequences.copy()
    num_anomalies = max(1, int(anomaly_ratio * num_sequences))
    idx = np.random.choice(num_sequences, size=num_anomalies, replace=False)

    # Amplify packet count feature across the full sequence for selected indices.
    anomalous[idx, :, packet_count_feature_index] *= magnitude
    # Clip back to [0, 1] range (assuming normalized inputs).
    anomalous = np.clip(anomalous, 0.0, 1.0)

    return anomalous


def simulate_abnormal_timing(
    normal_sequences: np.ndarray,
    timing_feature_index: Optional[int] = None,
    small_factor: float = 0.1,
    large_factor: float = 5.0,
    anomaly_ratio: float = 0.2,
) -> np.ndarray:
    """
    Simulate abnormal timing by creating very small or very large inter-arrival times.

    Parameters
    ----------
    normal_sequences : np.ndarray
        Base sequences, shape (num_sequences, sequence_length, num_features).
    timing_feature_index : int, optional
        Index of timing / inter-arrival-time feature. If None, uses the second feature (1) if available.
    small_factor : float, optional
        Factor to shrink timing values, simulating bursts (default: 0.1).
    large_factor : float, optional
        Factor to increase timing values, simulating slow / congested traffic (default: 5.0).
    anomaly_ratio : float, optional
        Fraction of sequences to modify.

    Returns
    -------
    np.ndarray
        Array of anomalous sequences.
    """
    if normal_sequences.ndim != 3:
        raise ValueError("normal_sequences must have shape (num_sequences, sequence_length, num_features).")

    num_sequences, sequence_length, num_features = normal_sequences.shape
    if timing_feature_index is None:
        timing_feature_index = 1 if num_features > 1 else 0

    if not (0 <= timing_feature_index < num_features):
        raise ValueError("timing_feature_index is out of bounds.")

    anomalous = normal_sequences.copy()
    num_anomalies = max(1, int(anomaly_ratio * num_sequences))
    idx = np.random.choice(num_sequences, size=num_anomalies, replace=False)

    # Half of anomalies become "fast bursts" (very small intervals),
    # the other half "slow" (very large intervals).
    half = num_anomalies // 2
    burst_idx = idx[:half]
    slow_idx = idx[half:]

    anomalous[burst_idx, :, timing_feature_index] *= small_factor
    anomalous[slow_idx, :, timing_feature_index] *= large_factor
    anomalous = np.clip(anomalous, 0.0, 1.0)

    return anomalous


def simulate_abnormal_packet_size(
    normal_sequences: np.ndarray,
    size_feature_index: Optional[int] = None,
    spike_magnitude: float = 4.0,
    anomaly_ratio: float = 0.2,
    spike_fraction: float = 0.3,
) -> np.ndarray:
    """
    Simulate abnormal packet size by injecting spikes in the packet size feature.

    Parameters
    ----------
    normal_sequences : np.ndarray
        Base sequences, shape (num_sequences, sequence_length, num_features).
    size_feature_index : int, optional
        Index of the feature representing packet size. If None, uses the last feature.
    spike_magnitude : float, optional
        Multiplicative factor for spikes (default: 4.0).
    anomaly_ratio : float, optional
        Fraction of sequences to make anomalous.
    spike_fraction : float, optional
        Fraction of timesteps within an anomalous sequence that will receive spikes.

    Returns
    -------
    np.ndarray
        Array of anomalous sequences.
    """
    if normal_sequences.ndim != 3:
        raise ValueError("normal_sequences must have shape (num_sequences, sequence_length, num_features).")

    num_sequences, sequence_length, num_features = normal_sequences.shape
    if size_feature_index is None:
        size_feature_index = num_features - 1

    if not (0 <= size_feature_index < num_features):
        raise ValueError("size_feature_index is out of bounds.")

    anomalous = normal_sequences.copy()
    num_anomalies = max(1, int(anomaly_ratio * num_sequences))
    seq_idx = np.random.choice(num_sequences, size=num_anomalies, replace=False)

    num_spike_steps = max(1, int(spike_fraction * sequence_length))

    for i in seq_idx:
        spike_indices = np.random.choice(sequence_length, size=num_spike_steps, replace=False)
        anomalous[i, spike_indices, size_feature_index] *= spike_magnitude

    anomalous = np.clip(anomalous, 0.0, 1.0)

    return anomalous


def simulate_attacks(
    normal_sequences: np.ndarray,
    mode: Literal["packet_flood", "abnormal_timing", "abnormal_size"] = "packet_flood",
    **kwargs,
) -> np.ndarray:
    """
    High-level helper to generate anomalous sequences compatible with LSTM input.

    Parameters
    ----------
    normal_sequences : np.ndarray
        Base normal sequences, shape (num_sequences, sequence_length, num_features).
    mode : {'packet_flood', 'abnormal_timing', 'abnormal_size'}
        Type of attack pattern to simulate.
    kwargs :
        Extra arguments passed to the specific simulator.

    Returns
    -------
    np.ndarray
        Anomalous sequences of the same shape as input.
    """
    if mode == "packet_flood":
        return simulate_packet_flood(normal_sequences, **kwargs)
    if mode == "abnormal_timing":
        return simulate_abnormal_timing(normal_sequences, **kwargs)
    if mode == "abnormal_size":
        return simulate_abnormal_packet_size(normal_sequences, **kwargs)

    raise ValueError(f"Unknown attack simulation mode: {mode!r}")


if __name__ == "__main__":
    # Simple demonstration of simulated attack sequences.
    rng = np.random.default_rng(seed=42)

    num_sequences = 10
    sequence_length = 12
    num_features = 4

    # Assume normalized baseline traffic in [0, 1].
    baseline = rng.random((num_sequences, sequence_length, num_features), dtype=np.float32)

    flood = simulate_packet_flood(baseline, anomaly_ratio=0.3)
    timing = simulate_abnormal_timing(baseline, anomaly_ratio=0.3)
    size_spikes = simulate_abnormal_packet_size(baseline, anomaly_ratio=0.3)

    print("Baseline shape:", baseline.shape)
    print("Packet flood anomalies shape:", flood.shape)
    print("Abnormal timing anomalies shape:", timing.shape)
    print("Abnormal packet size anomalies shape:", size_spikes.shape)

