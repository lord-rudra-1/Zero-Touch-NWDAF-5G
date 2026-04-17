"""
Telemetry preprocessor — transforms raw Prometheus DataFrames into
normalised feature arrays ready for the LSTM‑Autoencoder.

Pipeline:
  1. Drop non-numeric columns (timestamp, elapsed)
  2. MinMax normalise each feature to [0, 1]
  3. Convert to LSTM sequences via sliding window
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from preprocessing.sequence_generator import generate_lstm_sequences


# ---------------------------------------------------------------------------
# Feature columns (must match telemetry/collector.py)
# ---------------------------------------------------------------------------
DEFAULT_FEATURE_COLUMNS: List[str] = [
    "app_ngap_messages_total",
    "app_nas_messages_total",
    "app_active_sessions",
    "app_registration_requests_total",
    "app_auth_failures_total",
    "app_request_latency_seconds",
]


def extract_features(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Extract and return only the numeric feature columns from a telemetry
    DataFrame as a 2‑D ``float32`` array.
    """
    cols = feature_columns or DEFAULT_FEATURE_COLUMNS
    available = [c for c in cols if c in df.columns]
    if not available:
        raise ValueError(
            f"None of the expected feature columns found in DataFrame. "
            f"Expected: {cols}, Got: {list(df.columns)}"
        )
    # Extract rates (derivatives) rather than cumulative counters
    df_diff = df[available].diff().fillna(0)
    return df_diff.values.astype(np.float32)


def normalise_features(
    data: np.ndarray,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalise a 2‑D feature array to [0, 1] using MinMaxScaler.

    Returns ``(normalised_array, fitted_scaler)``.  Pass the returned
    scaler back when normalising new/attack data so scaling is consistent.
    """
    if scaler is None:
        scaler = MinMaxScaler()
        normalised = scaler.fit_transform(data)
    else:
        normalised = scaler.transform(data)
    return normalised.astype(np.float32), scaler


def preprocess_telemetry(
    df: pd.DataFrame,
    sequence_length: int = 10,
    feature_columns: Optional[List[str]] = None,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Full preprocessing pipeline: extract → normalise → sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Raw telemetry collected by ``telemetry.collector``.
    sequence_length : int
        Sliding-window size for LSTM sequences.
    feature_columns : list[str], optional
        Column names to use as features.
    scaler : MinMaxScaler, optional
        Pre-fitted scaler.  If *None* a new one is fitted.

    Returns
    -------
    sequences : np.ndarray
        3-D array ``(N, T, F)`` ready for the LSTM‑AE.
    scaler : MinMaxScaler
        The fitted scaler (save it for inference-time normalisation).
    """
    raw = extract_features(df, feature_columns=feature_columns)
    normalised, scaler = normalise_features(raw, scaler=scaler)
    sequences = generate_lstm_sequences(normalised, sequence_length=sequence_length)
    return sequences, scaler


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick demo with simulated data
    from telemetry.collector import generate_simulated_telemetry

    df = generate_simulated_telemetry(duration_seconds=120, poll_interval=1.0)
    seqs, sc = preprocess_telemetry(df, sequence_length=10)
    print(f"Input rows   : {len(df)}")
    print(f"Sequences    : {seqs.shape}")
    print(f"Feature range: [{seqs.min():.4f}, {seqs.max():.4f}]")
