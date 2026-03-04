from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_features(
    df: pd.DataFrame,
    exclude_columns: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """
    Perform feature engineering for network anomaly detection.

    Steps:
    - Select numerical features that describe traffic behavior
    - Optionally exclude non-feature columns (e.g. labels, IDs)
    - Normalize features to [0, 1] range using MinMaxScaler
    - Return the normalized feature matrix as a NumPy array
    """
    # Ensure we have a copy of the dataframe so original data is not modified in-place.
    data = df.copy()

    # Exclude columns that should not be used as features (e.g. labels, identifiers).
    if exclude_columns:
        data = data.drop(columns=list(exclude_columns), errors="ignore")

    # Select only numerical columns, which typically represent traffic statistics
    # such as packet counts, byte counts, durations, and protocol-specific metrics.
    numeric_data = data.select_dtypes(include=[np.number])

    # Initialize MinMaxScaler to scale each feature into the [0, 1] range.
    scaler = MinMaxScaler()

    # Fit the scaler on the numerical data and transform it to obtain
    # a normalized NumPy array suitable for feeding into an LSTM autoencoder.
    normalized_features = scaler.fit_transform(numeric_data.values)

    return normalized_features


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from preprocessing.load_data import load_network_dataset

    parser = argparse.ArgumentParser(
        description="Feature engineering for network anomaly detection.",
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the cleaned or raw network intrusion CSV file.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["label", "Label", "attack", "Attack"],
        help="Column names to exclude from features (e.g. label columns or IDs).",
    )

    args = parser.parse_args()
    csv_path = Path(args.csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load and clean the dataset using the loader defined earlier.
    df_clean = load_network_dataset(csv_path)

    # Prepare normalized feature matrix for model training.
    features = prepare_features(df_clean, exclude_columns=args.exclude)

    print(f"Feature matrix shape: {features.shape}")
