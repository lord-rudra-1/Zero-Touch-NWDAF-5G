from typing import Tuple

import numpy as np
import pandas as pd

from preprocessing.feature_engineering import prepare_features
from preprocessing.sequence_generator import generate_lstm_sequences


def split_train_test_sequences(
    df: pd.DataFrame,
    label_column: str = "Label",
    normal_label: str = "Normal",
    sequence_length: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split network traffic data into LSTM-ready training and test sets.

    - Training uses only "normal" traffic sequences.
    - Testing uses only attack / non-normal sequences.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe containing numerical features and a label column.
    label_column : str, optional
        Name of the column containing labels (default: "Label").
    normal_label : str, optional
        Value in `label_column` that indicates normal traffic (default: "Normal").
    sequence_length : int, optional
        Length of the time sequence window for LSTM (default: 10).

    Returns
    -------
    X_train : np.ndarray
        Normal traffic sequences for training,
        shape (num_normal_sequences, sequence_length, num_features).
    X_test : np.ndarray
        Attack / anomalous traffic sequences for testing,
        shape (num_attack_sequences, sequence_length, num_features).
    y_test : np.ndarray
        Labels for each test sequence, shape (num_attack_sequences,).
        Each label corresponds to the label of the last sample in the sequence.
    """
    if label_column not in df.columns:
        raise ValueError(f"Expected label column '{label_column}' not found in dataframe.")

    # Separate normal and attack traffic based on the label column.
    normal_df = df[df[label_column] == normal_label]
    attack_df = df[df[label_column] != normal_label]

    if normal_df.empty:
        raise ValueError("No normal traffic samples found for training.")
    if attack_df.empty:
        raise ValueError("No attack / anomalous samples found for testing.")

    # Prepare numerical, normalized features for each subset.
    X_normal = prepare_features(normal_df, exclude_columns=[label_column])
    X_attack = prepare_features(attack_df, exclude_columns=[label_column])

    # Convert continuous samples into overlapping LSTM sequences.
    X_train = generate_lstm_sequences(X_normal, sequence_length=sequence_length)
    X_test = generate_lstm_sequences(X_attack, sequence_length=sequence_length)

    # Derive sequence-level labels for the test set.
    # Each sequence spans `sequence_length` consecutive rows.
    # We assign the label of the last row in each window to the sequence.
    y_attack = attack_df[label_column].to_numpy()
    y_test = y_attack[sequence_length - 1 :]

    if len(y_test) != X_test.shape[0]:
        raise RuntimeError(
            "Mismatch between number of test sequences and labels: "
            f"{X_test.shape[0]} sequences vs {len(y_test)} labels."
        )

    # Print counts for transparency.
    print(f"Number of normal samples: {len(normal_df)}")
    print(f"Number of attack samples: {len(attack_df)}")
    print(f"Number of normal training sequences: {X_train.shape[0]}")
    print(f"Number of attack test sequences: {X_test.shape[0]}")

    return X_train, X_test, y_test


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Split network traffic data into LSTM training (normal) and test (attack) sets.",
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the network traffic CSV file containing a 'Label' column.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="Label",
        help="Name of the label column (default: 'Label').",
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
        help="Length of the LSTM sequence window (default: 10).",
    )

    args = parser.parse_args()
    csv_path = Path(args.csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load and clean the raw dataset while preserving the label column.
    df_raw = pd.read_csv(csv_path)
    df_clean = df_raw.dropna().drop_duplicates()

    # The cleaned dataframe must still contain the label column.
    if args.label_column not in df_clean.columns:
        raise ValueError(
            f"Label column '{args.label_column}' not found after loading the dataset. "
            "Make sure the CSV includes this column and it is not dropped during preprocessing."
        )

    X_train, X_test, y_test = split_train_test_sequences(
        df_clean,
        label_column=args.label_column,
        normal_label=args.normal_label,
        sequence_length=args.sequence_length,
    )

    print(f"Final shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

