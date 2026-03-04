import pandas as pd
import numpy as np
from typing import Union


def load_network_dataset(csv_path: Union[str, "os.PathLike"]) -> pd.DataFrame:
    """
    Load and clean a network intrusion dataset from a CSV file.

    Steps:
    - Read CSV with pandas
    - Drop rows with missing values
    - Remove duplicate rows
    - Keep only numerical columns
    - Print dataset shape and column names
    - Return cleaned DataFrame
    """
    # Load raw dataset
    df = pd.read_csv(csv_path)

    # Handle missing values (drop rows containing any NaNs)
    df = df.dropna()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Select only numerical columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Print dataset information
    print(f"Cleaned dataset shape: {numeric_df.shape}")
    print("Numerical columns:")
    print(list(numeric_df.columns))

    return numeric_df


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Load and clean a network intrusion dataset (CSV)."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the network intrusion CSV file.",
    )

    args = parser.parse_args()
    csv_path = Path(args.csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    load_network_dataset(csv_path)
