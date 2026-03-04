from typing import Literal

import numpy as np


def percentile_threshold(errors: np.ndarray, percentile: float = 95.0) -> float:
    """
    Compute anomaly threshold as a given percentile of reconstruction errors.

    Parameters
    ----------
    errors : np.ndarray
        1D array of reconstruction errors.
    percentile : float, optional
        Percentile to use (default: 95.0).

    Returns
    -------
    float
        Threshold value.
    """
    threshold = float(np.percentile(errors, percentile))
    print(
        f"[Threshold] Percentile method: using {percentile:.1f}th percentile "
        f"of errors -> threshold = {threshold:.6f}"
    )
    return threshold


def statistical_threshold(errors: np.ndarray, k: float = 3.0) -> float:
    """
    Compute anomaly threshold using mean + k * std of reconstruction errors.

    Parameters
    ----------
    errors : np.ndarray
        1D array of reconstruction errors.
    k : float, optional
        Multiplier for standard deviation (default: 3.0).

    Returns
    -------
    float
        Threshold value.
    """
    mu = float(np.mean(errors))
    sigma = float(np.std(errors))
    threshold = mu + k * sigma
    print(
        f"[Threshold] Statistical method: mean = {mu:.6f}, std = {sigma:.6f}, "
        f"k = {k:.1f} -> threshold = {threshold:.6f}"
    )
    return threshold


def select_threshold(
    errors: np.ndarray,
    method: Literal["percentile", "statistical"] = "percentile",
    percentile: float = 95.0,
    k: float = 3.0,
) -> float:
    """
    Convenience function to select anomaly threshold using a chosen method.

    Parameters
    ----------
    errors : np.ndarray
        1D array of reconstruction errors.
    method : {'percentile', 'statistical'}
        Thresholding method to use.
    percentile : float, optional
        Percentile for the percentile method.
    k : float, optional
        Standard deviation multiplier for the statistical method.

    Returns
    -------
    float
        Selected threshold value.
    """
    if method == "percentile":
        return percentile_threshold(errors, percentile=percentile)
    if method == "statistical":
        return statistical_threshold(errors, k=k)

    raise ValueError(f"Unknown threshold method: {method!r}")


if __name__ == "__main__":
    # Simple demonstration with synthetic errors.
    rng = np.random.default_rng(seed=42)
    synthetic_errors = rng.gamma(shape=2.0, scale=0.01, size=1000)

    print("Computing thresholds on synthetic reconstruction errors...")
    t_percentile = percentile_threshold(synthetic_errors, percentile=95.0)
    t_stat = statistical_threshold(synthetic_errors, k=3.0)

    print(f"Percentile threshold: {t_percentile:.6f}")
    print(f"Statistical threshold: {t_stat:.6f}")

