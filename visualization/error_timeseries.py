from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_error_timeseries(
    errors: np.ndarray,
    threshold: Optional[float] = None,
    timestamps: Optional[Sequence[Union[str, float]]] = None,
    title: str = "Reconstruction Error Time Series",
    xlabel: str = "Sequence Index",
) -> None:
    """
    Plot reconstruction error across time and highlight anomalies.

    Requirements:
    - x-axis = sequence index or timestamp
    - y-axis = reconstruction error
    - mark anomaly points in red
    - draw threshold line
    """
    if errors.ndim != 1:
        raise ValueError(f"`errors` must be a 1D array, got shape {errors.shape}.")

    n = len(errors)
    if timestamps is None:
        x = np.arange(n)
        x_label = xlabel
    else:
        if len(timestamps) != n:
            raise ValueError(
                f"Length of timestamps ({len(timestamps)}) does not match "
                f"number of errors ({n})."
            )
        x = np.array(timestamps)
        x_label = "Timestamp"

    # Identify anomalies if threshold provided
    anomaly_mask = None
    if threshold is not None:
        anomaly_mask = errors > threshold

    # Configure publication-quality style
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

    # Plot reconstruction error as a line
    ax.plot(
        x,
        errors,
        color="steelblue",
        linewidth=1.6,
        label="Reconstruction Error",
    )

    # Plot threshold line
    if threshold is not None:
        ax.axhline(
            threshold,
            color="crimson",
            linestyle="--",
            linewidth=2.0,
            label=f"Threshold = {threshold:.4f}",
        )

    # Mark anomaly points in red
    if anomaly_mask is not None and np.any(anomaly_mask):
        ax.scatter(
            x[anomaly_mask],
            errors[anomaly_mask],
            color="darkred",
            marker="x",
            s=40,
            zorder=5,
            label="Anomalies",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Reconstruction Error", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)

    # Place legend in a clean location
    ax.legend(fontsize=10, frameon=True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage with synthetic data.
    rng = np.random.default_rng(seed=42)
    num_points = 300
    base_errors = rng.gamma(shape=2.0, scale=0.01, size=num_points)

    # Inject higher errors to simulate anomalies.
    anomaly_indices = rng.choice(num_points, size=8, replace=False)
    base_errors[anomaly_indices] += rng.uniform(0.05, 0.15, size=len(anomaly_indices))

    threshold_example = np.percentile(base_errors, 95)

    # Without timestamps (sequence index on x-axis)
    plot_error_timeseries(
        base_errors,
        threshold=threshold_example,
        timestamps=None,
        title="Reconstruction Error Over Sequences (Synthetic Example)",
        xlabel="Sequence Index",
    )

