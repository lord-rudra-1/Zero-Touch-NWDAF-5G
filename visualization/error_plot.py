from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_reconstruction_error_distribution(
    errors: np.ndarray,
    threshold: Optional[float] = None,
    bins: int = 60,
    title: str = "Distribution of Reconstruction Errors",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the distribution of reconstruction errors for anomaly detection.

    Parameters
    ----------
    errors : np.ndarray
        1D array of reconstruction errors.
    threshold : float, optional
        Threshold above which points are considered anomalous.
    bins : int, optional
        Number of bins in the histogram.
    title : str, optional
        Title of the plot, suitable for research presentation.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=bins, color="steelblue", alpha=0.8, edgecolor="black")
    plt.xlabel("Reconstruction Error", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(title, fontsize=14)

    if threshold is not None:
        plt.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.4f}")
        plt.legend(fontsize=10)

    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_reconstruction_error_over_time(
    errors: np.ndarray,
    threshold: Optional[float] = None,
    title: str = "Reconstruction Error Over Time",
    time_label: str = "Time Index",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot reconstruction error as a function of time and mark anomalies.

    Parameters
    ----------
    errors : np.ndarray
        1D array of reconstruction errors ordered by time.
    threshold : float, optional
        Threshold above which points are considered anomalous.
    title : str, optional
        Title of the plot, suitable for research presentation.
    time_label : str, optional
        Label for the x-axis (e.g., 'Time Index' or 'Flow Index').
    """
    time_indices = np.arange(len(errors))

    plt.figure(figsize=(10, 5))
    plt.plot(time_indices, errors, label="Reconstruction Error", color="steelblue", linewidth=1.5)
    plt.xlabel(time_label, fontsize=12)
    plt.ylabel("Reconstruction Error", fontsize=12)
    plt.title(title, fontsize=14)

    anomaly_mask = None
    if threshold is not None:
        plt.axhline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.4f}")
        anomaly_mask = errors > threshold

    # Highlight anomalies as scatter points.
    if anomaly_mask is not None and np.any(anomaly_mask):
        plt.scatter(
            time_indices[anomaly_mask],
            errors[anomaly_mask],
            color="darkred",
            marker="x",
            s=40,
            label="Potential Anomalies",
        )

    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # Example usage with synthetic data.
    np.random.seed(42)
    num_points = 500
    base_errors = np.random.gamma(shape=2.0, scale=0.01, size=num_points)

    # Inject a few higher-error points to simulate anomalies.
    anomaly_indices = np.random.choice(num_points, size=10, replace=False)
    base_errors[anomaly_indices] += np.random.uniform(0.05, 0.15, size=len(anomaly_indices))

    threshold_example = np.percentile(base_errors, 95)

    plot_reconstruction_error_distribution(
        base_errors,
        threshold=threshold_example,
        title="Reconstruction Error Distribution (Synthetic Example)",
    )

    plot_reconstruction_error_over_time(
        base_errors,
        threshold=threshold_example,
        title="Reconstruction Error vs Time (Synthetic Example)",
        time_label="Sequence Index",
    )

