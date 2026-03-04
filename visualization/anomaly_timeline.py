from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_anomaly_timeline(
    errors: np.ndarray,
    anomaly_flags: np.ndarray,
    timestamps: Optional[Sequence[Union[str, float]]] = None,
    threshold: Optional[float] = None,
    title: str = "Anomaly Detection Timeline",
    xlabel: str = "Time",
) -> None:
    """
    Visualize anomaly detection timeline as time vs reconstruction error.

    Parameters
    ----------
    errors : np.ndarray
        1D array of reconstruction errors ordered by time.
    anomaly_flags : np.ndarray
        1D binary array (0/1), same length as errors; 1 indicates anomaly.
    timestamps : sequence, optional
        Timestamps aligned with errors; if None, indices are used.
    threshold : float, optional
        Threshold line to draw on the plot.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    """
    errors = np.asarray(errors, dtype=float)
    anomaly_flags = np.asarray(anomaly_flags, dtype=int)

    if errors.ndim != 1:
        raise ValueError(f"`errors` must be 1D, got shape {errors.shape}.")
    if anomaly_flags.ndim != 1:
        raise ValueError(f"`anomaly_flags` must be 1D, got shape {anomaly_flags.shape}.")
    if errors.shape[0] != anomaly_flags.shape[0]:
        raise ValueError(
            f"Length mismatch: errors ({errors.shape[0]}) vs anomaly_flags ({anomaly_flags.shape[0]})."
        )

    n = len(errors)
    if timestamps is None:
        x = np.arange(n)
        x_label = "Sequence Index" if xlabel == "Time" else xlabel
    else:
        if len(timestamps) != n:
            raise ValueError(
                f"Length of timestamps ({len(timestamps)}) does not match number of errors ({n})."
            )
        x = np.array(timestamps)
        x_label = xlabel

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

    ax.plot(
        x,
        errors,
        color="steelblue",
        linewidth=1.6,
        label="Reconstruction Error",
    )

    if threshold is not None:
        ax.axhline(
            threshold,
            color="crimson",
            linestyle="--",
            linewidth=2.0,
            label=f"Threshold = {threshold:.4f}",
        )

    # Highlight anomaly regions using red shading.
    is_anom = anomaly_flags.astype(bool)
    if np.any(is_anom):
        ax.fill_between(
            x,
            errors,
            where=is_anom,
            color="red",
            alpha=0.25,
            label="Anomaly Region",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Reconstruction Error", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, frameon=True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage with synthetic data.
    rng = np.random.default_rng(seed=42)
    num_points = 300
    errors_demo = rng.gamma(shape=2.0, scale=0.01, size=num_points)

    # Inject anomalies.
    anomaly_indices = rng.choice(num_points, size=12, replace=False)
    errors_demo[anomaly_indices] += rng.uniform(0.05, 0.15, size=len(anomaly_indices))

    anomaly_flags_demo = np.zeros(num_points, dtype=int)
    anomaly_flags_demo[anomaly_indices] = 1

    threshold_demo = np.percentile(errors_demo, 95)

    plot_anomaly_timeline(
        errors_demo,
        anomaly_flags_demo,
        timestamps=None,
        threshold=threshold_demo,
        title="Anomaly Detection Timeline (Synthetic Example)",
        xlabel="Sequence Index",
    )

