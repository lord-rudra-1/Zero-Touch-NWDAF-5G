from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve


def _set_publication_style() -> None:
    """Configure matplotlib / seaborn style for high-quality figures."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )


def plot_training_loss(
    losses: Sequence[float],
    title: str = "Training Loss vs Epochs",
) -> None:
    """Plot training reconstruction loss over epochs."""
    _set_publication_style()
    epochs = np.arange(1, len(losses) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, losses, marker="o", color="steelblue", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (MSE)")
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    plt.show()


def plot_reconstruction_error_distribution_final(
    errors: np.ndarray,
    threshold: Optional[float] = None,
    title: str = "Reconstruction Error Distribution",
) -> None:
    """High-quality reconstruction error distribution plot."""
    _set_publication_style()
    errors = np.asarray(errors, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(errors, bins=50, kde=True, color="steelblue", ax=ax)
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Frequency")
    ax.set_title(title)

    if threshold is not None:
        ax.axvline(threshold, color="crimson", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.4f}")
        ax.legend()

    fig.tight_layout()
    plt.show()


def plot_error_time_with_spikes(
    errors: np.ndarray,
    anomaly_flags: np.ndarray,
    threshold: Optional[float] = None,
    timestamps: Optional[Sequence[Union[str, float]]] = None,
    title: str = "Reconstruction Error Over Time with Anomaly Spikes",
    xlabel: str = "Time Index",
) -> None:
    """
    Plot reconstruction error vs time and highlight anomaly spikes.
    """
    _set_publication_style()
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
        x_label = xlabel
    else:
        if len(timestamps) != n:
            raise ValueError(
                f"Length of timestamps ({len(timestamps)}) does not match number of errors ({n})."
            )
        x = np.array(timestamps)
        x_label = "Timestamp"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, errors, color="steelblue", linewidth=1.5, label="Reconstruction Error")

    if threshold is not None:
        ax.axhline(threshold, color="crimson", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.4f}")

    # Mark anomaly spikes
    mask = anomaly_flags.astype(bool)
    if np.any(mask):
        ax.scatter(
            x[mask],
            errors[mask],
            color="darkred",
            s=40,
            marker="x",
            label="Anomaly",
            zorder=5,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Reconstruction Error")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    plt.show()


def plot_roc_curve_final(
    y_true: np.ndarray,
    scores: np.ndarray,
    title: str = "ROC Curve for Anomaly Detection",
) -> float:
    """
    Plot ROC curve and return AUC, styled for reports.
    """
    _set_publication_style()
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    if y_true.shape != scores.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs scores {scores.shape}")

    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"Model (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Guess")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.show()

    return roc_auc


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    labels: Optional[Sequence[str]] = None,
) -> None:
    """
    Plot confusion matrix heatmap for binary anomaly detection.
    """
    _set_publication_style()
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if labels is None:
        labels = ["Normal (0)", "Anomaly (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Demonstration with synthetic data for report-ready figures.
    rng = np.random.default_rng(seed=42)

    # 1. Training loss vs epochs
    fake_losses = np.linspace(0.05, 0.01, num=20) + rng.normal(scale=0.002, size=20)
    plot_training_loss(fake_losses)

    # 2. Reconstruction error distribution
    errors_demo = rng.gamma(shape=2.0, scale=0.01, size=1000)
    thresh_demo = np.percentile(errors_demo, 95)
    plot_reconstruction_error_distribution_final(errors_demo, threshold=thresh_demo)

    # 3. Error vs time with anomaly spikes
    anomaly_flags_demo = np.zeros_like(errors_demo, dtype=int)
    spike_idx = rng.choice(errors_demo.size, size=20, replace=False)
    errors_demo[spike_idx] += rng.uniform(0.05, 0.15, size=spike_idx.size)
    anomaly_flags_demo[spike_idx] = 1
    plot_error_time_with_spikes(errors_demo, anomaly_flags_demo, threshold=thresh_demo)

    # 4. ROC curve
    y_true_demo = rng.integers(0, 2, size=500)
    scores_demo = rng.random(size=500) + 0.6 * y_true_demo
    plot_roc_curve_final(y_true_demo, scores_demo)

    # 5. Confusion matrix heatmap
    y_pred_demo = (scores_demo > np.median(scores_demo)).astype(int)
    plot_confusion_matrix_heatmap(y_true_demo, y_pred_demo)

