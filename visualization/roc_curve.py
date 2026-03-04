from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def plot_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    title: str = "ROC Curve for Anomaly Detection",
    pos_label: int = 1,
    line_label: Optional[str] = "LSTM Autoencoder",
) -> float:
    """
    Plot ROC curve and compute AUC for anomaly detection.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0 = normal, 1 = anomaly).
    scores : np.ndarray
        Anomaly scores (e.g., reconstruction errors). Higher = more anomalous.
    title : str, optional
        Title for the plot.
    pos_label : int, optional
        Label of the positive class (default: 1 for anomaly).
    line_label : str, optional
        Label for the ROC curve in the legend.

    Returns
    -------
    float
        Area Under the ROC Curve (AUC).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    if y_true.shape != scores.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs scores {scores.shape}")

    # Compute ROC curve and AUC.
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    # Configure publication-style plot.
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)

    # Plot ROC curve
    label = f"{line_label} (AUC = {roc_auc:.3f})" if line_label is not None else f"AUC = {roc_auc:.3f}"
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=label)

    # Plot random baseline
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Guess")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, frameon=True)

    fig.tight_layout()
    plt.show()

    print(f"ROC AUC: {roc_auc:.4f}")
    return roc_auc


if __name__ == "__main__":
    # Example usage with synthetic labels and scores.
    rng = np.random.default_rng(seed=42)
    n = 500

    # Synthetic binary labels (0 = normal, 1 = anomaly)
    y_true_demo = rng.integers(0, 2, size=n)
    # Synthetic anomaly scores (higher = more anomalous)
    scores_demo = rng.random(size=n) + 0.5 * y_true_demo

    plot_roc_curve(y_true_demo, scores_demo, title="ROC Curve (Synthetic Example)")

