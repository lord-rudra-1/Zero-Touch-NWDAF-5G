from typing import Dict, Tuple

import numpy as np


def confusion_matrix_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix for binary anomaly detection.

    Assumes:
        1 = anomaly (positive class)
        0 = normal  (negative class)

    Returns
    -------
    tp, fp, fn, tn : int
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    return tp, fp, fn, tn


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 score, and confusion matrix for anomaly detection.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels (0 = normal, 1 = anomaly).
    y_pred : array-like
        Predicted anomaly flags (0 = normal, 1 = anomaly).

    Returns
    -------
    Dict[str, float]
        Dictionary containing precision, recall, f1, and confusion matrix entries.
    """
    tp, fp, fn, tn = confusion_matrix_binary(y_true, y_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }

    print("Anomaly Detection Metrics")
    print("------------------------")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print()
    print("Confusion Matrix (binary)")
    print(f"TP (anomaly correctly detected)   : {tp}")
    print(f"FP (normal incorrectly detected)  : {fp}")
    print(f"FN (missed anomaly)               : {fn}")
    print(f"TN (normal correctly ignored)     : {tn}")

    return metrics


if __name__ == "__main__":
    # Simple example
    y_true_example = np.array([0, 0, 1, 1, 1, 0, 1])
    y_pred_example = np.array([0, 1, 1, 1, 0, 0, 1])

    compute_metrics(y_true_example, y_pred_example)

