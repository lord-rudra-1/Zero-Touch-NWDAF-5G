from typing import Dict

import numpy as np

from evaluation.metrics import compute_metrics


def summarize_results(y_true, y_pred) -> Dict[str, float]:
    """
    Summarize anomaly detection performance for research reporting.

    Prints and returns:
    - total sequences
    - normal sequences
    - true anomaly sequences
    - detected anomalies (predicted positives)
    - detection rate (TP / true anomalies, in %)
    - false positives
    - precision
    - recall
    - F1 score
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}")

    total_sequences = int(y_true_arr.size)
    normal_sequences = int(np.sum(y_true_arr == 0))
    true_anomalies = int(np.sum(y_true_arr == 1))
    detected_anomalies = int(np.sum(y_pred_arr == 1))

    # Compute core metrics and confusion matrix components.
    metrics = compute_metrics(y_true_arr, y_pred_arr)

    tp = int(metrics["tp"])
    fp = int(metrics["fp"])
    fn = int(metrics["fn"])
    # tn = int(metrics["tn"])  # not printed explicitly below

    detection_rate = (tp / true_anomalies * 100.0) if true_anomalies > 0 else 0.0

    print("\n=== Anomaly Detection Results Summary ===")
    print(f"Total sequences         : {total_sequences}")
    print(f"Normal sequences        : {normal_sequences}")
    print(f"True anomaly sequences  : {true_anomalies}")
    print(f"Detected anomalies      : {detected_anomalies}")
    print(f"Detection rate (TP / true anomalies) : {detection_rate:.2f}%")
    print(f"False positives (FP)    : {fp}")
    print("----------------------------------------")
    print(f"Precision               : {metrics['precision']:.4f}")
    print(f"Recall                  : {metrics['recall']:.4f}")
    print(f"F1 Score                : {metrics['f1_score']:.4f}")

    # Add high-level numbers into the returned dict as well.
    metrics_summary = {
        "total_sequences": float(total_sequences),
        "normal_sequences": float(normal_sequences),
        "true_anomalies": float(true_anomalies),
        "detected_anomalies": float(detected_anomalies),
        "detection_rate_percent": float(detection_rate),
        "false_positives": float(fp),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
    }

    return metrics_summary


if __name__ == "__main__":
    # Simple example for demonstration.
    y_true_example = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1])
    y_pred_example = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0])

    summarize_results(y_true_example, y_pred_example)

