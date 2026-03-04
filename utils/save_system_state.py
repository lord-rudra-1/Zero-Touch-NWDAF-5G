import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Union

import numpy as np
import torch


def save_system_state(
    model: torch.nn.Module,
    reconstruction_errors: Union[Iterable[float], np.ndarray],
    threshold: float,
    metrics: Dict[str, Any],
    base_results_dir: Union[str, Path] = "results",
    model_filename: str = "lstm_autoencoder.pth",
) -> Path:
    """
    Save experiment artifacts for an anomaly detection run.

    Artifacts:
    - trained model weights
    - reconstruction errors
    - threshold value
    - evaluation metrics

    Files are stored under:
      {base_results_dir}/run_YYYYMMDD_HHMMSS/
    """
    base_results_dir = Path(base_results_dir)
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base_results_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save model
    model_path = run_dir / model_filename
    torch.save(model.state_dict(), model_path)

    # 2) Save reconstruction errors
    errors = np.asarray(reconstruction_errors, dtype=float)
    errors_npy_path = run_dir / "reconstruction_errors.npy"
    np.save(errors_npy_path, errors)

    errors_csv_path = run_dir / "reconstruction_errors.csv"
    try:
        import pandas as pd

        df_errors = pd.DataFrame({"reconstruction_error": errors})
        df_errors.to_csv(errors_csv_path, index=False)
    except Exception:
        # If pandas is not available for some reason, skip CSV export.
        pass

    # 3) Save threshold and metrics as JSON
    summary = {
        "threshold": float(threshold),
        "metrics": metrics,
    }
    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved system state under: {run_dir}")
    print(f"  Model path     : {model_path}")
    print(f"  Errors (.npy)  : {errors_npy_path}")
    print(f"  Errors (.csv)  : {errors_csv_path}")
    print(f"  Summary (JSON) : {summary_path}")

    return run_dir


if __name__ == "__main__":
    # Minimal smoke test with a dummy model and synthetic data.
    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.linear(x)

    dummy_model = _DummyModel()
    dummy_errors = np.random.rand(10)
    dummy_threshold = 0.5
    dummy_metrics = {"precision": 0.9, "recall": 0.85, "f1_score": 0.875}

    save_system_state(
        model=dummy_model,
        reconstruction_errors=dummy_errors,
        threshold=dummy_threshold,
        metrics=dummy_metrics,
    )

