"""
Live anomaly monitor — Phase 4 real‑time detection & zero‑touch mitigation.

Runs an infinite loop that:
  1. Polls Ella Core's ``/metrics`` every second
  2. Maintains a sliding window of the last ``T`` samples
  3. Feeds each window into the pre-trained LSTM‑Autoencoder
  4. Computes reconstruction error and compares to a rolling adaptive threshold
  5. Anomaly is confirmed ONLY after ``anomaly_streak_required`` consecutive
     steps above threshold (prevents single-spike false positives)
  6. Mitigation fires and then honoured by a ``cooldown_seconds`` before the
     next autonomous action (prevents rapid repeated blocks)

Can also run in **simulated** mode for demos without Ella Core.

FIX NOTES (v2):
  - Rolling adaptive threshold replaces the static saved threshold:
      adaptive = max(saved_threshold, rolling_mean + 3*rolling_std) over a
      128-step window.  During normal traffic this self-corrects upward;
      during an actual attack the rolling stats lag behind the true spike.
  - Anomaly streak logic: anomaly declared only after 3 consecutive windows
      exceed the adaptive threshold.  Resets to 0 on any normal window.
  - Cooldown: after mitigation fires, another 5 seconds must pass before
      the next mitigation call regardless of error score.
  - Enhanced logging: prints current threshold, rolling avg error,
      streak count, and cooldown status every step.
  - Feature dimension guard: asserts live feature count matches training.
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from core.ella_config import EllaConfig, get_config
from evaluation.reconstruction_error import compute_reconstruction_errors, load_trained_model
from inference.mitigation import block_subscriber, save_mitigation_log
from telemetry.collector import fetch_metrics, DEFAULT_METRIC_NAMES
from telemetry.preprocessor import DEFAULT_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Scaler helpers
# ---------------------------------------------------------------------------

def _load_scaler_params(scaler_path: Path) -> dict:
    """Load scaler parameters saved during training."""
    with scaler_path.open() as f:
        return json.load(f)


def _normalise_with_params(
    raw: np.ndarray,
    scaler_params: dict,
) -> np.ndarray:
    """
    Normalise a 2-D array using saved MinMaxScaler parameters.
    Avoids needing the sklearn scaler object at inference time.
    """
    data_min = np.array(scaler_params["min"], dtype=np.float32)
    data_max = np.array(scaler_params["max"], dtype=np.float32)
    scale    = np.array(scaler_params["scale"], dtype=np.float32)

    result = (raw - data_min) * scale
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Live monitor
# ---------------------------------------------------------------------------

def run_live_monitor(
    config: Optional[EllaConfig] = None,
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
    poll_interval: float = 1.0,
    max_iterations: Optional[int] = None,
    auto_mitigate: bool = True,
    verbose: bool = True,
    # Stability parameters
    anomaly_streak_required: int = 3,
    cooldown_seconds: float = 5.0,
    rolling_window_size: int = 128,
) -> None:
    """
    Start the real-time anomaly detection loop.

    Parameters
    ----------
    config : EllaConfig, optional
        Ella Core configuration.
    model_path : Path, optional
        Path to trained model (default: models/lstm_autoencoder.pth).
    metadata_path : Path, optional
        Path to training metadata JSON (contains base threshold).
    scaler_path : Path, optional
        Path to scaler parameters JSON.
    poll_interval : float
        Seconds between metric polls.
    max_iterations : int, optional
        Stop after this many iterations (None = infinite).
    auto_mitigate : bool
        Whether to automatically block offending subscribers.
    verbose : bool
        Print each detection result.
    anomaly_streak_required : int
        Number of **consecutive** anomalous windows before declaring anomaly.
        Default 3 — eliminates single-spike false positives.
    cooldown_seconds : float
        Seconds to wait after mitigation before next mitigation action.
    rolling_window_size : int
        History length (in steps) used to compute the adaptive threshold.
    """
    cfg = config or get_config()
    project_root = Path(__file__).resolve().parents[1]

    # Resolve paths
    if model_path is None:
        model_path = project_root / "models" / "lstm_autoencoder.pth"
    if metadata_path is None:
        metadata_path = project_root / "models" / "training_metadata.json"
    if scaler_path is None:
        scaler_path = project_root / "models" / "scaler_params.json"

    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    with metadata_path.open() as f:
        metadata = json.load(f)

    saved_threshold  = metadata["threshold"]
    sequence_length  = metadata["sequence_length"]
    num_features     = metadata["num_features"]
    mean_train_error = metadata.get("mean_training_error", 0.0)
    std_train_error  = metadata.get("std_training_error", 0.0)

    scaler_params = _load_scaler_params(scaler_path)

    # Feature dimension guard
    feature_names = DEFAULT_FEATURE_COLUMNS[:num_features]
    if len(feature_names) != num_features:
        raise ValueError(
            f"Feature dimension mismatch: training used {num_features} features "
            f"but DEFAULT_FEATURE_COLUMNS has {len(DEFAULT_FEATURE_COLUMNS)}."
        )

    model = load_trained_model(model_path=model_path, num_features=num_features)

    print(f"\n{'='*62}")
    print(f" NWDAF Live Monitor — Zero‑Touch Detection (v2)")
    print(f"{'='*62}")
    print(f" Model            : {model_path.name}")
    print(f" Base threshold   : {saved_threshold:.6f}")
    print(f" Train error mean : {mean_train_error:.6f} ± {std_train_error:.6f}")
    print(f" Seq length       : {sequence_length}")
    print(f" Features ({num_features})    : {feature_names}")
    print(f" Streak required  : {anomaly_streak_required} consecutive windows")
    print(f" Cooldown         : {cooldown_seconds}s after mitigation")
    print(f" Mitigation       : {'AUTO' if auto_mitigate else 'DISABLED'}")
    print(f"{'='*62}")
    print(f"\n{'Time':>12}  {'Error':>10}  {'Threshold':>10}  "
          f"{'RollAvg':>9}  {'Streak':>6}  {'Status'}")
    print(f"{'-'*80}")

    # Sliding window buffer for sequence
    window: deque = deque(maxlen=sequence_length)

    # Rolling error history for adaptive threshold
    error_history: deque = deque(maxlen=rolling_window_size)

    # Stability state
    anomaly_streak: int = 0
    last_mitigation_time: float = 0.0

    iteration   = 0
    anomaly_count = 0
    total_count   = 0

    try:
        while max_iterations is None or iteration < max_iterations:
            tick = time.time()

            # 1. Fetch latest metrics
            snapshot = fetch_metrics(config=cfg, metric_names=feature_names)
            feature_vec = np.array(
                [snapshot.get(m, 0.0) for m in feature_names],
                dtype=np.float32,
            )
            
            # Extract rates (derivatives) rather than cumulative counters
            if not hasattr(run_live_monitor, "_last_vec"):
                run_live_monitor._last_vec = np.zeros_like(feature_vec)
            
            diff_vec = feature_vec - run_live_monitor._last_vec
            run_live_monitor._last_vec = feature_vec.copy()
            
            window.append(diff_vec)

            if len(window) < sequence_length:
                if verbose:
                    now = datetime.now().strftime("%H:%M:%S")
                    print(f"{now:>12}  {'buffering':>10}  {'—':>10}  "
                          f"{'—':>9}  {'—':>6}  "
                          f"({len(window)}/{sequence_length} samples)")
                iteration += 1
                time.sleep(max(0, poll_interval - (time.time() - tick)))
                continue

            # 2. Build sequence and normalise
            raw_seq  = np.array(list(window), dtype=np.float32)   # (T, F)
            norm_seq = _normalise_with_params(raw_seq, scaler_params)  # (T, F)
            batch    = norm_seq[np.newaxis, ...]                   # (1, T, F)

            # 3. Compute reconstruction error
            error = float(compute_reconstruction_errors(model, batch)[0])
            total_count += 1
            error_history.append(error)

            # 4. Compute adaptive threshold
            #    = max(saved_threshold, rolling_mean + 3*rolling_std)
            #    This self‑levels during normal traffic and is robust to the
            #    LSTM's absolute error magnitude.
            if len(error_history) >= 10:
                roll_arr = np.array(error_history)
                roll_mean = float(roll_arr.mean())
                roll_std  = float(roll_arr.std())
                adaptive_threshold = max(saved_threshold, roll_mean + 3.0 * roll_std)
            else:
                roll_mean = error
                roll_std  = 0.0
                adaptive_threshold = saved_threshold

            # 5. Streak logic — must exceed threshold CONSECUTIVELY
            above_threshold = error > adaptive_threshold
            if above_threshold:
                anomaly_streak += 1
            else:
                anomaly_streak = 0   # reset on any normal step

            is_confirmed_anomaly = anomaly_streak >= anomaly_streak_required

            # 6. Status
            if is_confirmed_anomaly:
                status = "⚠  ANOMALY"
                anomaly_count += 1
            else:
                status = "   Normal "

            # 7. Mitigation with cooldown
            action = ""
            now_t  = time.time()
            in_cooldown = (now_t - last_mitigation_time) < cooldown_seconds

            if is_confirmed_anomaly and auto_mitigate and not in_cooldown:
                imsi = cfg.subscriber_imsi
                block_subscriber(imsi, config=cfg)
                action = f"→ BLOCKED {imsi}"
                last_mitigation_time = now_t
            elif is_confirmed_anomaly and in_cooldown:
                remaining = cooldown_seconds - (now_t - last_mitigation_time)
                action = f"[cooldown {remaining:.1f}s]"

            # 8. Logging
            if verbose:
                now = datetime.now().strftime("%H:%M:%S")
                streak_display = f"{anomaly_streak}/{anomaly_streak_required}"
                print(f"{now:>12}  {error:>10.6f}  {adaptive_threshold:>10.6f}  "
                      f"{roll_mean:>9.6f}  {streak_display:>6}  "
                      f"{status}  {action}")

            iteration += 1
            time.sleep(max(0, poll_interval - (time.time() - tick)))

    except KeyboardInterrupt:
        print("\n[monitor] Interrupted by user")

    # Summary
    print(f"\n{'='*62}")
    print(f" Detection Summary")
    print(f"{'='*62}")
    print(f" Total windows analysed : {total_count}")
    print(f" Anomalies confirmed    : {anomaly_count}")
    print(f" Anomaly rate           : "
          f"{(anomaly_count/max(total_count,1))*100:.1f}%")
    if error_history:
        arr = np.array(error_history)
        print(f" Final rolling error    : {arr.mean():.6f} ± {arr.std():.6f}")

    if auto_mitigate:
        save_mitigation_log()

    print(f"{'='*62}")


# ---------------------------------------------------------------------------
# Simulated mode for demos
# ---------------------------------------------------------------------------

def run_simulated_monitor(
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
    normal_samples: int = 200,
    attack_samples: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Run anomaly detection on simulated telemetry (no Ella Core needed).

    Generates normal + attack data using the parametric sine-wave model
    (same distribution as live fetch_metrics), feeds it through the model,
    and reports detection performance.

    Returns a summary dict.
    """
    from telemetry.collector import generate_simulated_telemetry
    from simulation.attack_generator import generate_attack_telemetry
    from telemetry.preprocessor import preprocess_telemetry
    from evaluation.metrics import compute_metrics
    from evaluation.threshold import select_threshold

    project_root = Path(__file__).resolve().parents[1]
    if model_path is None:
        model_path = project_root / "models" / "lstm_autoencoder.pth"
    if metadata_path is None:
        metadata_path = project_root / "models" / "training_metadata.json"
    if scaler_path is None:
        scaler_path = project_root / "models" / "scaler_params.json"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")

    with metadata_path.open() as f:
        metadata = json.load(f)

    threshold      = metadata["threshold"]
    sequence_length = metadata["sequence_length"]
    num_features   = metadata["num_features"]

    model = load_trained_model(model_path=model_path, num_features=num_features)
    scaler_params = _load_scaler_params(scaler_path)

    # Generate test data — use parametric model for BOTH normal and attack
    print("\n[sim-monitor] Generating normal test data (parametric sine-wave)...")
    df_normal = generate_simulated_telemetry(
        duration_seconds=normal_samples,
        poll_interval=1.0,
        attack_start_frac=1.0,            # all normal
        ue_clone_count_during_attack=0,   # zero clones
    )
    print("[sim-monitor] Generating attack test data (parametric + clone surge)...")
    df_attack = generate_simulated_telemetry(
        duration_seconds=attack_samples,
        poll_interval=1.0,
        attack_start_frac=0.0,             # entire window is attack
        ue_clone_count_during_attack=20,   # 20 clones → definitive spike
        seed=999,
    )

    # Preprocess with saved scaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.data_min_ = np.array(scaler_params["min"])
    scaler.data_max_ = np.array(scaler_params["max"])
    scaler.scale_    = np.array(scaler_params["scale"])
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.n_features_in_ = num_features
    scaler.feature_range  = (0, 1)
    scaler.min_           = -scaler.data_min_ * scaler.scale_
    scaler.n_samples_seen_ = 1

    X_normal, _ = preprocess_telemetry(df_normal, sequence_length=sequence_length, scaler=scaler)
    X_attack, _ = preprocess_telemetry(df_attack, sequence_length=sequence_length, scaler=scaler)

    errors_normal = compute_reconstruction_errors(model, X_normal)
    errors_attack = compute_reconstruction_errors(model, X_attack)

    errors_all = np.concatenate([errors_normal, errors_attack])
    y_true = np.concatenate([
        np.zeros(len(errors_normal), dtype=int),
        np.ones(len(errors_attack), dtype=int),
    ])
    y_pred = (errors_all > threshold).astype(int)

    metrics = compute_metrics(y_true, y_pred)

    print(f"\n{'='*62}")
    print(f" Simulated Detection Results")
    print(f"{'='*62}")
    print(f" Normal sequences  : {len(errors_normal)}")
    print(f" Attack sequences  : {len(errors_attack)}")
    print(f" Saved threshold   : {threshold:.6f}")
    print(f" Normal mean error : {errors_normal.mean():.6f} (max: {errors_normal.max():.6f})")
    print(f" Attack mean error : {errors_attack.mean():.6f} (max: {errors_attack.max():.6f})")
    print(f" Separation ratio  : {errors_attack.mean()/max(errors_normal.mean(),1e-9):.1f}x")
    print(f" Precision         : {metrics['precision']:.4f}")
    print(f" Recall            : {metrics['recall']:.4f}")
    print(f" F1 Score          : {metrics['f1_score']:.4f}")
    print(f"{'='*62}")

    return {
        "threshold": threshold,
        "normal_mean_error":  float(errors_normal.mean()),
        "normal_max_error":   float(errors_normal.max()),
        "attack_mean_error":  float(errors_attack.mean()),
        "separation_ratio":   float(errors_attack.mean() / max(errors_normal.mean(), 1e-9)),
        **metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NWDAF Live Anomaly Monitor")
    parser.add_argument("--simulated", action="store_true",
                        help="Run on simulated data instead of live Ella Core")
    parser.add_argument("--interval",   type=float, default=1.0)
    parser.add_argument("--iterations", type=int,   default=None)
    parser.add_argument("--no-mitigate", action="store_true")
    parser.add_argument("--streak",     type=int,   default=3,
                        help="Consecutive anomalous windows before declaring anomaly")
    parser.add_argument("--cooldown",   type=float, default=5.0,
                        help="Seconds between mitigation actions")
    args = parser.parse_args()

    if args.simulated:
        run_simulated_monitor()
    else:
        run_live_monitor(
            poll_interval=args.interval,
            max_iterations=args.iterations,
            auto_mitigate=not args.no_mitigate,
            anomaly_streak_required=args.streak,
            cooldown_seconds=args.cooldown,
        )
