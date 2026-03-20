## Zero Touch NWDAF: Unsupervised Deep Autoencoders for Real‑Time 5G Signaling Anomaly Detection

This project implements a **zero‑touch, unsupervised anomaly detection system** for 5G signaling traffic, inspired by the **NWDAF (Network Data Analytics Function)** defined in 3GPP 5G architecture.

An **LSTM autoencoder** is trained only on _normal_ signaling sequences. At runtime, reconstruction error is used as an anomaly score. High errors indicate patterns that deviate from learned “normal” behavior and are flagged as anomalies. The project includes:

- A full preprocessing and sequence generation pipeline.
- LSTM autoencoder model and training scripts.
- Synthetic signaling and attack simulation (no external dataset required).
- Evaluation and visualization modules (ROC, confusion matrix, timelines).
- A **Streamlit NWDAF Analytics Dashboard** for interactive analysis.
- A real‑time console simulator for streaming detection.

---

## 1. Project Structure

High‑level layout:

```text
ai-in-5g-project/
  dataset/
    synthetic_signaling.py        # Synthetic normal + attack signaling sequences
  preprocessing/
    load_data.py                  # (CSV loader, optional)
    feature_engineering.py        # Numeric feature selection + MinMax scaling
    sequence_generator.py         # Tabular → LSTM sequences via sliding window
    data_split.py                 # Split into normal (train) and attack (test)
  models/
    lstm_autoencoder.py           # LSTM autoencoder definition (encoder/decoder)
  training/
    train_model.py                # Training script (CSV-based pipeline)
  inference/
    anomaly_detector.py           # Batch anomaly detection using trained model
    realtime_detector.py          # Real-time streaming detector (synthetic or CSV)
  evaluation/
    reconstruction_error.py       # Per-sequence reconstruction error + histogram
    threshold.py                  # Threshold selection (percentile / statistical)
    attack_simulator.py           # Packet flood, abnormal timing, size spikes
    test_attacks.py               # Evaluate model on simulated attacks
    metrics.py                    # Precision, recall, F1, confusion matrix
    results_summary.py            # High-level text summary for reports
  visualization/
    error_plot.py                 # Error distribution + error vs time
    error_timeseries.py           # Publication-quality time-series plot
    anomaly_timeline.py           # Error vs time with shaded anomaly regions
    roc_curve.py                  # ROC curve + AUC
    final_plots.py                # Set of final figures for the report
  dashboard/
    app.py                        # Streamlit "Zero Touch NWDAF Analytics Dashboard"
  results/
    save_results.py               # Save errors, thresholds, flags as CSV
  utils/
    save_system_state.py          # Save model, errors, threshold, metrics per run
  docs/
    architecture.py               # Generates architecture_diagram.png
  demo.py                         # End-to-end demo script (supports synthetic mode)
  main.py                         # Integrated CSV-based pipeline (optional)
  requirements.txt                # Python dependencies
  .gitignore                      # Git ignore rules
```

---

## 2. Requirements

- **Python**: 3.11+ recommended
- OS: Windows, Linux, or macOS
- Packages: installed via `requirements.txt`:
  - `numpy`, `pandas`, `scikit-learn`, `torch`, `matplotlib`, `seaborn`
  - `streamlit`, `plotly`, plus supporting libraries

---

## 3. Environment Setup

From the project root:

### 3.1 Create and activate a virtual environment

**Windows (PowerShell)**:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell complains about execution policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

**Linux / macOS (bash/zsh)**:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.2 Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 4. Running the System (No External Dataset Required)

The project is designed to work **without any external CSV file**, using internally generated synthetic 5G signaling data.

### 4.1 End-to-end demo (synthetic mode)

Runs the full anomaly detection pipeline (training, thresholding, metrics, and plots) using **synthetic normal + attack sequences**:

```bash
python demo.py --synthetic
```

What this does:

1. Generates synthetic normal training sequences and attack test sequences:
   - Normal: 500 sequences.
   - Attack: 200 sequences created by `evaluation.attack_simulator`.
2. Trains an LSTM autoencoder on normal sequences (if no model saved yet).
3. Computes reconstruction errors for normal and attack sequences.
4. Selects an anomaly threshold (by default 95th percentile of normal errors).
5. Detects anomalies and prints:
   - Precision, recall, F1 score.
   - Confusion matrix (TP, FP, FN, TN).
   - Overall detection rate and false positives.
6. Produces high-quality plots:
   - Reconstruction error distribution.
   - Error vs time with anomaly spikes.
   - Anomaly timeline.
   - ROC curve and AUC.
   - Confusion matrix heatmap.

This is the recommended command for **offline demonstrations** and generating figures for the report.

### 4.2 Real-time anomaly detection simulation (synthetic streaming)

Simulates a **real-time signaling stream** using synthetic data:

```bash
python -m inference.realtime_detector --synthetic
```

Behavior:

- Generates synthetic normal and attack sequences.
- Loads the trained LSTM autoencoder (`models/lstm_autoencoder.pth`).
- Computes a threshold from normal errors.
- Streams sequences one by one, printing:

```text
Time | Error | Status
----------------------
00:00:00 | 0.0345 | Normal
00:00:02 | 0.9123 | Anomaly
...
```

- Includes a short delay between predictions to mimic real-time operation.

This is ideal for **live console demos** showing real-time behavior.

### 4.3 NWDAF Analytics Dashboard (Streamlit UI)

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

Then open the local URL (e.g., `http://localhost:8501`) in a browser.

In the sidebar:

- Check **“Use synthetic signaling data (no CSV)”**.
- Adjust:
  - `Sequence length` (default 10).
  - `Threshold percentile` (default 95.0).
  - `Training epochs` and `Batch size`.
- Click **Run Analysis**.

The dashboard will:

1. Generate synthetic normal and attack sequences.
2. Train or load the LSTM autoencoder.
3. Compute reconstruction errors and select a threshold.
4. Display:
   - **Network statistics**:
     - Total sequences analyzed.
     - Number of anomalies detected.
     - Anomaly percentage.

   - **Reconstruction error over time** (Plotly):
     - X-axis: sequence index.
     - Y-axis: reconstruction error.
     - Color: normal vs anomaly.
     - Horizontal threshold line.

   - **Detected anomalies table**:
     - Columns: `timestamp`, `reconstruction_error`, `anomaly_flag`.

   - **Evaluation metrics**:
     - Precision, recall, F1 score.
     - Confusion matrix details (TP, FP, FN, TN).

This UI serves as a **Zero Touch NWDAF analytics dashboard** for presentations.

---

## 5. Optional: CSV-Based Pipeline

Although the system can run entirely without external data, it can also operate on a real CSV dataset (network intrusion / signaling dataset).

Expected CSV format (for `main.py`, `training/train_model.py`, and non-synthetic `demo.py`):

- Numeric feature columns (traffic statistics).
- A label column (default name `Label`) with:
  - `"Normal"` for normal traffic.
  - Any other value for attack/anomaly samples.

Example usage:

```bash
python main.py data/your_dataset.csv
python demo.py data/your_dataset.csv
python -m inference.realtime_detector data/your_dataset.csv
```

---

## 6. Implementation Overview

### 6.1 LSTM Autoencoder

- **Encoder**:
  - LSTM (64 units) → LSTM (32 units).
  - Final hidden state projected to a 16‑dimensional latent vector.
- **Decoder**:
  - Latent vector expanded and repeated across time.
  - LSTM (32 units) → LSTM (64 units).
  - Time-distributed dense layer reconstructing original feature dimension.
- **Loss function**:
  - Mean Squared Error (MSE) between input and reconstruction.
- **Input shape**:
  - `(batch_size, sequence_length, num_features)`.

### 6.2 Anomaly Detection Logic

- Train on **normal sequences only**.
- For each sequence, compute an average MSE reconstruction error.
- Compute a **threshold** using:
  - Percentile method (e.g., 95th percentile of normal errors).
  - Or statistical method: `mean + 3 * std`.
- If `error > threshold` ⇒ anomaly; else ⇒ normal.

### 6.3 Evaluation and Visualization

- **Metrics**:
  - Precision, recall, F1 score.
  - Confusion matrix (TP, FP, FN, TN).
- **Plots**:
  - Training loss vs epochs.
  - Reconstruction error distributions.
  - Error vs time with highlighted anomaly spikes.
  - ROC curve and AUC.
  - Confusion matrix heatmap.
  - Anomaly timelines with shaded anomaly regions.

---

## 7. Saving Results and System State

- `results/save_results.py`:
  - Exports `reconstruction_error`, `threshold`, and `anomaly_flag` to `results/anomaly_detection_results.csv`.
- `utils/save_system_state.py`:
  - Creates a timestamped folder under `results/run_YYYYMMDD_HHMMSS/` and saves:
    - Model weights (`.pth`).
    - Reconstruction errors (`.npy` and `.csv`).
    - Threshold and evaluation metrics (`summary.json`).

These utilities support **reproducibility** and post-hoc analysis.

---

## 8. GitHub and Cross-Platform Notes

- The project uses **relative imports** and `pathlib` so it is OS-agnostic.
- To run on Linux/macOS:
  - Use `source .venv/bin/activate` instead of `Activate.ps1`.
  - All commands (`python demo.py --synthetic`, `python -m inference.realtime_detector --synthetic`, `streamlit run dashboard/app.py`) remain the same.
- `.gitignore` is configured to avoid committing:
  - Virtual environment (`.venv/`).
  - Python caches (`__pycache__/`, `*.pyc`).
  - Models (`models/*.pth`) and result folders (`results/`).

---

## 9. Summary

This repository provides a complete, **unsupervised, zero‑touch** anomaly detection pipeline for 5G signaling data:

- LSTM autoencoder modeling of normal signaling sequences.
- Synthetic data and attack generation (no external dataset required).
- Threshold-based anomaly scoring and evaluation metrics.
- Real-time console simulation and a NWDAF-style analytics dashboard.
- Research-quality visualizations and reporting utilities.

It is ready to be:

- **Run locally** for experimentation and demos.
- **Pushed to GitHub** and executed on other operating systems with the same setup and run commands.
