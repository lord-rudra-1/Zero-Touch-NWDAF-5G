from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

# Ensure project root is on sys.path so that `dataset` and other top-level
# packages can be imported when Streamlit runs this script from the
# `dashboard` directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.synthetic_signaling import generate_synthetic_dataset
from evaluation.metrics import compute_metrics
from evaluation.reconstruction_error import compute_reconstruction_errors, load_trained_model
from evaluation.threshold import select_threshold
from models.lstm_autoencoder import LSTMAutoencoder
from preprocessing.data_split import split_train_test_sequences
from results.save_results import save_anomaly_detection_results


@st.cache_resource
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_or_load_model(X_train: np.ndarray, epochs: int = 20, batch_size: int = 32, lr: float = 1e-3):
    """
    Train a new LSTM autoencoder on normal sequences or load from disk if available.
    """
    device = get_device()
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    model_path = models_dir / "lstm_autoencoder.pth"

    num_features = X_train.shape[-1]

    if model_path.is_file():
        model = load_trained_model(model_path=model_path, num_features=num_features, device=device)
        return model, model_path

    models_dir.mkdir(parents=True, exist_ok=True)

    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    model = LSTMAutoencoder(num_features=num_features).to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    train_dataset = TensorDataset(X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_samples = 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            batch_size_actual = batch_x.size(0)

            recon, _ = model(batch_x)
            loss = criterion(recon, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size_actual
            num_samples += batch_size_actual

        avg_loss = epoch_loss / max(num_samples, 1)
        st.write(f"Epoch {epoch}/{epochs} - Training Loss: {avg_loss:.6f}")

    # Save trained model
    torch.save(model.state_dict(), model_path)
    return model, model_path


def main():
    st.set_page_config(
        page_title="Zero Touch NWDAF Analytics Dashboard",
        layout="wide",
    )

    st.title("Zero Touch NWDAF Analytics Dashboard")
    st.markdown(
        "Interactive dashboard for LSTM autoencoder-based network anomaly detection."
    )

    st.sidebar.header("Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload network traffic CSV", type=["csv"])
    use_synthetic = st.sidebar.checkbox("Use synthetic signaling data (no CSV)", value=False)
    label_column = st.sidebar.text_input("Label column name", value="Label")
    normal_label = st.sidebar.text_input("Normal label value", value="Normal")
    sequence_length = st.sidebar.number_input("Sequence length", min_value=5, max_value=100, value=10, step=1)
    threshold_percentile = st.sidebar.slider(
        "Threshold percentile (based on normal errors)", min_value=80.0, max_value=99.9, value=95.0, step=0.1
    )
    epochs = st.sidebar.number_input("Training epochs", min_value=5, max_value=100, value=20, step=1)
    batch_size = st.sidebar.number_input("Batch size", min_value=8, max_value=256, value=32, step=8)

    run_button = st.sidebar.button("Run Analysis")

    if not use_synthetic and not uploaded_file:
        st.info("Upload a CSV file or enable synthetic data to start the analysis.")
        return

    if not run_button:
        st.stop()

    # ------------------------------------------------------------------
    # Load and preprocess dataset OR generate synthetic data
    # ------------------------------------------------------------------
    st.subheader("1. Load / Generate Data and Preprocess")

    if use_synthetic:
        st.write("Using internally generated synthetic signaling data (no CSV).")
        X_train, X_test, y_test = generate_synthetic_dataset(
            num_train_normal=500,
            num_test_attack=200,
            sequence_length=int(sequence_length),
            num_features=4,
        )
        st.write(f"Synthetic normal training sequences: {X_train.shape}")
        st.write(f"Synthetic attack test sequences   : {X_test.shape}")
    else:
        df = pd.read_csv(uploaded_file)
        st.write("Raw dataset shape:", df.shape)

        if label_column not in df.columns:
            st.error(f"Label column '{label_column}' not found in uploaded dataset.")
            st.stop()

        df_clean = df.dropna().drop_duplicates()
        st.write("After cleaning (drop NaN + duplicates):", df_clean.shape)

        # ------------------------------------------------------------------
        # Feature engineering + sequence generation
        # ------------------------------------------------------------------
        st.subheader("2. Generate LSTM Sequences")

        try:
            X_train, X_test, y_test = split_train_test_sequences(
                df_clean,
                label_column=label_column,
                normal_label=normal_label,
                sequence_length=int(sequence_length),
            )
        except Exception as e:
            st.error(f"Error during sequence generation: {e}")
            st.stop()

        st.write(f"Training sequences (normal): {X_train.shape}")
        st.write(f"Test sequences (attack): {X_test.shape}")

    # ------------------------------------------------------------------
    # Train or load model
    # ------------------------------------------------------------------
    st.subheader("3. Train / Load LSTM Autoencoder")
    with st.spinner("Training or loading model..."):
        model, model_path = train_or_load_model(
            X_train,
            epochs=int(epochs),
            batch_size=int(batch_size),
        )
    st.success(f"Model ready. Path: {model_path}")

    device = get_device()

    # ------------------------------------------------------------------
    # Compute reconstruction errors
    # ------------------------------------------------------------------
    st.subheader("4. Compute Reconstruction Errors and Threshold")

    errors_normal = compute_reconstruction_errors(model, X_train, device=device)
    errors_attack = compute_reconstruction_errors(model, X_test, device=device)

    st.write(
        f"Normal sequences: {len(errors_normal)} | "
        f"mean error = {errors_normal.mean():.6f}, std = {errors_normal.std():.6f}"
    )
    st.write(
        f"Attack sequences: {len(errors_attack)} | "
        f"mean error = {errors_attack.mean():.6f}, std = {errors_attack.std():.6f}"
    )

    threshold = select_threshold(
        errors_normal,
        method="percentile",
        percentile=float(threshold_percentile),
    )

    errors_all = np.concatenate([errors_normal, errors_attack], axis=0)
    y_true = np.concatenate(
        [
            np.zeros_like(errors_normal, dtype=int),
            np.ones_like(errors_attack, dtype=int),
        ]
    )
    y_pred = (errors_all > threshold).astype(int)

    # Save results to CSV
    save_anomaly_detection_results(
        reconstruction_errors=errors_all,
        threshold=threshold,
        anomaly_flags=y_pred,
    )

    # ------------------------------------------------------------------
    # 1. Network statistics
    # ------------------------------------------------------------------
    st.subheader("5. Network Statistics")

    total_sequences = len(errors_all)
    total_anomalies = int(y_pred.sum())
    anomaly_percentage = (total_anomalies / total_sequences) * 100.0 if total_sequences > 0 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total sequences analyzed", f"{total_sequences}")
    col2.metric("Anomalies detected", f"{total_anomalies}")
    col3.metric("Anomaly percentage", f"{anomaly_percentage:.2f}%")

    # ------------------------------------------------------------------
    # 2. Reconstruction error plot (error vs time with threshold)
    # ------------------------------------------------------------------
    st.subheader("6. Reconstruction Error Over Time")

    time_index = np.arange(total_sequences)
    df_plot = pd.DataFrame(
        {
            "time": time_index,
            "reconstruction_error": errors_all,
            "anomaly_flag": y_pred,
        }
    )

    fig = px.line(
        df_plot,
        x="time",
        y="reconstruction_error",
        color="anomaly_flag",
        color_discrete_map={0: "steelblue", 1: "red"},
        labels={"time": "Sequence Index", "reconstruction_error": "Reconstruction Error", "anomaly_flag": "Anomaly"},
        title="Reconstruction Error vs Time",
    )
    # Add threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="black", annotation_text="Threshold")

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 3. Detected anomaly table
    # ------------------------------------------------------------------
    st.subheader("7. Detected Anomalies Table")

    df_results = pd.DataFrame(
        {
            "timestamp": time_index,
            "reconstruction_error": errors_all,
            "anomaly_flag": y_pred,
        }
    )

    anomalies_only = df_results[df_results["anomaly_flag"] == 1]
    st.dataframe(anomalies_only, use_container_width=True)

    # ------------------------------------------------------------------
    # 4. Evaluation metrics
    # ------------------------------------------------------------------
    st.subheader("8. Evaluation Metrics")
    metrics = compute_metrics(y_true, y_pred)
    st.json(metrics)

    st.success("Analysis completed. Use the sidebar to adjust parameters or upload a new dataset.")


if __name__ == "__main__":
    main()

