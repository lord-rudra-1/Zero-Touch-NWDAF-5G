<p align="center">
  <h1 align="center">🛡️ Zero-Touch NWDAF</h1>
  <p align="center">
    <em>Unsupervised Deep Autoencoders for Real-Time 5G Signaling Anomaly Detection & Self-Healing</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg" alt="PyTorch 2.x">
  <img src="https://img.shields.io/badge/5G_Core-Ella_Core-green.svg" alt="Ella Core">
  <img src="https://img.shields.io/badge/RAN-UERANSIM-orange.svg" alt="UERANSIM">
  <img src="https://img.shields.io/badge/License-Academic-lightgrey.svg" alt="License">
</p>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Core Concepts](#core-concepts)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Demo Mode (No External Dependencies)](#demo-mode-no-external-dependencies)
  - [Training Only](#training-only)
  - [Detection Only](#detection-only)
  - [Full Pipeline (With Ella Core)](#full-pipeline-with-ella-core)
- [Phase-by-Phase Breakdown](#phase-by-phase-breakdown)
- [Configuration](#configuration)
- [Results & Metrics](#results--metrics)

---

## Project Overview

The **3rd Generation Partnership Project (3GPP)** defines the **Network Data Analytics Function (NWDAF)** as the standard mechanism for introducing AI-driven intelligence into the 5G Core. This project implements a **fully automated, closed-loop NWDAF** designed specifically to detect and mitigate **zero-day signaling storms** on the 5G control plane.

### The Problem

A **signaling storm** occurs when a massive volume of control-plane messages (registrations, authentications, session requests) overwhelms the 5G core functions (AMF, SMF, UDM). Unlike volumetric DDoS attacks, these exploit the **signaling protocol itself** and are invisible to traditional firewalls.

### Our Solution

Instead of relying on supervised machine learning — which requires labelled datasets of known attacks — this system utilises an **unsupervised LSTM-Autoencoder**. The neural network is trained **exclusively on healthy network traffic** to learn the temporal and structural baseline of the network. When an unknown attack occurs, the model produces a **spike in reconstruction error**, and an automated closed-loop response blocks the malicious subscriber in real time.

---

## Core Concepts

### 1. LSTM-Autoencoder (Unsupervised Anomaly Detection)

```
┌──────────────────────────────────────────────────────┐
│                  LSTM-Autoencoder                     │
│                                                      │
│  Input (T, F) ──► Encoder ──► Latent Z ──► Decoder ──► Reconstruction (T, F)
│                   (64→32→16)   (16-D)    (16→32→64)  │
│                                                      │
│  Loss = MSE(Input, Reconstruction)                   │
│  Anomaly = Loss > Threshold                          │
└──────────────────────────────────────────────────────┘
```

- **Encoder**: Three LSTM layers (64 → 32 → 16 hidden units) compress temporal sequences into a 16-dimensional latent representation
- **Decoder**: Three LSTM layers mirror the encoder to reconstruct the original input
- **Training**: Only on **normal** traffic — the model learns what "healthy" looks like
- **Detection**: Anomalous traffic produces high reconstruction error because the model has never seen attack patterns

### 2. Activity-Based Mobility Model (ABMM)

Rather than static ping loops, our ABMM simulates **human-like UE behaviour** based on time-of-day schedules:

| Time Period | Simulated Activity | Signaling Events |
|---|---|---|
| 00:00 – 07:00 | Home (idle/sleep) | Minimal: periodic TAU |
| 07:00 – 09:00 | Commute (Home → Coffee → Work) | Registration, Handover, PDU Setup |
| 09:00 – 12:00 | Work (data sessions) | Active PDU, Service Requests |
| 12:00 – 13:00 | Lunch break (Coffee) | Handover, Short data bursts |
| 13:00 – 17:00 | Work (data sessions) | Active PDU, Service Requests |
| 17:00 – 19:00 | Commute (Work → Park → Home) | Handover, Deregistration/Registration |
| 19:00 – 24:00 | Home (streaming/browsing) | Steady data, eventual idle |

This generates **temporally grounded, spatially meaningful** movement traces that produce realistic signaling patterns on the 5G Core.

### 3. Closed-Loop Self-Healing

```
Ella Core ──► Prometheus /metrics ──► Telemetry Collector
                                           │
                                     ┌─────▼─────┐
                                     │  Sliding   │
                                     │  Window    │
                                     │  Buffer    │
                                     └─────┬─────┘
                                           │
                                     ┌─────▼─────┐
                                     │ LSTM-AE    │
                                     │ Inference  │
                                     └─────┬─────┘
                                           │
                                     Reconstruction Error > Threshold?
                                           │
                                    ┌──────┴──────┐
                                    │ YES         │ NO
                                    ▼             ▼
                              ┌──────────┐  ┌──────────┐
                              │ BLOCK    │  │ Continue  │
                              │ via API  │  │ Monitoring│
                              └──────────┘  └──────────┘
```

When an anomaly is detected, the system **automatically** calls Ella Core's REST API to block or throttle the offending subscriber — achieving **zero-touch mitigation** without human intervention.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NWDAF System                             │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ UERANSIM │◄──►│  Ella Core   │◄──►│  Prometheus /metrics │  │
│  │ (gNB+UE) │    │  (5G Core)   │    │  (Telemetry Source)  │  │
│  └──────────┘    └──────────────┘    └──────────┬───────────┘  │
│       │                │                         │              │
│       │           REST API                       │              │
│       │           (block/throttle)                │              │
│  ┌────▼────┐          ▲                  ┌───────▼──────┐      │
│  │  ABMM   │          │                  │  Telemetry   │      │
│  │(Mobility)│          │                  │  Collector   │      │
│  └─────────┘    ┌─────┴──────┐           └───────┬──────┘      │
│                 │ Mitigation │                    │              │
│  ┌─────────┐   │  Module    │           ┌────────▼─────┐       │
│  │ Attack  │   └────────────┘           │ Preprocessor │       │
│  │Generator│                            │ (Normalise)  │       │
│  └─────────┘                            └────────┬─────┘       │
│                                                  │              │
│                                         ┌────────▼─────┐       │
│                                         │  LSTM-AE     │       │
│                                         │  (Train /    │       │
│                                         │   Detect)    │       │
│                                         └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Technology | Role |
|---|---|---|
| **5G Core** | [Ella Core](https://ellacore.io/) | Go-based, single binary, eBPF-accelerated data plane |
| **RAN & UE Simulation** | [UERANSIM](https://github.com/aligungr/UERANSIM) | C++ implementation of 5G gNB and UE |
| **Data Ingestion** | Prometheus HTTP API | JSON/HTTP metric scraping at 1 Hz |
| **AI/ML Engine** | Python + PyTorch | LSTM-Autoencoder for unsupervised anomaly detection |
| **Data Processing** | Pandas + Scikit-Learn + NumPy | Feature extraction, normalisation, sequence generation |
| **Automation** | Python `requests` | REST API interaction for closed-loop mitigation |
| **Visualisation** | Matplotlib + Seaborn | Error distributions, ROC curves, anomaly timelines |

---

## Project Structure

```
ai-in-5g-project/
│
├── main.py                          # Master orchestrator (demo/train/detect/full)
├── requirements.txt                 # Python dependencies
├── .gitignore
├── readme.md                        # ← You are here
│
├── core/                            # Phase 1: Ella Core integration
│   ├── ella_config.py               #   Configuration constants (PLMN, slice, keys, URLs)
│   ├── ella_setup.py                #   Deploy / start / stop core + subscriber API
│   └── verify_connectivity.py       #   End-to-end health checks
│
├── ran/                             # Phase 1: UERANSIM integration
│   ├── gnb_config.py                #   gNB YAML config generator + process manager
│   └── ue_config.py                 #   UE YAML config generator + process manager
│
├── telemetry/                       # Phase 2: Data collection
│   ├── collector.py                 #   Polls /metrics, computes counter deltas
│   └── preprocessor.py              #   MinMax normalisation → LSTM sequences
│
├── simulation/                      # Phase 2: Traffic generation
│   ├── abmm.py                      #   Activity-Based Mobility Model
│   └── attack_generator.py          #   Signaling storm generator
│
├── models/                          # Phase 3: AI model
│   └── lstm_autoencoder.py          #   LSTM Encoder-Decoder architecture
│
├── training/                        # Phase 3: Model training
│   └── train_model.py               #   Training loop + model save
│
├── preprocessing/                   # Phase 3: Data preparation
│   └── sequence_generator.py        #   Sliding window → LSTM sequences
│
├── pipeline/                        # Phase 3: End-to-end training
│   └── train_pipeline.py            #   Telemetry → preprocess → train → threshold
│
├── inference/                       # Phase 4: Real-time detection
│   ├── live_monitor.py              #   Live polling + LSTM-AE inference + alerting
│   └── mitigation.py                #   Block/throttle subscribers via REST API
│
├── evaluation/                      # Evaluation & metrics
│   ├── reconstruction_error.py      #   Per-sequence MSE computation
│   ├── threshold.py                 #   Threshold selection (percentile / statistical)
│   ├── metrics.py                   #   Precision, recall, F1, confusion matrix
│   └── results_summary.py           #   Text summary generator
│
└── visualization/                   # Plotting & figures
    ├── error_plot.py                #   Error distribution + error-over-time
    ├── error_timeseries.py          #   Time-series with anomaly markers
    ├── anomaly_timeline.py          #   Shaded anomaly regions
    ├── roc_curve.py                 #   ROC curve + AUC
    └── final_plots.py              #   Report-quality figure set
```

---

## Installation

### Prerequisites

- **Python** 3.10 or later
- **pip** (package installer)
- *Optional*: Ella Core binary + UERANSIM (for full pipeline mode)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ai-in-5g-project.git
cd ai-in-5g-project

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Optional: Ella Core & UERANSIM

For the **full pipeline** (live telemetry + real signaling), install:

```bash
# Ella Core (check https://ellacore.io/docs for latest instructions)
# Binary should be available at $PATH as 'ella-core'

# UERANSIM (build from source)
git clone https://github.com/aligungr/UERANSIM
cd UERANSIM
make
```

---

## Usage

### Demo Mode (No External Dependencies)

Run the complete pipeline with simulated data — no Ella Core or UERANSIM required:

```bash
python3 main.py --demo
```

This will:
1. Generate simulated normal telemetry (300 samples)
2. Preprocess and create LSTM sequences
3. Train the LSTM-Autoencoder (50 epochs)
4. Compute anomaly threshold (95th percentile)
5. Generate simulated attack telemetry
6. Run detection and report Precision, Recall, F1

**Quick demo** (fewer epochs for faster execution):

```bash
python3 main.py --demo --epochs 20 --train-duration 120
```

### Training Only

Train the model on pre-collected telemetry CSV or simulated data:

```bash
# With simulated data
python3 main.py --train --epochs 50

# With a pre-collected CSV
python3 main.py --train --csv data/normal_telemetry.csv --epochs 100
```

### Detection Only

Run anomaly detection using a pre-trained model:

```bash
# Simulated mode (demo)
python3 main.py --detect --simulated

# Live mode (requires Ella Core running + trained model)
python3 main.py --detect --interval 1.0
```

### Full Pipeline (With Ella Core)

Requires Ella Core and UERANSIM installed:

```bash
# 1. Generate UERANSIM configs
python3 -m ran.gnb_config
python3 -m ran.ue_config

# 2. Verify environment
python3 -m core.verify_connectivity

# 3. Run full pipeline
python3 main.py --full --abmm-hours 4 --epochs 50
```

---

## Phase-by-Phase Breakdown

### Phase 1: Environment Setup & Core Initialization

| Module | Purpose |
|---|---|
| `core/ella_config.py` | Centralised configuration (PLMN: 00101, SST: 1, subscriber keys, API endpoints) |
| `core/ella_setup.py` | Binary discovery, process lifecycle, subscriber provisioning via REST API |
| `core/verify_connectivity.py` | Health checks: API responding, metrics flowing, subscriber registered |
| `ran/gnb_config.py` | Generates UERANSIM-compatible gNB YAML (AMF address, NGAP, slices) |
| `ran/ue_config.py` | Generates UE YAML with matching crypto keys (SUPI, Ki, OPc) |

### Phase 2: Data Generation & Telemetry Extraction

| Module | Purpose |
|---|---|
| `telemetry/collector.py` | Polls Ella Core's `/metrics` at 1 Hz, computes counter deltas (NGAP msg/s, NAS msg/s, etc.) |
| `telemetry/preprocessor.py` | MinMax normalises to [0,1], converts to sliding-window sequences (N, T, F) |
| `simulation/abmm.py` | Orchestrates UERANSIM UEs through Home→Work→Coffee→Park mobility schedules |
| `simulation/attack_generator.py` | Spawns 50+ cloned UEs for rapid attach/detach signaling floods |

### Phase 3: AI Model Development

| Module | Purpose |
|---|---|
| `models/lstm_autoencoder.py` | LSTM Encoder (64→32→16) + Decoder (16→32→64) architecture |
| `training/train_model.py` | Training loop with Adam optimiser, MSE loss, accepts numpy arrays |
| `preprocessing/sequence_generator.py` | Sliding window converter: 2D (N, F) → 3D (N, T, F) |
| `pipeline/train_pipeline.py` | End-to-end: collect → preprocess → train → threshold → save model |

### Phase 4: Real-Time Detection & Zero-Touch Mitigation

| Module | Purpose |
|---|---|
| `inference/live_monitor.py` | Continuous polling loop → LSTM-AE inference → anomaly flag |
| `inference/mitigation.py` | `block_subscriber()`, `throttle_subscriber()`, `unblock_subscriber()` via REST API |

---

## Configuration

All configuration is centralised in `core/ella_config.py` and can be overridden via environment variables:

| Environment Variable | Default | Description |
|---|---|---|
| `ELLA_HOST` | `127.0.0.1` | Ella Core host address |
| `ELLA_API_PORT` | `9090` | REST API port |
| `ELLA_METRICS_PORT` | `9090` | Prometheus metrics port |
| `ELLA_API_TOKEN` | *(empty)* | Bearer token for API auth |

### Model Hyperparameters

| Parameter | Default | CLI Flag |
|---|---|---|
| Sequence length | 10 | `--seq-len` |
| Training epochs | 50 | `--epochs` |
| Batch size | 32 | `--batch-size` |
| Learning rate | 1e-3 | *(hardcoded, adjustable in code)* |
| Threshold percentile | 95% | *(adjustable in code)* |

---

## Results & Metrics

After running `python3 main.py --demo`, you should see output similar to:

```
============================================================
 Simulated Detection Results
============================================================
 Normal sequences  : 191
 Attack sequences  : 91
 Threshold         : 0.002345
 Normal mean error : 0.000812
 Attack mean error : 0.145672
 Precision         : 0.9890
 Recall            : 1.0000
 F1 Score          : 0.9945
============================================================
```

The attack reconstruction error is **~180× higher** than the normal baseline, demonstrating clear separation between healthy and anomalous traffic patterns.

After training, the following files are saved under `models/`:

| File | Contents |
|---|---|
| `lstm_autoencoder.pth` | Trained model weights (PyTorch state_dict) |
| `training_metadata.json` | Threshold, training stats, hyperparameters |
| `scaler_params.json` | MinMaxScaler parameters for inference normalisation |

---

<p align="center">
  <em>Built as part of a research project on AI-driven 5G Core Network Security</em>
</p>
