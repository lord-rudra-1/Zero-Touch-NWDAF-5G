## Alignment Plan: Making the Project Match the “Zero‑Touch NWDAF” Brief

This document explains, **in detail**, how to evolve the current project into the full system described in your teammate’s brief:

> Zero‑Touch NWDAF – Unsupervised Deep Autoencoders for Real‑Time 5G Signaling Anomaly Detection using **Open5GS + UERANSIM + PCAP + LSTM‑AE**.

Right now your code already implements:

- The **unsupervised LSTM‑Autoencoder**.
- Sequence generation, anomaly detection, thresholding, metrics, and visualizations.
- A **real‑time simulation** and a **NWDAF‑style dashboard** using **synthetic data**.

What is missing is the **real 5G network environment and packet capture/feature extraction**. This plan tells you how to add those parts step‑by‑step.

---

## 0. Recommended Environment

While you can prototype some steps on Windows, the **full Open5GS + UERANSIM + tshark pipeline is much easier on Linux**, typically Ubuntu.

- **Recommended OS**: Ubuntu 22.04 LTS (or 20.04).
- **Privileges**: You need `sudo` to install packages and capture on interfaces.
- **Tools to install** (via `apt` or equivalent):
  - `git`, `cmake`, `gcc`, `g++`, `make`
  - `libuv1-dev`, `libmicrohttpd-dev`, `libgnutls28-dev`, etc. (Open5GS deps)
  - `wireshark`, `tshark`
  - `python3`, `python3-venv`, `python3-pip`

You can keep **this codebase** as‑is, push it to GitHub, then **clone it on an Ubuntu VM** where you install Open5GS, UERANSIM, and run the PCAP parts.

---

## 1. Phase 1 – Set Up 5G Core (Open5GS) and RAN/UE (UERANSIM)

### 1.1 Install and configure Open5GS (5G Core)

1. On Ubuntu:

   ```bash
   sudo apt update
   sudo apt install -y git build-essential meson ninja-build flex bison \
       libyaml-dev libgnutls28-dev libgcrypt-dev libmicrohttpd-dev \
       libmongoc-dev libbson-dev libtins-dev libcurl4-gnutls-dev
   ```

2. Clone and build Open5GS:

   ```bash
   git clone https://github.com/open5gs/open5gs.git
   cd open5gs
   meson build --prefix=`pwd`/install
   ninja -C build
   ninja -C build install
   ```

3. Configure a **basic standalone 5GC**:

   - Edit configuration files under `config/` or `install/etc/open5gs/`:
     - `amf.yaml`, `smf.yaml`, `upf.yaml`, `nrf.yaml`, etc.
   - Set PLMN (e.g., `MCC=001`, `MNC=01`), TAC, NSSAI, and APN.
   - Ensure `amf` N2 address and `upf` addresses match your Ubuntu host.

4. Start core services (simplified):

   ```bash
   cd open5gs/install
   ./bin/open5gs-nrfd &
   ./bin/open5gs-amfd &
   ./bin/open5gs-smfd &
   ./bin/open5gs-upfd &
   # ... and other NFs as needed (pcfd, ausfd, udmd, etc.)
   ```

   For real deployments, use systemd service files, but background processes are fine for a lab.

### 1.2 Install and configure UERANSIM (RAN/UE simulator)

1. On the same (or another) Ubuntu VM:

   ```bash
   git clone https://github.com/aligungr/UERANSIM.git
   cd UERANSIM
   make
   ```

2. Configure **gNB** and **UE** profiles:

   - `config/open5gs-gnb.yaml` – set:
     - gNB IP (N2 interface).
     - AMF IP (Open5GS `amf` address).
     - PLMN, tracking area code, etc. consistent with Open5GS.
   - `config/open5gs-ue.yaml` – set:
     - `supi` (IMSI), `key`, `opc` matching subscribers configured in Open5GS (via its web UI or `subscribers` config).

3. Start gNB and UE:

   ```bash
   # In UERANSIM folder
   sudo ./build/nr-gnb -c config/open5gs-gnb.yaml
   sudo ./build/nr-ue  -c config/open5gs-ue.yaml
   ```

   If configured correctly, the UE should register to the 5GC and get an IP (e.g., via `ogstun`).

---

## 2. Phase 2 – Data Generation with PCAPs

Goal: **Create normal and attack PCAP files** that resemble the brief.

### 2.1 Create folder structure for PCAPs

In your project (this repo):

```bash
mkdir -p dataset/pcap/normal
mkdir -p dataset/pcap/attack
```

### 2.2 Capture baseline “Normal PCAP”

1. Identify interfaces:
   - `ogstun` (user-plane tunnel) – useful for data.
   - `lo` or `n2` interface between gNB and AMF (control‑plane NGAP/NAS).

2. Run baseline registration loops (normal behavior):

   - Keep Open5GS and one UE running.
   - Optionally write a small UERANSIM script to repeatedly **attach / detach** or reconnect.

3. Capture with `tshark`:

   ```bash
   sudo tshark -i n2 -w dataset/pcap/normal/normal.pcapng -F pcapng
   ```

   - Let it run while normal registration and data traffic occurs.
   - Stop capture (Ctrl+C) after enough traffic (e.g., 5–10 minutes).

### 2.3 Capture “Attack PCAP” (signaling storm)

Goal: create many rapid attach/registration attempts (“signaling storm”).

1. Write a simple bash loop to spawn multiple UEs or repeated registration:

   ```bash
   # example.sh
   # Simulate a signaling storm by restarting UERANSIM UE repeatedly
   for i in $(seq 1 100); do
     sudo ./build/nr-ue -c config/open5gs-ue.yaml &
     sleep 0.5
     pkill -f nr-ue   # aggressively disconnect
   done
   ```

2. Start tshark capture:

   ```bash
   sudo tshark -i n2 -w dataset/pcap/attack/attack.pcapng -F pcapng
   ```

3. Run the attack script and then stop tshark.

Now you have:

- `dataset/pcap/normal/normal.pcapng`
- `dataset/pcap/attack/attack.pcapng`

---

## 3. Phase 3 – Packet Feature Extraction (PCAP → Sequences)

Your current pipeline expects **normalized numeric sequences** of shape:

```text
(num_sequences, sequence_length, num_features)
```

We must create a module that:

1. Reads PCAP files.
2. Extracts per‑packet features.
3. Groups them into sequences (windows of length 10).
4. Normalizes features to [0, 1].
5. Returns arrays compatible with the existing model.

### 3.1 Choose a PCAP library

Use either:

- **Scapy** (`pip install scapy`) – more manual but very flexible.
- **PyShark** (wrapper around tshark).

A folder/module suggestion:

```text
dataset/
  synthetic_signaling.py          # already exists
  pcap_to_sequences.py            # NEW
```

### 3.2 Design features

Good features for signaling anomaly detection:

- **Packet size**: `len(packet)`.
- **Inter‑arrival time**: delta between timestamps.
- **Direction**: UE → core vs core → UE (0/1).
- **Protocol / message type**:
  - E.g., NGAP, NAS, SCTP; you can encode as small integers.
- Optionally: per‑flow statistics (counts, moving averages).

Example minimal feature vector per packet:

```text
[ size_bytes, delta_t, direction, proto_id ]
```

where:

- `size_bytes`: raw length.
- `delta_t`: `current_ts - previous_ts` (start with 0 for first).
- `direction`: 0 for gNB→AMF, 1 for AMF→gNB (based on source/dest IP).
- `proto_id`: small integer for protocol (e.g., NGAP=1, NAS=2, other=0).

### 3.3 Implement `pcap_to_sequences.py`

Core functions:

```python
def pcap_to_dataframe(pcap_path: str) -> pd.DataFrame:
    # columns: ["timestamp", "size", "delta_t", "direction", "proto_id"]

def df_to_sequences(df: pd.DataFrame, sequence_length: int = 10) -> np.ndarray:
    # sliding window over rows → (num_sequences, sequence_length, num_features)

def load_normal_and_attack_sequences(
    normal_pcap_path: str,
    attack_pcap_path: str,
    sequence_length: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # X_train: from normal PCAP
    # X_attack: from attack PCAP
    # y_attack: labels (all ones)
```

Implementation outline:

1. Loop over packets in PCAP (Scapy or PyShark).
2. Extract the fields and append to a list of dicts.
3. Convert to `pandas.DataFrame`.
4. Sort by timestamp.
5. Compute `delta_t` via `df["timestamp"].diff().fillna(0.0)`.
6. Normalize numeric columns with Min–Max (or reuse `feature_engineering.py`).
7. Convert to NumPy and build sequences using a sliding window (similar to `sequence_generator.py`).

### 3.4 Integrate with your training pipeline

Instead of using `dataset.synthetic_signaling.generate_synthetic_dataset`, you can:

- Add a new option to `demo.py` and `train_model.py`:

  ```bash
  python demo.py --pcap \
      --normal-pcap dataset/pcap/normal/normal.pcapng \
      --attack-pcap dataset/pcap/attack/attack.pcapng
  ```

- In code:
  - If `--pcap` is set, call `load_normal_and_attack_sequences(...)` to obtain `X_train` and `X_test`.
  - Then follow the same steps: train LSTM‑AE, compute errors, set threshold, detect anomalies, plot, etc.

This keeps **synthetic** and **PCAP‑based** paths side‑by‑side.

---

## 4. Phase 4 – Real-Time Detection on Live Interface

The brief’s final phase is a **live script** that monitors traffic and fires an alert when a signaling storm starts.

### 4.1 Design of the live detector

The live detector should:

1. Attach to an interface (e.g., `n2` for NGAP or `ogstun`).
2. Continuously read packets (using `tshark -l` for line‑buffered output or PyShark live capture).
3. Convert each incoming packet into the same **feature vector** as in the PCAP pipeline.
4. Maintain a **rolling window** of the last `T` packets (e.g., 10) to form a sequence.
5. Feed sequences into the LSTM autoencoder (using the **already trained model**).
6. Compute reconstruction error and compare with the pre‑computed threshold.
7. Print or log:

   ```text
   Time | Error | Status (Normal / ALERT)
   ```

### 4.2 Implementation strategy

You can extend `inference/realtime_detector.py` with a `--live` mode:

```bash
python -m inference.realtime_detector --live --iface n2
```

Internally:

1. Load the trained model and threshold (trained on normal PCAP sequences).
2. Start a live capture:

   - **Option A** (`tshark` via subprocess):

     ```python
     proc = subprocess.Popen(
         ["tshark", "-i", iface, "-T", "fields", "-e", "frame.time_epoch", "-e", "frame.len", ...],
         stdout=subprocess.PIPE,
         text=True,
     )
     ```

   - **Option B** (PyShark):

     ```python
     capture = pyshark.LiveCapture(interface=iface, display_filter="ngap || nas-5gs")
     for pkt in capture.sniff_continuously():
         ...
     ```

3. For each packet line:
   - Parse timestamp, size, direction, protocol.
   - Update a fixed-length Python deque for the last `T` packets.
   - When the deque has length `T`, convert it to a `(1, T, num_features)` NumPy array and send it to the model.
4. Compute reconstruction error, compare to threshold, and print:

   ```python
   status = "Anomaly" if error > threshold else "Normal"
   print(f"{human_readable_time} | {error:.4f} | {status}")
   ```

This preserves the **same model and feature scaling** as in offline training.

---

## 5. Phase 5 – NWDAF Dashboard Integration with Real Data

Once you have PCAP generation and/or live capture implemented, you can:

1. **Add PCAP upload** to the dashboard:
   - Allow uploading PCAP files (e.g., `normal.pcapng` and `attack.pcapng`).
   - Run the same `pcap_to_sequences.py` on the uploaded files.
   - Display metrics and plots, just as you now do for synthetic data.

2. **Optional live view** (more advanced):
   - Expose an API endpoint that a separate live detector script writes to (e.g., via HTTP POST with errors and flags).
   - Have Streamlit periodically poll or refresh from a local results file to show near‑real‑time anomaly timelines.

For your current course project, **PCAP upload + offline analytics** is already very strong.

---

## 6. Summary of Changes Needed

To fully align the current codebase with the original brief:

1. **Environment**:
   - Set up Open5GS + UERANSIM on an Ubuntu VM.

2. **Data generation**:
   - Capture **normal** and **attack** PCAPs on relevant interfaces using `tshark`.

3. **PCAP → sequences**:
   - Implement `dataset/pcap_to_sequences.py` that:
     - Reads PCAPs.
     - Extracts per‑packet signaling features.
     - Builds normalized sequences of length 10.
     - Produces `X_train` (normal) and `X_attack` (attack).

4. **Integrate with existing pipeline**:
   - Add PCAP-based options to `demo.py`, `train_model.py`, and optionally `main.py`.
   - Keep synthetic mode as a fallback/demo mode.

5. **Real-time live detector**:
   - Extend `realtime_detector.py` with a `--live` mode that listens on an interface, converts packets to features, and runs inference/alerting.

6. **Dashboard enhancements** (optional but nice):
   - Support PCAP uploads and offline PCAP analysis in the Streamlit app.

With these steps, your project will not only be conceptually aligned (which it already is) but also **architecturally and operationally aligned** with the original plan: a **Zero‑Touch NWDAF‑style analytics engine** driven by **real 5G signaling traffic** from Open5GS + UERANSIM. 

