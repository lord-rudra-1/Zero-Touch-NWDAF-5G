# Zero-Touch NWDAF — Execution Runbook

This runbook provides a step-by-step guide to deploying and validating the 5G Signaling Anomaly Detection pipeline. It covers everything from fresh environment setup to a 25-UE scaled demonstration.

---

## 📋 1. Prerequisites & Environment Setup

### System Requirements
*   **OS**: Ubuntu 20.04 or 22.04 LTS (Linux is required for SCTP and UERANSIM support).
*   **Python**: 3.10 or higher.
*   **Infrastructure**: [UERANSIM](https://github.com/aligungr/UERANSIM) and [Ella Core](https://ellacore.io/) installed and in `$PATH`.

### Local Setup
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd ai-in-5g-project

# 2. Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Dependencies
pip install -r requirements.txt
```

---

## ⚙️ 2. Configuration & Security Alignment

Before starting the UEs, you MUST align the security credentials between the UE configs and the Ella Core database.

### Step A: Discover API Token
The system uses a Bearer token to talk to Ella Core. Find your token:
```bash
# Search your history or check Ella Core settings
export ELLA_API_TOKEN="ellacore_MzR9YhsEKcBt_n84hgUW6As00rak1XlPb55dS"
```

### Step B: Bulk-Provision Subscribers
This command pushes 25 unique subscribers to the core with the correct `OPC`, `Key`, and `SQN` values to prevent "MAC Mismatch" errors.
```bash
python3 tools/provision_clones.py
```

---

## 🚀 3. Execution Use Cases

### Level 1: The Quick Demo (Simulated)
Run this if you don't have Ella Core or UERANSIM installed. It uses synthetic telemetry to prove the AI logic works.
```bash
python3 main.py --demo --epochs 30
```

### Level 2: The Scaled Lab (Full 5G)
Deploy 25 UEs and establish a "Normal" baseline.

1.  **Generate Configs**:
    ```bash
    python3 -m ran.gnb_config
    python3 tools/ue_generator.py
    ```
2.  **Start the Network**:
    ```bash
    # Open Terminal 1: Start gNB
    sudo ./UERANSIM/build/nr-gnb -c ./config/gnb.yaml
    
    # Open Terminal 2: Start 25 UEs
    sudo ./tools/launch_ues.sh
    ```
3.  **Verify Attachment**:
    ```bash
    grep -c "PDU Session establishment is successful" logs/ue_*.log
    # Total should be 25
    ```

### Level 3: AI Monitoring & Mitigation
Launch the live observer to protect the network.
```bash
# With the 25 UEs already running:
python3 main.py --detect --interval 1.0 --iterations 300
```

---

## 🛡️ 4. Anomaly Detection & Attack Scenarios

### Launching a Signaling Storm
In a new terminal, simulate 50 bots attempting to flood the core:
```bash
python3 -m simulation.attack_generator --clones 25 --duration 30 --cleanup
```

### Observing the Result
*   **Reconstruction Error**: Watch the `live_monitor` output. Error should spike from ~0.50 to 8.00+.
*   **Mitigation**: Check `results/mitigation_log.json` to see the automated `BLOCK` commands sent to the core.

---

## 🛠️ 5. Troubleshooting Cheat Sheet

| Issue | Root Cause | Fix |
|:---|:---|:---|
| **MAC Mismatch** | Key/OPc out of sync | Run `python3 tools/provision_clones.py` |
| **Connection Refused (5002)** | Core API is down | `sudo snap restart ella-core` |
| **PDU Session Failure** | Slices (SST/SD) mismatch | Check values in `core/ella_config.py` vs core UI |
| **Zombied Processes** | UERANSIM didn't close | `sudo pkill -9 nr-gnb && sudo pkill -9 nr-ue` |

---

**Project Lead**: AI In 5G Team  
**Last Updated**: April 16, 2026
