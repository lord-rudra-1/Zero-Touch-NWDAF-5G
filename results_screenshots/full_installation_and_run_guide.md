# 5G Infrastructure Installation & Live Run Guide

Because the system requires `sudo` (administrator) privileges to install system-level network components, you will need to run these installation commands manually in your terminal. Once installed, you can proceed with the live execution.

---

## Part 1: Installation Guide

### Step 1: Install UERANSIM Dependencies
UERANSIM simulates the 5G UE (User Equipment) and gNB (Base Station). It requires a C++ build environment and SCTP networking libraries.
Run this in your terminal:
```bash
sudo apt update
sudo apt install -y make cmake gcc g++ libsctp-dev lksctp-tools iproute2 git
```

### Step 2: Clone and Build UERANSIM
You need to clone the UERANSIM repository directly inside your project folder so the scripts can find it.
```bash
cd /home/codehasira23/ai-in-5g-project
git clone https://github.com/aligungr/UERANSIM
cd UERANSIM
make
cd ..
```
*Wait for the compilation to finish. It will create `nr-gnb` and `nr-ue` binaries in the `UERANSIM/build` directory.*

### Step 3: Install Ella Core
Ella Core acts as the 5G Core Network. Based on the runbook, it runs as a snap service.
```bash
sudo snap install ella-core
```
*(If this requires a specific channel or is a local snap file, follow your standard Ella Core distribution instructions).*

Once installed, check that the service is running:
```bash
sudo snap services ella-core
```

---

## Part 2: Live Execution Guide

Now that the infrastructure is installed, you can run the real live environment! You will need multiple terminal tabs open.

### 1. Security Alignment & Provisioning
First, discover your Ella Core API token and export it. Then provision the 25 UE profiles into the core so they can authenticate.

```bash
# Terminal 1 (in ai-in-5g-project directory)
export ELLA_API_TOKEN="ellacore_MzR9YhsEKcBt_n84hgUW6As00rak1XlPb55dS"

source .venv/bin/activate
python3 tools/provision_clones.py
```
📸 **Screenshot 1:** Capture the terminal showing 25 subscribers successfully provisioned.

### 2. Generate Network Configurations
Generate the YAML configurations for the gNB and the 25 UEs.
```bash
# Terminal 1
python3 -m ran.gnb_config
python3 tools/ue_generator.py
```
📸 **Screenshot 2:** Capture the successful generation of `gnb.yaml` and `ue_*.yaml`.

### 3. Start the 5G Network
```bash
# Terminal 2: Start the Base Station (gNB)
cd /home/codehasira23/ai-in-5g-project
sudo ./UERANSIM/build/nr-gnb -c ./config/gnb.yaml
```

Wait a few seconds for it to connect, then:
```bash
# Terminal 3: Start 25 UEs simultaneously
cd /home/codehasira23/ai-in-5g-project
sudo ./tools/launch_ues.sh
```

**Verify the connections:**
```bash
# Terminal 1
grep -c "PDU Session establishment is successful" logs/ue_*.log
```
📸 **Screenshot 3:** Capture the gNB terminal running, the UEs launching, and the `grep` command returning 25 successful sessions.

### 4. Start AI Live Monitoring
With the network running, start the AI anomaly detector.
```bash
# Terminal 4
cd /home/codehasira23/ai-in-5g-project
source .venv/bin/activate
python3 main.py --detect --interval 1.0 --iterations 300
```
📸 **Screenshot 4:** Capture the live monitor running smoothly with a low error rate (e.g., ~0.02).

### 5. Launch the Attack & Observe Mitigation
Inject the signaling storm bot attack.
```bash
# Terminal 5
cd /home/codehasira23/ai-in-5g-project
source .venv/bin/activate
python3 -m simulation.attack_generator --clones 25 --duration 30 --cleanup
```

📸 **Screenshot 5:** Switch back to Terminal 4 (Live Monitor) and capture the moment the Reconstruction Error spikes drastically and triggers the `⚠ ANOMALY` status.
📸 **Screenshot 6:** Capture Terminal 4 showing the automated `→ BLOCKED imsi-...` mitigation commands firing.

---

### Cleanup
When done, kill the UERANSIM processes:
```bash
sudo pkill -9 nr-gnb && sudo pkill -9 nr-ue
```
