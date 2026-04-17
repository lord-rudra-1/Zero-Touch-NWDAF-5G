#!/bin/bash
set -e

# Change to the correct project directory
cd "/home/codehasira23/AI IN %G/ai-in-5g-project"

echo "=========================================================="
echo "    ZERO-TOUCH NWDAF PIPELINE END-TO-END LIVE DEMO        "
echo "=========================================================="
echo ""

# 1. Cleanup
echo "[1/5] Cleaning up any stale UERANSIM processes..."
.venv/bin/python3 -m telemetry.collector --cleanup
echo ""

# 2. Baseline Collection
echo "[2/5] Collecting 60s of LIVE baseline telemetry from Ella Core..."
.venv/bin/python3 -m telemetry.collector --duration 60 --interval 1.0 --output data/demo_normal.csv
echo ""

# 3. Model Training
echo "[3/5] Training LSTM Autoencoder on fresh baseline telemetry..."
.venv/bin/python3 -m pipeline.train_pipeline --csv data/demo_normal.csv --epochs 50 --batch-size 16 --seq-len 10
echo ""

# 4. Start Live Monitor & Attack
echo "[4/5] Starting live zero-touch anomaly monitor in background..."
.venv/bin/python3 -m inference.live_monitor --interval 1.0 --iterations 30 --streak 3 --cooldown 5.0 > data/demo_monitor.log 2>&1 &
MON_PID=$!

echo "      [Wait 8s] Allowing live monitor to fill its sequence sliding-window..."
sleep 8

echo "[5/5] Launching Phase 2 Signaling Storm Attack via UERANSIM..."
.venv/bin/python3 -m simulation.attack_generator --clones 25 --duration 10 --cleanup

echo "      [Wait] Waiting for live monitor to conclude post-attack analysis..."
wait $MON_PID
echo ""

echo "=========================================================="
echo "                   DEMO EXECUTION RESULTS                 "
echo "=========================================================="
echo ""
echo "--- Live Monitor Logs Output ---"
cat data/demo_monitor.log
echo ""
echo "--- Real Mitigation Firewall Logs ---"
cat results/mitigation_log.json
echo ""
echo "✅ End-to-end Demonstration Concluded!"
