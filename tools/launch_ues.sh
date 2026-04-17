#!/bin/bash

# Configuration
UE_BINARY="./UERANSIM/build/nr-ue"
CONFIG_DIR="./ue_configs"
LOG_DIR="./logs"
COUNT=25
DELAY=0.5

# Check binary
if [ ! -f "$UE_BINARY" ]; then
    echo "Error: UE binary not found at $UE_BINARY"
    exit 1
fi

# Ensure log directory exists
mkdir -p "$LOG_DIR"

echo "Starting $COUNT UEs..."

for i in $(seq -f "%02g" 1 $COUNT); do
    CONFIG="$CONFIG_DIR/ue_$i.yaml"
    LOG="$LOG_DIR/ue_$i.log"
    
    if [ -f "$CONFIG" ]; then
        echo "Launching UE $i with config $CONFIG..."
        # Launching in background with nohup to keep running
        nohup "$UE_BINARY" -c "$CONFIG" > "$LOG" 2>&1 &
        sleep $DELAY
    else
        echo "Warning: Config $CONFIG not found."
    fi
done

echo "All UEs launched. Check $LOG_DIR for outputs."
echo "Active nr-ue processes: $(pgrep -fc nr-ue)"
