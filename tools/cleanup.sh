#!/bin/bash

echo "Stopping all UERANSIM UE processes..."
sudo pkill -f nr-ue || echo "No nr-ue processes found."

echo "Cleaning logs..."
rm -rf logs/*.log
mkdir -p logs

echo "Cleanup complete."
