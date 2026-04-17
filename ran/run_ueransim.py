import time
import signal
import sys
import subprocess
from pathlib import Path

from ran.gnb_config import generate_gnb_config, write_gnb_config, start_gnb, stop_gnb
from ran.ue_config import generate_ue_config, write_ue_config, start_ue, stop_ue
from core.ella_config import get_config

def main():
    print("="*60)
    print(" UERANSIM Traffic Generator ")
    print("="*60)
    
    cfg = get_config()
    ueransim_path = Path("UERANSIM")
    
    if not ueransim_path.exists():
        print(f"[Error] UERANSIM not found at {ueransim_path.absolute()}")
        sys.exit(1)

    print("[1] Generating configurations...")
    gnb_cfg = generate_gnb_config(config=cfg, amf_ip="192.168.71.130", gnb_ip="192.168.71.130", link_ip="192.168.71.130")
    ue_cfg = generate_ue_config(config=cfg, gnb_ip="192.168.71.130")
    
    write_gnb_config(gnb_cfg)
    write_ue_config(ue_cfg)
    
    # Catch Ctrl+C
    def cleanup(sig, frame):
        print("\n[cleanup] Stopping UERANSIM...")
        stop_ue()
        stop_gnb()
        print("[cleanup] Done.")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    print("\n[2] Starting gNodeB...")
    start_gnb(config_path="config/gnb.yaml", ueransim_path=ueransim_path)
    
    # Give gNB time to connect to Ella Core (AMF)
    time.sleep(3)
    
    print("\n[3] Starting UE...")
    start_ue(config_path="config/ue.yaml", ueransim_path=ueransim_path)
    
    # Give UE time to register and create PDU session
    time.sleep(5)
    
    print("\n[4] Generating Traffic over uesimtun0...")
    print("Running ping to 8.8.8.8...")
    
    try:
        ping_proc = subprocess.Popen(
            ["ping", "-I", "uesimtun0", "8.8.8.8"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Stream ping output
        for line in ping_proc.stdout:
            print(f"  [ping] {line.strip()}")
            
    except Exception as e:
        print(f"[Error] Failed to start ping: {e}")
        
    # Wait indefinitely until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup(None, None)

if __name__ == "__main__":
    main()
