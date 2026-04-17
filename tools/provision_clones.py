import requests
import sys

# Constants
TOKEN = "ellacore_MzR9YhsEKcBt_n84hgUW6As00rak1XlPb55dS"
BASE_URL = "https://127.0.0.1:5002/api/v1"
SUBSCRIBER_PREFIX = "0010100000000"  # We will append 01-25

# Standard security profile from ella_config.py
KEY = "465B5CE8B199B49FAA5F0A2EE238A6BC"
OPC = "E8ED289DEBA952E4283B54E88E6183CA"

def provision_subscriber(imsi_digits):
    url = f"{BASE_URL}/subscribers"
    imsi_url = f"{url}/{imsi_digits}"
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    
    # 1. Delete if exists
    try:
        requests.delete(imsi_url, headers=headers, verify=False, timeout=5)
    except:
        pass

    # 2. Create fresh
    payload = {
        "imsi": imsi_digits,
        "key": KEY,
        "opc": OPC,
        "policyName": "default",
        "sequenceNumber": "000000000000",
        "sst": 1,
        "sd": "010203"
    }
    
    try:
        resp = requests.post(url, json=payload, headers=headers, verify=False, timeout=10)
        if resp.status_code in (200, 201):
            print(f"[SUCCESS] Provisioned {imsi_digits}")
            return True
        else:
            print(f"[FAILED] {imsi_digits}: HTTP {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        print(f"[ERROR] {imsi_digits}: {str(e)}")
        return False

if __name__ == "__main__":
    print(f"Starting bulk provisioning of 25 subscribers to {BASE_URL}...")
    success_count = 0
    for i in range(1, 26):
        imsi = f"{SUBSCRIBER_PREFIX}{i:02d}"
        if provision_subscriber(imsi):
            success_count += 1
    
    print(f"\nFinal Status: {success_count}/25 provisioned successfully.")
    if success_count == 25:
        sys.exit(0)
    else:
        sys.exit(1)
