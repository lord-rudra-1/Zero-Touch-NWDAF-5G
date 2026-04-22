import os
import yaml

def generate_ue_configs(count=25, output_dir="ue_configs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Base settings from config/ue.yaml and config/gnb.yaml
    base_config = {
        'mcc': '001',
        'mnc': '01',
        'protectionScheme': 0,
        'homeNetworkPublicKey': '5a8d38864820197c3394b92613b20b91633cbd897119273bf8e4a6f4eec0a650',
        'homeNetworkPublicKeyId': 1,
        'routingIndicator': '0000',
        'key': '465B5CE8B199B49FAA5F0A2EE238A6BC',
        'op': 'E8ED289DEBA952E4283B54E88E6183CA',
        'opType': 'OPC',
        'amf': '8000',
        'tunNetmask': '255.255.255.0',
        'gnbSearchList': ['192.168.71.130'],
        'sessions': [
            {
                'type': 'IPv4',
                'apn': 'internet',
                'slice': {'sst': 1, 'sd': '000000'}
            }
        ],
        'configured-nssai': [{'sst': 1, 'sd': '000000'}],
        'default-nssai': [{'sst': 1, 'sd': '000000'}],
        'uacAic': {'mps': False, 'mcs': False},
        'uacAcc': {
            'normalClass': 0,
            'class11': False,
            'class12': False,
            'class13': False,
            'class14': False,
            'class15': False
        },
        'integrity': {'IA1': True, 'IA2': True, 'IA3': True},
        'ciphering': {'EA1': True, 'EA2': True, 'EA3': True},
        'integrityMaxRate': {'uplink': 'full', 'downlink': 'full'}
    }

    for i in range(1, count + 1):
        # Generate unique IDs
        ue_id = f"{i:02d}"
        imsi = f"imsi-0010100000000{ue_id}"
        imei = f"3569380356438{ue_id}"
        
        config = base_config.copy()
        config['supi'] = imsi
        config['imei'] = imei
        
        file_path = os.path.join(output_dir, f"ue_{ue_id}.yaml")
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Generated {file_path}")

if __name__ == "__main__":
    generate_ue_configs()
