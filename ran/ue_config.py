"""
UERANSIM User Equipment (UE) configuration generator and process manager.

Generates ``nr-ue`` YAML configuration files and provides start/stop
helpers for individual UE instances.
"""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.ella_config import EllaConfig, get_config


# ---------------------------------------------------------------------------
# Global handle to the running UE process
# ---------------------------------------------------------------------------
_ue_process: Optional[subprocess.Popen] = None


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def generate_ue_config(
    gnb_ip: str = "127.0.0.1",
    plmn: Optional[str] = None,
    imsi: Optional[str] = None,
    key: Optional[str] = None,
    opc: Optional[str] = None,
    config: Optional[EllaConfig] = None,
    mcc: Optional[str] = None,
    mnc: Optional[str] = None,
    sst: Optional[int] = None,
    sd: Optional[str] = None,
    gnb_search_list: Optional[List[str]] = None,
    apn: str = "internet",
    session_type: str = "IPv4",
) -> Dict[str, Any]:
    """
    Generate a complete UERANSIM ``nr-ue`` configuration dictionary.

    Parameters
    ----------
    gnb_ip : str
        IP address of the gNB that this UE should connect to.
    plmn : str, optional
        5-digit PLMN.  Derived from *config* if omitted.
    imsi : str, optional
        UE IMSI (e.g. ``"imsi-001010000000001"``).
    key : str, optional
        Subscriber permanent key (hex).  Must match the value registered in
        Ella Core.
    opc : str, optional
        Operator key (hex).
    config : EllaConfig, optional
        Project configuration.

    Returns
    -------
    dict
        Configuration dictionary ready to be serialized to YAML.
    """
    cfg = config or get_config()
    _mcc = mcc or cfg.mcc
    _mnc = mnc or cfg.mnc
    _imsi = imsi or cfg.subscriber_imsi
    _key = key or cfg.subscriber_key
    _opc = opc or cfg.subscriber_opc
    _sst = sst if sst is not None else cfg.sst
    _sd = sd or cfg.sd

    # Strip the "imsi-" prefix if present (UERANSIM expects plain digits in supi)
    supi = _imsi if not _imsi.startswith("imsi-") else _imsi

    ue_cfg: Dict[str, Any] = {
        # Subscriber identity
        "supi": supi,
        "mcc": _mcc,
        "mnc": _mnc,
        "protectionScheme": 0,
        "homeNetworkPublicKey": "5a8d38864820197c3394b92613b20b91633cbd897119273bf8e4a6f4eec0a650",
        "homeNetworkPublicKeyId": 1,
        "routingIndicator": "0000",

        # Security
        "key": _key,
        # UERANSIM expects `op` even when opType is OPC.
        # So store OPc value in `op` and mark the type accordingly.
        "op": _opc,
        "opType": "OPC",
        "amf": "8000",     # Authentication Management Field

        # IMEI
        "imei": "356938035643803",
        "imeiSv": "4370816125816151",
        "tunNetmask": "255.255.255.0",

        # gNB search list
        "gnbSearchList": gnb_search_list or [gnb_ip],

        # Initial PDU session / APN
        "sessions": [
            {
                "type": session_type,
                "apn": apn,
                "slice": {
                    "sst": _sst,
                    "sd": _sd,
                },
            }
        ],

        # Configured NSSAI
        "configured-nssai": [
            {
                "sst": _sst,
                "sd": _sd,
            }
        ],

        # Default NSSAI
        "default-nssai": [
            {
                "sst": _sst,
                "sd": _sd,
            }
        ],

        # UAC fields required by recent UERANSIM versions
        "uacAic": {"mps": False, "mcs": False},
        "uacAcc": {
            "normalClass": 0,
            "class11": False,
            "class12": False,
            "class13": False,
            "class14": False,
            "class15": False,
        },

        # UAC Access Identities / Control Class
        "integrity": {"IA1": True, "IA2": True, "IA3": True},
        "ciphering": {"EA1": True, "EA2": True, "EA3": True},

        # Misc
        "integrityMaxRate": {"uplink": "full", "downlink": "full"},
    }

    return ue_cfg


def write_ue_config(
    config_dict: Dict[str, Any],
    output_path: Path | str = "config/ue.yaml",
) -> Path:
    """Serialize the UE config dict to a YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"[ue_config] Written UE config to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def start_ue(
    config_path: Path | str,
    ueransim_path: Optional[Path | str] = None,
) -> subprocess.Popen:
    """
    Launch the UERANSIM ``nr-ue`` process.

    Parameters
    ----------
    config_path : Path or str
        Path to the UE YAML configuration file.
    ueransim_path : Path or str, optional
        Root directory of UERANSIM.  If ``None``, expects ``nr-ue`` on
        ``$PATH``.

    Returns
    -------
    subprocess.Popen
        Handle to the running process.
    """
    global _ue_process

    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"UE config not found: {config_path}")

    if ueransim_path:
        binary = Path(ueransim_path) / "build" / "nr-ue"
    else:
        binary = Path("nr-ue")

    cmd = [str(binary), "-c", str(config_path)]
    print(f"[ue_config] Starting UE: {' '.join(cmd)}")

    _ue_process = subprocess.Popen(
        cmd
    )
    print(f"[ue_config] UE started (PID {_ue_process.pid})")
    return _ue_process


def stop_ue(timeout: float = 10.0) -> None:
    """Gracefully stop the running UE process."""
    global _ue_process

    if _ue_process is None:
        print("[ue_config] No active UE process to stop.")
        return

    print(f"[ue_config] Sending SIGTERM to UE (PID {_ue_process.pid})...")
    _ue_process.send_signal(signal.SIGTERM)

    try:
        _ue_process.wait(timeout=timeout)
        print("[ue_config] UE stopped gracefully.")
    except subprocess.TimeoutExpired:
        print("[ue_config] UE did not stop in time; sending SIGKILL.")
        _ue_process.kill()
        _ue_process.wait()

    _ue_process = None


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = generate_ue_config()
    print("Generated UE config:")
    print(yaml.dump(cfg, default_flow_style=False, sort_keys=False))

    # Optionally write to disk
    write_ue_config(cfg, output_path="config/ue.yaml")
