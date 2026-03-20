"""
UERANSIM gNodeB (gNB) configuration generator and process manager.

Generates the ``nr-gnb`` YAML configuration file that connects a
simulated base station to Ella Core and provides start/stop helpers.
"""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.ella_config import EllaConfig, get_config


# ---------------------------------------------------------------------------
# Global handle to the running gNB process
# ---------------------------------------------------------------------------
_gnb_process: Optional[subprocess.Popen] = None


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def generate_gnb_config(
    amf_ip: str = "127.0.0.1",
    gnb_ip: str = "127.0.0.1",
    plmn: Optional[str] = None,
    tac: Optional[int] = None,
    config: Optional[EllaConfig] = None,
    link_ip: str = "127.0.0.1",
    ngap_port: int = 38412,
    gtp_port: int = 2152,
    mcc: Optional[str] = None,
    mnc: Optional[str] = None,
    sst: Optional[int] = None,
    sd: Optional[str] = None,
    gnb_id: int = 1,
    gnb_id_length: int = 32,
) -> Dict[str, Any]:
    """
    Generate a complete UERANSIM ``nr-gnb`` configuration dictionary.

    Parameters
    ----------
    amf_ip : str
        IP address where Ella Core's N2 (NGAP/SCTP) interface is listening.
    gnb_ip : str
        IP address that the gNB should bind to.
    plmn : str, optional
        5-digit PLMN string (e.g. ``"00101"``).  Derived from *config* if
        omitted.
    tac : int, optional
        Tracking Area Code.
    config : EllaConfig, optional
        Project configuration; defaults are pulled from here.

    Returns
    -------
    dict
        Configuration dictionary ready to be serialized to YAML.
    """
    cfg = config or get_config()
    _mcc = mcc or cfg.mcc
    _mnc = mnc or cfg.mnc
    _tac = tac if tac is not None else cfg.tac
    _sst = sst if sst is not None else cfg.sst
    _sd = sd or cfg.sd

    gnb_cfg: Dict[str, Any] = {
        # gNB identity
        "mcc": _mcc,
        "mnc": _mnc,
        "nci": f"0x00000000{gnb_id:01X}",
        "idLength": gnb_id_length,
        "tac": _tac,

        # Network interfaces
        "linkIp": link_ip,
        "ngapIp": gnb_ip,
        "gtpIp": gnb_ip,

        # AMF connection
        "amfConfigs": [
            {
                "address": amf_ip,
                "port": ngap_port,
            }
        ],

        # Supported S-NSSAI
        "slices": [
            {
                "sst": _sst,
                "sd": _sd,
            }
        ],

        # Misc
        "ignoreStreamIds": True,
    }

    return gnb_cfg


def write_gnb_config(
    config_dict: Dict[str, Any],
    output_path: Path | str = "config/gnb.yaml",
) -> Path:
    """Serialize the gNB config dict to a YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"[gnb_config] Written gNB config to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def start_gnb(
    config_path: Path | str,
    ueransim_path: Optional[Path | str] = None,
) -> subprocess.Popen:
    """
    Launch the UERANSIM ``nr-gnb`` process.

    Parameters
    ----------
    config_path : Path or str
        Path to the gNB YAML configuration file.
    ueransim_path : Path or str, optional
        Root directory of the UERANSIM installation.  If ``None``, expects
        ``nr-gnb`` to be on ``$PATH``.

    Returns
    -------
    subprocess.Popen
        Handle to the running process.
    """
    global _gnb_process

    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"gNB config not found: {config_path}")

    if ueransim_path:
        binary = Path(ueransim_path) / "build" / "nr-gnb"
    else:
        binary = Path("nr-gnb")

    cmd = ["sudo", str(binary), "-c", str(config_path)]
    print(f"[gnb_config] Starting gNB: {' '.join(cmd)}")

    _gnb_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(f"[gnb_config] gNB started (PID {_gnb_process.pid})")
    return _gnb_process


def stop_gnb(timeout: float = 10.0) -> None:
    """Gracefully stop the running gNB process."""
    global _gnb_process

    if _gnb_process is None:
        print("[gnb_config] No active gNB process to stop.")
        return

    print(f"[gnb_config] Sending SIGTERM to gNB (PID {_gnb_process.pid})...")
    _gnb_process.send_signal(signal.SIGTERM)

    try:
        _gnb_process.wait(timeout=timeout)
        print("[gnb_config] gNB stopped gracefully.")
    except subprocess.TimeoutExpired:
        print("[gnb_config] gNB did not stop in time; sending SIGKILL.")
        _gnb_process.kill()
        _gnb_process.wait()

    _gnb_process = None


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = generate_gnb_config()
    print("Generated gNB config:")
    print(yaml.dump(cfg, default_flow_style=False, sort_keys=False))

    # Optionally write to disk
    write_gnb_config(cfg, output_path="config/gnb.yaml")
