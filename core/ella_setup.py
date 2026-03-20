"""
Ella Core deployment helper.

Provides functions to:
  - check if the Ella Core binary is available
  - start / stop the Ella Core process
  - wait until the core is healthy
  - create subscribers via the REST API
"""

from __future__ import annotations

import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from core.ella_config import EllaConfig, get_config


# ---------------------------------------------------------------------------
# Global handle to the running Ella Core process
# ---------------------------------------------------------------------------
_ella_process: Optional[subprocess.Popen] = None


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

def find_ella_binary(search_name: str = "ella-core") -> Optional[Path]:
    """
    Locate the Ella Core binary on the system.

    Search order:
      1. ``ELLA_CORE_BIN`` environment variable
      2. ``$PATH`` via ``shutil.which``
      3. Common installation paths
    """
    import os

    env_path = os.getenv("ELLA_CORE_BIN")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p

    which = shutil.which(search_name)
    if which:
        return Path(which)

    # Common install locations
    for candidate in [
        Path("/usr/local/bin") / search_name,
        Path.home() / "ella-core" / search_name,
        Path.home() / "bin" / search_name,
    ]:
        if candidate.is_file():
            return candidate

    return None


def check_ella_installed() -> bool:
    """Return ``True`` if the Ella Core binary can be found."""
    return find_ella_binary() is not None


# ---------------------------------------------------------------------------
# Start / Stop
# ---------------------------------------------------------------------------

def start_ella_core(
    config_path: Optional[Path] = None,
    binary_path: Optional[Path] = None,
    extra_args: Optional[list] = None,
) -> subprocess.Popen:
    """
    Launch Ella Core as a background subprocess.

    Parameters
    ----------
    config_path : Path, optional
        Path to the Ella Core configuration file.  If ``None`` the binary's
        built-in defaults are used.
    binary_path : Path, optional
        Explicit path to the binary.  If ``None``, :func:`find_ella_binary`
        is used.
    extra_args : list, optional
        Additional CLI arguments.

    Returns
    -------
    subprocess.Popen
        Handle to the running process.
    """
    global _ella_process

    if binary_path is None:
        binary_path = find_ella_binary()
        if binary_path is None:
            raise FileNotFoundError(
                "Ella Core binary not found.  Set the ELLA_CORE_BIN "
                "environment variable or install ella-core to $PATH."
            )

    cmd: list[str] = [str(binary_path)]
    if config_path:
        cmd += ["--config", str(config_path)]
    if extra_args:
        cmd += extra_args

    print(f"[ella_setup] Starting Ella Core: {' '.join(cmd)}")
    _ella_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(f"[ella_setup] Ella Core started (PID {_ella_process.pid})")
    return _ella_process


def stop_ella_core(timeout: float = 10.0) -> None:
    """Gracefully stop the running Ella Core process."""
    global _ella_process

    if _ella_process is None:
        print("[ella_setup] No active Ella Core process to stop.")
        return

    print(f"[ella_setup] Sending SIGTERM to Ella Core (PID {_ella_process.pid})...")
    _ella_process.send_signal(signal.SIGTERM)

    try:
        _ella_process.wait(timeout=timeout)
        print("[ella_setup] Ella Core stopped gracefully.")
    except subprocess.TimeoutExpired:
        print("[ella_setup] Ella Core did not stop in time; sending SIGKILL.")
        _ella_process.kill()
        _ella_process.wait()

    _ella_process = None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def wait_for_ready(
    config: Optional[EllaConfig] = None,
    timeout: float = 60.0,
    poll_interval: float = 2.0,
) -> bool:
    """
    Block until Ella Core's ``/metrics`` endpoint responds or *timeout* elapses.

    Returns ``True`` if the core became healthy within the timeout.
    """
    cfg = config or get_config()
    url = cfg.metrics_url
    deadline = time.time() + timeout

    print(f"[ella_setup] Waiting for Ella Core at {url} (timeout={timeout}s)...")

    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                print("[ella_setup] Ella Core is healthy ✓")
                return True
        except requests.ConnectionError:
            pass

        time.sleep(poll_interval)

    print("[ella_setup] Ella Core did not become healthy within the timeout ✗")
    return False


# ---------------------------------------------------------------------------
# Subscriber management
# ---------------------------------------------------------------------------

def create_subscriber(
    imsi: str,
    key: str,
    opc: str,
    config: Optional[EllaConfig] = None,
) -> Dict[str, Any]:
    """
    Create a new subscriber in Ella Core via the REST API.

    Parameters
    ----------
    imsi : str
        Subscriber IMSI (e.g. ``"imsi-001010000000001"``).
    key : str
        Subscriber permanent key (hex).
    opc : str
        Operator key (hex).
    config : EllaConfig, optional
        Ella Core configuration.  Defaults to :func:`get_config`.

    Returns
    -------
    dict
        JSON response body from the API.
    """
    cfg = config or get_config()
    url = f"{cfg.api_base_url}/subscribers"

    payload = {
        "imsi": imsi,
        "key": key,
        "opc": opc,
        "sst": cfg.sst,
        "sd": cfg.sd,
    }

    print(f"[ella_setup] Creating subscriber {imsi} via POST {url}")
    resp = requests.post(url, json=payload, headers=cfg.auth_headers, timeout=10)

    if resp.status_code in (200, 201):
        print(f"[ella_setup] Subscriber {imsi} created successfully ✓")
    else:
        print(
            f"[ella_setup] Failed to create subscriber {imsi}: "
            f"HTTP {resp.status_code} — {resp.text}"
        )

    return resp.json() if resp.text else {}


def create_default_subscriber(config: Optional[EllaConfig] = None) -> Dict[str, Any]:
    """Create the project's default test subscriber."""
    cfg = config or get_config()
    return create_subscriber(
        imsi=cfg.subscriber_imsi,
        key=cfg.subscriber_key,
        opc=cfg.subscriber_opc,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Ella Core Setup Utility")
    print("=" * 40)
    binary = find_ella_binary()
    if binary:
        print(f"Binary found at: {binary}")
    else:
        print("Binary NOT found — install Ella Core first.")
        sys.exit(1)

    cfg = get_config()
    print(f"API URL     : {cfg.api_base_url}")
    print(f"Metrics URL : {cfg.metrics_url}")
    print()
    print("To start the core programmatically:")
    print("  from core.ella_setup import start_ella_core, wait_for_ready")
    print("  proc = start_ella_core()")
    print("  wait_for_ready()")
