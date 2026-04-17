"""
Zero‑Touch Mitigation Module.

When an anomaly is detected, this module communicates with Ella Core's
REST API to:
  - **Block** the offending subscriber (by IMSI)
  - **Throttle** connections from suspicious sources
  - **Log** all mitigation actions for audit

This is the "self-healing" component of the closed-loop NWDAF architecture.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from core.ella_config import EllaConfig, get_config


# ---------------------------------------------------------------------------
# Mitigation action log
# ---------------------------------------------------------------------------
_mitigation_log: List[Dict[str, Any]] = []


def _log_action(action: str, imsi: str, success: bool, detail: str = "") -> None:
    entry = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "action": action,
        "imsi": imsi,
        "success": success,
        "detail": detail,
    }
    _mitigation_log.append(entry)
    status = "✓" if success else "✗"
    print(f"[mitigation] {status} {action} — IMSI: {imsi}  {detail}")


def get_mitigation_log() -> List[Dict[str, Any]]:
    """Return a copy of all mitigation actions taken."""
    return list(_mitigation_log)


# ---------------------------------------------------------------------------
# Block subscriber
# ---------------------------------------------------------------------------

def block_subscriber(
    imsi: str,
    config: Optional[EllaConfig] = None,
) -> bool:
    """
    Instantly block a malicious subscriber via Ella Core's REST API.

    Sends ``POST /api/v1/subscribers/{imsi}/block`` with a Bearer token
    to terminate the attacker's network access.

    Parameters
    ----------
    imsi : str
        IMSI of the subscriber to block (e.g. ``"imsi-001010000000001"``).
    config : EllaConfig, optional
        Ella Core configuration.

    Returns
    -------
    bool
        ``True`` if the block was successful.
    """
    cfg = config or get_config()
    url = f"{cfg.api_base_url}/subscribers/{imsi}/block"

    try:
        resp = requests.post(url, headers=cfg.auth_headers, timeout=5, **cfg.requests_kwargs)
        success = resp.status_code in (200, 201, 204)
        _log_action("BLOCK", imsi, success,
                     f"HTTP {resp.status_code}")
        return success
    except requests.RequestException as exc:
        _log_action("BLOCK", imsi, False, str(exc))
        return False


# ---------------------------------------------------------------------------
# Throttle connections
# ---------------------------------------------------------------------------

def throttle_subscriber(
    imsi: str,
    rate_limit: int = 10,
    config: Optional[EllaConfig] = None,
) -> bool:
    """
    Throttle a subscriber's signaling rate via the REST API.

    Parameters
    ----------
    imsi : str
        Subscriber IMSI.
    rate_limit : int
        Maximum allowed signaling messages per second.
    config : EllaConfig, optional
        Ella Core configuration.

    Returns
    -------
    bool
        ``True`` if throttling was applied successfully.
    """
    cfg = config or get_config()
    url = f"{cfg.api_base_url}/subscribers/{imsi}/throttle"
    payload = {"rate_limit_per_second": rate_limit}

    try:
        resp = requests.post(
            url, json=payload, headers=cfg.auth_headers, timeout=5, **cfg.requests_kwargs
        )
        success = resp.status_code in (200, 201, 204)
        _log_action("THROTTLE", imsi, success,
                     f"rate={rate_limit}/s, HTTP {resp.status_code}")
        return success
    except requests.RequestException as exc:
        _log_action("THROTTLE", imsi, False, str(exc))
        return False


# ---------------------------------------------------------------------------
# Unblock (recovery)
# ---------------------------------------------------------------------------

def unblock_subscriber(
    imsi: str,
    config: Optional[EllaConfig] = None,
) -> bool:
    """Unblock a previously blocked subscriber."""
    cfg = config or get_config()
    url = f"{cfg.api_base_url}/subscribers/{imsi}/unblock"

    try:
        resp = requests.post(url, headers=cfg.auth_headers, timeout=5, **cfg.requests_kwargs)
        success = resp.status_code in (200, 201, 204)
        _log_action("UNBLOCK", imsi, success,
                     f"HTTP {resp.status_code}")
        return success
    except requests.RequestException as exc:
        _log_action("UNBLOCK", imsi, False, str(exc))
        return False


# ---------------------------------------------------------------------------
# Save mitigation log
# ---------------------------------------------------------------------------

def save_mitigation_log(output_path: Path | str = "results/mitigation_log.json") -> Path:
    """Persist the mitigation log to JSON for audit / reporting."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(_mitigation_log, f, indent=2)

    print(f"[mitigation] Saved {len(_mitigation_log)} actions to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Mitigation Module — Manual Test")
    print("=" * 40)
    cfg = get_config()
    print(f"API URL: {cfg.api_base_url}")
    print(f"Test IMSI: {cfg.subscriber_imsi}")
    print()
    print("Available functions:")
    print("  block_subscriber(imsi)")
    print("  throttle_subscriber(imsi, rate_limit)")
    print("  unblock_subscriber(imsi)")
