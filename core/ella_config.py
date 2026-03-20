"""
Ella Core configuration constants and helpers.

Centralises every tuneable parameter so that the rest of the pipeline
(setup, telemetry, mitigation) can import a single, consistent config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Default network parameters
# ---------------------------------------------------------------------------
DEFAULT_PLMN_MCC = "001"
DEFAULT_PLMN_MNC = "01"
DEFAULT_PLMN_ID = f"{DEFAULT_PLMN_MCC}{DEFAULT_PLMN_MNC}"

DEFAULT_SST = 1          # Slice/Service Type  (eMBB)
DEFAULT_SD = "010203"    # Slice Differentiator (hex)

DEFAULT_TAC = 1          # Tracking Area Code

# ---------------------------------------------------------------------------
# Ella Core network addresses
# ---------------------------------------------------------------------------
DEFAULT_ELLA_HOST = os.getenv("ELLA_HOST", "127.0.0.1")
DEFAULT_ELLA_API_PORT = int(os.getenv("ELLA_API_PORT", "9090"))
DEFAULT_ELLA_METRICS_PORT = int(os.getenv("ELLA_METRICS_PORT", "9090"))

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
DEFAULT_ELLA_API_TOKEN = os.getenv("ELLA_API_TOKEN", "")


# ---------------------------------------------------------------------------
# Subscriber defaults (must match UERANSIM UE config)
# ---------------------------------------------------------------------------
DEFAULT_SUBSCRIBER_KEY = "465B5CE8B199B49FAA5F0A2EE238A6BC"
DEFAULT_SUBSCRIBER_OPC = "E8ED289DEBA952E4283B54E88E6183CA"
DEFAULT_SUBSCRIBER_IMSI = "imsi-001010000000001"


@dataclass
class EllaConfig:
    """Container for all Ella Core configuration parameters."""

    # PLMN
    mcc: str = DEFAULT_PLMN_MCC
    mnc: str = DEFAULT_PLMN_MNC

    # Slice
    sst: int = DEFAULT_SST
    sd: str = DEFAULT_SD

    # Tracking area
    tac: int = DEFAULT_TAC

    # Network addresses
    host: str = DEFAULT_ELLA_HOST
    api_port: int = DEFAULT_ELLA_API_PORT
    metrics_port: int = DEFAULT_ELLA_METRICS_PORT

    # Auth
    api_token: str = DEFAULT_ELLA_API_TOKEN

    # Default subscriber
    subscriber_imsi: str = DEFAULT_SUBSCRIBER_IMSI
    subscriber_key: str = DEFAULT_SUBSCRIBER_KEY
    subscriber_opc: str = DEFAULT_SUBSCRIBER_OPC

    # Additional subscribers (list of dicts with imsi/key/opc)
    extra_subscribers: List[Dict[str, str]] = field(default_factory=list)

    # ---- derived helpers ---------------------------------------------------

    @property
    def plmn_id(self) -> str:
        return f"{self.mcc}{self.mnc}"

    @property
    def api_base_url(self) -> str:
        """Base URL for the Ella Core REST API."""
        return f"http://{self.host}:{self.api_port}/api/v1"

    @property
    def metrics_url(self) -> str:
        """Prometheus-compatible /metrics endpoint."""
        return f"http://{self.host}:{self.metrics_port}/metrics"

    @property
    def auth_headers(self) -> Dict[str, str]:
        """HTTP headers with Bearer token for API calls."""
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers


# ---------------------------------------------------------------------------
# Convenience singleton for quick imports
# ---------------------------------------------------------------------------
_default_config: EllaConfig | None = None


def get_config() -> EllaConfig:
    """Return the project-wide default config (lazily created)."""
    global _default_config
    if _default_config is None:
        _default_config = EllaConfig()
    return _default_config


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = get_config()
    print(f"PLMN ID       : {cfg.plmn_id}")
    print(f"SST / SD      : {cfg.sst} / {cfg.sd}")
    print(f"API base URL  : {cfg.api_base_url}")
    print(f"Metrics URL   : {cfg.metrics_url}")
    print(f"Default IMSI  : {cfg.subscriber_imsi}")
