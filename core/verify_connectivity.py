"""
End-to-end connectivity and health verification for the NWDAF pipeline.

Run as:
    python -m core.verify_connectivity

Checks performed:
  1. Ella Core health (API responding)
  2. Prometheus /metrics endpoint is serving telemetry
  3. Default subscriber is registered
"""

from __future__ import annotations

import sys
from typing import Optional

import requests

from core.ella_config import EllaConfig, get_config


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_core_health(config: Optional[EllaConfig] = None) -> bool:
    """Verify that Ella Core's API is responding."""
    cfg = config or get_config()
    url = cfg.api_base_url
    try:
        resp = requests.get(url, headers=cfg.auth_headers, timeout=5)
        ok = resp.status_code < 500
        status = "✓" if ok else f"✗ (HTTP {resp.status_code})"
        print(f"[check] Ella Core API ({url}): {status}")
        return ok
    except requests.ConnectionError:
        print(f"[check] Ella Core API ({url}): ✗ (connection refused)")
        return False
    except requests.Timeout:
        print(f"[check] Ella Core API ({url}): ✗ (timeout)")
        return False


def check_metrics_endpoint(config: Optional[EllaConfig] = None) -> bool:
    """Verify that the Prometheus /metrics endpoint is serving data."""
    cfg = config or get_config()
    url = cfg.metrics_url
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200 and len(resp.text) > 0:
            # Count the number of metric lines (lines not starting with '#')
            metric_lines = [
                line for line in resp.text.splitlines()
                if line and not line.startswith("#")
            ]
            print(f"[check] Metrics endpoint ({url}): ✓ ({len(metric_lines)} metrics)")
            return True
        else:
            print(f"[check] Metrics endpoint ({url}): ✗ (HTTP {resp.status_code})")
            return False
    except requests.ConnectionError:
        print(f"[check] Metrics endpoint ({url}): ✗ (connection refused)")
        return False
    except requests.Timeout:
        print(f"[check] Metrics endpoint ({url}): ✗ (timeout)")
        return False


def check_subscriber_registered(
    imsi: Optional[str] = None,
    config: Optional[EllaConfig] = None,
) -> bool:
    """Verify that a subscriber exists in Ella Core's database."""
    cfg = config or get_config()
    imsi = imsi or cfg.subscriber_imsi
    url = f"{cfg.api_base_url}/subscribers/{imsi}"
    try:
        resp = requests.get(url, headers=cfg.auth_headers, timeout=5)
        if resp.status_code == 200:
            print(f"[check] Subscriber {imsi}: ✓ (registered)")
            return True
        elif resp.status_code == 404:
            print(f"[check] Subscriber {imsi}: ✗ (not found)")
            return False
        else:
            print(f"[check] Subscriber {imsi}: ✗ (HTTP {resp.status_code})")
            return False
    except requests.ConnectionError:
        print(f"[check] Subscriber {imsi}: ✗ (API unreachable)")
        return False
    except requests.Timeout:
        print(f"[check] Subscriber {imsi}: ✗ (timeout)")
        return False


# ---------------------------------------------------------------------------
# Combined check
# ---------------------------------------------------------------------------

def run_full_check(config: Optional[EllaConfig] = None) -> bool:
    """
    Run all connectivity checks and return ``True`` only if all pass.
    """
    cfg = config or get_config()

    print("=" * 50)
    print(" NWDAF Pipeline — Connectivity Verification")
    print("=" * 50)
    print()

    results = {
        "Ella Core API": check_core_health(cfg),
        "Metrics endpoint": check_metrics_endpoint(cfg),
        "Subscriber registered": check_subscriber_registered(config=cfg),
    }

    print()
    print("-" * 50)
    all_ok = all(results.values())
    for name, passed in results.items():
        mark = "PASS" if passed else "FAIL"
        print(f"  {mark}  {name}")

    print("-" * 50)
    if all_ok:
        print("Overall: ALL CHECKS PASSED ✓")
    else:
        failed = [n for n, p in results.items() if not p]
        print(f"Overall: {len(failed)} CHECK(S) FAILED ✗")

    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    success = run_full_check()
    sys.exit(0 if success else 1)
