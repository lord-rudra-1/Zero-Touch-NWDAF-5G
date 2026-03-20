"""
Activity‑Based Mobility Model (ABMM) for UERANSIM orchestration.

Instead of static ping loops, this model simulates human-like UE behaviour
by alternating between **movement** and **dwell** phases based on time‑of‑day
and contextual preferences.

Locations:
  - Home    (evening / night)
  - Work    (daytime)
  - Coffee  (mid-morning / afternoon break)
  - Park    (weekend / evening walk)

Each location transition triggers UERANSIM UE actions:
  registration → data session → handover → deregistration

This produces temporally grounded, spatially meaningful movement traces
that drive realistic signaling on the 5G control plane.
"""

from __future__ import annotations

import random
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from core.ella_config import EllaConfig, get_config


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

class Location(Enum):
    HOME = "Home"
    WORK = "Work"
    COFFEE = "Coffee Shop"
    PARK = "Park"


@dataclass
class LocationProfile:
    """Properties of a location that influence UE behaviour."""
    name: str
    coords: tuple  # (lat, lon) placeholder
    avg_dwell_minutes: float = 30.0
    speed_kmh: float = 5.0  # walking speed to/from


# Default location profiles
LOCATIONS: Dict[Location, LocationProfile] = {
    Location.HOME: LocationProfile("Home", (28.6139, 77.2090), avg_dwell_minutes=240, speed_kmh=0),
    Location.WORK: LocationProfile("Work", (28.6200, 77.2150), avg_dwell_minutes=180, speed_kmh=30),
    Location.COFFEE: LocationProfile("Coffee Shop", (28.6180, 77.2120), avg_dwell_minutes=25, speed_kmh=5),
    Location.PARK: LocationProfile("Park", (28.6160, 77.2100), avg_dwell_minutes=45, speed_kmh=5),
}


# ---------------------------------------------------------------------------
# Time-of-day schedules
# ---------------------------------------------------------------------------

def _get_schedule(hour: int) -> List[Location]:
    """
    Return a likely location sequence based on hour of day (0..23).
    Models a typical urban commuter pattern.
    """
    if 0 <= hour < 7:
        return [Location.HOME]
    elif 7 <= hour < 9:
        return [Location.HOME, Location.COFFEE, Location.WORK]
    elif 9 <= hour < 12:
        return [Location.WORK]
    elif 12 <= hour < 13:
        return [Location.WORK, Location.COFFEE, Location.WORK]
    elif 13 <= hour < 17:
        return [Location.WORK]
    elif 17 <= hour < 19:
        return [Location.WORK, Location.PARK, Location.HOME]
    elif 19 <= hour < 22:
        return [Location.HOME]
    else:
        return [Location.HOME]


# ---------------------------------------------------------------------------
# ABMM Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class ABMMOrchestrator:
    """
    Orchestrates one or more UERANSIM UEs through time-of-day mobility
    patterns, generating realistic signaling sequences.
    """
    ueransim_path: Optional[Path] = None
    ue_config_path: Optional[Path] = None
    config: Optional[EllaConfig] = None
    simulated_hours: int = 24
    time_compression: float = 10.0  # 1 sim-hour = time_compression real-seconds
    seed: int = 42

    # Internal state
    _current_location: Location = Location.HOME
    _events: List[Dict] = field(default_factory=list)

    def _log_event(self, event_type: str, detail: str) -> None:
        entry = {
            "time": time.time(),
            "event": event_type,
            "detail": detail,
            "location": self._current_location.value,
        }
        self._events.append(entry)
        print(f"[ABMM] {event_type:20s}  |  {detail}")

    def _simulate_registration(self) -> None:
        """Trigger a UE registration event."""
        self._log_event("REGISTRATION", f"UE attaching at {self._current_location.value}")
        if self.ueransim_path and self.ue_config_path:
            try:
                subprocess.Popen(
                    ["sudo", str(self.ueransim_path / "build" / "nr-ue"),
                     "-c", str(self.ue_config_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                pass  # Binary not available — simulation mode

    def _simulate_deregistration(self) -> None:
        """Trigger a UE deregistration event."""
        self._log_event("DEREGISTRATION", f"UE detaching from {self._current_location.value}")
        # In a real setup, we would kill the nr-ue process

    def _simulate_handover(self, from_loc: Location, to_loc: Location) -> None:
        """Simulate a handover between locations."""
        self._log_event("HANDOVER", f"{from_loc.value} → {to_loc.value}")

    def _simulate_data_session(self) -> None:
        """Simulate a PDU session (data transfer)."""
        self._log_event("DATA_SESSION", f"Active data at {self._current_location.value}")

    def _dwell(self, location: Location) -> None:
        """Stay at a location for a duration scaled by time compression."""
        profile = LOCATIONS[location]
        real_seconds = (profile.avg_dwell_minutes * 60) / self.time_compression
        # Add randomness (±30%)
        jitter = random.uniform(0.7, 1.3)
        sleep_time = max(0.5, real_seconds * jitter)

        self._log_event("DWELL", f"Staying at {location.value} for {sleep_time:.1f}s (sim)")
        self._simulate_data_session()
        time.sleep(sleep_time)

    def run(self) -> List[Dict]:
        """
        Execute the full ABMM simulation across simulated hours.

        Returns the list of logged events.
        """
        random.seed(self.seed)
        self._events = []

        print(f"[ABMM] Starting {self.simulated_hours}h simulation "
              f"(compression={self.time_compression}x)")
        print(f"[ABMM] {'='*60}")

        self._simulate_registration()

        for hour in range(self.simulated_hours):
            schedule = _get_schedule(hour)
            self._log_event("HOUR_START", f"Simulated hour {hour:02d}:00")

            for target_loc in schedule:
                if target_loc != self._current_location:
                    # Movement phase
                    self._simulate_handover(self._current_location, target_loc)
                    self._current_location = target_loc

                # Dwell phase
                self._dwell(target_loc)

        self._simulate_deregistration()
        print(f"[ABMM] {'='*60}")
        print(f"[ABMM] Simulation complete — {len(self._events)} events")
        return self._events


# ---------------------------------------------------------------------------
# Quick-run function for the pipeline
# ---------------------------------------------------------------------------

def run_abmm(
    duration_hours: int = 24,
    time_compression: float = 60.0,
    ueransim_path: Optional[Path] = None,
    ue_config_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Convenience wrapper to run the ABMM and return events.
    High time_compression values make the simulation faster for demos.
    """
    orchestrator = ABMMOrchestrator(
        ueransim_path=ueransim_path,
        ue_config_path=ue_config_path,
        simulated_hours=duration_hours,
        time_compression=time_compression,
    )
    return orchestrator.run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Activity-Based Mobility Model")
    parser.add_argument("--hours", type=int, default=4,
                        help="Number of simulated hours (default: 4)")
    parser.add_argument("--compression", type=float, default=120.0,
                        help="Time compression factor (default: 120x)")
    args = parser.parse_args()

    events = run_abmm(
        duration_hours=args.hours,
        time_compression=args.compression,
    )
    print(f"\nTotal events generated: {len(events)}")
