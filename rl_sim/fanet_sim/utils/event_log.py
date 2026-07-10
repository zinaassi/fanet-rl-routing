"""
event_log.py — Stage 1 raw-event logger for the FANET simulator.

The simulator writes one JSON object per line (JSONL) to a per-episode file.
No metrics are computed here — that is the job of scripts/analyze.py (Stage 2).

This split is deliberate: new metrics can be added later by extending the
analysis script, without re-running expensive simulations.

Record schemas (all records share ``record_type``):

    record_type = "episode_meta"
        episode_id, seed, num_drones, num_M, num_C, episode_length,
        mobility_params, traffic_load, connectivity_model_params, started_at

    record_type = "packet_event"
        event in {"generated", "forwarded", "delivered", "dropped"}
        packet_id, time, src_drone, current_drone, next_hop (or None),
        hop_index, drop_reason (or None), is_control

    record_type = "step_state"
        time, frac_connected_to_gs, num_components

    record_type = "drone_state"
        time, drone_id, type, position, energy_radio, energy_motion
"""

from __future__ import annotations

import json
import os
import time as _time
from typing import Any, Dict, Optional


class EventLogger:
    """Append-only JSONL writer for raw simulation events.

    Open one EventLogger per episode. Call ``close()`` (or use it as a
    context manager) when the episode ends so the file is flushed.

    Attributes:
        path:        Absolute path of the log file being written.
        episode_id:  The episode this logger belongs to.
    """

    def __init__(self, path: str, episode_id: int) -> None:
        """Open *path* for writing (truncates any existing file).

        Args:
            path:       Absolute or relative path to the output .jsonl file.
            episode_id: Integer episode identifier (recorded on every event).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.episode_id = episode_id
        self._fh = open(path, "w", encoding="utf-8")
        self._closed = False

    # ------------------------------------------------------------------
    # Context-manager sugar
    # ------------------------------------------------------------------

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Flush and close the underlying file."""
        if not self._closed:
            self._fh.flush()
            self._fh.close()
            self._closed = True

    # ------------------------------------------------------------------
    # Low-level writer
    # ------------------------------------------------------------------

    def _write(self, record: Dict[str, Any]) -> None:
        record.setdefault("episode_id", self.episode_id)
        self._fh.write(json.dumps(record, default=_json_default))
        self._fh.write("\n")

    # ------------------------------------------------------------------
    # Schema-specific helpers
    # ------------------------------------------------------------------

    def log_episode_meta(
        self,
        *,
        seed: Any,
        num_drones: int,
        num_M: int,
        num_C: int,
        episode_length: int,
        mobility_params: Dict[str, Any],
        traffic_load: Dict[str, Any],
        connectivity_model_params: Dict[str, Any],
        anchor: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record per-episode metadata (must include the RNG seed).

        Args:
            anchor: Optional citation for the literature source this setup
                is anchored to (e.g. IQMR). Embedded verbatim in the record
                so every log self-documents which paper its parameters
                were chosen to match.
        """
        record: Dict[str, Any] = {
            "record_type": "episode_meta",
            "seed": seed,
            "num_drones": num_drones,
            "num_M": num_M,
            "num_C": num_C,
            "episode_length": episode_length,
            "mobility_params": mobility_params,
            "traffic_load": traffic_load,
            "connectivity_model_params": connectivity_model_params,
            "started_at": _time.time(),
        }
        if anchor is not None:
            record["anchor"] = anchor
        self._write(record)

    def log_packet_event(
        self,
        *,
        event: str,
        time: float,
        packet_id: int,
        src_drone: int,
        current_drone: Any,
        hop_index: int,
        is_control: bool,
        next_hop: Optional[Any] = None,
        drop_reason: Optional[str] = None,
    ) -> None:
        """Record a single packet lifecycle event.

        Args:
            event:         "generated" | "forwarded" | "delivered" | "dropped".
            time:          Simulation time of the event.
            packet_id:     Unique packet identifier.
            src_drone:     ID of the M-drone that originated the packet.
            current_drone: ID (or "GS") of the drone emitting this event.
            hop_index:     Hop count *after* this event.
            is_control:    True for routing/control/overhead traffic.
            next_hop:      ID (or "GS") of the next hop, for forwarded events.
            drop_reason:   String reason (e.g. "no_route", "ttl_expired").
        """
        self._write({
            "record_type": "packet_event",
            "event": event,
            "time": time,
            "packet_id": packet_id,
            "src_drone": src_drone,
            "current_drone": current_drone,
            "next_hop": next_hop,
            "hop_index": hop_index,
            "drop_reason": drop_reason,
            "is_control": is_control,
        })

    def log_step_state(
        self,
        *,
        time: float,
        frac_connected_to_gs: float,
        num_components: int,
    ) -> None:
        """Record the per-timestep network-state snapshot."""
        self._write({
            "record_type": "step_state",
            "time": time,
            "frac_connected_to_gs": frac_connected_to_gs,
            "num_components": num_components,
        })

    def log_drone_state(
        self,
        *,
        time: float,
        drone_id: int,
        drone_type: str,
        position: Any,
        energy_radio: float,
        energy_motion: float,
    ) -> None:
        """Record a per-drone state snapshot."""
        self._write({
            "record_type": "drone_state",
            "time": time,
            "drone_id": drone_id,
            "type": drone_type,
            "position": list(position),
            "energy_radio": energy_radio,
            "energy_motion": energy_motion,
        })


# ----------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Fallback serializer for NumPy types and other non-JSON-native values."""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    except ImportError:
        pass
    if hasattr(obj, "name"):  # Enum
        return obj.name
    return str(obj)
