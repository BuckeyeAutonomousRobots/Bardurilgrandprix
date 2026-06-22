from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.types import AttitudeCommand, FSMTransition, RacePlan, VehicleState


def _vec3(value: Optional[tuple[float, float, float]]) -> Optional[list[float]]:
    if value is None:
        return None
    return [float(value[0]), float(value[1]), float(value[2])]


def _attitude_rad(vehicle: VehicleState) -> list[float]:
    return [vehicle.roll_rad, vehicle.pitch_rad, vehicle.yaw_rad]


def _vehicle_snapshot(vehicle: VehicleState, mono_time_s: float) -> dict[str, Any]:
    return {
        "mono_time_s": mono_time_s,
        "position_ned_m": _vec3(vehicle.position_ned_m),
        "velocity_ned_mps": _vec3(vehicle.velocity_ned_mps),
        "attitude_rad": _attitude_rad(vehicle),
        "armed": vehicle.armed,
        "position_wall_time_s": vehicle.position_wall_time_s,
    }


@dataclass
class _PendingAction:
    action_id: int
    action_type: str
    mono_time_s: float
    position_wall_time_s: Optional[float]
    before: dict[str, Any]
    details: dict[str, Any]
    created_mono_s: float = field(default_factory=time.monotonic)


class DroneActionLogger:
    """Records drone actions with vehicle state before and after each action."""

    def __init__(
        self,
        log_path: str,
        *,
        post_timeout_s: float = 0.15,
        max_pending: int = 64,
    ) -> None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._fh = path.open("a", encoding="utf-8", buffering=1)
        self._post_timeout_s = post_timeout_s
        self._max_pending = max_pending
        self._next_id = 1
        self._pending: list[_PendingAction] = []

    def close(self) -> None:
        self.flush_pending(force=True)
        self._fh.flush()
        self._fh.close()

    @property
    def log_path(self) -> str:
        return str(self._path)

    def record(
        self,
        action_type: str,
        vehicle: VehicleState,
        *,
        mono_time_s: float,
        details: dict[str, Any],
    ) -> int:
        action_id = self._next_id
        self._next_id += 1
        pending = _PendingAction(
            action_id=action_id,
            action_type=action_type,
            mono_time_s=mono_time_s,
            position_wall_time_s=vehicle.position_wall_time_s,
            before=_vehicle_snapshot(vehicle, mono_time_s),
            details=details,
        )
        self._pending.append(pending)
        if len(self._pending) > self._max_pending:
            self._pending.pop(0)
        return action_id

    def record_control(
        self,
        vehicle: VehicleState,
        *,
        mono_time_s: float,
        plan: RacePlan,
        command: AttitudeCommand,
        active_gate_index: int,
        map_dist_center_m: Optional[float],
        map_plane_signed_m: Optional[float],
    ) -> int:
        return self.record(
            "control_command",
            vehicle,
            mono_time_s=mono_time_s,
            details={
                "fsm_state": plan.state,
                "active_gate_index": active_gate_index,
                "map_dist_center_m": map_dist_center_m,
                "map_plane_signed_m": map_plane_signed_m,
                "plan_forward_speed_mps": plan.forward_speed_mps,
                "plan_commit": plan.commit,
                "command": {
                    "roll_rad": command.roll_rad,
                    "pitch_rad": command.pitch_rad,
                    "yaw_rad": command.yaw_rad,
                    "thrust": command.thrust,
                    "roll_rate_rps": command.roll_rate_rps,
                    "pitch_rate_rps": command.pitch_rate_rps,
                    "yaw_rate_rps": command.yaw_rate_rps,
                },
                "saturation": command.saturation.as_dict(),
            },
        )

    def record_fsm_transition(
        self,
        vehicle: VehicleState,
        *,
        mono_time_s: float,
        transition: FSMTransition,
        active_gate_index: int,
        map_plane_signed_m: Optional[float],
    ) -> int:
        return self.record(
            "fsm_transition",
            vehicle,
            mono_time_s=mono_time_s,
            details={
                "old_state": transition.old_state,
                "new_state": transition.new_state,
                "reason": transition.reason,
                "active_gate_index": active_gate_index,
                "map_plane_signed_m": map_plane_signed_m,
            },
        )

    def update_vehicle(self, vehicle: VehicleState, mono_time_s: float) -> None:
        if not self._pending:
            return
        still_pending: list[_PendingAction] = []
        for pending in self._pending:
            if self._should_complete(pending, vehicle, mono_time_s):
                self._write_completed(pending, vehicle, mono_time_s)
            else:
                still_pending.append(pending)
        self._pending = still_pending

    def flush_pending(self, *, force: bool = False) -> None:
        if not self._pending:
            return
        now_s = time.monotonic()
        still_pending: list[_PendingAction] = []
        for pending in self._pending:
            if force or (now_s - pending.created_mono_s) >= self._post_timeout_s:
                self._write_completed(pending, None, now_s)
            else:
                still_pending.append(pending)
        self._pending = still_pending

    def _should_complete(self, pending: _PendingAction, vehicle: VehicleState, mono_time_s: float) -> bool:
        if pending.position_wall_time_s is None:
            if vehicle.position_ned_m is not None:
                return True
            return (mono_time_s - pending.mono_time_s) >= self._post_timeout_s

        if vehicle.position_wall_time_s is not None and vehicle.position_wall_time_s > pending.position_wall_time_s:
            return True
        if vehicle.position_ned_m is not None and pending.before.get("position_ned_m") is None:
            return True
        return (mono_time_s - pending.mono_time_s) >= self._post_timeout_s

    def _write_completed(
        self,
        pending: _PendingAction,
        vehicle: Optional[VehicleState],
        mono_time_s: float,
    ) -> None:
        after: Optional[dict[str, Any]] = None
        if vehicle is not None:
            after = _vehicle_snapshot(vehicle, mono_time_s)
            after["delta_dt_s"] = mono_time_s - pending.mono_time_s
            before_pos = pending.before.get("position_ned_m")
            after_pos = after.get("position_ned_m")
            if before_pos is not None and after_pos is not None:
                after["position_delta_ned_m"] = [
                    after_pos[0] - before_pos[0],
                    after_pos[1] - before_pos[1],
                    after_pos[2] - before_pos[2],
                ]

        row = {
            "ts": time.time(),
            "event": "drone_action",
            "action_id": pending.action_id,
            "action_type": pending.action_type,
            "mono_time_s": pending.mono_time_s,
            "before": pending.before,
            "after": after,
            **pending.details,
        }
        self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        if pending.action_type in {"fsm_transition", "arm", "sim_reset", "position_jump"}:
            self._fh.flush()
