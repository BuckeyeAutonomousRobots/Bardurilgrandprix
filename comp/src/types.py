from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VehicleState:
    wall_time_s: float = 0.0
    heartbeat_wall_time_s: float = 0.0
    system_status: int = 0
    armed: bool = False
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0
    roll_rate_rps: float = 0.0
    pitch_rate_rps: float = 0.0
    yaw_rate_rps: float = 0.0
    position_ned_m: Optional[tuple[float, float, float]] = None
    velocity_ned_mps: Optional[tuple[float, float, float]] = None
    position_wall_time_s: Optional[float] = None
    accel_mps2: Optional[tuple[float, float, float]] = None
    gyro_rps: Optional[tuple[float, float, float]] = None
    timesync_offset_ns: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def heartbeat_is_fresh(self, now_s: float, timeout_s: float = 1.0) -> bool:
        return self.heartbeat_wall_time_s > 0.0 and (now_s - self.heartbeat_wall_time_s) <= timeout_s


@dataclass
class VisionFrame:
    frame_id: int
    sim_time_ns: int
    wall_time_s: float
    image_shape: tuple[int, int]
    image_bgr: Any


@dataclass
class GateDetection:
    frame_id: int
    timestamp_s: float
    center_px: tuple[float, float]
    bbox: tuple[int, int, int, int]
    corners_px: tuple[tuple[float, float], ...]
    confidence: float
    area_fraction: float
    source: str = "opencv"


@dataclass
class GateTrack:
    frame_id: int = -1
    timestamp_s: float = 0.0
    center_px: tuple[float, float] = (0.0, 0.0)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0
    area_fraction: float = 0.0
    source: str = "none"
    visible: bool = False
    predicted: bool = False
    missed_frames: int = 0

    def is_fresh(self, now_s: float, timeout_s: float) -> bool:
        return self.timestamp_s > 0.0 and (now_s - self.timestamp_s) <= timeout_s


@dataclass
class EstimatedState:
    now_s: float
    vehicle: VehicleState
    gate_track: Optional[GateTrack]
    link_ready: bool
    vision_ready: bool
    gate_bearing_x_rad: float = 0.0
    gate_bearing_y_rad: float = 0.0
    gate_range_m: Optional[float] = None
    gate_range_x_m: Optional[float] = None
    gate_range_y_m: Optional[float] = None
    gate_depth_confidence: float = 0.0
    gate_confidence: float = 0.0
    race_started: bool = False
    has_track_map: bool = False
    active_gate_index: int = -1
    map_gate_bearing_x_rad: Optional[float] = None
    map_gate_bearing_y_rad: Optional[float] = None
    map_gate_range_m: Optional[float] = None
    map_approach_bearing_x_rad: Optional[float] = None
    map_approach_bearing_y_rad: Optional[float] = None
    map_approach_range_m: Optional[float] = None
    map_commit_bearing_x_rad: Optional[float] = None
    map_commit_bearing_y_rad: Optional[float] = None
    map_commit_range_m: Optional[float] = None
    map_plane_signed_m: Optional[float] = None
    map_within_gate_bounds: bool = False
    map_lateral_error_m: Optional[float] = None
    map_vertical_error_m: Optional[float] = None
    map_dist_center_m: Optional[float] = None
    map_gate_commit_active: bool = False
    map_gate_commit_strength: float = 0.0
    map_exit_yaw_rad: Optional[float] = None
    map_commit_speed_mps: Optional[float] = None
    map_gate_altitude_m: Optional[float] = None
    sim_last_gate_race_time: float = -1.0


@dataclass
class RacePlan:
    state: str
    forward_speed_mps: float
    yaw_scan_rate_rps: float = 0.0
    stabilize_heading: bool = False
    commit: bool = False
    use_map_navigation: bool = False
    climb_rate_mps: float = 0.0


@dataclass
class CommandSaturation:
    thrust_min: bool = False
    thrust_max: bool = False
    roll_rate_limit: bool = False
    pitch_rate_limit: bool = False
    yaw_rate_limit: bool = False

    @property
    def any(self) -> bool:
        return any(
            (
                self.thrust_min,
                self.thrust_max,
                self.roll_rate_limit,
                self.pitch_rate_limit,
                self.yaw_rate_limit,
            )
        )

    def as_dict(self) -> dict[str, bool]:
        return {
            "thrust_min": self.thrust_min,
            "thrust_max": self.thrust_max,
            "roll_rate_limit": self.roll_rate_limit,
            "pitch_rate_limit": self.pitch_rate_limit,
            "yaw_rate_limit": self.yaw_rate_limit,
        }


@dataclass
class FSMTransition:
    old_state: str
    new_state: str
    reason: str
    timestamp_s: float


@dataclass
class AttitudeCommand:
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    thrust: float
    roll_rate_rps: float
    pitch_rate_rps: float
    yaw_rate_rps: float
    quaternion_wxyz: tuple[float, float, float, float]
    saturation: CommandSaturation = field(default_factory=CommandSaturation)


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._vehicle = VehicleState()
        self._latest_frame: Optional[VisionFrame] = None
        self._gate_track: Optional[GateTrack] = None
        self._last_vision_frame_time_s = 0.0
        self._race_status: dict[str, Any] = {}
        self._track_gates: list[dict[str, Any]] = []

    def update_vehicle(self, vehicle: VehicleState) -> None:
        with self._lock:
            self._vehicle = copy.deepcopy(vehicle)

    def get_vehicle(self) -> VehicleState:
        with self._lock:
            return copy.deepcopy(self._vehicle)

    def update_frame(self, frame: VisionFrame) -> None:
        with self._lock:
            self._latest_frame = frame
            self._last_vision_frame_time_s = frame.wall_time_s

    def get_frame(self) -> Optional[VisionFrame]:
        with self._lock:
            return self._latest_frame

    def update_gate_track(self, gate_track: Optional[GateTrack]) -> None:
        with self._lock:
            self._gate_track = copy.deepcopy(gate_track)

    def get_gate_track(self) -> Optional[GateTrack]:
        with self._lock:
            return copy.deepcopy(self._gate_track)

    def vision_is_ready(
        self,
        now_s: Optional[float] = None,
        timeout_s: Optional[float] = None,
    ) -> bool:
        with self._lock:
            last_frame_time_s = self._last_vision_frame_time_s
        if last_frame_time_s <= 0.0:
            return False
        if now_s is None or timeout_s is None:
            return True
        return (now_s - last_frame_time_s) <= timeout_s

    def update_race_status(self, race_status: dict[str, Any]) -> None:
        with self._lock:
            self._race_status = copy.deepcopy(race_status)

    def get_race_status(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._race_status)

    def update_track_gates(self, track_gates: list[dict[str, Any]]) -> None:
        with self._lock:
            self._track_gates = copy.deepcopy(track_gates)

    def get_track_gates(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._track_gates)
