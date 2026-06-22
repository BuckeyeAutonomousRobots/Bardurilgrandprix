from __future__ import annotations

import math
import time
from typing import Optional


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

from src.estimation.vision_depth import estimate_gate_depth_m
from src.planning.map_guidance import (
    bearing_from_vehicle,
    exit_yaw_ned,
    gate_commit_info,
    gate_plane_metrics,
)
from src.types import EstimatedState, GateTrack, VehicleState


class StateEstimator:
    def __init__(self, camera_config: dict, sim_config: dict) -> None:
        self.camera = camera_config
        self.sim = sim_config
        self._bearing_x = 0.0
        self._bearing_y = 0.0
        self._range_m: Optional[float] = None
        self._range_x_m: Optional[float] = None
        self._range_y_m: Optional[float] = None
        self._depth_confidence = 0.0
        self._confidence = 0.0
        self._alpha = 0.35
        self._map_gate_position_ned: Optional[tuple[float, float, float]] = None
        self._map_blend = 0.0

    def _clear_visual_gate_state(self) -> None:
        self._bearing_x = 0.0
        self._bearing_y = 0.0
        self._range_m = None
        self._range_x_m = None
        self._range_y_m = None
        self._depth_confidence = 0.0
        self._confidence = 0.0

    def reset_for_next_gate(self) -> None:
        self._clear_visual_gate_state()
        self._map_gate_position_ned = None
        self._map_blend = 0.0

    @staticmethod
    def _resolve_gate_list_index(track_gates: list[dict], active_idx: int) -> Optional[int]:
        for i, gate in enumerate(track_gates):
            if int(gate.get("gate_id", -1)) == active_idx:
                return i
        if 0 <= active_idx < len(track_gates):
            return active_idx
        return None

    @staticmethod
    def _select_active_gate(track_gates: list[dict], active_idx: int) -> Optional[dict]:
        gate = next((g for g in track_gates if int(g.get("gate_id", -1)) == active_idx), None)
        if gate is not None:
            return gate
        if 0 <= active_idx < len(track_gates):
            return track_gates[active_idx]
        return None

    @staticmethod
    def _race_started(race_status: Optional[dict]) -> bool:
        if not race_status:
            return False
        start_ms = race_status.get("race_start_boot_time_ms")
        if start_ms is None or int(start_ms) < 0:
            return False
        updated = race_status.get("updated_wall_time")
        if updated is None:
            return False
        if time.time() - float(updated) > 1.5:
            return False
        finish_ns = race_status.get("race_finish_time_ns", -1)
        return finish_ns is None or int(finish_ns) < 0

    def estimate(
        self,
        now_s: float,
        vehicle: VehicleState,
        gate_track: Optional[GateTrack],
        vision_ready: bool,
        race_status: Optional[dict] = None,
        track_gates: Optional[list[dict]] = None,
        force_race_started: Optional[bool] = None,
        gate_corners_px: Optional[tuple[tuple[float, float], ...]] = None,
    ) -> EstimatedState:
        link_ready = vehicle.heartbeat_is_fresh(now_s, timeout_s=float(self.sim.get("link_timeout_s", 1.0)))
        race_started = (
            bool(force_race_started)
            if force_race_started is not None
            else self._race_started(race_status)
        )
        active_gate_index = -1
        has_track_map = False
        map_bearing_x = None
        map_bearing_y = None
        map_range = None
        map_path_bearing_x = None
        map_path_bearing_y = None
        map_path_range = None
        map_commit_bearing_x = None
        map_commit_bearing_y = None
        map_commit_range = None
        map_plane_signed = None
        map_within_bounds = False
        map_lateral_error = None
        map_vertical_error = None
        map_dist_center = None
        map_gate_commit_active = False
        map_gate_commit_strength = 0.0
        map_exit_yaw = None
        map_commit_speed = None
        map_gate_altitude = None
        sim_last_gate_race_time = -1.0
        use_map_first_gate = bool(self.sim.get("use_map_first_gate", True))
        vision_primary = bool(self.sim.get("vision_primary_navigation", False))
        map_navigation_enabled = use_map_first_gate and not vision_primary
        gate_pass_radius = float(self.sim.get("gate_pass_radius_m", 1.5))
        min_speed = float(self.sim.get("map_min_speed_mps", 2.5))
        cruise_speed = float(self.sim.get("map_cruise_speed_mps", 6.0))

        if race_status:
            sim_last_gate_race_time = float(race_status.get("last_gate_race_time", -1.0))
            active_gate_index = int(race_status.get("active_gate_index", -1))

        if (
            map_navigation_enabled
            and vehicle.position_ned_m is not None
            and race_status
            and track_gates
        ):
            active_idx = active_gate_index
            gate = self._select_active_gate(track_gates, active_idx)
            gate_idx = self._resolve_gate_list_index(track_gates, active_idx)
            if gate is not None and gate_idx is not None:
                pos = vehicle.position_ned_m
                gx, gy, gz = [float(v) for v in gate["position_ned"]]
                gate_center = (gx, gy, gz)
                map_gate_altitude = -gz
                map_bearing_x, map_bearing_y, map_range = bearing_from_vehicle(
                    pos, vehicle.yaw_rad, gate_center,
                )

                # Homing target is always gate center (altitude + horizontal) so layout
                # only changes where the gate is, not how we approach it.
                path_target = gate_center
                map_path_bearing_x, map_path_bearing_y, map_path_range = bearing_from_vehicle(
                    pos, vehicle.yaw_rad, path_target,
                )

                commit = gate_commit_info(
                    track_gates,
                    gate_idx,
                    pos,
                    min_speed_mps=min_speed,
                    cruise_speed_mps=cruise_speed,
                )
                if commit is not None:
                    map_gate_commit_active = commit.active
                    map_gate_commit_strength = commit.strength
                    map_dist_center = commit.dist_center_m
                    map_commit_speed = commit.commit_speed_mps
                    map_commit_bearing_x, map_commit_bearing_y, map_commit_range = bearing_from_vehicle(
                        pos, vehicle.yaw_rad, commit.through_ned,
                    )

                plane = gate_plane_metrics(
                    track_gates, gate_idx, pos, gate_pass_radius=gate_pass_radius,
                )
                if plane is not None:
                    map_plane_signed = plane.signed_dist_m
                    map_within_bounds = plane.within_bounds
                    map_lateral_error = plane.lateral_m
                    map_vertical_error = plane.vertical_m

                map_exit_yaw = exit_yaw_ned(track_gates, gate_idx)

                map_max_range = float(self.sim.get("map_max_range_m", 80.0))
                allow_first_gate_seed = race_started and use_map_first_gate and active_idx == 0
                if map_range is not None and (map_range <= map_max_range or allow_first_gate_seed):
                    has_track_map = True
                    self._map_gate_position_ned = gate_center
                else:
                    map_bearing_x = None
                    map_bearing_y = None
                    map_range = None

        vision_gate_fresh = gate_track is not None and gate_track.is_fresh(
            now_s, float(self.sim.get("gate_timeout_s", 0.35))
        )
        if not vision_ready:
            self._clear_visual_gate_state()
        elif vision_gate_fresh:
            u_des = float(self.camera["cx"])
            v_des = float(self.camera["desired_v_px"])
            fx = float(self.camera["fx"])
            fy = float(self.camera["fy"])
            theta_x = math.atan2(gate_track.center_px[0] - u_des, fx)
            tilt_rad = math.radians(float(self.camera.get("tilt_up_deg", 0.0)))
            theta_y = math.atan2(gate_track.center_px[1] - v_des, fy) - tilt_rad

            a = self._alpha
            self._bearing_x = (1.0 - a) * self._bearing_x + a * theta_x
            self._bearing_y = (1.0 - a) * self._bearing_y + a * theta_y
            self._confidence = (1.0 - a) * self._confidence + a * gate_track.confidence

            meas_range, meas_rx, meas_ry, depth_conf = estimate_gate_depth_m(
                gate_track,
                self.camera,
                corners_px=gate_corners_px,
            )
            a_depth = min(0.55, a + 0.15)
            self._range_m = meas_range if self._range_m is None else ((1.0 - a_depth) * self._range_m + a_depth * meas_range)
            self._range_x_m = meas_rx if self._range_x_m is None else ((1.0 - a) * self._range_x_m + a * meas_rx)
            self._range_y_m = meas_ry if self._range_y_m is None else ((1.0 - a) * self._range_y_m + a * meas_ry)
            self._depth_confidence = (1.0 - a) * self._depth_confidence + a * depth_conf
        else:
            self._confidence *= 0.92
            self._depth_confidence *= 0.90
            self._bearing_x *= 0.80
            self._bearing_y *= 0.80
            self._range_m = None
            self._range_x_m = None
            self._range_y_m = None

        bearing_x = self._bearing_x
        bearing_y = self._bearing_y
        gate_range_m = self._range_m
        gate_range_x_m = self._range_x_m
        gate_range_y_m = self._range_y_m
        gate_depth_confidence = self._depth_confidence
        gate_confidence = self._confidence
        prefer_visual = vision_gate_fresh and gate_confidence >= float(self.sim.get("detect_confidence", 0.5)) * 0.8
        map_first_nav = race_started and map_navigation_enabled and has_track_map
        if map_first_nav:
            prefer_visual = False
        use_map_bearing = race_started and map_path_bearing_x is not None and not prefer_visual
        if map_first_nav:
            self._map_blend = 1.0
        elif use_map_bearing:
            target_blend = 1.0
            blend_rate = float(self.sim.get("map_blend_rate", 0.15))
            self._map_blend = clamp(self._map_blend + (target_blend - self._map_blend) * blend_rate, 0.0, 1.0)
        else:
            target_blend = 0.0
            blend_rate = float(self.sim.get("map_blend_rate", 0.15))
            self._map_blend = clamp(self._map_blend + (target_blend - self._map_blend) * blend_rate, 0.0, 1.0)
        if self._map_blend > 0.0 and map_path_bearing_x is not None:
            map_bx = map_path_bearing_x
            map_by = map_path_bearing_y if map_path_bearing_y is not None else bearing_y
            map_br = map_path_range if map_path_range is not None else map_range
            bearing_x = (1.0 - self._map_blend) * bearing_x + self._map_blend * map_bx
            bearing_y = (1.0 - self._map_blend) * bearing_y + self._map_blend * map_by
            if gate_range_m is not None and map_br is not None:
                gate_range_m = (1.0 - self._map_blend) * gate_range_m + self._map_blend * map_br
            elif map_br is not None:
                gate_range_m = map_br

        return EstimatedState(
            now_s=now_s,
            vehicle=vehicle,
            gate_track=gate_track,
            link_ready=link_ready,
            vision_ready=vision_ready,
            gate_bearing_x_rad=bearing_x,
            gate_bearing_y_rad=bearing_y,
            gate_range_m=gate_range_m,
            gate_range_x_m=gate_range_x_m,
            gate_range_y_m=gate_range_y_m,
            gate_depth_confidence=gate_depth_confidence,
            gate_confidence=gate_confidence,
            race_started=race_started,
            has_track_map=has_track_map,
            active_gate_index=active_gate_index,
            map_gate_bearing_x_rad=map_bearing_x,
            map_gate_bearing_y_rad=map_bearing_y,
            map_gate_range_m=map_range,
            map_approach_bearing_x_rad=map_path_bearing_x,
            map_approach_bearing_y_rad=map_path_bearing_y,
            map_approach_range_m=map_path_range,
            map_commit_bearing_x_rad=map_commit_bearing_x,
            map_commit_bearing_y_rad=map_commit_bearing_y,
            map_commit_range_m=map_commit_range,
            map_plane_signed_m=map_plane_signed,
            map_within_gate_bounds=map_within_bounds,
            map_lateral_error_m=map_lateral_error,
            map_vertical_error_m=map_vertical_error,
            map_dist_center_m=map_dist_center,
            map_gate_commit_active=map_gate_commit_active,
            map_gate_commit_strength=map_gate_commit_strength,
            map_exit_yaw_rad=map_exit_yaw,
            map_commit_speed_mps=map_commit_speed,
            map_gate_altitude_m=map_gate_altitude,
            sim_last_gate_race_time=sim_last_gate_race_time,
        )
