from __future__ import annotations

import math

from dataclasses import dataclass

from src.types import EstimatedState, FSMTransition, RacePlan


@dataclass
class RaceFSM:
    config: dict
    state: str = "WAIT_LINK"
    _state_since_s: float = 0.0
    _stable_frames: int = 0
    _target_gate_index: int = -1
    _last_sim_gate_race_time: float = -1.0
    _gate_lost_since_s: float | None = None
    _commit_peak_area_fraction: float = 0.0
    last_transition: FSMTransition | None = None

    @property
    def target_gate_index(self) -> int:
        return self._target_gate_index

    def _gate_track_usable(self, gate_track, now_s: float, *, vision_primary: bool) -> bool:
        if gate_track is None:
            return False
        timeout_key = "vision_gate_timeout_s" if vision_primary else "gate_timeout_s"
        timeout_s = float(self.config.get(timeout_key, self.config["gate_timeout_s"]))
        if gate_track.is_fresh(now_s, timeout_s):
            return True
        if not vision_primary:
            return False
        dropout_s = float(self.config.get("gate_dropout_s", self.config["gate_timeout_s"]))
        return (
            gate_track.predicted
            and gate_track.timestamp_s > 0.0
            and (now_s - gate_track.timestamp_s) <= dropout_s
        )

    def _update_gate_lost_timer(self, gate_usable: bool, now_s: float) -> float | None:
        if gate_usable:
            self._gate_lost_since_s = None
            return None
        if self._gate_lost_since_s is None:
            self._gate_lost_since_s = now_s
        return self._gate_lost_since_s

    def _vision_align_recover_timeout_s(self, vision_primary: bool) -> float:
        if vision_primary:
            return float(self.config.get("vision_align_recover_timeout_s", 1.25))
        return float(self.config["recover_timeout_s"])

    def _vision_gate_locked(
        self,
        est: EstimatedState,
        gate_track,
        *,
        gate_usable: bool,
        gate_strong: bool,
        vision_primary: bool,
    ) -> bool:
        """Smoothed gate confidence can stay high while the track is briefly predicted."""
        if not vision_primary:
            return gate_strong
        if gate_strong:
            return True
        detect_conf = float(self.config["detect_confidence"])
        return (
            gate_usable
            and gate_track is not None
            and est.gate_confidence >= detect_conf
        )

    def update(
        self,
        est: EstimatedState,
        *,
        recover_requested: bool = False,
        recover_reason: str = "",
    ) -> RacePlan:
        now_s = est.now_s
        self.last_transition = None
        if self._state_since_s <= 0.0:
            self._state_since_s = now_s

        # Global emergency descent check - can trigger from any state
        vz = 0.0
        if est.vehicle.velocity_ned_mps is not None:
            vz = est.vehicle.velocity_ned_mps[2]
        if (
            vz > float(self.config.get("emergency_descent_threshold_mps", 10.0))
            and self.state not in {"EMERGENCY_RECOVERY", "TAKEOFF", "WAIT_START", "WAIT_VISION", "WAIT_LINK"}
        ):
            self._transition("EMERGENCY_RECOVERY", now_s, "falling_too_fast")
            return self._build_plan()

        gate_track = est.gate_track
        vision_primary = bool(self.config.get("vision_primary_navigation", False))
        gate_fresh_timeout_s = float(
            self.config.get("vision_gate_timeout_s", self.config["gate_timeout_s"])
            if vision_primary
            else self.config["gate_timeout_s"]
        )
        gate_fresh = gate_track is not None and gate_track.is_fresh(now_s, gate_fresh_timeout_s)
        gate_usable = self._gate_track_usable(gate_track, now_s, vision_primary=vision_primary)
        gate_lost_since_s = self._update_gate_lost_timer(gate_usable, now_s)
        align_recover_timeout_s = self._vision_align_recover_timeout_s(vision_primary)
        map_ready = (
            not vision_primary
            and est.race_started
            and est.has_track_map
            and est.active_gate_index >= 0
        )
        speed = 0.0
        horizontal_speed = 0.0
        if est.vehicle.velocity_ned_mps is not None:
            vx, vy, vz = est.vehicle.velocity_ned_mps
            horizontal_speed = float((vx * vx + vy * vy) ** 0.5)
            speed = float((vx * vx + vy * vy + vz * vz) ** 0.5)
        stabilize_roll_pitch_rad = float(self.config.get("stabilize_roll_pitch_rad", 0.20))
        align_roll_pitch_rad = float(
            self.config.get("map_align_roll_pitch_rad", self.config.get("align_roll_pitch_rad", 0.32))
        )
        stabilize_speed_mps = float(self.config.get("stabilize_speed_mps", 0.80))
        recover_speed_mps = float(self.config.get("recover_speed_mps", 2.50))
        search_min_time_s = float(self.config.get("search_min_time_s", 1.0))
        search_min_area_fraction = float(self.config.get("search_min_area_fraction", 0.02))
        attitude_stable = (
            abs(est.vehicle.roll_rad) <= stabilize_roll_pitch_rad
            and abs(est.vehicle.pitch_rad) <= stabilize_roll_pitch_rad
        )
        search_attitude_limit = float(
            self.config.get(
                "vision_search_attitude_roll_pitch_rad",
                stabilize_roll_pitch_rad if not vision_primary else 0.32,
            )
        )
        search_attitude_stable = (
            abs(est.vehicle.roll_rad) <= search_attitude_limit
            and abs(est.vehicle.pitch_rad) <= search_attitude_limit
        )
        vision_acquire_max_speed = float(self.config.get("vision_acquire_max_speed_mps", 1.2))
        motion_stable = horizontal_speed <= stabilize_speed_mps
        search_motion_limit = float(
            self.config.get(
                "search_motion_stable_speed_mps",
                stabilize_speed_mps if not vision_primary else 7.5,
            )
        )
        search_motion_stable = horizontal_speed <= search_motion_limit
        stabilize_max_vz_mps = float(self.config.get("stabilize_max_vz_mps", 3.0))
        vz_ok = abs(vz) <= stabilize_max_vz_mps
        vision_gate_below_rad = float(self.config.get("vision_gate_below_bearing_rad", 0.08))
        lower_gate_visible = vision_primary and est.gate_bearing_y_rad >= vision_gate_below_rad
        vision_acquire_max_climb_mps = float(self.config.get("vision_acquire_max_climb_mps", 0.80))
        if lower_gate_visible:
            vision_acquire_max_climb_mps = float(
                self.config.get("vision_lower_gate_acquire_max_climb_mps", 0.35)
            )
        vision_vertical_acquire_ready = vz >= -vision_acquire_max_climb_mps
        takeoff_attitude_limit = float(
            self.config.get(
                "vision_takeoff_attitude_roll_pitch_rad",
                self.config.get("stabilize_roll_pitch_rad", 0.22),
            )
            if vision_primary
            else float(self.config.get("stabilize_roll_pitch_rad", 0.22))
        )
        takeoff_attitude_stable = (
            abs(est.vehicle.roll_rad) <= takeoff_attitude_limit
            and abs(est.vehicle.pitch_rad) <= takeoff_attitude_limit
        )
        gate_strong = (
            gate_fresh
            and gate_track is not None
            and gate_track.visible
            and not gate_track.predicted
            and gate_track.area_fraction >= search_min_area_fraction
            and est.gate_confidence >= float(self.config["detect_confidence"])
        )
        vision_gate_locked = self._vision_gate_locked(
            est,
            gate_track,
            gate_usable=gate_usable,
            gate_strong=gate_strong,
            vision_primary=vision_primary,
        )
        count_gate = self.state in {"SEARCH_GATE", "RECOVER", "ALIGN_GATE", "APPROACH_GATE"}
        gate_trackable = gate_fresh and gate_track is not None and gate_track.visible
        if vision_primary and gate_track is not None and self.state in {"APPROACH_GATE", "COMMIT_GATE"}:
            self._commit_peak_area_fraction = max(
                self._commit_peak_area_fraction,
                float(gate_track.area_fraction),
            )
        if vision_primary and self.state == "SEARCH_GATE":
            gate_trackable = gate_trackable and est.gate_confidence >= float(self.config["detect_confidence"]) * 0.65
        if count_gate and gate_strong:
            self._stable_frames += 1
        elif count_gate and vision_primary and self.state == "SEARCH_GATE" and gate_trackable:
            self._stable_frames += 1
        elif count_gate and vision_primary and self.state == "SEARCH_GATE" and gate_fresh:
            self._stable_frames = max(0, self._stable_frames - 1)
        else:
            self._stable_frames = 0

        aligned = (
            abs(est.gate_bearing_x_rad) <= float(self.config["align_bearing_rad"])
            and abs(est.gate_bearing_y_rad) <= float(self.config["align_vertical_rad"])
        )
        if vision_primary:
            vision_align_b = float(self.config.get("vision_align_bearing_rad", 0.24))
            vision_align_v = float(self.config.get("vision_align_vertical_rad", 0.24))
            aligned = (
                abs(est.gate_bearing_x_rad) <= vision_align_b
                and abs(est.gate_bearing_y_rad) <= vision_align_v
            )
        vision_partial_aligned = False
        if vision_primary:
            partial_b = float(self.config.get("vision_align_bearing_rad", 0.24)) * 1.5
            partial_v = float(self.config.get("vision_align_vertical_rad", 0.24)) * 1.5
            vision_partial_aligned = (
                abs(est.gate_bearing_x_rad) <= partial_b
                and abs(est.gate_bearing_y_rad) <= partial_v
            )
        tightly_aligned = (
            abs(est.gate_bearing_x_rad) <= float(self.config["approach_bearing_rad"])
            and abs(est.gate_bearing_y_rad) <= float(self.config["align_vertical_rad"])
        )
        range_ready = (
            est.gate_range_m is not None and est.gate_range_m <= float(self.config["commit_range_m"])
        ) or (
            gate_track is not None and gate_track.area_fraction >= float(self.config["commit_area_fraction"])
        )
        start_gate_commit_ready = (
            gate_track is not None
            and gate_strong
            and gate_track.area_fraction >= float(self.config.get("start_gate_commit_area_fraction", 0.30))
            and abs(est.gate_bearing_x_rad) <= float(self.config.get("start_gate_commit_bearing_rad", 0.10))
            and abs(est.gate_bearing_y_rad) <= float(self.config["align_vertical_rad"])
        )
        map_horiz_aligned = (
            map_ready
            and abs(est.gate_bearing_x_rad) <= float(self.config.get("map_align_bearing_rad", self.config["align_bearing_rad"]))
        )
        map_aligned = (
            map_horiz_aligned
            and abs(est.gate_bearing_y_rad) <= float(self.config.get("map_align_vertical_rad", self.config["align_vertical_rad"]))
        )
        map_tightly_aligned = (
            map_ready
            and abs(est.gate_bearing_x_rad) <= float(self.config.get("map_approach_bearing_rad", self.config["approach_bearing_rad"]))
            and abs(est.gate_bearing_y_rad) <= float(self.config.get("map_align_vertical_rad", self.config["align_vertical_rad"]))
        )
        map_range_ready = (
            map_ready
            and est.gate_range_m is not None
            and est.gate_range_m <= float(self.config.get("map_commit_range_m", self.config["commit_range_m"]))
        )
        map_max_lateral_m = float(self.config.get("map_max_lateral_error_m", 0.75))
        map_max_vertical_m = float(self.config.get("map_max_vertical_error_m", 0.75))
        map_lateral_ready = (
            not map_ready
            or est.map_lateral_error_m is None
            or abs(est.map_lateral_error_m) <= map_max_lateral_m
            or (
                est.map_dist_center_m is not None
                and est.map_dist_center_m
                > float(self.config.get("map_align_lateral_relax_dist_m", 12.0))
            )
        )
        map_vertical_ready = (
            not map_ready
            or est.map_vertical_error_m is None
            or abs(est.map_vertical_error_m) <= map_max_vertical_m
            or (
                est.map_dist_center_m is not None
                and est.map_dist_center_m
                > float(self.config.get("map_align_lateral_relax_dist_m", 12.0))
            )
        )
        map_gate_volume_ready = (
            map_ready
            and est.map_within_gate_bounds
            and est.map_dist_center_m is not None
            and est.map_dist_center_m
            <= float(self.config.get("map_gate_volume_dist_m", 3.5))
        )
        map_before_gate = (
            not map_ready
            or est.map_plane_signed_m is None
            or est.map_plane_signed_m
            < float(self.config.get("map_approach_max_plane_m", 1.0))
        )
        map_past_gate_overshoot = (
            map_ready
            and est.map_plane_signed_m is not None
            and est.map_plane_signed_m
            > float(self.config.get("map_approach_overshoot_plane_m", 2.5))
        )
        map_commit_line_ready = (
            map_ready
            and est.map_gate_commit_active
            and est.map_within_gate_bounds
            and est.map_plane_signed_m is not None
            and est.map_plane_signed_m >= float(self.config.get("map_commit_line_trigger_m", -0.35))
        )
        map_classical_commit_ready = (
            map_ready
            and map_horiz_aligned
            and map_lateral_ready
            and map_vertical_ready
            and est.map_gate_commit_active
            and est.map_gate_commit_strength >= float(self.config.get("map_commit_strength_min", 0.40))
            and est.map_dist_center_m is not None
            and est.map_dist_center_m <= float(self.config.get("map_classical_commit_dist_m", 8.0))
            and est.map_within_gate_bounds
        )
        gate_index_changed = (
            self._target_gate_index >= 0
            and est.active_gate_index >= 0
            and est.active_gate_index != self._target_gate_index
        )
        sim_gate_passed = (
            self._target_gate_index >= 0
            and est.sim_last_gate_race_time > 0.0
            and est.sim_last_gate_race_time != self._last_sim_gate_race_time
            and est.active_gate_index > self._target_gate_index
        )
        sim_gate_timing = (
            self._target_gate_index >= 0
            and est.sim_last_gate_race_time > 0.0
            and est.sim_last_gate_race_time != self._last_sim_gate_race_time
        )
        if sim_gate_passed or sim_gate_timing:
            self._last_sim_gate_race_time = est.sim_last_gate_race_time

        if recover_requested and self.state not in {"WAIT_LINK", "WAIT_VISION"}:
            self._transition("RECOVER", now_s, recover_reason or "recover_requested")
            return self._build_plan()

        if (gate_index_changed or sim_gate_passed or sim_gate_timing) and self.state not in {
            "WAIT_LINK", "WAIT_VISION", "WAIT_START", "PASS_GATE",
        }:
            self._target_gate_index = max(self._target_gate_index, est.active_gate_index)
            reason = "active_gate_index_advanced"
            if sim_gate_timing and not gate_index_changed and not sim_gate_passed:
                reason = "sim_gate_race_time"
            self._transition("PASS_GATE", now_s, reason)
            return self._build_plan()

        if self.state == "WAIT_LINK":
            if est.link_ready:
                self._transition("WAIT_VISION", now_s, "link_ready")
        elif self.state == "WAIT_VISION":
            if est.vision_ready:
                self._transition("WAIT_START", now_s, "vision_ready")
        elif self.state == "WAIT_START":
            alt_ok = True
            if est.vehicle.position_ned_m is not None:
                alt_ok = abs(float(est.vehicle.position_ned_m[2])) <= float(
                    self.config.get("spawn_altitude_max_m", 50.0)
                )
            if est.race_started and est.vehicle.armed and alt_ok:
                self._target_gate_index = est.active_gate_index
                self._transition("TAKEOFF", now_s, "race_started")
        elif self.state == "EMERGENCY_RECOVERY":
            vz = 0.0
            if est.vehicle.velocity_ned_mps is not None:
                vz = est.vehicle.velocity_ned_mps[2]
            # Exit when descent rate is under control
            if vz <= float(self.config.get("emergency_recovery_vz_mps", 5.0)):
                self._transition("TAKEOFF", now_s, "descent_arrested")
        elif self.state == "TAKEOFF":
            self._stable_frames = 0
            vz = 0.0
            if est.vehicle.velocity_ned_mps is not None:
                vz = est.vehicle.velocity_ned_mps[2]
            alt = 0.0
            if est.vehicle.position_ned_m is not None:
                px, py, pz = est.vehicle.position_ned_m
                alt = -float(pz)
                sane_xy = float(self.config.get("vision_position_sane_xy_m", 40.0))
                sane_alt = float(self.config.get("vision_takeoff_alt_sane_max_m", 6.0))
                if vision_primary and (
                    math.hypot(float(px), float(py)) > sane_xy
                    or alt > sane_alt
                    or alt < -0.5
                ):
                    alt = min(
                        float(self.config.get("vision_takeoff_min_alt_m", 0.65)),
                        float(self.config.get("vision_takeoff_altitude_m", 1.0)),
                    )
            if vision_primary:
                takeoff_alt = float(
                    self.config.get(
                        "vision_takeoff_altitude_m",
                        self.config.get("takeoff_altitude_m", 5.0),
                    )
                )
            else:
                takeoff_alt = float(self.config.get("takeoff_altitude_m", 5.0))
            takeoff_max_s = float(self.config.get("takeoff_max_time_s", 10.0))
            if vision_primary:
                takeoff_max_s = float(
                    self.config.get("vision_takeoff_max_time_s", takeoff_max_s)
                )
            timed_out = (now_s - self._state_since_s) >= takeoff_max_s
            alt_fraction = float(
                self.config.get("vision_takeoff_alt_fraction", 0.72)
                if vision_primary
                else 0.85
            )
            at_altitude = alt >= takeoff_alt * alt_fraction
            climb_settle_mps = float(self.config.get("takeoff_climb_settle_mps", 0.0))
            if climb_settle_mps <= 0.0:
                climb_settle_mps = float(self.config.get("takeoff_climb_rate_mps", 2.0)) * 2.0
            if vision_primary:
                climb_settle_mps = float(
                    self.config.get("vision_takeoff_climb_settle_mps", climb_settle_mps)
                )
            climb_settled = abs(vz) <= climb_settle_mps
            vision_min_alt = float(self.config.get("vision_takeoff_min_alt_m", takeoff_alt * 0.70))
            vision_min_time_s = float(self.config.get("vision_takeoff_min_time_s", 0.25))
            takeoff_elapsed_s = now_s - self._state_since_s
            vision_gate_takeoff_done = (
                vision_primary
                and gate_strong
                and alt >= vision_min_alt
                and takeoff_elapsed_s >= vision_min_time_s
            )
            vision_min_alt_done = (
                vision_primary
                and alt >= vision_min_alt
                and takeoff_elapsed_s >= vision_min_time_s
            )
            takeoff_complete = est.vehicle.armed and (
                timed_out
                or (
                    takeoff_attitude_stable
                    and (
                        vision_gate_takeoff_done
                        or vision_min_alt_done
                        or (at_altitude and climb_settled)
                    )
                )
            )
            if takeoff_complete:
                reason = "takeoff_complete"
                if vision_gate_takeoff_done and not (at_altitude and climb_settled):
                    reason = "vision_gate_visible_takeoff"
                elif vision_min_alt_done and not (at_altitude and climb_settled):
                    reason = "vision_min_alt_takeoff"
                self._target_gate_index = est.active_gate_index
                skip_stabilize = vision_primary and bool(
                    self.config.get("vision_takeoff_skip_stabilize", True)
                )
                if skip_stabilize and gate_strong and vision_vertical_acquire_ready:
                    self._transition("ALIGN_GATE", now_s, reason)
                elif skip_stabilize and vision_vertical_acquire_ready:
                    self._transition("SEARCH_GATE", now_s, reason)
                else:
                    self._transition("STABILIZE", now_s, reason)
        elif self.state == "STABILIZE":
            self._stable_frames = 0
            stabilize_time_s = float(self.config["stabilize_time_s"])
            if vision_primary:
                stabilize_time_s = float(self.config.get("vision_stabilize_time_s", 0.45))
            if (
                est.vehicle.armed
                and attitude_stable
                and motion_stable
                and vz_ok
                and (now_s - self._state_since_s) >= stabilize_time_s
            ):
                self._target_gate_index = est.active_gate_index
                if map_ready and bool(self.config.get("use_map_first_gate", True)) and not vision_primary:
                    self._transition("ALIGN_GATE", now_s, "stabilize_complete_map_gate")
                elif (
                    vision_primary
                    and gate_strong
                    and vision_partial_aligned
                    and horizontal_speed <= vision_acquire_max_speed * 1.5
                ):
                    self._transition("ALIGN_GATE", now_s, "stabilize_gate_visible")
                else:
                    self._transition("SEARCH_GATE", now_s, "stabilize_complete")
        elif self.state == "SEARCH_GATE":
            if start_gate_commit_ready and motion_stable:
                self._transition("COMMIT_GATE", now_s, "start_gate_commit")
            elif (
                vision_primary
                and gate_strong
                and gate_track is not None
                and gate_track.area_fraction >= float(self.config.get("vision_acquire_area_fraction", 0.015))
                and horizontal_speed <= vision_acquire_max_speed * 1.5
                and search_attitude_stable
                and vision_vertical_acquire_ready
            ):
                self._transition("ALIGN_GATE", now_s, "vision_gate_strong_acquire")
            elif (
                vision_primary
                and gate_fresh
                and gate_track is not None
                and gate_track.visible
                and est.gate_confidence >= float(self.config["detect_confidence"])
                and gate_track.area_fraction >= float(self.config.get("vision_acquire_area_fraction", 0.04))
                and search_attitude_stable
                and horizontal_speed <= vision_acquire_max_speed
                and vision_vertical_acquire_ready
            ):
                self._transition("ALIGN_GATE", now_s, "vision_gate_acquire")
            elif (
                (now_s - self._state_since_s) >= search_min_time_s
                and self._stable_frames >= int(self.config["stable_frames_required"])
                and aligned
                and attitude_stable
                and (search_motion_stable if vision_primary else motion_stable)
                and (vision_vertical_acquire_ready if vision_primary else True)
            ):
                self._transition("ALIGN_GATE", now_s, "stable_gate_track")
        elif self.state == "ALIGN_GATE":
            if gate_index_changed:
                self._transition("PASS_GATE", now_s, "active_gate_index_advanced")
            elif map_past_gate_overshoot:
                self._transition("STABILIZE", now_s, "align_past_gate_stabilize")
            elif map_ready:
                if (
                    range_ready
                    and tightly_aligned
                    and map_before_gate
                    and map_lateral_ready
                    and map_vertical_ready
                ):
                    self._transition("COMMIT_GATE", now_s, "visual_commit_ready_with_map")
                elif map_classical_commit_ready:
                    self._transition("COMMIT_GATE", now_s, "map_classical_commit_ready")
                elif map_commit_line_ready:
                    self._transition("COMMIT_GATE", now_s, "map_plane_commit_ready")
                elif (
                    not (
                        abs(est.vehicle.roll_rad) <= align_roll_pitch_rad
                        and abs(est.vehicle.pitch_rad) <= align_roll_pitch_rad
                    )
                    and speed > recover_speed_mps
                    and abs(vz) <= float(self.config.get("stabilize_max_vz_mps", 3.0))
                ):
                    self._transition("RECOVER", now_s, "unstable_during_align")
                elif map_horiz_aligned or aligned:
                    if (
                        map_range_ready
                        and map_tightly_aligned
                        and map_lateral_ready
                        and map_vertical_ready
                        and map_gate_volume_ready
                    ):
                        self._transition("COMMIT_GATE", now_s, "map_aligned_and_range_ready")
                    elif map_horiz_aligned and map_lateral_ready and map_vertical_ready and map_before_gate:
                        self._transition("APPROACH_GATE", now_s, "map_or_visual_aligned")
                elif (
                    (now_s - self._state_since_s) >= float(self.config.get("align_max_time_s", 4.0))
                    and map_ready
                    and map_before_gate
                    and abs(est.gate_bearing_x_rad) <= float(self.config.get("map_align_bearing_rad", 0.10)) * 1.5
                ):
                    self._transition("APPROACH_GATE", now_s, "align_timeout_horiz_ok")
            elif (
                gate_lost_since_s is not None
                and (now_s - gate_lost_since_s) > align_recover_timeout_s
                and not (
                    vision_primary
                    and est.gate_confidence
                    >= float(self.config["detect_confidence"]) * 0.55
                )
            ):
                self._transition("RECOVER", now_s, "gate_timeout_align")
            elif (
                vision_primary
                and not vision_gate_locked
                and horizontal_speed > float(self.config.get("vision_recover_speed_mps", 4.5))
                and (now_s - self._state_since_s) >= float(self.config.get("vision_overspeed_min_time_s", 0.20))
            ):
                self._transition("RECOVER", now_s, "vision_overspeed")
            elif (
                not vision_primary
                and not (
                    abs(est.vehicle.roll_rad) <= align_roll_pitch_rad
                    and abs(est.vehicle.pitch_rad) <= align_roll_pitch_rad
                )
                and speed > recover_speed_mps
                and abs(vz) <= float(self.config.get("stabilize_max_vz_mps", 3.0))
            ):
                self._transition("RECOVER", now_s, "unstable_during_align")
            elif vision_primary and gate_strong and horizontal_speed <= float(
                self.config.get("vision_approach_max_speed_mps", 1.6)
            ) and (now_s - self._state_since_s) >= float(self.config.get("vision_align_min_time_s", 0.35)):
                self._transition("APPROACH_GATE", now_s, "vision_align_speed_ready")
            elif vision_primary and gate_strong and aligned:
                if range_ready and tightly_aligned:
                    self._transition("COMMIT_GATE", now_s, "vision_aligned_and_range_ready")
                else:
                    self._transition("APPROACH_GATE", now_s, "vision_aligned")
            elif vision_primary and gate_strong:
                if vision_partial_aligned and (now_s - self._state_since_s) >= 0.25:
                    self._transition("APPROACH_GATE", now_s, "vision_partial_align")
            elif aligned:
                if range_ready and tightly_aligned:
                    self._transition("COMMIT_GATE", now_s, "aligned_and_range_ready")
                else:
                    self._transition("APPROACH_GATE", now_s, "aligned")
        elif self.state == "APPROACH_GATE":
            if gate_index_changed:
                self._transition("PASS_GATE", now_s, "active_gate_index_advanced")
            elif map_past_gate_overshoot:
                self._transition("ALIGN_GATE", now_s, "approach_overshoot_realign")
            elif map_ready:
                if (
                    range_ready
                    and tightly_aligned
                    and map_lateral_ready
                    and map_vertical_ready
                    and map_gate_volume_ready
                    and map_before_gate
                ):
                    self._transition("COMMIT_GATE", now_s, "visual_approach_range_ready_with_map")
                elif map_classical_commit_ready:
                    self._transition("COMMIT_GATE", now_s, "map_classical_commit_ready")
                elif (
                    map_horiz_aligned
                    and map_lateral_ready
                    and map_vertical_ready
                    and map_gate_volume_ready
                    and est.map_dist_center_m is not None
                    and est.map_dist_center_m <= float(self.config.get("map_early_commit_dist_m", 7.0))
                ):
                    self._transition("COMMIT_GATE", now_s, "map_early_commit_dist")
                elif map_commit_line_ready and map_lateral_ready and map_vertical_ready:
                    self._transition("COMMIT_GATE", now_s, "map_plane_commit_ready")
                elif not attitude_stable and speed > recover_speed_mps:
                    self._transition("RECOVER", now_s, "unstable_during_approach")
                elif map_range_ready and map_tightly_aligned and map_lateral_ready and map_vertical_ready and map_gate_volume_ready:
                    self._transition("COMMIT_GATE", now_s, "map_approach_range_ready")
                elif not map_aligned and not aligned and (
                    est.map_dist_center_m is None
                    or est.map_dist_center_m
                    > float(self.config.get("map_align_lateral_relax_dist_m", 12.0))
                ):
                    self._transition("ALIGN_GATE", now_s, "map_lost_alignment")
            elif (
                gate_lost_since_s is not None
                and (now_s - gate_lost_since_s) > align_recover_timeout_s
                and not (
                    vision_primary
                    and est.gate_confidence
                    >= float(self.config["detect_confidence"]) * 0.55
                )
            ):
                self._transition("RECOVER", now_s, "gate_timeout_approach")
            elif not vision_primary and not attitude_stable and speed > recover_speed_mps:
                self._transition("RECOVER", now_s, "unstable_during_approach")
            elif vision_primary and gate_strong and gate_track is not None:
                dwell_s = float(self.config.get("vision_approach_commit_time_s", 2.0))
                if (
                    gate_strong
                    and horizontal_speed <= float(self.config.get("vision_commit_max_speed_mps", 2.5))
                    and (now_s - self._state_since_s) >= dwell_s
                ):
                    self._transition("COMMIT_GATE", now_s, "vision_approach_dwell_commit")
                loose_b = float(self.config.get("vision_align_bearing_rad", 0.24)) * 2.0
                loose_v = float(self.config.get("vision_align_vertical_rad", 0.24)) * 2.0
                commit_speed_cap = float(self.config.get("vision_commit_max_speed_mps", 2.8))
                loosely_aligned = (
                    abs(est.gate_bearing_x_rad) <= loose_b
                    and abs(est.gate_bearing_y_rad) <= loose_v
                )
                commit_area = float(
                    self.config.get(
                        "vision_commit_area_fraction",
                        self.config.get("commit_area_fraction", 0.10),
                    )
                )
                if (
                    loosely_aligned
                    and horizontal_speed <= commit_speed_cap
                    and gate_track.area_fraction >= commit_area
                ):
                    self._transition("COMMIT_GATE", now_s, "vision_area_commit")
                elif loosely_aligned and range_ready:
                    self._transition("COMMIT_GATE", now_s, "vision_approach_range_ready")
            elif range_ready and tightly_aligned:
                self._transition("COMMIT_GATE", now_s, "approach_range_ready")
            elif not vision_primary and not aligned:
                self._transition("ALIGN_GATE", now_s, "lost_alignment")
        elif self.state == "COMMIT_GATE":
            if gate_track is not None:
                self._commit_peak_area_fraction = max(
                    self._commit_peak_area_fraction,
                    float(gate_track.area_fraction),
                )
            if gate_index_changed or sim_gate_passed or sim_gate_timing:
                self._target_gate_index = max(self._target_gate_index, est.active_gate_index)
                reason = "active_gate_index_advanced"
                if sim_gate_timing and not gate_index_changed and not sim_gate_passed:
                    reason = "sim_gate_race_time"
                self._transition("PASS_GATE", now_s, reason)
            elif map_ready:
                commit_elapsed_s = now_s - self._state_since_s
                post_gate_margin_m = float(self.config.get("map_post_gate_margin_m", 1.5))
                commit_fail_s = float(self.config.get("map_commit_fail_time_s", 1.25))
                min_plane_m = float(self.config.get("map_pass_min_plane_m", 0.0))
                cross_plane_m = float(self.config.get("map_gate_cross_plane_m", 0.5))
                pass_radius_m = float(self.config.get("gate_pass_radius_m", 1.5))
                crossed_forward = (
                    est.map_plane_signed_m is not None
                    and est.map_plane_signed_m >= post_gate_margin_m
                    and est.map_within_gate_bounds
                )
                near_plane = (
                    est.map_plane_signed_m is not None
                    and est.map_plane_signed_m > -1.5
                )
                local_plane_crossed = (
                    map_gate_volume_ready
                    and est.map_plane_signed_m is not None
                    and est.map_plane_signed_m >= cross_plane_m
                    and est.map_plane_signed_m <= post_gate_margin_m + 1.0
                    and commit_elapsed_s >= 0.05
                )
                local_near_center = (
                    map_gate_volume_ready
                    and est.map_dist_center_m is not None
                    and est.map_dist_center_m <= pass_radius_m
                    and est.map_plane_signed_m is not None
                    and est.map_plane_signed_m >= min_plane_m
                    and commit_elapsed_s >= 0.10
                )
                if local_plane_crossed:
                    self._transition("PASS_GATE", now_s, "map_plane_crossed")
                elif local_near_center:
                    self._transition("PASS_GATE", now_s, "map_near_center_at_plane")
                elif crossed_forward and commit_elapsed_s >= commit_fail_s:
                    self._transition("PASS_GATE", now_s, "crossed_plane_without_gate_advance")
                elif commit_elapsed_s >= commit_fail_s and not near_plane:
                    self._transition("RECOVER", now_s, "commit_timeout_without_gate_advance")
                elif (
                    est.map_plane_signed_m is not None
                    and est.map_plane_signed_m > cross_plane_m
                    and not est.map_within_gate_bounds
                    and commit_elapsed_s >= 0.35
                ):
                    self._transition("ALIGN_GATE", now_s, "commit_overshoot_realign")
            elif vision_primary:
                commit_elapsed_s = now_s - self._state_since_s
                loss_timeout_s = float(self.config.get("vision_commit_pass_loss_s", 0.50))
                min_commit_elapsed_s = float(self.config.get("vision_commit_min_elapsed_s", 0.35))
                commit_max_time_s = float(self.config.get("vision_commit_max_time_s", 6.0))
                min_peak_area = float(
                    self.config.get(
                        "vision_acquire_area_fraction",
                        self.config.get("vision_commit_area_fraction", 0.08),
                    )
                )
                gate_lost_long_enough = (
                    gate_lost_since_s is not None
                    and (now_s - gate_lost_since_s) >= loss_timeout_s
                )
                if (
                    gate_lost_long_enough
                    and commit_elapsed_s >= min_commit_elapsed_s
                    and self._commit_peak_area_fraction >= min_peak_area
                ):
                    self._transition("PASS_GATE", now_s, "gate_passed_or_lost_post_commit")
                elif commit_elapsed_s >= commit_max_time_s:
                    if self._commit_peak_area_fraction >= min_peak_area:
                        self._transition("PASS_GATE", now_s, "vision_commit_timeout_pass")
                    else:
                        self._transition("RECOVER", now_s, "vision_commit_timeout_recover")
            elif (not gate_fresh and (now_s - self._state_since_s) > 0.15) or (
                est.gate_range_m is not None and est.gate_range_m < 1.0
            ):
                self._transition("PASS_GATE", now_s, "gate_passed_or_lost_post_commit")
        elif self.state == "PASS_GATE":
            if est.active_gate_index > self._target_gate_index:
                self._target_gate_index = est.active_gate_index
                if vision_primary:
                    self._transition("SEARCH_GATE", now_s, "next_gate_search")
                    return self._build_plan()
            if (now_s - self._state_since_s) > float(self.config["pass_timeout_s"]):
                if est.active_gate_index > self._target_gate_index:
                    self._target_gate_index = est.active_gate_index
                    if map_ready:
                        self._transition("ALIGN_GATE", now_s, "pass_confirmed_next_gate")
                    else:
                        self._transition("SEARCH_GATE", now_s, "pass_timeout_complete")
                elif vision_primary and est.active_gate_index <= self._target_gate_index:
                    if gate_usable and gate_strong:
                        self._transition("COMMIT_GATE", now_s, "pass_unconfirmed_recommit")
                    else:
                        self._transition("APPROACH_GATE", now_s, "pass_unconfirmed_reapproach")
                elif (
                    map_ready
                    and est.active_gate_index <= self._target_gate_index
                    and est.map_dist_center_m is not None
                    and est.map_dist_center_m
                    <= float(self.config.get("map_commit_range_m", self.config.get("commit_range_m", 8.0)))
                    and (
                        est.map_plane_signed_m is None
                        or est.map_plane_signed_m
                        < float(self.config.get("map_post_gate_margin_m", 1.5))
                    )
                ):
                    self._transition("COMMIT_GATE", now_s, "pass_unconfirmed_recommit")
                elif map_ready:
                    self._transition("STABILIZE", now_s, "pass_complete_stabilize")
                else:
                    self._transition("SEARCH_GATE", now_s, "pass_timeout_complete")
        elif self.state == "RECOVER":
            recover_align_speed = float(
                self.config.get(
                    "vision_recover_align_max_speed_mps",
                    self.config.get("vision_approach_max_speed_mps", 1.6),
                )
            )
            speed_ok = horizontal_speed <= recover_align_speed
            recover_elapsed_s = now_s - self._state_since_s
            min_recover_dwell_s = float(self.config.get("vision_recover_min_time_s", 0.30))
            if (
                self._stable_frames >= int(self.config["stable_frames_required"])
                and attitude_stable
                and speed_ok
                and vz_ok
                and recover_elapsed_s >= min_recover_dwell_s
            ):
                if vision_primary:
                    if gate_strong:
                        self._transition("ALIGN_GATE", now_s, "recovered_gate_track")
                    else:
                        self._transition("SEARCH_GATE", now_s, "recovered_scan")
                else:
                    self._transition("ALIGN_GATE", now_s, "recovered_gate_track")
            elif (now_s - self._state_since_s) > float(self.config["pass_timeout_s"]):
                if est.race_started and (not attitude_stable or not motion_stable):
                    self._transition("STABILIZE", now_s, "recover_to_stabilize")
                else:
                    self._transition("SEARCH_GATE", now_s, "recover_timeout")

        return self._build_plan()

    def force_pass_gate(self, now_s: float, reason: str) -> RacePlan:
        if self._target_gate_index < 0:
            self._target_gate_index = 0
        self._transition("PASS_GATE", now_s, reason)
        return self._build_plan()

    def _transition(self, new_state: str, now_s: float, reason: str) -> None:
        if new_state == self.state:
            return
        old_state = self.state
        self.state = new_state
        self._state_since_s = now_s
        self._stable_frames = 0
        if new_state == "COMMIT_GATE":
            pass
        elif new_state not in {"ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE"}:
            self._gate_lost_since_s = None
            self._commit_peak_area_fraction = 0.0
        self.last_transition = FSMTransition(
            old_state=old_state,
            new_state=new_state,
            reason=reason,
            timestamp_s=now_s,
        )

    def _map_nav_enabled(self) -> bool:
        return bool(self.config.get("use_map_first_gate", True)) and not bool(
            self.config.get("vision_primary_navigation", False),
        )

    def _build_plan(self) -> RacePlan:
        if self.state == "WAIT_LINK":
            return RacePlan(self.state, 0.0, stabilize_heading=True)
        if self.state == "WAIT_VISION":
            return RacePlan(self.state, 0.0, stabilize_heading=True)
        if self.state == "WAIT_START":
            return RacePlan(self.state, 0.0, stabilize_heading=True)
        if self.state == "EMERGENCY_RECOVERY":
            climb = float(self.config.get("emergency_recovery_climb_rate_mps", 15.0))
            return RacePlan(self.state, 0.0, stabilize_heading=True, climb_rate_mps=climb)
        if self.state == "TAKEOFF":
            return RacePlan(self.state, 0.0, stabilize_heading=True)
        if self.state == "STABILIZE":
            return RacePlan(self.state, 0.0, stabilize_heading=True)
        if self.state == "SEARCH_GATE":
            return RacePlan(self.state, float(self.config["search_speed_mps"]), yaw_scan_rate_rps=float(self.config["search_yaw_rate_rps"]))
        if self.state == "ALIGN_GATE":
            return RacePlan(self.state, float(self.config["align_speed_mps"]), use_map_navigation=self._map_nav_enabled())
        if self.state == "APPROACH_GATE":
            return RacePlan(self.state, float(self.config["approach_speed_mps"]), use_map_navigation=self._map_nav_enabled())
        if self.state == "COMMIT_GATE":
            return RacePlan(
                self.state,
                float(self.config["commit_speed_mps"]),
                commit=True,
                use_map_navigation=self._map_nav_enabled(),
            )
        if self.state == "PASS_GATE":
            return RacePlan(
                self.state,
                float(self.config["commit_speed_mps"]),
                commit=True,
                use_map_navigation=self._map_nav_enabled(),
            )
        return RacePlan(self.state, 0.0, yaw_scan_rate_rps=float(self.config["recover_yaw_rate_rps"]))
