from __future__ import annotations

import math

from src.control.quaternion import euler_to_quaternion_wxyz
from src.types import AttitudeCommand, CommandSaturation, EstimatedState, RacePlan


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class AttitudeController:
    def __init__(self, gains: dict) -> None:
        self.gains = gains
        self._search_yaw_sp = 0.0
        self._search_initialized = False
        self._yaw_hold_sp = 0.0
        self._yaw_hold_initialized = False
        self._hover_pitch_sp = float(gains.get("hover_pitch_rad", 0.0))
        self._commit_yaw_sp = 0.0
        self._commit_initialized = False
        self._speed_cmd_mps = 0.0
        self._last_plan_state = ""
        self._hover_altitude_m: float | None = None
        self._altitude_captured = False
        self._alt_integral = 0.0
        self._takeoff_alt_slew_m: float | None = None
        self._align_yaw_hold_sp = 0.0
        self._align_yaw_hold_initialized = False
        self._vision_base_hover_altitude_m: float | None = None
        self._launch_pitch_captured = False

    def reset_transient(self) -> None:
        self._speed_cmd_mps = 0.0
        self._alt_integral = 0.0
        self._hover_altitude_m = None
        self._altitude_captured = False
        self._takeoff_alt_slew_m = None
        self._last_plan_state = ""
        self.reset_for_next_gate()
        self._launch_pitch_captured = False

    def reset_for_next_gate(self) -> None:
        self._speed_cmd_mps = 0.0
        self._vision_base_hover_altitude_m = None
        self._altitude_captured = False
        self._commit_initialized = False
        self._search_initialized = False
        self._yaw_hold_initialized = False
        self._align_yaw_hold_initialized = False

    def _capture_launch_pitch_trim(self, vehicle) -> None:
        """Sim craft rests nose-high; capture that bias so approach commands nose-down."""
        if not bool(self.gains.get("capture_launch_pitch_trim", True)):
            return
        limit = float(self.gains.get("launch_pitch_trim_limit_rad", 0.55))
        self._hover_pitch_sp = clamp(float(vehicle.pitch_rad), -limit, limit)
        self._launch_pitch_captured = True

    def _collective_baseline(self, vz: float, *, allow_sink_boost: bool = True) -> float:
        thrust = float(self.gains["hover_thrust"]) + float(self.gains["thrust_vz_d"]) * float(vz)
        # NED: vz > 0 means descending — add lift when sinking, not when climbing.
        sink_threshold = float(self.gains.get("sink_boost_vz_threshold", 0.15))
        if allow_sink_boost and vz > sink_threshold:
            thrust += min(
                float(self.gains.get("sink_boost_max", 0.12)),
                float(vz) * float(self.gains.get("sink_boost_gain", 0.08)),
            )
        return thrust

    def _hover_thrust(self, vz: float) -> float:
        return clamp(
            self._collective_baseline(vz),
            float(self.gains["thrust_min"]),
            float(self.gains["thrust_max"]),
        )

    def _altitude_pid_thrust(
        self,
        vehicle,
        vz: float,
        dt: float,
        *,
        kp_key: str,
        ki_key: str,
        allow_overspeed_cut: bool = True,
        allow_sink_boost: bool = True,
        thrust_cap: float | None = None,
    ) -> float:
        alt_error = 0.0
        if self._hover_altitude_m is not None and vehicle.position_ned_m is not None:
            current_alt = -vehicle.position_ned_m[2]
            alt_error = self._hover_altitude_m - current_alt
            if abs(alt_error) >= 1.0:
                self._alt_integral *= 0.92
            i_limit = float(self.gains.get("alt_integral_limit", 0.5))
            self._alt_integral = clamp(self._alt_integral + alt_error * dt, -i_limit, i_limit)
        alt_kp = float(self.gains.get(kp_key, 0.12))
        alt_ki = float(self.gains.get(ki_key, 0.02))
        thrust_raw = (
            self._collective_baseline(float(vz), allow_sink_boost=allow_sink_boost)
            + alt_kp * alt_error
            + alt_ki * self._alt_integral
        )
        max_climb = float(self.gains.get("max_climb_rate_mps", 2.5))
        if allow_overspeed_cut and float(vz) < -max_climb:
            thrust_raw -= float(self.gains.get("climb_overspeed_thrust_gain", 0.12)) * (-float(vz) - max_climb)
        t_min = float(self.gains["thrust_min"])
        t_max = float(self.gains["thrust_max"])
        thrust = clamp(thrust_raw, t_min, t_max)
        if thrust_cap is not None:
            thrust = min(thrust, thrust_cap)
        if thrust != thrust_raw:
            if (thrust >= t_max and alt_error > 0.0) or (thrust <= t_min and alt_error < 0.0):
                self._alt_integral *= 0.9
        return thrust

    def _climb_brake_pitch(self, vz: float, alt_error: float = 0.0) -> float:
        max_climb = float(self.gains.get("max_climb_rate_mps", 3.0))
        if float(vz) >= -max_climb and alt_error >= -0.5:
            return 0.0
        overspeed = max(0.0, -float(vz) - max_climb)
        brake = float(self.gains.get("climb_brake_pitch_gain", 0.04)) * overspeed
        if alt_error < -0.5:
            brake += float(self.gains.get("climb_brake_alt_gain", 0.03)) * min(abs(alt_error), 4.0)
        limit = float(self.gains.get("climb_brake_pitch_limit_rad", 0.25))
        return clamp(brake, 0.0, limit)

    def _vertical_pitch_trim(self, theta_y: float, *, scale: float = 1.0) -> float:
        """Aim at gate height with pitch, not collective thrust (thrust climb = straight up)."""
        kp = float(self.gains.get("vertical_gate_pitch_kp", 0.35)) * scale
        limit = float(self.gains.get("vertical_gate_pitch_limit_rad", 0.10)) * scale
        # Higher pitch setpoint = nose-down in this sim's rate loop. Gate below image
        # center (theta_y > 0) needs more nose-down; gate above needs less.
        return clamp(kp * theta_y, -limit, limit)

    def _vision_lower_gate_pitch_boost(
        self,
        est: EstimatedState,
        plan: RacePlan,
        *,
        using_map: bool,
    ) -> float:
        if using_map or not self._vision_primary_enabled():
            return 0.0
        if plan.state not in {"ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE", "PASS_GATE"} and not plan.commit:
            return 0.0
        if est.gate_confidence < float(self.gains["min_gate_confidence"]):
            return 0.0
        gate_below_rad = float(self.gains.get("vision_gate_below_bearing_rad", 0.08))
        theta_y = float(est.gate_bearing_y_rad)
        if theta_y <= gate_below_rad:
            return 0.0
        extra_bearing = theta_y - gate_below_rad
        boost = extra_bearing * float(
            self.gains.get("vision_lower_gate_pitch_boost_per_rad", 0.55)
        )
        if plan.state == "ALIGN_GATE":
            boost *= float(self.gains.get("vision_lower_gate_align_pitch_boost_scale", 1.20))
        proximity = self._vision_approach_proximity(est)
        taper = clamp(
            1.0
            - proximity
            * float(self.gains.get("vision_lower_gate_pitch_boost_close_taper", 0.0)),
            float(self.gains.get("vision_lower_gate_pitch_boost_close_floor", 0.25)),
            1.0,
        )
        boost *= taper
        return min(
            boost,
            float(self.gains.get("vision_lower_gate_pitch_boost_max_rad", 0.12)),
        )

    def _clamp_gate_pitch_sp(
        self,
        pitch_sp: float,
        pitch_now: float,
        plan: RacePlan,
        *,
        horizontal_speed: float = 0.0,
        using_map: bool,
        gate_theta_y: float = 0.0,
    ) -> float:
        base = self._hover_pitch_sp
        if self._vision_primary_enabled() and not using_map and plan.state in {
            "SEARCH_GATE",
            "ALIGN_GATE",
            "APPROACH_GATE",
            "COMMIT_GATE",
            "PASS_GATE",
        }:
            up_limit = float(self.gains.get("vision_pitch_up_limit_rad", 0.06))
            brake_speed = float(self.gains.get("vision_align_speed_brake_mps", 1.8))
            if plan.state in {"SEARCH_GATE", "RECOVER"} or horizontal_speed > brake_speed:
                up_limit = max(
                    up_limit,
                    float(
                        self.gains.get(
                            "vision_brake_pitch_up_limit_rad",
                            self.gains.get("search_brake_pitch_limit_rad", 0.24),
                        )
                    ),
                )
            down_limit = float(self.gains.get("vision_approach_pitch_limit_rad", 0.38))
            if plan.state == "ALIGN_GATE":
                down_limit = float(self.gains.get("vision_align_pitch_limit_rad", 0.22))
            lo = base - up_limit
            hi = base + down_limit
            gate_below_rad = float(self.gains.get("vision_gate_below_bearing_rad", 0.08))
            if (
                gate_theta_y >= gate_below_rad
                and (
                    plan.state in {"ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE", "PASS_GATE"}
                    or plan.commit
                )
            ):
                extra_bearing = gate_theta_y - gate_below_rad
                hi += min(
                    float(self.gains.get("vision_lower_gate_extra_pitch_limit_rad", 0.10)),
                    extra_bearing
                    * float(self.gains.get("vision_lower_gate_extra_pitch_limit_per_rad", 0.90)),
                )
                min_delta = float(self.gains.get("vision_min_nose_down_pitch_delta_rad", 0.05))
                lo = max(lo, pitch_now + min_delta)
            return clamp(pitch_sp, lo, hi)
        if plan.state in {"ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE", "PASS_GATE"} or plan.commit:
            limit = self._vision_approach_pitch_limit_rad(plan, using_map)
            return clamp(pitch_sp, base - limit, base + limit)
        limit = float(self.gains.get("pitch_limit_rad", 0.80))
        return clamp(pitch_sp, base - limit, base + limit)

    def _vision_vertical_pitch_scale(self, plan: RacePlan | None) -> float:
        if plan is None or not self._vision_primary_enabled():
            return 1.0
        if plan.state in {"APPROACH_GATE", "COMMIT_GATE", "PASS_GATE"} or plan.commit:
            return float(self.gains.get("vision_approach_vertical_pitch_scale", 1.0))
        if plan.state == "ALIGN_GATE":
            return float(self.gains.get("vision_align_vertical_pitch_scale", 0.75))
        return float(self.gains.get("vision_search_vertical_pitch_scale", 0.45))

    def _vision_approach_proximity(self, est: EstimatedState) -> float:
        if est.gate_range_m is not None:
            close_m = float(self.gains.get("vision_approach_close_range_m", 4.0))
            far_m = float(self.gains.get("vision_approach_far_range_m", 12.0))
            span = max(far_m - close_m, 1.0)
            return clamp(1.0 - (float(est.gate_range_m) - close_m) / span, 0.0, 1.0)
        if (
            est.gate_track is not None
            and est.gate_confidence >= float(self.gains["min_gate_confidence"])
        ):
            area_ref = float(self.gains.get("vision_commit_area_fraction", 0.08))
            return clamp(float(est.gate_track.area_fraction) / max(area_ref, 1e-3), 0.0, 1.0)
        return 0.0

    def _vision_gate_center_alt_offset_m(self, est: EstimatedState) -> float:
        """Lower hover target when the gate sits below the image aim point."""
        kp = float(self.gains.get("vision_gate_center_alt_kp_m", 1.8))
        limit = float(self.gains.get("vision_gate_center_alt_limit_m", 1.0))
        theta_y = float(est.gate_bearing_y_rad)
        depth_gain = 1.0
        if est.gate_range_m is not None:
            close_m = float(self.gains.get("vision_approach_close_range_m", 4.0))
            raw_depth_gain = 1.0 + (close_m - float(est.gate_range_m)) / max(close_m, 1.0)
            if theta_y >= 0.0:
                depth_gain = clamp(raw_depth_gain, 1.0, 1.8)
            else:
                depth_gain = clamp(raw_depth_gain, 0.8, 1.6)
        offset_m = kp * theta_y * depth_gain
        gate_below_rad = float(self.gains.get("vision_gate_below_bearing_rad", 0.08))
        if theta_y > gate_below_rad:
            offset_m += float(self.gains.get("vision_lower_gate_extra_alt_gain", 0.9)) * (
                theta_y - gate_below_rad
            )
        return clamp(offset_m, -limit, limit)

    def _vision_lower_gate_extra_drop_m(self, est: EstimatedState, plan: RacePlan) -> float:
        if plan.state not in {"ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE", "PASS_GATE"} and not plan.commit:
            return 0.0
        if est.gate_confidence < float(self.gains["min_gate_confidence"]):
            return 0.0
        gate_below_rad = float(self.gains.get("vision_gate_below_bearing_rad", 0.08))
        theta_y = float(est.gate_bearing_y_rad)
        if theta_y <= gate_below_rad:
            return 0.0
        extra_bearing = theta_y - gate_below_rad
        extra_drop = extra_bearing * float(
            self.gains.get("vision_lower_gate_extra_drop_per_rad_m", 1.8)
        )
        if plan.state == "ALIGN_GATE":
            extra_drop *= float(self.gains.get("vision_lower_gate_align_drop_scale", 1.15))
        return min(
            extra_drop,
            float(self.gains.get("vision_lower_gate_extra_drop_max_m", 0.45)),
        )

    def _vision_center_bearing_limits(self, plan: RacePlan) -> tuple[float, float]:
        bx = float(self.gains.get("vision_center_bearing_rad", 0.06))
        by = float(self.gains.get("vision_center_vertical_rad", 0.08))
        if plan.state in {"APPROACH_GATE", "COMMIT_GATE", "PASS_GATE"} or plan.commit:
            tighten = float(self.gains.get("vision_approach_center_tighten", 0.55))
            bx *= tighten
            by *= tighten
        return bx, by

    def _vision_centered_enough(self, theta_x: float, theta_y: float, plan: RacePlan) -> bool:
        bx, by = self._vision_center_bearing_limits(plan)
        return abs(theta_x) <= bx and abs(theta_y) <= by

    def _vision_straight_line_roll(self, body_vy: float, plan: RacePlan) -> float:
        if not self._vision_primary_enabled():
            return 0.0
        if plan.state not in {"APPROACH_GATE", "COMMIT_GATE", "PASS_GATE"} and not plan.commit:
            return 0.0
        kp = float(self.gains.get("vision_straight_lateral_vy_kp", 0.55))
        limit = float(self.gains.get("vision_straight_roll_limit_rad", 0.14))
        return clamp(-kp * body_vy, -limit, limit)

    def _vision_center_fixation_gain(self, plan: RacePlan, theta_x: float, theta_y: float) -> float:
        if not self._vision_primary_enabled():
            return 1.0
        if plan.state not in {"ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE", "PASS_GATE"} and not plan.commit:
            return 1.0
        base = float(self.gains.get("vision_center_fixation_gain", 1.8))
        if self._vision_centered_enough(theta_x, theta_y, plan):
            return base * float(self.gains.get("vision_center_locked_gain", 1.25))
        return base

    def _vision_hover_altitude_target(self, est: EstimatedState, plan: RacePlan, base_alt: float) -> float:
        if plan.state not in {"ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE", "PASS_GATE"} and not plan.commit:
            return base_alt
        floor_alt = float(self.gains.get("vision_approach_min_alt_m", 0.8))
        if plan.state == "ALIGN_GATE":
            align_alt = self.gains.get("vision_align_hover_altitude_m")
            if align_alt is not None:
                target = min(base_alt, float(align_alt))
                if est.gate_confidence >= float(self.gains["min_gate_confidence"]):
                    target -= self._vision_gate_center_alt_offset_m(est)
                    target -= self._vision_lower_gate_extra_drop_m(est, plan)
                return max(floor_alt, target)
        working_base = base_alt
        if plan.state == "ALIGN_GATE":
            align_frac = float(self.gains.get("vision_align_alt_drop_fraction", 0.10))
            working_base = base_alt * (1.0 - align_frac)
        proximity = self._vision_approach_proximity(est)
        if (
            plan.state == "ALIGN_GATE"
            and est.gate_confidence >= float(self.gains["min_gate_confidence"])
        ):
            proximity = max(
                proximity,
                float(self.gains.get("vision_align_proximity_floor", 0.40)),
            )
        max_drop = float(self.gains.get("vision_approach_alt_drop_m", 0.5))
        drop = max_drop * proximity
        if plan.commit or plan.state in {"COMMIT_GATE", "PASS_GATE"}:
            drop = max(drop, max_drop * float(self.gains.get("vision_commit_alt_drop_fraction", 0.85)))
        center_offset = 0.0
        if est.gate_confidence >= float(self.gains["min_gate_confidence"]):
            center_offset = self._vision_gate_center_alt_offset_m(est)
        lower_gate_extra_drop = self._vision_lower_gate_extra_drop_m(est, plan)
        return max(floor_alt, working_base - drop - center_offset - lower_gate_extra_drop)

    def _update_vision_hover_altitude(self, est: EstimatedState, dt: float, plan: RacePlan) -> None:
        if self._hover_altitude_m is None:
            return
        if self._vision_base_hover_altitude_m is None:
            self._vision_base_hover_altitude_m = self._hover_altitude_m
        target = self._vision_hover_altitude_target(
            est,
            plan,
            self._vision_base_hover_altitude_m,
        )
        if plan.state == "ALIGN_GATE":
            max_rate = float(self.gains.get("vision_align_alt_slew_rate_mps", 4.0))
        else:
            max_rate = float(self.gains.get("vision_gate_alt_slew_rate_mps", 2.0))
        gate_below_rad = float(self.gains.get("vision_gate_below_bearing_rad", 0.08))
        if (
            est.gate_confidence >= float(self.gains["min_gate_confidence"])
            and float(est.gate_bearing_y_rad) > gate_below_rad
            and target < self._hover_altitude_m
        ):
            max_rate += min(
                float(self.gains.get("vision_lower_gate_extra_alt_slew_max_mps", 2.5)),
                (float(est.gate_bearing_y_rad) - gate_below_rad)
                * float(self.gains.get("vision_lower_gate_extra_alt_slew_per_rad_mps", 12.0)),
            )
        delta = clamp(target - self._hover_altitude_m, -max_rate * dt, max_rate * dt)
        self._hover_altitude_m += delta

    def _map_vertical_pitch_trim(self, est: EstimatedState) -> float:
        if est.map_vertical_error_m is None:
            return 0.0
        kp = float(self.gains.get("map_vertical_error_pitch_kp", 0.18))
        limit = float(self.gains.get("vertical_gate_pitch_limit_rad", 0.10))
        # Positive map_vertical_error_m = drone below gate plane in NED (need to climb).
        return clamp(-kp * float(est.map_vertical_error_m), -limit, limit)

    def _map_before_gate(self, est: EstimatedState, margin_m: float) -> bool:
        return est.map_plane_signed_m is None or float(est.map_plane_signed_m) < margin_m

    def _capture_hover_altitude(self, vehicle) -> None:
        if self._hover_altitude_m is None and vehicle.position_ned_m is not None:
            self._hover_altitude_m = -vehicle.position_ned_m[2]
            self._alt_integral = 0.0

    def _vision_search_body_damping(self, body_vx: float, body_vy: float, *, aggressive: bool = False) -> tuple[float, float]:
        speed = math.hypot(body_vx, body_vy)
        if aggressive:
            gain_scale = clamp(speed / 1.2, 3.0, 6.0)
            roll_limit = float(self.gains.get("search_brake_roll_limit_rad", 0.30))
            pitch_limit = float(self.gains.get("search_brake_pitch_limit_rad", 0.24))
        else:
            gain_scale = clamp(speed / 2.0, 1.0, 2.8)
            roll_limit = float(self.gains.get("search_roll_limit_rad", 0.18))
            pitch_limit = float(self.gains.get("search_pitch_limit_rad", 0.14))
        roll_sp = clamp(
            -float(self.gains.get("search_vy_kp", self.gains.get("stabilize_vy_kp", 0.12))) * body_vy * gain_scale,
            -roll_limit,
            roll_limit,
        )
        pitch_sp = clamp(
            self._hover_pitch_sp
            + float(self.gains.get("search_pitch_trim_rad", 0.04))
            - float(self.gains.get("search_vx_kp", self.gains.get("stabilize_vx_kp", 0.10))) * body_vx * gain_scale,
            self._hover_pitch_sp - pitch_limit,
            self._hover_pitch_sp + pitch_limit,
        )
        return roll_sp, pitch_sp

    def _vision_search_command(
        self,
        vehicle,
        est: EstimatedState,
        plan: RacePlan,
        dt: float,
        *,
        yaw_now: float,
        vz: float,
        body_vx: float,
        body_vy: float,
        steer: bool,
    ) -> AttitudeCommand:
        self._capture_hover_altitude(vehicle)
        horizontal_speed = math.hypot(body_vx, body_vy)
        scan_max_speed = float(self.gains.get("search_scan_max_speed_mps", 1.0))
        brake_speed = float(self.gains.get("search_brake_speed_mps", 2.8))
        braking = horizontal_speed > scan_max_speed
        roll_sp, pitch_sp = self._vision_search_body_damping(
            body_vx,
            body_vy,
            aggressive=braking or horizontal_speed > brake_speed,
        )
        if steer and not braking:
            steer_roll_limit = float(self.gains.get("vision_search_steer_roll_limit_rad", 0.10))
            roll_sp = clamp(roll_sp, -steer_roll_limit, steer_roll_limit)
            yaw_delta = clamp(
                float(self.gains.get("search_yaw_kp", self.gains["yaw_kp"])) * est.gate_bearing_x_rad
                - float(self.gains["yaw_rate_d"]) * vehicle.yaw_rate_rps,
                -float(self.gains.get("search_yaw_step_limit_rad", 0.18)),
                float(self.gains.get("search_yaw_step_limit_rad", 0.18)),
            )
            yaw_sp = wrap_pi(yaw_now + yaw_delta)
            gate_below = self._vision_gate_below(est)
            up_limit = float(self.gains.get("search_pitch_limit_rad", 0.14))
            down_limit = float(self.gains.get("search_pitch_limit_rad", 0.14))
            if gate_below:
                down_limit = float(self.gains.get("vision_align_pitch_limit_rad", 0.40))
            pitch_sp = clamp(
                pitch_sp
                + self._vertical_pitch_trim(
                    est.gate_bearing_y_rad,
                    scale=self._vision_vertical_pitch_scale(plan),
                ),
                self._hover_pitch_sp - up_limit,
                self._hover_pitch_sp + down_limit,
            )
            align = self._vision_forward_alignment(est.gate_bearing_x_rad)
            forward_nudge = float(self.gains.get("vision_search_forward_pitch_rad", 0.12)) * align
            pitch_sp = clamp(
                pitch_sp + forward_nudge,
                self._hover_pitch_sp - up_limit,
                self._hover_pitch_sp + down_limit,
            )
        elif steer and braking:
            yaw_delta = clamp(
                0.35 * float(self.gains.get("search_yaw_kp", self.gains["yaw_kp"])) * est.gate_bearing_x_rad,
                -float(self.gains.get("search_yaw_step_limit_rad", 0.18)),
                float(self.gains.get("search_yaw_step_limit_rad", 0.18)),
            )
            yaw_sp = wrap_pi(yaw_now + yaw_delta)
        elif braking:
            if not self._search_initialized:
                self._search_yaw_sp = yaw_now
                self._search_initialized = True
            yaw_sp = yaw_now
        else:
            if not self._search_initialized:
                self._search_yaw_sp = yaw_now
                self._search_initialized = True
            scan_rps = float(self.gains.get("search_yaw_scan_rps", plan.yaw_scan_rate_rps))
            if float(vz) < -0.20:
                scan_rps *= 0.35
            if horizontal_speed > 1.5:
                scan_rps *= max(0.10, 1.0 - (horizontal_speed - 1.5) / 3.0)
            self._search_yaw_sp = wrap_pi(self._search_yaw_sp + scan_rps * dt)
            yaw_sp = self._search_yaw_sp
        thrust = self._vision_altitude_hold_thrust(
            vehicle, float(vz), dt, est, plan, using_map=False
        )
        return self._build_command(vehicle, roll_sp, pitch_sp, yaw_sp, thrust)

    def _gate_pass_pitch_trim(self, est: EstimatedState, plan, using_map: bool) -> float:
        """Small nose-down pitch to slip through gate center — only while still before the plane."""
        if not using_map:
            return 0.0
        trim = float(self.gains.get("map_gate_pass_pitch_trim_rad", 0.0))
        if trim <= 0.0:
            return 0.0
        margin = float(self.gains.get("map_gate_pass_pitch_plane_margin_m", 0.5))
        if not self._map_before_gate(est, margin):
            return 0.0
        if plan.commit or plan.state == "COMMIT_GATE":
            return trim
        dist = est.map_dist_center_m
        if dist is None:
            return 0.0
        drop_dist = float(self.gains.get("map_gate_pass_pitch_dist_m", 4.0))
        if float(dist) >= drop_dist:
            return 0.0
        proximity = 1.0 - float(dist) / drop_dist
        return trim * clamp(proximity, 0.0, 1.0)

    def _backward_drift_pitch_brake(self, body_vx: float, est: EstimatedState, using_map: bool) -> float:
        """Counter backward body-frame drift with a small nose-down / forward pitch bump."""
        if body_vx >= -0.35:
            return 0.0
        if using_map and est.map_plane_signed_m is not None and float(est.map_plane_signed_m) > 2.0:
            return 0.0
        kp = float(self.gains.get("backward_drift_pitch_kp", 0.08))
        limit = float(self.gains.get("backward_drift_pitch_limit_rad", 0.12))
        return clamp(-kp * body_vx, 0.0, limit)

    def _approach_altitude_drop_m(self, est: EstimatedState, plan: RacePlan | None) -> float:
        """Lower hover target on final approach — only before gate plane, never in ALIGN."""
        if plan is None:
            return 0.0
        max_drop = float(self.gains.get("map_gate_pass_alt_drop_m", 0.0))
        if max_drop <= 0.0:
            return 0.0
        if plan.state not in {"APPROACH_GATE", "COMMIT_GATE"} and not plan.commit:
            return 0.0
        margin = float(self.gains.get("map_gate_pass_alt_drop_plane_margin_m", 0.5))
        if not self._map_before_gate(est, margin):
            return 0.0
        if plan.commit or plan.state == "COMMIT_GATE":
            return max_drop
        dist = est.map_dist_center_m
        if dist is None:
            return 0.0
        drop_dist = float(self.gains.get("map_gate_pass_alt_drop_dist_m", 6.0))
        if float(dist) >= drop_dist:
            return 0.0
        proximity = 1.0 - float(dist) / drop_dist
        return max_drop * clamp(proximity, 0.0, 1.0)

    def _update_hover_altitude_for_gate(
        self,
        est: EstimatedState,
        dt: float,
        plan: RacePlan | None = None,
    ) -> None:
        if plan is not None and self._vision_primary_enabled():
            self._update_vision_hover_altitude(est, dt, plan)
            return
        if est.map_gate_altitude_m is None:
            return
        target_alt = float(est.map_gate_altitude_m) - self._approach_altitude_drop_m(est, plan)
        if self._hover_altitude_m is None:
            self._hover_altitude_m = target_alt
            return
        max_rate = float(self.gains.get("map_gate_alt_slew_rate_mps", 2.5))
        delta = clamp(target_alt - self._hover_altitude_m, -max_rate * dt, max_rate * dt)
        self._hover_altitude_m += delta

    def _approach_forward_scale(self, est: EstimatedState, using_map: bool) -> float:
        """Hold forward motion until altitude and lateral error are small enough."""
        if using_map:
            scale = 1.0
            if est.map_vertical_error_m is not None:
                vert_err = abs(float(est.map_vertical_error_m))
                vert_ok = float(self.gains.get("map_vertical_align_before_forward_m", 0.6))
                if vert_err > vert_ok:
                    scale = min(scale, max(0.05, 1.0 - (vert_err - vert_ok) / 2.5))
            if est.map_lateral_error_m is not None:
                lat_err = abs(float(est.map_lateral_error_m))
                lat_ok = float(self.gains.get("map_lateral_align_before_forward_m", 0.8))
                if lat_err > lat_ok:
                    scale = min(scale, max(0.05, 1.0 - (lat_err - lat_ok) / 2.0))
            if (
                est.map_dist_center_m is not None
                and est.map_within_gate_bounds is False
            ):
                dist = float(est.map_dist_center_m)
                slow_radius = float(self.gains.get("map_collision_slow_radius_m", 6.0))
                if dist < slow_radius:
                    proximity = 1.0 - dist / slow_radius
                    oob_scale = 1.0 - proximity * float(
                        self.gains.get("map_collision_oob_slow_gain", 0.85),
                    )
                    floor = float(self.gains.get("map_collision_min_forward_scale", 0.15))
                    scale = min(scale, max(floor, oob_scale))
            return scale
        if abs(est.gate_bearing_y_rad) > float(self.gains.get("vision_vertical_align_rad", 0.10)):
            if float(est.gate_bearing_y_rad) > 0.0:
                extra_bearing = float(est.gate_bearing_y_rad) - float(
                    self.gains.get("vision_vertical_align_rad", 0.10)
                )
                floor = float(self.gains.get("vision_lower_gate_forward_scale_min", 0.25))
                gain = float(self.gains.get("vision_lower_gate_forward_scale_per_rad", 2.2))
                return max(floor, 1.0 - extra_bearing * gain)
            return max(0.05, 1.0 - abs(est.gate_bearing_y_rad) / 0.35)
        if abs(est.gate_bearing_x_rad) > float(self.gains.get("vision_center_bearing_rad", 0.06)):
            return max(0.10, 1.0 - abs(est.gate_bearing_x_rad) / 0.30)
        return 1.0

    def _vision_primary_enabled(self) -> bool:
        return bool(self.gains.get("vision_primary_navigation", False))

    def _vision_forward_alignment(self, theta_x: float) -> float:
        gate_rad = float(self.gains.get("vision_align_bearing_rad", 0.24))
        return 1.0 - min(abs(theta_x) / max(gate_rad, 0.08), 1.0)

    def _vision_horizontal_speed_mps(self, est: EstimatedState) -> float:
        velocity = est.vehicle.velocity_ned_mps
        if velocity is None:
            return 0.0
        return float(math.hypot(float(velocity[0]), float(velocity[1])))

    def _vision_yaw_step_limit_rad(self, horizontal_speed: float) -> float:
        base = min(
            float(self.gains.get("yaw_step_limit_rad", 0.30)),
            float(self.gains.get("vision_yaw_step_limit_rad", 0.14)),
        )
        speed_ref = float(self.gains.get("vision_yaw_speed_ref_mps", 1.4))
        scale = clamp(1.0 - 0.70 * (horizontal_speed / max(speed_ref, 0.5)), 0.18, 1.0)
        return base * scale

    def _vision_roll_yaw_theta_split(
        self,
        theta_x: float,
        horizontal_speed: float,
    ) -> tuple[float, float]:
        """Roll centers the gate in the image; yaw only assists coarse alignment."""
        speed_ref = float(self.gains.get("vision_roll_yaw_speed_ref_mps", 1.0))
        roll_share = clamp(0.60 + 0.32 * (horizontal_speed / max(speed_ref, 0.5)), 0.60, 0.94)
        coarse = float(self.gains.get("vision_yaw_coarse_bearing_rad", 0.16))
        if abs(theta_x) > coarse:
            roll_share = min(roll_share, 0.72)
        yaw_share = 1.0 - roll_share
        return roll_share * theta_x, yaw_share * theta_x

    def _vision_track_gate_yaw(
        self,
        vehicle,
        theta_x: float,
        horizontal_speed: float,
        *,
        gain_scale: float = 1.0,
    ) -> float:
        _, yaw_theta = self._vision_roll_yaw_theta_split(theta_x, horizontal_speed)
        yaw_kp = float(self.gains["yaw_kp"]) * gain_scale
        step = self._vision_yaw_step_limit_rad(horizontal_speed)
        yaw_delta = clamp(
            yaw_kp * yaw_theta - float(self.gains["yaw_rate_d"]) * vehicle.yaw_rate_rps,
            -step,
            step,
        )
        return wrap_pi(vehicle.yaw_rad + yaw_delta)

    def _lower_gate_extra_bearing_rad(self, est: EstimatedState) -> float:
        gate_below_rad = float(self.gains.get("vision_gate_below_bearing_rad", 0.08))
        return max(0.0, float(est.gate_bearing_y_rad) - gate_below_rad)

    def _vision_lower_gate_forward_pitch_cap_rad(
        self,
        est: EstimatedState,
        plan: RacePlan,
        *,
        using_map: bool,
    ) -> float | None:
        if using_map or not self._vision_primary_enabled():
            return None
        if plan.state not in {"ALIGN_GATE", "APPROACH_GATE"}:
            return None
        if est.gate_confidence < float(self.gains["min_gate_confidence"]):
            return None
        extra_bearing = self._lower_gate_extra_bearing_rad(est)
        if extra_bearing <= 0.0:
            return None
        base_cap = float(self.gains.get("vision_lower_gate_forward_pitch_cap_rad", 0.09))
        per_rad = float(self.gains.get("vision_lower_gate_forward_pitch_cap_per_rad", 0.25))
        floor = float(self.gains.get("vision_lower_gate_forward_pitch_cap_min_rad", 0.02))
        cap = max(floor, base_cap - extra_bearing * per_rad)
        proximity = self._vision_approach_proximity(est)
        cap = max(
            floor,
            cap
            - proximity
            * float(self.gains.get("vision_lower_gate_forward_pitch_cap_near_delta_rad", 0.0)),
        )
        if plan.state == "ALIGN_GATE":
            cap = min(
                cap,
                float(
                    self.gains.get(
                        "vision_lower_gate_align_forward_pitch_cap_rad",
                        base_cap,
                    )
                ),
            )
        return cap

    def _vision_forward_scale(self, est: EstimatedState, plan: RacePlan, using_map: bool) -> float:
        if using_map or not self._vision_primary_enabled():
            return self._approach_forward_scale(est, using_map)
        vert_scale = self._approach_forward_scale(est, using_map)
        align = self._vision_forward_alignment(est.gate_bearing_x_rad)
        if plan.state == "SEARCH_GATE":
            scale = float(self.gains.get("vision_search_forward_scale", 0.20)) * align * vert_scale
        elif plan.state == "ALIGN_GATE":
            base = float(self.gains.get("vision_align_forward_scale", 0.35))
            scale = base * (0.4 + 0.6 * align) * vert_scale
        else:
            scale = vert_scale
        extra_bearing = self._lower_gate_extra_bearing_rad(est)
        if extra_bearing > 0.0:
            lower_gate_floor = float(
                self.gains.get(
                    "vision_lower_gate_align_forward_scale_min",
                    self.gains.get("vision_lower_gate_forward_scale_min", 0.25),
                )
            )
            lower_gate_gain = float(
                self.gains.get("vision_lower_gate_align_forward_scale_per_rad", 2.8)
            )
            scale = min(scale, max(lower_gate_floor, 1.0 - extra_bearing * lower_gate_gain))
        hs = self._vision_horizontal_speed_mps(est)
        speed_cap = float(self.gains.get("vision_forward_max_speed_mps", 1.8))
        if hs > speed_cap:
            scale *= max(0.10, 1.0 - (hs - speed_cap) / 2.5)
        return scale

    def _vision_lower_gate_speed_cap_mps(
        self,
        est: EstimatedState,
        plan: RacePlan,
        *,
        using_map: bool,
    ) -> float | None:
        if using_map or not self._vision_primary_enabled():
            return None
        if plan.state not in {"SEARCH_GATE", "ALIGN_GATE", "APPROACH_GATE"}:
            return None
        extra_bearing = self._lower_gate_extra_bearing_rad(est)
        if extra_bearing <= 0.0:
            return None
        base_cap = float(self.gains.get("vision_lower_gate_speed_cap_mps", 0.55))
        gain = float(self.gains.get("vision_lower_gate_speed_cap_per_rad", 0.9))
        floor = float(self.gains.get("vision_lower_gate_speed_cap_min_mps", 0.12))
        cap = max(floor, base_cap - extra_bearing * gain)
        if est.gate_range_m is not None:
            far_m = float(self.gains.get("vision_lower_gate_speed_cap_far_range_m", 20.0))
            near_m = float(self.gains.get("vision_approach_close_range_m", 4.0))
            proximity = clamp(
                1.0 - (float(est.gate_range_m) - near_m) / max(far_m - near_m, 1.0),
                0.0,
                1.0,
            )
            cap = min(
                cap,
                base_cap - proximity * float(self.gains.get("vision_lower_gate_speed_cap_near_delta_mps", 0.20)),
            )
            cap = max(floor, cap)
        return cap

    def _vision_approach_pitch_limit_rad(self, plan: RacePlan, using_map: bool) -> float:
        if using_map or not self._vision_primary_enabled():
            return float(
                self.gains.get("align_pitch_limit_rad", self.gains.get("stabilize_pitch_limit_rad", 0.18))
            )
        if plan.state in {"APPROACH_GATE", "COMMIT_GATE", "PASS_GATE"}:
            return float(self.gains.get("vision_approach_pitch_limit_rad", 0.38))
        if plan.state == "ALIGN_GATE":
            return float(self.gains.get("vision_align_pitch_limit_rad", 0.22))
        return float(self.gains.get("search_pitch_limit_rad", 0.12))

    def _vision_speed_target_mps(self, plan: RacePlan, using_map: bool) -> float | None:
        if using_map or not self._vision_primary_enabled():
            return None
        if plan.state == "SEARCH_GATE":
            return float(self.gains.get("vision_search_speed_mps", 0.6))
        if plan.state == "ALIGN_GATE":
            return float(self.gains.get("vision_align_speed_mps", 0.9))
        return None

    def _gate_speed_target_mps(self, est: EstimatedState, plan, using_map: bool) -> float:
        target_speed = float(plan.forward_speed_mps)
        vision_speed = self._vision_speed_target_mps(plan, using_map)
        if vision_speed is not None:
            target_speed = max(target_speed, vision_speed)
        lower_gate_speed_cap = self._vision_lower_gate_speed_cap_mps(
            est,
            plan,
            using_map=using_map,
        )
        if lower_gate_speed_cap is not None:
            target_speed = min(target_speed, lower_gate_speed_cap)
        if (
            using_map
            and (plan.commit or plan.state in {"COMMIT_GATE", "PASS_GATE"})
            and est.map_commit_speed_mps is not None
        ):
            target_speed = max(target_speed, float(est.map_commit_speed_mps))
            target_speed = min(
                target_speed,
                float(self.gains.get("map_commit_speed_cap_mps", target_speed)),
            )
            if est.map_within_gate_bounds is False:
                target_speed = min(
                    target_speed,
                    float(self.gains.get("map_commit_speed_cap_oob_mps", target_speed)),
                )
        if not using_map or est.map_dist_center_m is None or est.map_within_gate_bounds:
            return target_speed
        dist = float(est.map_dist_center_m)
        slow_radius = float(self.gains.get("map_collision_slow_radius_m", 6.0))
        if dist >= slow_radius:
            return target_speed
        oob_cap = float(self.gains.get("map_approach_speed_cap_oob_mps", 1.8))
        proximity = 1.0 - dist / slow_radius
        blend = clamp(proximity, 0.0, 1.0)
        return min(target_speed, oob_cap + (1.0 - blend) * (target_speed - oob_cap))

    def _gate_vertical_theta_y(self, est: EstimatedState, using_map: bool, theta_y: float) -> float:
        if using_map and est.map_gate_bearing_y_rad is not None:
            return float(est.map_gate_bearing_y_rad)
        return theta_y

    def _current_alt_error(self, vehicle) -> float:
        if self._hover_altitude_m is None or vehicle.position_ned_m is None:
            return 0.0
        return self._hover_altitude_m - (-vehicle.position_ned_m[2])

    def _forward_pitch(self, est, plan, using_map: bool, body_vx: float) -> float:
        """Compute forward pitch for gate approach - up to 45 deg when aligned."""
        theta_x = est.gate_bearing_x_rad
        pitch_speed_kp = float(self.gains["pitch_speed_kp"]) * (float(self.gains.get("map_pitch_gain_scale", 1.0)) if using_map else 1.0)
        base_pitch = float(self.gains["pitch_trim_rad"]) + pitch_speed_kp * (self._speed_cmd_mps - body_vx)
        forward_pitch = 0.0
        if est.gate_track is not None and est.gate_confidence >= float(self.gains["min_gate_confidence"]):
            alignment = 1.0 - min(abs(theta_x) / 0.3, 1.0)
            forward_kp = float(self.gains.get("forward_flight_pitch_kp", 0.50))
            forward_limit = float(self.gains.get("forward_flight_pitch_limit_rad", 0.70))
            forward_pitch = clamp(forward_kp * alignment * 0.7, -forward_limit, forward_limit)
        if (
            using_map
            and est.map_within_gate_bounds is False
            and est.map_dist_center_m is not None
            and float(est.map_dist_center_m)
            < float(self.gains.get("map_collision_slow_radius_m", 6.0))
        ):
            forward_pitch *= float(self.gains.get("map_collision_forward_pitch_scale", 0.35))
        command_pitch = base_pitch + forward_pitch
        lower_gate_cap = self._vision_lower_gate_forward_pitch_cap_rad(
            est,
            plan,
            using_map=using_map,
        )
        if lower_gate_cap is not None:
            command_pitch = min(command_pitch, self._hover_pitch_sp + lower_gate_cap)
        return command_pitch

    def _gate_pitch_sp(self, vehicle, vz: float, est, plan, using_map: bool, body_vx: float, theta_y: float) -> float:
        vert_theta_y = self._gate_vertical_theta_y(est, using_map, theta_y)
        forward_scale = self._vision_forward_scale(est, plan, using_map)
        desired = self._forward_pitch(est, plan, using_map, body_vx) * forward_scale
        vert_scale = self._vision_vertical_pitch_scale(plan) if not using_map else 1.0
        vertical_trim = self._vertical_pitch_trim(vert_theta_y, scale=vert_scale)
        vertical_trim += self._vision_lower_gate_pitch_boost(est, plan, using_map=using_map)
        if using_map:
            vertical_trim += self._map_vertical_pitch_trim(est)
        return clamp(
            desired
            + vertical_trim
            + self._gate_pass_pitch_trim(est, plan, using_map)
            + self._backward_drift_pitch_brake(body_vx, est, using_map)
            + self._climb_brake_pitch(float(vz), self._current_alt_error(vehicle)),
            self._hover_pitch_sp - float(self.gains["pitch_limit_rad"]),
            self._hover_pitch_sp + float(self.gains["pitch_limit_rad"]),
        )

    def _vision_gate_below(self, est: EstimatedState) -> bool:
        return float(est.gate_bearing_y_rad) >= float(
            self.gains.get("vision_gate_below_bearing_rad", 0.08)
        )

    def _vision_altitude_hold_thrust(
        self,
        vehicle,
        vz: float,
        dt: float,
        est: EstimatedState,
        plan: RacePlan,
        *,
        using_map: bool,
    ) -> float:
        allow_sink_boost = True
        thrust_cap: float | None = None
        if (
            self._vision_primary_enabled()
            and not using_map
            and est.gate_confidence >= float(self.gains["min_gate_confidence"])
            and self._vision_gate_below(est)
            and (
                plan.state in {"SEARCH_GATE", "ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE", "PASS_GATE"}
                or plan.commit
            )
        ):
            allow_sink_boost = False
            current_alt = self._sane_vehicle_altitude_m(vehicle)
            target_alt = self._hover_altitude_m
            if current_alt is not None and target_alt is not None and current_alt > target_alt + 0.02:
                self._alt_integral = min(self._alt_integral, 0.0)
                cut = float(self.gains.get("vision_gate_descent_thrust_cut", 0.10))
                extra_bearing = max(
                    0.0,
                    float(est.gate_bearing_y_rad)
                    - float(self.gains.get("vision_gate_below_bearing_rad", 0.08)),
                )
                cut += extra_bearing * float(
                    self.gains.get("vision_gate_descent_extra_thrust_cut_per_rad", 0.18)
                )
                if plan.state == "ALIGN_GATE":
                    cut += extra_bearing * float(
                        self.gains.get("vision_lower_gate_align_extra_thrust_cut_per_rad", 0.10)
                    )
                proximity = self._vision_approach_proximity(est)
                if proximity > 0.0:
                    cut += proximity * float(
                        self.gains.get("vision_lower_gate_descent_close_cut_max", 0.0)
                    )
                alt_error = float(current_alt - target_alt)
                if alt_error > 0.08:
                    cut += min(
                        float(self.gains.get("vision_gate_descent_thrust_cut_max_extra", 0.08)),
                        (alt_error - 0.08)
                        * float(self.gains.get("vision_gate_descent_alt_error_cut_gain", 0.04)),
                    )
                cut = min(
                    cut,
                    float(self.gains.get("vision_gate_descent_thrust_cut_max", 0.18)),
                )
                thrust_cap = float(self.gains["hover_thrust"]) - cut
        return self._altitude_pid_thrust(
            vehicle,
            vz,
            dt,
            kp_key="alt_hold_kp",
            ki_key="alt_hold_ki",
            allow_sink_boost=allow_sink_boost,
            thrust_cap=thrust_cap,
        )

    def _altitude_hold_thrust(self, vehicle, vz: float, dt: float) -> float:
        return self._altitude_pid_thrust(
            vehicle,
            vz,
            dt,
            kp_key="alt_hold_kp",
            ki_key="alt_hold_ki",
        )

    def _vehicle_altitude_m(self, vehicle) -> float | None:
        if vehicle.position_ned_m is None:
            return None
        return -float(vehicle.position_ned_m[2])

    def _sane_vehicle_altitude_m(self, vehicle) -> float | None:
        if vehicle.position_ned_m is None:
            return None
        px, py, pz = vehicle.position_ned_m
        alt = -float(pz)
        sane_xy = float(self.gains.get("vision_position_sane_xy_m", 40.0))
        sane_alt = float(self.gains.get("vision_takeoff_alt_sane_max_m", 6.0))
        if math.hypot(float(px), float(py)) > sane_xy or alt > sane_alt or alt < -0.5:
            return None
        return alt

    def _takeoff_command(self, vehicle, yaw_sp: float, dt: float, climb_rate_mps: float = 0.0) -> AttitudeCommand:
        vx_ned, vy_ned, vz = (vehicle.velocity_ned_mps or (0.0, 0.0, 0.0))
        cos_yaw = math.cos(vehicle.yaw_rad)
        sin_yaw = math.sin(vehicle.yaw_rad)
        body_vx = cos_yaw * float(vx_ned) + sin_yaw * float(vy_ned)
        body_vy = -sin_yaw * float(vx_ned) + cos_yaw * float(vy_ned)
        roll_sp = clamp(
            -float(self.gains.get("stabilize_vy_kp", 0.04)) * body_vy,
            -float(self.gains.get("stabilize_roll_limit_rad", self.gains["roll_limit_rad"])),
            float(self.gains.get("stabilize_roll_limit_rad", self.gains["roll_limit_rad"])),
        )
        pitch_limit = float(self.gains.get("stabilize_pitch_limit_rad", self.gains["pitch_limit_rad"]))
        pitch_sp = clamp(
            self._hover_pitch_sp + float(self.gains.get("search_pitch_trim_rad", 0.0)) + self._climb_brake_pitch(float(vz), self._current_alt_error(vehicle)),
            self._hover_pitch_sp - pitch_limit,
            self._hover_pitch_sp + pitch_limit,
        )
        takeoff_alt = float(self.gains.get("takeoff_altitude_m", 4.0))
        if bool(self.gains.get("vision_primary_navigation", False)):
            takeoff_alt = float(self.gains.get("vision_takeoff_altitude_m", takeoff_alt))
        if self._takeoff_alt_slew_m is not None:
            max_climb = float(self.gains.get("takeoff_climb_rate_mps", 2.0))
            self._takeoff_alt_slew_m = min(takeoff_alt, self._takeoff_alt_slew_m + max_climb * dt)
            self._hover_altitude_m = self._takeoff_alt_slew_m
        thrust = self._altitude_pid_thrust(
            vehicle,
            float(vz),
            dt,
            kp_key="stabilize_alt_kp",
            ki_key="stabilize_alt_ki",
            allow_overspeed_cut=False,
        )
        takeoff_thrust = float(self.gains.get("takeoff_thrust", self.gains["hover_thrust"] + 0.12))
        current_alt = self._sane_vehicle_altitude_m(vehicle)
        target_alt = self._hover_altitude_m if self._hover_altitude_m is not None else takeoff_alt
        overshoot_m = float(self.gains.get("vision_takeoff_overshoot_m", 0.20))
        if current_alt is not None and current_alt < target_alt - 0.05:
            thrust = max(thrust, takeoff_thrust)
        elif current_alt is not None and current_alt > target_alt + overshoot_m:
            thrust = min(
                thrust,
                float(self.gains["hover_thrust"]) - float(self.gains.get("vision_takeoff_overshoot_thrust_cut", 0.06)),
            )
        max_climb = float(self.gains.get("takeoff_climb_rate_mps", 2.0))
        if float(vz) < -max_climb * 1.2:
            thrust = min(thrust, float(self.gains["hover_thrust"]) + float(vz) * float(self.gains["thrust_vz_d"]))
        thrust = clamp(thrust, float(self.gains["thrust_min"]), float(self.gains["thrust_max"]))
        return self._build_command(vehicle, roll_sp, pitch_sp, yaw_sp, thrust)

    def update(self, est: EstimatedState, plan: RacePlan, dt: float) -> AttitudeCommand:
        vehicle = est.vehicle
        yaw_now = vehicle.yaw_rad
        roll_now = vehicle.roll_rad
        pitch_now = vehicle.pitch_rad
        vx_ned, vy_ned, vz = (vehicle.velocity_ned_mps or (0.0, 0.0, 0.0))
        cos_yaw = math.cos(yaw_now)
        sin_yaw = math.sin(yaw_now)
        body_vx = cos_yaw * float(vx_ned) + sin_yaw * float(vy_ned)
        body_vy = -sin_yaw * float(vx_ned) + cos_yaw * float(vy_ned)
        using_map = plan.use_map_navigation and est.has_track_map and est.race_started

        if plan.state != self._last_plan_state:
            if plan.state == "STABILIZE" and vehicle.position_ned_m is not None:
                self._hover_altitude_m = -vehicle.position_ned_m[2]
                self._altitude_captured = True
                self._alt_integral = 0.0
                self._vision_base_hover_altitude_m = None
                self._capture_launch_pitch_trim(vehicle)
            elif plan.state == "TAKEOFF":
                self._altitude_captured = False
                self._vision_base_hover_altitude_m = None
                if not self._launch_pitch_captured:
                    self._capture_launch_pitch_trim(vehicle)
                takeoff_alt = float(self.gains.get("takeoff_altitude_m", 5.0))
                if bool(self.gains.get("vision_primary_navigation", False)):
                    takeoff_alt = float(self.gains.get("vision_takeoff_altitude_m", takeoff_alt))
                self._hover_altitude_m = takeoff_alt
                self._alt_integral = 0.0
                if vehicle.position_ned_m is not None:
                    self._takeoff_alt_slew_m = -vehicle.position_ned_m[2]
                else:
                    self._takeoff_alt_slew_m = 0.0
            elif plan.state == "ALIGN_GATE":
                self._align_yaw_hold_initialized = False
                align_alt = self.gains.get("vision_align_hover_altitude_m")
                if align_alt is not None and self._vision_primary_enabled():
                    target_align_alt = float(align_alt)
                    current_alt = self._sane_vehicle_altitude_m(vehicle)
                    if self._hover_altitude_m is not None:
                        target_align_alt = min(target_align_alt, self._hover_altitude_m)
                    if current_alt is not None:
                        target_align_alt = min(target_align_alt, current_alt)
                    self._hover_altitude_m = target_align_alt
                    self._vision_base_hover_altitude_m = target_align_alt
                    self._altitude_captured = True
                    self._alt_integral = 0.0
            elif plan.state == "APPROACH_GATE":
                self._align_yaw_hold_initialized = False
            elif plan.state in {"WAIT_LINK", "WAIT_VISION", "WAIT_START", "SEARCH_GATE", "RECOVER", "EMERGENCY_RECOVERY"}:
                self._vision_base_hover_altitude_m = None
            self._last_plan_state = plan.state
        target_speed = self._gate_speed_target_mps(est, plan, using_map)
        max_accel = float(self.gains.get("speed_ramp_accel_mps2", 1.5))
        max_decel = float(self.gains.get("speed_ramp_decel_mps2", 1.0))
        lower_gate_speed_cap = self._vision_lower_gate_speed_cap_mps(
            est,
            plan,
            using_map=using_map,
        )
        if lower_gate_speed_cap is not None:
            max_decel = max(
                max_decel,
                float(self.gains.get("vision_lower_gate_speed_decel_mps2", 2.8)),
            )
        if target_speed > self._speed_cmd_mps:
            self._speed_cmd_mps = min(self._speed_cmd_mps + max_accel * dt, target_speed)
        else:
            self._speed_cmd_mps = max(self._speed_cmd_mps - max_decel * dt, target_speed)
        if (
            lower_gate_speed_cap is not None
            and not using_map
            and plan.state == "ALIGN_GATE"
        ):
            self._speed_cmd_mps = min(self._speed_cmd_mps, lower_gate_speed_cap)
        if (
            using_map
            and est.map_within_gate_bounds is False
            and est.map_dist_center_m is not None
            and float(est.map_dist_center_m)
            < float(self.gains.get("map_collision_slow_radius_m", 6.0))
        ):
            oob_cap = float(self.gains.get("map_approach_speed_cap_oob_mps", 1.8))
            if abs(body_vx) > oob_cap:
                self._speed_cmd_mps = min(self._speed_cmd_mps, oob_cap)

        if not self._altitude_captured and vehicle.position_ned_m is not None and plan.state in {"SEARCH_GATE", "ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE", "PASS_GATE", "RECOVER", "EMERGENCY_RECOVERY"}:
            if est.map_gate_altitude_m is not None:
                self._hover_altitude_m = est.map_gate_altitude_m
            else:
                self._hover_altitude_m = -vehicle.position_ned_m[2]
            self._altitude_captured = True
            self._alt_integral = 0.0
            if not self._launch_pitch_captured:
                self._capture_launch_pitch_trim(vehicle)

        if plan.state in {"SEARCH_GATE", "ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE", "PASS_GATE", "RECOVER"}:
            self._update_hover_altitude_for_gate(est, dt, plan)

        if (
            plan.state not in {"WAIT_LINK", "WAIT_VISION", "WAIT_START", "STABILIZE"}
            and (abs(roll_now) >= float(self.gains.get("upright_recover_tilt_rad", 0.90))
                 or abs(pitch_now) >= float(self.gains.get("upright_recover_tilt_rad", 0.90)))
        ):
            return self._level_recovery_command(vehicle, yaw_now)

        if (
            plan.state in {"SEARCH_GATE", "ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE", "PASS_GATE", "RECOVER"}
            and (abs(roll_now) >= float(self.gains.get("level_recover_tilt_rad", 0.35))
                 or abs(pitch_now) >= float(self.gains.get("level_recover_tilt_rad", 0.35)))
        ):
            return self._level_recovery_command(vehicle, yaw_now)

        if plan.state in {"WAIT_LINK", "WAIT_VISION", "WAIT_START", "STABILIZE"}:
            if not self._yaw_hold_initialized:
                self._yaw_hold_sp = yaw_now
                self._yaw_hold_initialized = True
            self._commit_initialized = False
            return self._stabilize_command(vehicle, self._yaw_hold_sp, dt, hold_altitude=plan.state == "STABILIZE")

        if plan.state in {"TAKEOFF", "EMERGENCY_RECOVERY"}:
            if not self._yaw_hold_initialized:
                self._yaw_hold_sp = yaw_now
                self._yaw_hold_initialized = True
            self._commit_initialized = False
            return self._takeoff_command(vehicle, self._yaw_hold_sp, dt, plan.climb_rate_mps)

        vision_primary = bool(self.gains.get("vision_primary_navigation", False))
        vision_search_conf = float(self.gains.get("vision_search_steering_conf", 0.12))
        steerable_gate = (
            est.gate_track is not None
            and est.gate_track.visible
            and est.gate_confidence >= vision_search_conf
        )

        if plan.state == "RECOVER" and not using_map:
            self._commit_initialized = False
            return self._vision_search_command(
                vehicle,
                est,
                plan,
                dt,
                yaw_now=yaw_now,
                vz=float(vz),
                body_vx=body_vx,
                body_vy=body_vy,
                steer=False,
            )

        if plan.state == "SEARCH_GATE":
            self._commit_initialized = False
            if not using_map:
                return self._vision_search_command(
                    vehicle,
                    est,
                    plan,
                    dt,
                    yaw_now=yaw_now,
                    vz=float(vz),
                    body_vx=body_vx,
                    body_vy=body_vy,
                    steer=steerable_gate,
                )
            yaw_delta = clamp(
                float(self.gains["yaw_kp"]) * est.gate_bearing_x_rad - float(self.gains["yaw_rate_d"]) * vehicle.yaw_rate_rps,
                -float(self.gains["yaw_step_limit_rad"]),
                float(self.gains["yaw_step_limit_rad"]),
            )
            yaw_sp = wrap_pi(yaw_now + yaw_delta)
            pitch_sp = self._gate_pitch_sp(vehicle, float(vz), est, plan, using_map, body_vx, est.gate_bearing_y_rad)
            thrust = self._vision_altitude_hold_thrust(
                vehicle, float(vz), dt, est, plan, using_map=using_map
            )
            return self._build_command(vehicle, 0.0, pitch_sp, yaw_sp, thrust)

        if (
            not using_map
            and plan.state not in {"ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE", "PASS_GATE"}
            and (est.gate_track is None or est.gate_confidence < float(self.gains["min_gate_confidence"]))
        ):
            if not self._search_initialized:
                self._search_yaw_sp = yaw_now
                self._search_initialized = True
            self._commit_initialized = False
            yaw_scan = plan.yaw_scan_rate_rps * (0.25 if float(vz) < -0.20 else 1.0)
            self._search_yaw_sp = wrap_pi(self._search_yaw_sp + yaw_scan * dt)
            pitch_sp = clamp(
                self._hover_pitch_sp + float(self.gains["search_pitch_trim_rad"]),
                self._hover_pitch_sp - float(self.gains["pitch_limit_rad"]),
                self._hover_pitch_sp + float(self.gains["pitch_limit_rad"]),
            )
            return self._build_command(
                vehicle,
                0.0,
                pitch_sp,
                self._search_yaw_sp,
                self._hover_thrust(float(vz)),
            )

        if plan.commit:
            roll_kp = float(self.gains["roll_kp"]) * 0.9
            commit_theta_x = est.gate_bearing_x_rad
            commit_theta_y = est.gate_bearing_y_rad
            if using_map and est.map_gate_bearing_x_rad is not None:
                commit_theta_x = est.map_gate_bearing_x_rad
                if est.map_gate_bearing_y_rad is not None:
                    commit_theta_y = est.map_gate_bearing_y_rad
            if using_map:
                if est.map_commit_bearing_x_rad is not None and est.map_commit_bearing_y_rad is not None:
                    blend = float(est.map_gate_commit_strength)
                    commit_theta_x = (1.0 - blend) * commit_theta_x + blend * est.map_commit_bearing_x_rad
                    commit_theta_y = (1.0 - blend) * commit_theta_y + blend * est.map_commit_bearing_y_rad
            if not self._commit_initialized:
                if using_map and est.map_exit_yaw_rad is not None:
                    yaw_err = wrap_pi(est.map_exit_yaw_rad - yaw_now)
                    yaw_delta = clamp(
                        0.45 * yaw_err,
                        -0.35 * float(self.gains["yaw_step_limit_rad"]),
                        0.35 * float(self.gains["yaw_step_limit_rad"]),
                    )
                    self._commit_yaw_sp = wrap_pi(yaw_now + yaw_delta)
                elif bool(self.gains.get("vision_primary_navigation", False)) and not using_map:
                    self._commit_yaw_sp = self._vision_track_gate_yaw(
                        vehicle,
                        commit_theta_x,
                        math.hypot(body_vx, body_vy),
                        gain_scale=float(self.gains.get("vision_approach_yaw_gain_scale", 0.40)),
                    )
                else:
                    yaw_delta = clamp(
                        0.6 * float(self.gains["yaw_kp"]) * commit_theta_x,
                        -0.5 * float(self.gains["yaw_step_limit_rad"]),
                        0.5 * float(self.gains["yaw_step_limit_rad"]),
                    )
                    self._commit_yaw_sp = wrap_pi(yaw_now + yaw_delta)
                self._commit_initialized = True
            elif using_map and est.map_exit_yaw_rad is not None:
                yaw_err = wrap_pi(est.map_exit_yaw_rad - self._commit_yaw_sp)
                yaw_delta = clamp(
                    0.20 * yaw_err - 0.12 * float(self.gains["yaw_rate_d"]) * vehicle.yaw_rate_rps,
                    -0.10 * float(self.gains["yaw_step_limit_rad"]),
                    0.10 * float(self.gains["yaw_step_limit_rad"]),
                )
                self._commit_yaw_sp = wrap_pi(self._commit_yaw_sp + yaw_delta)
            elif bool(self.gains.get("vision_primary_navigation", False)) and not using_map:
                self._commit_yaw_sp = self._vision_track_gate_yaw(
                    vehicle,
                    commit_theta_x,
                    math.hypot(body_vx, body_vy),
                    gain_scale=float(self.gains.get("vision_approach_yaw_gain_scale", 0.40)),
                )
            elif abs(commit_theta_x) > 0.08:
                yaw_delta = clamp(
                    0.15 * float(self.gains["yaw_kp"]) * commit_theta_x
                    - 0.15 * float(self.gains["yaw_rate_d"]) * vehicle.yaw_rate_rps,
                    -0.12 * float(self.gains["yaw_step_limit_rad"]),
                    0.12 * float(self.gains["yaw_step_limit_rad"]),
                )
                self._commit_yaw_sp = wrap_pi(self._commit_yaw_sp + yaw_delta)
            yaw_sp = self._commit_yaw_sp
            roll_limit = 0.55 * float(self.gains["roll_limit_rad"])
            if using_map:
                roll_limit = float(self.gains.get("map_commit_roll_limit_rad", 0.10))
            commit_roll_theta = commit_theta_x
            if bool(self.gains.get("vision_primary_navigation", False)) and not using_map:
                commit_roll_theta, _ = self._vision_roll_yaw_theta_split(
                    commit_theta_x,
                    math.hypot(body_vx, body_vy),
                )
            roll_sp = clamp(
                roll_kp * commit_roll_theta,
                -roll_limit,
                roll_limit,
            )
            if (
                bool(self.gains.get("vision_primary_navigation", False))
                and not using_map
                and math.hypot(body_vx, body_vy) > 0.8
            ):
                damp_roll, _ = self._vision_search_body_damping(
                    body_vx,
                    body_vy,
                    aggressive=math.hypot(body_vx, body_vy) > 1.5,
                )
                blend = clamp((math.hypot(body_vx, body_vy) - 0.8) / 2.0, 0.0, 0.55)
                roll_sp = clamp(roll_sp * (1.0 - blend) + damp_roll * blend, -roll_limit, roll_limit)
            if bool(self.gains.get("vision_primary_navigation", False)) and not using_map:
                straight_roll = self._vision_straight_line_roll(body_vy, plan)
                roll_sp = clamp(roll_sp + straight_roll, -roll_limit, roll_limit)
            if using_map and est.map_lateral_error_m is not None:
                lat_kp = float(self.gains.get("map_lateral_roll_kp", 0.07))
                lat_roll = clamp(
                    -lat_kp * float(est.map_lateral_error_m),
                    -roll_limit,
                    roll_limit,
                )
                roll_sp = clamp(roll_sp + lat_roll, -roll_limit, roll_limit)
            if using_map and est.map_within_gate_bounds:
                roll_sp = clamp(
                    0.35 * roll_kp * commit_theta_x,
                    -float(self.gains.get("map_commit_line_roll_limit_rad", 0.05)),
                    float(self.gains.get("map_commit_line_roll_limit_rad", 0.05)),
                )
            commit_pitch_trim = float(self.gains.get("commit_pitch_trim_rad", -0.10))
            if using_map:
                plane = est.map_plane_signed_m
                dist = est.map_dist_center_m
                if plane is not None:
                    if plane < -4.0:
                        commit_pitch_trim = float(self.gains.get("map_commit_far_pitch_trim_rad", 0.14))
                    elif plane < 0.0:
                        commit_pitch_trim = float(self.gains.get("map_commit_mid_pitch_trim_rad", 0.18)) + 0.015 * (-plane)
                    elif plane < 1.5:
                        commit_pitch_trim = float(self.gains.get("map_commit_line_pitch_trim_rad", 0.22))
                    else:
                        commit_pitch_trim = float(self.gains.get("map_commit_post_plane_pitch_trim_rad", 0.18))
                    if dist is not None and dist < 4.0 and plane < 1.0:
                        commit_pitch_trim = max(
                            commit_pitch_trim,
                            float(self.gains.get("map_commit_near_pitch_trim_rad", 0.24)),
                        )
                    if dist is not None and dist < 2.5 and plane < 0.5:
                        commit_pitch_trim = max(
                            commit_pitch_trim,
                            float(self.gains.get("map_commit_push_pitch_trim_rad", 0.32)),
                        )
                else:
                    strength = max(0.0, min(1.0, float(est.map_gate_commit_strength)))
                    commit_pitch_trim = float(self.gains.get("map_commit_pitch_trim_rad", commit_pitch_trim)) + strength * float(
                        self.gains.get("map_commit_strength_pitch_gain_rad", 0.04),
                    )
            elif bool(self.gains.get("vision_primary_navigation", False)) and not using_map:
                area_frac = float(est.gate_track.area_fraction) if est.gate_track is not None else 0.0
                area_ref = float(self.gains.get("vision_commit_area_fraction", 0.08))
                base_trim = float(self.gains.get("vision_commit_pitch_trim_rad", 0.26))
                area_scale = clamp(area_frac / max(area_ref, 1e-3), 0.6, 1.4)
                commit_pitch_trim = base_trim * area_scale
                target_commit_speed = float(self.gains.get("commit_speed_mps", 2.2))
                forward_speed = abs(body_vx)
                if forward_speed < target_commit_speed * 0.85:
                    commit_pitch_trim = max(
                        commit_pitch_trim,
                        float(self.gains.get("vision_commit_push_pitch_trim_rad", 0.30)),
                    )
            commit_speed_cap = float(self.gains.get("map_commit_speed_cap_mps", 3.5))
            if bool(self.gains.get("vision_primary_navigation", False)) and not using_map:
                commit_speed_cap = min(
                    commit_speed_cap,
                    float(self.gains.get("vision_commit_max_speed_mps", 2.5)),
                )
            if using_map and est.map_within_gate_bounds is False:
                commit_speed_cap = min(
                    commit_speed_cap,
                    float(self.gains.get("map_commit_speed_cap_oob_mps", 2.4)),
                )
            cross_plane_m = float(self.gains.get("map_commit_cross_plane_hold_m", 0.5))
            forward_speed = abs(body_vx)
            if forward_speed > commit_speed_cap:
                keep_push = (
                    using_map
                    and est.map_plane_signed_m is not None
                    and (
                        est.map_plane_signed_m < 0.0
                        or (
                            est.map_plane_signed_m < cross_plane_m
                            and not est.map_within_gate_bounds
                        )
                    )
                )
                if not keep_push:
                    commit_pitch_trim = min(
                        commit_pitch_trim,
                        float(self.gains.get("map_commit_overspeed_pitch_trim_rad", 0.06)),
                    )
            elif (
                not bool(self.gains.get("vision_primary_navigation", False))
                and body_vx < -0.5
            ):
                commit_pitch_trim = max(commit_pitch_trim, float(self.gains.get("map_commit_mid_pitch_trim_rad", 0.18)))

            # Blend forward pitch with commit pitch trim
            forward_scale = self._approach_forward_scale(est, using_map)
            if using_map and est.map_within_gate_bounds is False:
                forward_scale = min(
                    forward_scale,
                    float(self.gains.get("map_commit_oob_forward_scale", 0.55)),
                )
            forward_pitch = self._forward_pitch(est, plan, using_map, body_vx) * forward_scale
            vert_scale = self._vision_vertical_pitch_scale(plan) if not using_map else 1.0
            vertical_trim = self._vertical_pitch_trim(commit_theta_y, scale=vert_scale)
            if using_map:
                vertical_trim += self._map_vertical_pitch_trim(est)
            pitch_sp = clamp(
                forward_pitch
                + commit_pitch_trim
                + vertical_trim
                + self._gate_pass_pitch_trim(est, plan, using_map)
                + self._backward_drift_pitch_brake(body_vx, est, using_map)
                + self._climb_brake_pitch(float(vz), self._current_alt_error(vehicle)),
                self._hover_pitch_sp - float(self.gains["pitch_limit_rad"]),
                self._hover_pitch_sp + float(self.gains["pitch_limit_rad"]),
            )
            thrust = self._vision_altitude_hold_thrust(
                vehicle, float(vz), dt, est, plan, using_map=using_map
            )
            if (
                using_map
                and est.map_dist_center_m is not None
                and est.map_dist_center_m < 4.0
                and est.map_plane_signed_m is not None
                and est.map_plane_signed_m < 1.0
                and est.map_within_gate_bounds
            ):
                thrust = min(
                    float(self.gains["thrust_max"]),
                    thrust + float(self.gains.get("commit_thrust_boost", 0.05)),
                )
            return self._build_command(vehicle, roll_sp, pitch_sp, yaw_sp, thrust)

        self._search_initialized = False
        self._yaw_hold_initialized = False
        self._commit_initialized = False
        theta_x = est.gate_bearing_x_rad
        theta_y = est.gate_bearing_y_rad
        if using_map and est.map_gate_bearing_x_rad is not None:
            if plan.state in {"ALIGN_GATE", "APPROACH_GATE"}:
                theta_x = est.map_gate_bearing_x_rad
                if est.map_gate_bearing_y_rad is not None:
                    theta_y = est.map_gate_bearing_y_rad
            elif est.map_dist_center_m is not None:
                blend = clamp(1.0 - est.map_dist_center_m / float(self.gains.get("map_gate_center_blend_dist_m", 8.0)), 0.35, 1.0)
                gate_bx = est.map_gate_bearing_x_rad
                gate_by = est.map_gate_bearing_y_rad if est.map_gate_bearing_y_rad is not None else theta_y
                theta_x = (1.0 - blend) * theta_x + blend * gate_bx
                theta_y = (1.0 - blend) * theta_y + blend * gate_by
        roll_kp = float(self.gains["roll_kp"]) * (float(self.gains.get("map_roll_gain_scale", 1.0)) if using_map else 1.0)
        yaw_kp = float(self.gains["yaw_kp"]) * (float(self.gains.get("map_yaw_gain_scale", 1.0)) if using_map else 1.0)
        if vision_primary and not using_map:
            if plan.state == "ALIGN_GATE":
                roll_kp *= float(self.gains.get("vision_align_roll_gain_scale", 0.55))
                yaw_kp *= float(self.gains.get("vision_align_yaw_gain_scale", 0.75))
            elif plan.state in {"APPROACH_GATE", "COMMIT_GATE", "PASS_GATE"}:
                roll_kp *= float(self.gains.get("vision_approach_roll_gain_scale", 1.0))
                yaw_kp *= float(self.gains.get("vision_approach_yaw_gain_scale", 1.0))
            fix_gain = self._vision_center_fixation_gain(plan, theta_x, theta_y)
            roll_kp *= fix_gain
            yaw_kp *= fix_gain

        map_yaw_error = None
        if using_map and est.map_exit_yaw_rad is not None and plan.state == "COMMIT_GATE":
            map_yaw_error = wrap_pi(est.map_exit_yaw_rad - yaw_now)

        hs = self._vision_horizontal_speed_mps(est)
        roll_theta = theta_x
        if (
            using_map
            and abs(theta_x) < float(self.gains.get("map_forward_yaw_gate_rad", 0.12))
            and est.map_lateral_error_m is not None
        ):
            lat_scale = float(self.gains.get("map_lateral_roll_kp", 0.07))
            roll_theta = clamp(-lat_scale * float(est.map_lateral_error_m), -0.25, 0.25)
        elif vision_primary and not using_map:
            roll_theta, _ = self._vision_roll_yaw_theta_split(theta_x, hs)

        roll_limit = float(self.gains["roll_limit_rad"])
        if vision_primary and not using_map and plan.state in {"ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE", "PASS_GATE"}:
            roll_limit = float(self.gains.get("vision_approach_roll_limit_rad", roll_limit))
        roll_sp = clamp(
            roll_kp * roll_theta - float(self.gains["lateral_velocity_d"]) * body_vy,
            -roll_limit,
            roll_limit,
        )

        if vision_primary and not using_map:
            yaw_sp = self._vision_track_gate_yaw(vehicle, theta_x, hs, gain_scale=yaw_kp / max(float(self.gains["yaw_kp"]), 1e-6))
            self._align_yaw_hold_initialized = False
        else:
            yaw_delta = clamp(
                yaw_kp * theta_x - float(self.gains["yaw_rate_d"]) * vehicle.yaw_rate_rps,
                -float(self.gains["yaw_step_limit_rad"]),
                float(self.gains["yaw_step_limit_rad"]),
            )
            yaw_hold_bearing = float(self.gains.get("vision_yaw_hold_bearing_rad", 0.10))
            center_bx, _center_by = self._vision_center_bearing_limits(plan)
            hold_speed_cap = float(self.gains.get("vision_yaw_hold_max_speed_mps", 0.6))
            if (
                plan.state in {"ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE"}
                and hs <= hold_speed_cap
                and (abs(theta_x) <= yaw_hold_bearing or abs(theta_x) <= center_bx)
                and self._vision_centered_enough(theta_x, theta_y, plan)
            ):
                if not self._align_yaw_hold_initialized:
                    self._align_yaw_hold_sp = yaw_now
                    self._align_yaw_hold_initialized = True
                yaw_sp = self._align_yaw_hold_sp
            else:
                yaw_sp = wrap_pi(yaw_now + yaw_delta)
                if plan.state == "ALIGN_GATE":
                    self._align_yaw_hold_initialized = False
        if vision_primary and not using_map:
            straight_roll = self._vision_straight_line_roll(body_vy, plan)
            roll_sp = clamp(roll_sp + straight_roll, -roll_limit, roll_limit)
        if (
            vision_primary
            and not using_map
            and plan.state == "ALIGN_GATE"
        ):
            damp_roll, _ = self._vision_search_body_damping(
                body_vx,
                body_vy,
                aggressive=math.hypot(body_vx, body_vy) > 0.8,
            )
            damp_mix = clamp(math.hypot(body_vx, body_vy) / 5.0, 0.0, 0.22)
            roll_sp = clamp(
                roll_sp + damp_roll * damp_mix,
                -roll_limit,
                roll_limit,
            )
        if (
            using_map
            and est.map_lateral_error_m is not None
            and abs(theta_x) < float(self.gains.get("map_forward_yaw_gate_rad", 0.12))
        ):
            lat_kp = float(self.gains.get("map_lateral_roll_kp", 0.07))
            lat_limit = float(self.gains.get("map_approach_lateral_roll_limit_rad", 0.12))
            if (
                est.map_within_gate_bounds is False
                and est.map_dist_center_m is not None
                and float(est.map_dist_center_m)
                < float(self.gains.get("map_collision_slow_radius_m", 6.0))
            ):
                lat_limit = min(
                    lat_limit,
                    float(self.gains.get("map_oob_lateral_roll_limit_rad", 0.08)),
                )
            roll_sp = clamp(
                roll_sp - lat_kp * float(est.map_lateral_error_m),
                -lat_limit,
                lat_limit,
        )
        pitch_sp = self._gate_pitch_sp(vehicle, float(vz), est, plan, using_map, body_vx, theta_y)
        speed_brake_mps = float(self.gains.get("vision_align_speed_brake_mps", 1.8))
        lower_gate_extra_bearing = self._lower_gate_extra_bearing_rad(est)
        if (
            vision_primary
            and not using_map
            and plan.state in {"ALIGN_GATE", "APPROACH_GATE"}
            and lower_gate_extra_bearing > 0.0
        ):
            speed_brake_mps = min(
                speed_brake_mps,
                float(self.gains.get("vision_lower_gate_brake_speed_mps", 0.9)),
            )
        if (
            vision_primary
            and not using_map
            and plan.state in {"ALIGN_GATE", "APPROACH_GATE"}
            and hs > speed_brake_mps
        ):
            _, damp_pitch = self._vision_search_body_damping(
                body_vx,
                body_vy,
                aggressive=True,
            )
            brake_mix = clamp((hs - speed_brake_mps) / 1.5, 0.25, 0.65)
            if lower_gate_extra_bearing > 0.0:
                brake_mix = clamp(
                    brake_mix
                    + lower_gate_extra_bearing
                    * float(self.gains.get("vision_lower_gate_brake_mix_per_rad", 1.2)),
                    0.35,
                    float(self.gains.get("vision_lower_gate_brake_mix_max", 0.90)),
                )
            pitch_sp = pitch_sp + (damp_pitch - self._hover_pitch_sp) * brake_mix
        elif (
            vision_primary
            and not using_map
            and plan.state in {"ALIGN_GATE", "APPROACH_GATE"}
        ):
            _, damp_pitch = self._vision_search_body_damping(
                body_vx,
                body_vy,
                aggressive=math.hypot(body_vx, body_vy) > 0.8,
            )
            blend = clamp(math.hypot(body_vx, body_vy) / 2.5, 0.25, 0.85)
            if lower_gate_extra_bearing > 0.0:
                blend = clamp(
                    blend
                    + lower_gate_extra_bearing
                    * float(self.gains.get("vision_lower_gate_brake_blend_per_rad", 0.8)),
                    0.35,
                    float(self.gains.get("vision_lower_gate_brake_blend_max", 0.90)),
                )
            pitch_sp = pitch_sp + (damp_pitch - self._hover_pitch_sp) * blend
        forward_scale = self._vision_forward_scale(est, plan, using_map)
        if forward_scale < 0.95 and not (vision_primary and not using_map):
            pitch_sp = self._hover_pitch_sp + (pitch_sp - self._hover_pitch_sp) * forward_scale
        if using_map and abs(theta_x) > float(self.gains.get("map_forward_yaw_gate_rad", 0.12)):
            map_forward_scale = max(0.0, 1.0 - abs(theta_x) / 0.45)
            pitch_sp = self._hover_pitch_sp + (pitch_sp - self._hover_pitch_sp) * map_forward_scale
        pitch_sp = self._clamp_gate_pitch_sp(
            pitch_sp,
            pitch_now,
            plan,
            horizontal_speed=hs,
            using_map=using_map,
            gate_theta_y=theta_y,
        )
        thrust = self._vision_altitude_hold_thrust(
            vehicle, float(vz), dt, est, plan, using_map=using_map
        )
        return self._build_command(vehicle, roll_sp, pitch_sp, yaw_sp, thrust)

    def _build_command(self, vehicle, roll_sp: float, pitch_sp: float, yaw_sp: float, thrust: float) -> AttitudeCommand:
        roll_err = roll_sp - vehicle.roll_rad
        pitch_err = pitch_sp - vehicle.pitch_rad
        yaw_err = wrap_pi(yaw_sp - vehicle.yaw_rad)
        roll_rate_raw = float(self.gains["attitude_p"]) * roll_err
        pitch_rate_raw = float(self.gains["attitude_p"]) * pitch_err
        yaw_rate_raw = float(self.gains["yaw_attitude_p"]) * yaw_err
        roll_rate_limit = float(self.gains["roll_rate_limit_rps"])
        pitch_rate_limit = float(self.gains["pitch_rate_limit_rps"])
        yaw_rate_limit = float(self.gains["yaw_rate_limit_rps"])
        roll_rate = clamp(roll_rate_raw, -roll_rate_limit, roll_rate_limit)
        pitch_rate = clamp(pitch_rate_raw, -pitch_rate_limit, pitch_rate_limit)
        yaw_rate = clamp(yaw_rate_raw, -yaw_rate_limit, yaw_rate_limit)
        sat = CommandSaturation(
            thrust_min=thrust <= float(self.gains["thrust_min"]) + 1e-6,
            thrust_max=thrust >= float(self.gains["thrust_max"]) - 1e-6,
            roll_rate_limit=abs(roll_rate_raw) >= roll_rate_limit - 1e-6,
            pitch_rate_limit=abs(pitch_rate_raw) >= pitch_rate_limit - 1e-6,
            yaw_rate_limit=abs(yaw_rate_raw) >= yaw_rate_limit - 1e-6,
        )
        return AttitudeCommand(
            roll_rad=roll_sp,
            pitch_rad=pitch_sp,
            yaw_rad=yaw_sp,
            thrust=thrust,
            roll_rate_rps=roll_rate,
            pitch_rate_rps=pitch_rate,
            yaw_rate_rps=yaw_rate,
            quaternion_wxyz=euler_to_quaternion_wxyz(roll_sp, pitch_sp, yaw_sp),
            saturation=sat,
        )

    def _stabilize_command(self, vehicle, yaw_sp: float, dt: float, *, hold_altitude: bool = True) -> AttitudeCommand:
        vx_ned, vy_ned, vz = (vehicle.velocity_ned_mps or (0.0, 0.0, 0.0))
        cos_yaw = math.cos(vehicle.yaw_rad)
        sin_yaw = math.sin(vehicle.yaw_rad)
        body_vx = cos_yaw * float(vx_ned) + sin_yaw * float(vy_ned)
        body_vy = -sin_yaw * float(vx_ned) + cos_yaw * float(vy_ned)
        roll_sp = clamp(
            -float(self.gains.get("stabilize_vy_kp", 0.04)) * body_vy,
            -float(self.gains.get("stabilize_roll_limit_rad", self.gains["roll_limit_rad"])),
            float(self.gains.get("stabilize_roll_limit_rad", self.gains["roll_limit_rad"])),
        )
        pitch_sp = clamp(
            self._hover_pitch_sp - float(self.gains.get("stabilize_vx_kp", 0.03)) * body_vx,
            -float(self.gains.get("stabilize_pitch_limit_rad", self.gains["pitch_limit_rad"])),
            float(self.gains.get("stabilize_pitch_limit_rad", self.gains["pitch_limit_rad"])),
        )
        if hold_altitude:
            if self._hover_altitude_m is None and vehicle.position_ned_m is not None:
                self._hover_altitude_m = -vehicle.position_ned_m[2]
                self._alt_integral = 0.0
            alt_error = 0.0
            if self._hover_altitude_m is not None and vehicle.position_ned_m is not None:
                alt_error = self._hover_altitude_m - (-vehicle.position_ned_m[2])
            pitch_sp = clamp(
                pitch_sp + self._climb_brake_pitch(float(vz), alt_error),
                self._hover_pitch_sp - float(self.gains.get("stabilize_pitch_limit_rad", self.gains["pitch_limit_rad"])),
                self._hover_pitch_sp + float(self.gains.get("stabilize_pitch_limit_rad", self.gains["pitch_limit_rad"])),
            )
            thrust = self._altitude_pid_thrust(
                vehicle,
                float(vz),
                dt,
                kp_key="stabilize_alt_kp",
                ki_key="stabilize_alt_ki",
            )
        else:
            alt_err = 0.0
            if self._hover_altitude_m is not None and vehicle.position_ned_m is not None:
                alt_err = self._hover_altitude_m - (-vehicle.position_ned_m[2])
            pitch_sp = clamp(
                pitch_sp + self._climb_brake_pitch(float(vz), alt_err),
                self._hover_pitch_sp - float(self.gains.get("stabilize_pitch_limit_rad", self.gains["pitch_limit_rad"])),
                self._hover_pitch_sp + float(self.gains.get("stabilize_pitch_limit_rad", self.gains["pitch_limit_rad"])),
            )
            thrust = self._hover_thrust(float(vz))
        return self._build_command(vehicle, roll_sp, pitch_sp, yaw_sp, thrust)

    def _level_recovery_command(self, vehicle, yaw_sp: float) -> AttitudeCommand:
        kp = float(self.gains.get("level_recover_kp", 4.0))
        kd = float(self.gains.get("level_recover_kd", 0.4))
        vz = float((vehicle.velocity_ned_mps or (0.0, 0.0, 0.0))[2])
        thrust = self._hover_thrust(vz)
        yaw_err = wrap_pi(yaw_sp - vehicle.yaw_rad)
        roll_rate_raw = -kp * vehicle.roll_rad - kd * vehicle.roll_rate_rps
        pitch_rate_raw = -kp * vehicle.pitch_rad - kd * vehicle.pitch_rate_rps
        yaw_rate_raw = float(self.gains["yaw_attitude_p"]) * yaw_err - kd * vehicle.yaw_rate_rps
        roll_rate_limit = float(self.gains["roll_rate_limit_rps"])
        pitch_rate_limit = float(self.gains["pitch_rate_limit_rps"])
        yaw_rate_limit = float(self.gains["yaw_rate_limit_rps"])
        roll_rate = clamp(roll_rate_raw, -roll_rate_limit, roll_rate_limit)
        pitch_rate = clamp(pitch_rate_raw, -pitch_rate_limit, pitch_rate_limit)
        yaw_rate = clamp(yaw_rate_raw, -yaw_rate_limit, yaw_rate_limit)
        sat = CommandSaturation(
            thrust_min=thrust <= float(self.gains["thrust_min"]) + 1e-6,
            thrust_max=thrust >= float(self.gains["thrust_max"]) - 1e-6,
            roll_rate_limit=abs(roll_rate_raw) >= roll_rate_limit - 1e-6,
            pitch_rate_limit=abs(pitch_rate_raw) >= pitch_rate_limit - 1e-6,
            yaw_rate_limit=abs(yaw_rate_raw) >= yaw_rate_limit - 1e-6,
        )
        return AttitudeCommand(
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_rad=yaw_sp,
            thrust=thrust,
            roll_rate_rps=roll_rate,
            pitch_rate_rps=pitch_rate,
            yaw_rate_rps=yaw_rate,
            quaternion_wxyz=euler_to_quaternion_wxyz(0.0, 0.0, yaw_sp),
            saturation=sat,
        )
