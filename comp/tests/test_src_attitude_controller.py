from src.control.attitude_controller import AttitudeController
from src.types import EstimatedState, GateTrack, RacePlan, VehicleState


GAINS = {
    "min_gate_confidence": 0.15,
    "hover_thrust": 0.50,
    "thrust_min": 0.34,
    "thrust_max": 0.66,
    "roll_kp": 0.90,
    "yaw_kp": 1.20,
    "yaw_rate_d": 0.10,
    "lateral_velocity_d": 0.08,
    "pitch_speed_kp": 0.10,
    "vertical_gate_pitch_kp": 0.35,
    "vertical_gate_pitch_limit_rad": 0.10,
    "thrust_vz_d": 0.04,
    "pitch_trim_rad": 0.00,
    "search_pitch_trim_rad": 0.00,
    "roll_limit_rad": 0.28,
    "pitch_limit_rad": 0.24,
    "yaw_step_limit_rad": 0.50,
    "attitude_p": 3.0,
    "yaw_attitude_p": 2.4,
    "roll_rate_limit_rps": 1.0,
    "pitch_rate_limit_rps": 1.0,
    "yaw_rate_limit_rps": 0.8,
}


def test_gate_right_commands_positive_yaw_and_roll():
    controller = AttitudeController(GAINS)
    vehicle = VehicleState(yaw_rad=0.1, velocity_ned_mps=(0.0, 0.0, 0.0))
    gate = GateTrack(frame_id=1, timestamp_s=1.0, center_px=(340.0, 290.0), bbox=(0, 0, 100, 100), confidence=0.8, area_fraction=0.1, visible=True)
    est = EstimatedState(
        now_s=1.0,
        vehicle=vehicle,
        gate_track=gate,
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.12,
        gate_bearing_y_rad=0.0,
        gate_confidence=0.8,
    )
    plan = RacePlan("ALIGN_GATE", 0.8)
    cmd = controller.update(est, plan, dt=1.0 / 60.0)
    assert cmd.yaw_rad > 0.1
    assert cmd.roll_rad > 0.0


def test_gate_below_increases_forward_pitch():
    controller = AttitudeController(GAINS)
    vehicle = VehicleState(yaw_rad=0.0, pitch_rad=0.31, velocity_ned_mps=(0.0, 0.0, 0.0), position_ned_m=(0.0, 0.0, -2.0))
    controller._hover_pitch_sp = 0.31
    controller._launch_pitch_captured = True
    controller._hover_altitude_m = 2.0
    controller._altitude_captured = True
    gate = GateTrack(frame_id=1, timestamp_s=1.0, center_px=(320.0, 320.0), bbox=(0, 0, 100, 100), confidence=0.8, area_fraction=0.1, visible=True)
    est = EstimatedState(
        now_s=1.0,
        vehicle=vehicle,
        gate_track=gate,
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.0,
        gate_bearing_y_rad=0.1,
        gate_confidence=0.8,
    )
    plan = RacePlan("ALIGN_GATE", 0.8)
    cmd = controller.update(est, plan, dt=1.0 / 60.0)
    assert cmd.pitch_rad > vehicle.pitch_rad


def test_gate_above_does_not_command_sky_pitch():
    controller = AttitudeController(GAINS)
    vehicle = VehicleState(yaw_rad=0.0, pitch_rad=0.31, velocity_ned_mps=(0.0, 0.0, 0.0), position_ned_m=(0.0, 0.0, -2.0))
    controller._hover_pitch_sp = 0.31
    controller._launch_pitch_captured = True
    controller._hover_altitude_m = 2.0
    controller._altitude_captured = True
    gate = GateTrack(frame_id=1, timestamp_s=1.0, center_px=(320.0, 120.0), bbox=(0, 0, 100, 100), confidence=0.8, area_fraction=0.1, visible=True)
    est = EstimatedState(
        now_s=1.0,
        vehicle=vehicle,
        gate_track=gate,
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.0,
        gate_bearing_y_rad=-0.15,
        gate_confidence=0.8,
    )
    plan = RacePlan("ALIGN_GATE", 0.8)
    cmd = controller.update(est, plan, dt=1.0 / 60.0)
    assert cmd.pitch_rad <= vehicle.pitch_rad + 0.12


def test_vertical_pitch_trim_gate_below_commands_nose_down():
    controller = AttitudeController(GAINS)
    trim = controller._vertical_pitch_trim(0.12)
    assert trim > 0.0
    trim_above = controller._vertical_pitch_trim(-0.10)
    assert trim_above < 0.0


def test_lower_gate_pitch_boost_adds_extra_nose_down():
    gains = {
        **GAINS,
        "vision_primary_navigation": True,
        "vision_lower_gate_pitch_boost_per_rad": 0.75,
        "vision_lower_gate_pitch_boost_max_rad": 0.16,
        "vision_lower_gate_align_pitch_boost_scale": 1.25,
    }
    controller = AttitudeController(gains)
    low = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=GateTrack(1, 1.0, (320.0, 220.0), (0, 0, 100, 100), 0.9, 0.08, True),
        link_ready=True,
        vision_ready=True,
        gate_bearing_y_rad=0.22,
        gate_confidence=0.9,
    )
    mild = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=GateTrack(1, 1.0, (320.0, 220.0), (0, 0, 100, 100), 0.9, 0.08, True),
        link_ready=True,
        vision_ready=True,
        gate_bearing_y_rad=0.10,
        gate_confidence=0.9,
    )
    plan = RacePlan("ALIGN_GATE", 0.8)

    assert controller._vision_lower_gate_pitch_boost(low, plan, using_map=False) > 0.05
    assert controller._vision_lower_gate_pitch_boost(low, plan, using_map=False) > controller._vision_lower_gate_pitch_boost(mild, plan, using_map=False)


def test_launch_pitch_trim_prevents_sky_point_on_approach():
    gains = {
        **GAINS,
        "vision_primary_navigation": True,
        "vision_approach_pitch_limit_rad": 0.42,
        "vision_min_nose_down_pitch_delta_rad": 0.06,
        "forward_flight_pitch_kp": 0.80,
    }
    controller = AttitudeController(gains)
    vehicle = VehicleState(yaw_rad=0.0, pitch_rad=0.31, velocity_ned_mps=(0.4, 0.0, 0.0), position_ned_m=(0.0, 0.0, -1.8))
    controller._hover_pitch_sp = 0.31
    controller._launch_pitch_captured = True
    controller._hover_altitude_m = 1.8
    controller._altitude_captured = True
    est = EstimatedState(
        now_s=1.0,
        vehicle=vehicle,
        gate_track=GateTrack(1, 1.0, (320.0, 250.0), (0, 0, 100, 100), 0.9, 0.12, True),
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.10,
        gate_confidence=0.9,
        gate_range_m=6.0,
    )
    cmd = controller.update(est, RacePlan("APPROACH_GATE", 1.4), dt=1.0 / 60.0)
    assert cmd.pitch_rad > vehicle.pitch_rad


def test_gate_above_does_not_spike_thrust():
    controller = AttitudeController(GAINS)
    vehicle = VehicleState(yaw_rad=0.0, velocity_ned_mps=(0.0, 0.0, 0.0), position_ned_m=(0.0, 0.0, -2.0))
    controller._hover_altitude_m = 2.0
    controller._altitude_captured = True
    est = EstimatedState(
        now_s=1.0,
        vehicle=vehicle,
        gate_track=None,
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.0,
        gate_bearing_y_rad=-0.15,
        gate_confidence=0.8,
    )
    plan = RacePlan("ALIGN_GATE", 0.8)
    cmd = controller.update(est, plan, dt=1.0 / 60.0)
    assert cmd.thrust <= GAINS["hover_thrust"] + 0.05


def test_sink_boost_raises_thrust_when_sinking():
    controller = AttitudeController(GAINS)
    hover = controller._hover_thrust(0.0)
    sinking = controller._hover_thrust(0.5)
    assert sinking > hover


def test_altitude_pid_raises_thrust_when_below_target():
    gains = {**GAINS, "alt_hold_kp": 0.20, "alt_hold_ki": 0.0, "hover_thrust": 0.50}
    controller = AttitudeController(gains)
    controller._hover_altitude_m = 5.0
    vehicle = VehicleState(position_ned_m=(0.0, 0.0, -4.0), velocity_ned_mps=(0.0, 0.0, 0.0))
    thrust = controller._altitude_hold_thrust(vehicle, 0.0, 1.0 / 60.0)
    assert thrust > gains["hover_thrust"]


def test_altitude_pid_reduces_thrust_when_above_target():
    gains = {**GAINS, "alt_hold_kp": 0.20, "alt_hold_ki": 0.0, "hover_thrust": 0.50}
    controller = AttitudeController(gains)
    controller._hover_altitude_m = 5.0
    vehicle = VehicleState(position_ned_m=(0.0, 0.0, -6.0), velocity_ned_mps=(0.0, 0.0, 0.0))
    thrust = controller._altitude_hold_thrust(vehicle, 0.0, 1.0 / 60.0)
    assert thrust < gains["hover_thrust"]


def test_altitude_integral_builds_over_time():
    gains = {**GAINS, "alt_hold_kp": 0.0, "alt_hold_ki": 0.10, "hover_thrust": 0.50}
    controller = AttitudeController(gains)
    controller._hover_altitude_m = 5.0
    vehicle = VehicleState(position_ned_m=(0.0, 0.0, -4.5), velocity_ned_mps=(0.0, 0.0, 0.0))
    dt = 1.0 / 60.0
    first = controller._altitude_hold_thrust(vehicle, 0.0, dt)
    second = controller._altitude_hold_thrust(vehicle, 0.0, dt)
    assert second > first


def test_gate_pass_pitch_trim_before_plane_only():
    gains = {
        **GAINS,
        "map_gate_pass_pitch_trim_rad": 0.05,
        "map_gate_pass_pitch_dist_m": 4.0,
        "map_gate_pass_pitch_plane_margin_m": 0.5,
    }
    controller = AttitudeController(gains)
    est = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=None,
        link_ready=True,
        vision_ready=True,
        map_dist_center_m=2.0,
        map_plane_signed_m=-1.0,
    )
    commit = RacePlan("COMMIT_GATE", 2.2, commit=True)
    past = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=None,
        link_ready=True,
        vision_ready=True,
        map_dist_center_m=2.0,
        map_plane_signed_m=5.0,
    )
    assert controller._gate_pass_pitch_trim(est, commit, using_map=True) == 0.05
    assert controller._gate_pass_pitch_trim(past, commit, using_map=True) == 0.0


def test_approach_altitude_drop_only_before_plane_and_on_approach():
    gains = {
        **GAINS,
        "map_gate_pass_alt_drop_m": 0.25,
        "map_gate_pass_alt_drop_dist_m": 6.0,
        "map_gate_pass_alt_drop_plane_margin_m": 0.5,
    }
    controller = AttitudeController(gains)
    est = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=None,
        link_ready=True,
        vision_ready=True,
        map_dist_center_m=3.0,
        map_plane_signed_m=-1.0,
    )
    approach = RacePlan("APPROACH_GATE", 2.2, commit=False)
    align = RacePlan("ALIGN_GATE", 1.5, commit=False)
    past = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=None,
        link_ready=True,
        vision_ready=True,
        map_dist_center_m=3.0,
        map_plane_signed_m=3.0,
    )
    assert controller._approach_altitude_drop_m(est, align) == 0.0
    assert 0.0 < controller._approach_altitude_drop_m(est, approach) < 0.25
    assert controller._approach_altitude_drop_m(past, approach) == 0.0
    commit = RacePlan("COMMIT_GATE", 2.2, commit=True)
    assert controller._approach_altitude_drop_m(est, commit) == 0.25


def test_backward_drift_brake_adds_forward_pitch():
    controller = AttitudeController({**GAINS, "backward_drift_pitch_kp": 0.08})
    est = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=None,
        link_ready=True,
        vision_ready=True,
        map_plane_signed_m=-2.0,
    )
    brake = controller._backward_drift_pitch_brake(-1.0, est, using_map=True)
    assert 0.05 < brake <= 0.12


def test_vision_approach_commands_forward_pitch():
    gains = {
        **GAINS,
        "forward_flight_pitch_kp": 0.80,
        "forward_flight_pitch_limit_rad": 0.70,
        "vision_primary_navigation": True,
        "vision_approach_pitch_limit_rad": 0.38,
        "vision_align_forward_scale": 0.40,
    }
    controller = AttitudeController(gains)
    vehicle = VehicleState(yaw_rad=0.0, velocity_ned_mps=(0.0, 0.0, 0.0), position_ned_m=(0.0, 0.0, -2.0))
    controller._hover_altitude_m = 2.0
    controller._altitude_captured = True
    gate = GateTrack(
        frame_id=1,
        timestamp_s=1.0,
        center_px=(320.0, 290.0),
        bbox=(200, 100, 140, 140),
        confidence=0.9,
        area_fraction=0.12,
        visible=True,
    )
    est = EstimatedState(
        now_s=1.0,
        vehicle=vehicle,
        gate_track=gate,
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.03,
        gate_bearing_y_rad=0.02,
        gate_confidence=0.9,
    )
    plan = RacePlan("APPROACH_GATE", 1.8)
    for _ in range(30):
        cmd = controller.update(est, plan, dt=1.0 / 60.0)
    assert cmd.pitch_rad > float(GAINS.get("pitch_trim_rad", 0.0)) + 0.08


def test_oob_near_gate_reduces_forward_scale():
    gains = {
        **GAINS,
        "map_collision_slow_radius_m": 6.0,
        "map_collision_oob_slow_gain": 0.85,
        "map_collision_min_forward_scale": 0.15,
    }
    controller = AttitudeController(gains)
    est = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=None,
        link_ready=True,
        vision_ready=True,
        has_track_map=True,
        map_dist_center_m=1.0,
        map_within_gate_bounds=False,
        map_vertical_error_m=0.2,
        map_lateral_error_m=0.2,
    )
    scale = controller._approach_forward_scale(est, using_map=True)
    assert scale < 0.55
    speed = controller._gate_speed_target_mps(est, RacePlan("APPROACH_GATE", 2.5), using_map=True)
    assert speed < 2.5


def test_vision_roll_yaw_split_favors_roll_at_speed():
    gains = {**GAINS, "vision_primary_navigation": True}
    controller = AttitudeController(gains)
    roll_theta, yaw_theta = controller._vision_roll_yaw_theta_split(0.20, horizontal_speed=2.0)
    assert abs(roll_theta) > abs(yaw_theta)


def test_vision_primary_tracks_yaw_without_hold():
    gains = {
        **GAINS,
        "vision_primary_navigation": True,
        "vision_approach_roll_gain_scale": 1.5,
        "vision_approach_yaw_gain_scale": 0.4,
        "vision_yaw_step_limit_rad": 0.12,
    }
    controller = AttitudeController(gains)
    vehicle = VehicleState(yaw_rad=0.0, velocity_ned_mps=(1.2, 0.0, 0.0), position_ned_m=(0.0, 0.0, -2.0))
    controller._hover_altitude_m = 2.0
    controller._altitude_captured = True
    gate = GateTrack(
        frame_id=1,
        timestamp_s=1.0,
        center_px=(360.0, 290.0),
        bbox=(200, 100, 140, 140),
        confidence=0.9,
        area_fraction=0.12,
        visible=True,
    )
    est = EstimatedState(
        now_s=1.0,
        vehicle=vehicle,
        gate_track=gate,
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.08,
        gate_bearing_y_rad=0.02,
        gate_confidence=0.9,
    )
    plan = RacePlan("APPROACH_GATE", 1.4)
    cmd1 = controller.update(est, plan, dt=1.0 / 60.0)
    est2 = EstimatedState(
        now_s=1.02,
        vehicle=VehicleState(yaw_rad=cmd1.yaw_rad, velocity_ned_mps=(1.2, 0.0, 0.0), position_ned_m=(0.0, 0.0, -2.0)),
        gate_track=gate,
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.06,
        gate_bearing_y_rad=0.02,
        gate_confidence=0.9,
    )
    cmd2 = controller.update(est2, plan, dt=1.0 / 60.0)
    assert cmd1.roll_rad > 0.0
    assert cmd2.yaw_rad != cmd1.yaw_rad or cmd2.yaw_rate_rps != 0.0


def test_vision_align_uses_absolute_hover_altitude():
    gains = {
        **GAINS,
        "vision_primary_navigation": True,
        "vision_align_hover_altitude_m": 1.75,
        "vision_approach_min_alt_m": 0.73,
    }
    controller = AttitudeController(gains)
    gate = GateTrack(
        frame_id=1,
        timestamp_s=1.0,
        center_px=(320.0, 250.0),
        bbox=(200, 80, 140, 140),
        confidence=0.9,
        area_fraction=0.08,
        visible=True,
    )
    est = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=gate,
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.0,
        gate_range_m=6.0,
        gate_confidence=0.9,
    )
    align = RacePlan("ALIGN_GATE", 0.8, commit=False)
    align_target = controller._vision_hover_altitude_target(est, align, 2.5)
    assert align_target == 1.75


def test_vision_align_hover_altitude_is_ceiling_not_forced_climb():
    gains = {
        **GAINS,
        "vision_primary_navigation": True,
        "vision_align_hover_altitude_m": 1.75,
        "vision_approach_min_alt_m": 0.35,
    }
    controller = AttitudeController(gains)
    controller._hover_pitch_sp = 0.31
    controller._launch_pitch_captured = True
    controller._hover_altitude_m = 0.62
    controller._altitude_captured = True
    controller._last_plan_state = "SEARCH_GATE"
    vehicle = VehicleState(
        yaw_rad=0.0,
        pitch_rad=0.31,
        velocity_ned_mps=(0.0, 0.0, 0.0),
        position_ned_m=(0.0, 0.0, -0.62),
    )
    est = EstimatedState(
        now_s=1.0,
        vehicle=vehicle,
        gate_track=GateTrack(1, 1.0, (320.0, 250.0), (0, 0, 100, 100), 0.9, 0.08, True),
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.12,
        gate_range_m=6.0,
        gate_confidence=0.9,
    )

    controller.update(est, RacePlan("ALIGN_GATE", 0.8), dt=1.0 / 60.0)

    assert controller._hover_altitude_m <= 0.62
    assert controller._vision_base_hover_altitude_m <= 0.62


def test_vision_align_lowers_hover_altitude_by_fraction():
    gains = {
        **GAINS,
        "vision_primary_navigation": True,
        "vision_align_alt_drop_fraction": 0.20,
        "vision_align_proximity_floor": 0.50,
        "vision_approach_alt_drop_m": 0.53,
        "vision_approach_min_alt_m": 0.73,
    }
    controller = AttitudeController(gains)
    gate = GateTrack(
        frame_id=1,
        timestamp_s=1.0,
        center_px=(320.0, 250.0),
        bbox=(200, 80, 140, 140),
        confidence=0.9,
        area_fraction=0.08,
        visible=True,
    )
    est = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=gate,
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.10,
        gate_range_m=6.0,
        gate_confidence=0.9,
    )
    base = 0.9
    align = RacePlan("ALIGN_GATE", 0.8, commit=False)
    align_target = controller._vision_hover_altitude_target(est, align, base)
    assert align_target < base
    assert align_target <= 0.73


def test_vision_approach_drops_altitude_and_centers_on_gate():
    gains = {
        **GAINS,
        "vision_primary_navigation": True,
        "vision_approach_alt_drop_m": 0.6,
        "vision_approach_close_range_m": 3.0,
        "vision_approach_far_range_m": 10.0,
        "vision_gate_center_alt_kp_m": 2.0,
        "vision_gate_center_alt_limit_m": 1.0,
    }
    controller = AttitudeController(gains)
    gate = GateTrack(
        frame_id=1,
        timestamp_s=1.0,
        center_px=(320.0, 220.0),
        bbox=(200, 80, 140, 140),
        confidence=0.9,
        area_fraction=0.08,
        visible=True,
    )
    est = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=gate,
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.12,
        gate_range_m=6.0,
        gate_confidence=0.9,
    )
    approach = RacePlan("APPROACH_GATE", 2.0, commit=False)
    commit = RacePlan("COMMIT_GATE", 2.0, commit=True)
    base = 1.8
    approach_target = controller._vision_hover_altitude_target(est, approach, base)
    commit_target = controller._vision_hover_altitude_target(est, commit, base)
    assert approach_target < base
    assert commit_target < approach_target
    assert controller._vision_gate_center_alt_offset_m(est) > 0.0


def test_lower_gate_altitude_offset_stays_aggressive_at_range():
    gains = {
        **GAINS,
        "vision_primary_navigation": True,
        "vision_gate_center_alt_kp_m": 2.6,
        "vision_gate_center_alt_limit_m": 1.2,
        "vision_lower_gate_extra_alt_gain": 1.1,
        "vision_approach_close_range_m": 3.5,
    }
    controller = AttitudeController(gains)
    est = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=GateTrack(1, 1.0, (320.0, 220.0), (0, 0, 100, 100), 0.9, 0.08, True),
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.0,
        gate_bearing_y_rad=0.18,
        gate_range_m=8.0,
        gate_confidence=0.9,
    )

    offset = controller._vision_gate_center_alt_offset_m(est)

    assert offset > 0.45


def test_lower_gate_extra_drop_applies_in_align():
    gains = {
        **GAINS,
        "vision_primary_navigation": True,
        "vision_lower_gate_extra_drop_per_rad_m": 2.2,
        "vision_lower_gate_extra_drop_max_m": 0.55,
        "vision_lower_gate_align_drop_scale": 1.2,
    }
    controller = AttitudeController(gains)
    est = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=GateTrack(1, 1.0, (320.0, 220.0), (0, 0, 100, 100), 0.9, 0.08, True),
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.0,
        gate_bearing_y_rad=0.22,
        gate_confidence=0.9,
    )

    align = RacePlan("ALIGN_GATE", 0.8)
    approach = RacePlan("APPROACH_GATE", 1.4)

    assert controller._vision_lower_gate_extra_drop_m(est, align) > 0.25
    assert controller._vision_lower_gate_extra_drop_m(est, align) > controller._vision_lower_gate_extra_drop_m(est, approach)


def test_lower_gate_extra_drop_lowers_hover_target_further():
    gains = {
        **GAINS,
        "vision_primary_navigation": True,
        "vision_approach_alt_drop_m": 0.6,
        "vision_approach_min_alt_m": 0.35,
        "vision_gate_center_alt_kp_m": 2.0,
        "vision_gate_center_alt_limit_m": 1.0,
        "vision_lower_gate_extra_drop_per_rad_m": 2.2,
        "vision_lower_gate_extra_drop_max_m": 0.55,
        "vision_lower_gate_align_drop_scale": 1.2,
    }
    controller = AttitudeController(gains)
    low = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=GateTrack(1, 1.0, (320.0, 220.0), (0, 0, 100, 100), 0.9, 0.08, True),
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.22,
        gate_range_m=7.0,
        gate_confidence=0.9,
    )
    mild = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=GateTrack(1, 1.0, (320.0, 220.0), (0, 0, 100, 100), 0.9, 0.08, True),
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.10,
        gate_range_m=7.0,
        gate_confidence=0.9,
    )

    low_target = controller._vision_hover_altitude_target(low, RacePlan("ALIGN_GATE", 0.8), 1.4)
    mild_target = controller._vision_hover_altitude_target(mild, RacePlan("ALIGN_GATE", 0.8), 1.4)

    assert low_target < mild_target


def test_lower_gate_slew_rate_descends_faster():
    gains = {
        **GAINS,
        "vision_primary_navigation": True,
        "vision_gate_center_alt_kp_m": 2.0,
        "vision_gate_center_alt_limit_m": 1.0,
        "vision_lower_gate_extra_drop_per_rad_m": 2.2,
        "vision_lower_gate_extra_drop_max_m": 0.55,
        "vision_lower_gate_extra_alt_slew_per_rad_mps": 14.0,
        "vision_lower_gate_extra_alt_slew_max_mps": 3.0,
        "vision_gate_alt_slew_rate_mps": 2.0,
    }
    controller = AttitudeController(gains)
    controller._hover_altitude_m = 1.0
    controller._vision_base_hover_altitude_m = 1.0
    low = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=GateTrack(1, 1.0, (320.0, 220.0), (0, 0, 100, 100), 0.9, 0.08, True),
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.24,
        gate_range_m=7.0,
        gate_confidence=0.9,
    )
    mild = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=GateTrack(1, 1.0, (320.0, 220.0), (0, 0, 100, 100), 0.9, 0.08, True),
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.10,
        gate_range_m=7.0,
        gate_confidence=0.9,
    )
    plan = RacePlan("ALIGN_GATE", 0.8)

    controller._update_vision_hover_altitude(mild, 1.0 / 60.0, plan)
    mild_alt = controller._hover_altitude_m
    controller._hover_altitude_m = 1.0
    controller._vision_base_hover_altitude_m = 1.0
    controller._update_vision_hover_altitude(low, 1.0 / 60.0, plan)
    low_alt = controller._hover_altitude_m

    assert low_alt < mild_alt


def test_lower_gate_pitch_command_is_more_aggressive():
    gains = {
        **GAINS,
        "vision_primary_navigation": True,
        "vision_align_pitch_limit_rad": 0.22,
        "vision_lower_gate_pitch_boost_per_rad": 0.75,
        "vision_lower_gate_pitch_boost_max_rad": 0.16,
        "vision_lower_gate_align_pitch_boost_scale": 1.25,
        "vision_lower_gate_extra_pitch_limit_per_rad": 1.10,
        "vision_lower_gate_extra_pitch_limit_rad": 0.12,
        "forward_flight_pitch_kp": 0.80,
    }
    controller = AttitudeController(gains)
    controller._hover_pitch_sp = 0.31
    controller._launch_pitch_captured = True
    controller._hover_altitude_m = 1.0
    controller._altitude_captured = True
    vehicle = VehicleState(
        yaw_rad=0.0,
        pitch_rad=0.31,
        velocity_ned_mps=(0.2, 0.0, 0.0),
        position_ned_m=(0.0, 0.0, -1.0),
    )
    low = EstimatedState(
        now_s=1.0,
        vehicle=vehicle,
        gate_track=GateTrack(1, 1.0, (320.0, 220.0), (0, 0, 100, 100), 0.9, 0.08, True),
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.24,
        gate_confidence=0.9,
        gate_range_m=7.0,
    )
    mild = EstimatedState(
        now_s=1.0,
        vehicle=vehicle,
        gate_track=GateTrack(1, 1.0, (320.0, 220.0), (0, 0, 100, 100), 0.9, 0.08, True),
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.10,
        gate_confidence=0.9,
        gate_range_m=7.0,
    )
    plan = RacePlan("ALIGN_GATE", 0.8)

    low_cmd = controller.update(low, plan, dt=1.0 / 60.0)
    controller.reset_transient()
    controller._hover_pitch_sp = 0.31
    controller._launch_pitch_captured = True
    controller._hover_altitude_m = 1.0
    controller._altitude_captured = True
    mild_cmd = controller.update(mild, plan, dt=1.0 / 60.0)

    assert low_cmd.pitch_rad > mild_cmd.pitch_rad


def test_lower_gate_descent_thrust_cut_scales_with_vertical_error():
    gains = {
        **GAINS,
        "vision_primary_navigation": True,
        "vision_gate_descent_thrust_cut": 0.10,
        "vision_gate_descent_extra_thrust_cut_per_rad": 0.22,
        "vision_lower_gate_align_extra_thrust_cut_per_rad": 0.14,
        "vision_gate_descent_alt_error_cut_gain": 0.05,
        "vision_gate_descent_thrust_cut_max_extra": 0.08,
        "vision_gate_descent_thrust_cut_max": 0.20,
    }
    controller = AttitudeController(gains)
    controller._hover_altitude_m = 0.55
    controller._altitude_captured = True
    vehicle = VehicleState(
        position_ned_m=(0.0, 0.0, -0.95),
        velocity_ned_mps=(0.0, 0.0, 0.0),
    )
    plan = RacePlan("ALIGN_GATE", 0.8)
    mild = EstimatedState(
        now_s=1.0,
        vehicle=vehicle,
        gate_track=GateTrack(1, 1.0, (320.0, 220.0), (0, 0, 100, 100), 0.9, 0.08, True),
        link_ready=True,
        vision_ready=True,
        gate_bearing_y_rad=0.10,
        gate_confidence=0.9,
    )
    low = EstimatedState(
        now_s=1.0,
        vehicle=vehicle,
        gate_track=GateTrack(1, 1.0, (320.0, 220.0), (0, 0, 100, 100), 0.9, 0.08, True),
        link_ready=True,
        vision_ready=True,
        gate_bearing_y_rad=0.28,
        gate_confidence=0.9,
    )

    mild_thrust = controller._vision_altitude_hold_thrust(
        vehicle, 0.0, 1.0 / 60.0, mild, plan, using_map=False
    )
    low_thrust = controller._vision_altitude_hold_thrust(
        vehicle, 0.0, 1.0 / 60.0, low, plan, using_map=False
    )

    assert low_thrust < mild_thrust


def test_reset_for_next_gate_clears_vision_gate_state():
    controller = AttitudeController({**GAINS, "vision_primary_navigation": True})
    controller._vision_base_hover_altitude_m = 1.8
    controller._altitude_captured = True
    controller._commit_initialized = True
    controller._align_yaw_hold_initialized = True

    controller.reset_for_next_gate()

    assert controller._vision_base_hover_altitude_m is None
    assert controller._altitude_captured is False
    assert controller._commit_initialized is False
    assert controller._align_yaw_hold_initialized is False


def test_map_commit_speed_target_uses_guidance_but_respects_cap():
    controller = AttitudeController({**GAINS, "map_commit_speed_cap_mps": 3.5})
    est = EstimatedState(
        now_s=1.0,
        vehicle=VehicleState(),
        gate_track=None,
        link_ready=True,
        vision_ready=True,
        has_track_map=True,
        race_started=True,
        map_within_gate_bounds=True,
        map_commit_speed_mps=5.8,
    )
    plan = RacePlan("COMMIT_GATE", 2.2, commit=True, use_map_navigation=True)

    speed = controller._gate_speed_target_mps(est, plan, using_map=True)

    assert speed == 3.5


def test_vision_align_braking_can_pitch_up_past_launch_trim_when_fast():
    gains = {
        **GAINS,
        "vision_primary_navigation": True,
        "vision_align_speed_brake_mps": 1.8,
        "vision_brake_pitch_up_limit_rad": 0.30,
        "search_vx_kp": 0.38,
        "search_pitch_trim_rad": 0.04,
        "search_brake_pitch_limit_rad": 0.34,
        "vision_align_pitch_limit_rad": 0.24,
    }
    controller = AttitudeController(gains)
    controller._hover_pitch_sp = 0.31
    controller._launch_pitch_captured = True
    controller._hover_altitude_m = 2.0
    controller._altitude_captured = True
    vehicle = VehicleState(
        yaw_rad=0.0,
        pitch_rad=0.31,
        velocity_ned_mps=(2.6, 0.0, 0.0),
        position_ned_m=(0.0, 0.0, -2.0),
    )
    est = EstimatedState(
        now_s=1.0,
        vehicle=vehicle,
        gate_track=GateTrack(
            frame_id=1,
            timestamp_s=1.0,
            center_px=(320.0, 260.0),
            bbox=(200, 100, 140, 140),
            confidence=0.9,
            area_fraction=0.12,
            visible=True,
        ),
        link_ready=True,
        vision_ready=True,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.04,
        gate_confidence=0.9,
        gate_range_m=20.0,
    )

    cmd = controller.update(est, RacePlan("ALIGN_GATE", 0.8), dt=1.0 / 60.0)

    assert cmd.pitch_rad < 0.20
