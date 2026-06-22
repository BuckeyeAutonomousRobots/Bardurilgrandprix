from src.planning.race_fsm import RaceFSM
from src.types import EstimatedState, GateTrack, VehicleState


SIM_CONFIG = {
    "gate_timeout_s": 0.35,
    "detect_confidence": 0.18,
    "stable_frames_required": 2,
    "search_min_time_s": 0.0,
    "search_min_area_fraction": 0.0,
    "align_bearing_rad": 0.16,
    "align_vertical_rad": 0.16,
    "approach_bearing_rad": 0.10,
    "map_align_bearing_rad": 0.10,
    "map_align_vertical_rad": 0.12,
    "map_approach_bearing_rad": 0.06,
    "commit_range_m": 4.5,
    "map_commit_range_m": 5.5,
    "commit_area_fraction": 0.10,
    "pass_timeout_s": 0.55,
    "recover_timeout_s": 0.40,
    "stabilize_time_s": 1.0,
    "use_map_first_gate": True,
    "search_speed_mps": 0.0,
    "align_speed_mps": 0.8,
    "approach_speed_mps": 1.6,
    "commit_speed_mps": 2.4,
    "search_yaw_rate_rps": 0.35,
    "recover_yaw_rate_rps": 0.25,
    "takeoff_max_time_s": 0.5,
}


def test_fsm_reaches_commit_gate():
    fsm = RaceFSM(SIM_CONFIG)
    vehicle = VehicleState(heartbeat_wall_time_s=1.0, armed=True)
    est = EstimatedState(now_s=1.0, vehicle=vehicle, gate_track=None, link_ready=True, vision_ready=False)
    assert fsm.update(est).state == "WAIT_VISION"
    est.now_s = 1.1
    est.vision_ready = True
    assert fsm.update(est).state == "WAIT_START"
    est.now_s = 1.2
    est.race_started = True
    assert fsm.update(est).state == "TAKEOFF"
    est.now_s = 2.2
    assert fsm.update(est).state == "STABILIZE"
    est.now_s = 3.2
    assert fsm.update(est).state == "SEARCH_GATE"
    gate = GateTrack(frame_id=1, timestamp_s=3.3, center_px=(320.0, 290.0), bbox=(200, 100, 140, 140), confidence=0.9, area_fraction=0.12, visible=True)
    est.gate_track = gate
    est.gate_confidence = 0.9
    est.gate_bearing_x_rad = 0.02
    est.gate_bearing_y_rad = 0.02
    est.gate_range_m = 4.0
    est.now_s = 3.3
    fsm.update(est)
    est.now_s = 3.35
    fsm.update(est)
    est.now_s = 3.4
    fsm.update(est)  # transitions to ALIGN_GATE
    est.now_s = 3.45
    fsm.update(est)  # transitions to ALIGN_GATE
    est.now_s = 3.5
    plan = fsm.update(est)  # transitions to COMMIT_GATE
    assert plan.state == "COMMIT_GATE"


def test_fsm_vision_primary_disables_map_navigation_plan():
    fsm = RaceFSM(SIM_CONFIG | {"vision_primary_navigation": True, "use_map_first_gate": False})
    vehicle = VehicleState(heartbeat_wall_time_s=1.0, armed=True)
    est = EstimatedState(
        now_s=3.2,
        vehicle=vehicle,
        gate_track=None,
        link_ready=True,
        vision_ready=True,
        race_started=True,
        has_track_map=True,
        active_gate_index=0,
    )
    fsm.state = "ALIGN_GATE"
    plan = fsm.update(est)
    assert plan.use_map_navigation is False


def test_fsm_map_first_gate_passes_on_gate_index_advance():
    fsm = RaceFSM(SIM_CONFIG)
    vehicle = VehicleState(heartbeat_wall_time_s=1.0, armed=True)
    est = EstimatedState(
        now_s=1.0,
        vehicle=vehicle,
        gate_track=None,
        link_ready=True,
        vision_ready=True,
        race_started=True,
        has_track_map=True,
        active_gate_index=0,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.01,
        gate_range_m=6.0,
    )
    assert fsm.update(est).state == "WAIT_VISION"
    est.now_s = 1.1
    assert fsm.update(est).state == "WAIT_START"
    est.now_s = 1.2
    assert fsm.update(est).state == "TAKEOFF"
    est.now_s = 2.3
    plan = fsm.update(est)
    assert plan.state == "STABILIZE"
    est.now_s = 3.3
    plan = fsm.update(est)
    assert plan.state == "ALIGN_GATE"
    assert plan.use_map_navigation is True
    est.now_s = 3.4
    est.gate_range_m = 5.0
    plan = fsm.update(est)
    assert plan.state == "APPROACH_GATE"
    est.now_s = 3.45
    est.gate_range_m = 3.0
    est.map_dist_center_m = 2.5
    est.map_within_gate_bounds = True
    est.map_lateral_error_m = 0.1
    est.map_vertical_error_m = 0.1
    plan = fsm.update(est)
    assert plan.state == "COMMIT_GATE"
    est.now_s = 3.5
    est.active_gate_index = 1
    plan = fsm.update(est)
    assert plan.state == "PASS_GATE"


def test_fsm_map_commit_passes_when_at_plane_near_center():
    fsm = RaceFSM(SIM_CONFIG | {"map_commit_fail_time_s": 1.25, "map_post_gate_margin_m": 1.5, "map_pass_min_plane_m": 0.0, "gate_pass_radius_m": 1.5})
    vehicle = VehicleState(heartbeat_wall_time_s=1.0, armed=True)
    est = EstimatedState(
        now_s=1.0,
        vehicle=vehicle,
        gate_track=None,
        link_ready=True,
        vision_ready=True,
        race_started=True,
        has_track_map=True,
        active_gate_index=0,
        gate_bearing_x_rad=0.01,
        gate_bearing_y_rad=0.01,
        gate_range_m=5.0,
        map_plane_signed_m=-0.5,
        map_within_gate_bounds=True,
        map_dist_center_m=5.0,
    )
    assert fsm.update(est).state == "WAIT_VISION"
    est.now_s = 1.1
    assert fsm.update(est).state == "WAIT_START"
    est.now_s = 1.2
    assert fsm.update(est).state == "TAKEOFF"
    est.now_s = 2.3
    assert fsm.update(est).state == "STABILIZE"
    est.now_s = 3.3
    assert fsm.update(est).state == "ALIGN_GATE"
    est.now_s = 3.4
    assert fsm.update(est).state == "APPROACH_GATE"
    est.now_s = 3.45
    est.gate_range_m = 3.0
    est.map_dist_center_m = 2.5
    est.map_within_gate_bounds = True
    est.map_lateral_error_m = 0.1
    est.map_vertical_error_m = 0.1
    assert fsm.update(est).state == "COMMIT_GATE"
    est.now_s = 3.7
    est.gate_range_m = 0.4
    est.map_dist_center_m = 0.4
    est.map_plane_signed_m = -0.5
    assert fsm.update(est).state == "COMMIT_GATE"
    est.map_plane_signed_m = 0.0
    est.map_within_gate_bounds = True
    assert fsm.update(est).state == "PASS_GATE"


def test_vision_takeoff_skips_stabilize_when_gate_visible():
    cfg = {
        **SIM_CONFIG,
        "vision_primary_navigation": True,
        "use_map_first_gate": False,
        "vision_takeoff_altitude_m": 1.25,
        "vision_takeoff_min_alt_m": 0.85,
        "vision_takeoff_min_time_s": 0.2,
        "vision_takeoff_skip_stabilize": True,
        "takeoff_max_time_s": 3.0,
        "search_min_area_fraction": 0.008,
    }
    fsm = RaceFSM(cfg)
    gate = GateTrack(
        frame_id=1,
        timestamp_s=1.0,
        center_px=(320.0, 290.0),
        bbox=(200, 100, 140, 140),
        confidence=0.9,
        area_fraction=0.12,
        visible=True,
    )
    vehicle = VehicleState(
        heartbeat_wall_time_s=1.0,
        armed=True,
        position_ned_m=(0.0, 0.0, -0.9),
        velocity_ned_mps=(0.0, 0.0, -0.2),
    )
    est = EstimatedState(
        now_s=1.0,
        vehicle=vehicle,
        gate_track=gate,
        link_ready=True,
        vision_ready=True,
        race_started=True,
        gate_confidence=0.9,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.02,
    )
    fsm.state = "TAKEOFF"
    fsm._state_since_s = 0.7
    plan = fsm.update(est)
    assert plan.state == "ALIGN_GATE"


def test_vision_takeoff_exits_on_timeout_with_launch_pitch():
    cfg = {
        **SIM_CONFIG,
        "vision_primary_navigation": True,
        "use_map_first_gate": False,
        "vision_takeoff_max_time_s": 0.4,
        "vision_takeoff_min_time_s": 0.1,
        "vision_takeoff_skip_stabilize": True,
        "vision_takeoff_attitude_roll_pitch_rad": 0.48,
    }
    fsm = RaceFSM(cfg)
    vehicle = VehicleState(
        heartbeat_wall_time_s=1.0,
        armed=True,
        roll_rad=0.0,
        pitch_rad=0.41,
        position_ned_m=(0.0, 0.0, -1.0),
        velocity_ned_mps=(0.0, 0.0, -0.1),
    )
    est = EstimatedState(
        now_s=1.0,
        vehicle=vehicle,
        gate_track=None,
        link_ready=True,
        vision_ready=True,
        race_started=True,
    )
    fsm.state = "TAKEOFF"
    fsm._state_since_s = 0.5
    plan = fsm.update(est)
    assert plan.state in {"ALIGN_GATE", "SEARCH_GATE", "STABILIZE"}


def test_recover_timeout_returns_to_stabilize_when_unstable():
    fsm = RaceFSM(SIM_CONFIG)
    fsm.state = "RECOVER"
    fsm._state_since_s = 1.0
    vehicle = VehicleState(
        heartbeat_wall_time_s=10.0,
        armed=True,
        roll_rad=0.5,
        velocity_ned_mps=(3.0, 0.0, 0.0),
    )
    est = EstimatedState(
        now_s=2.0,
        vehicle=vehicle,
        gate_track=None,
        link_ready=True,
        vision_ready=True,
        race_started=True,
    )
    assert fsm.update(est).state == "STABILIZE"


def test_fsm_vision_align_tolerates_predicted_gate_dropout():
    fsm = RaceFSM(
        SIM_CONFIG
        | {
            "vision_primary_navigation": True,
            "use_map_first_gate": False,
            "vision_align_recover_timeout_s": 1.5,
            "gate_dropout_s": 0.55,
            "vision_gate_timeout_s": 0.65,
        }
    )
    fsm.state = "ALIGN_GATE"
    fsm._state_since_s = 1.0
    vehicle = VehicleState(heartbeat_wall_time_s=1.0, armed=True)
    gate = GateTrack(
        frame_id=1,
        timestamp_s=1.0,
        center_px=(320.0, 290.0),
        bbox=(200, 100, 140, 140),
        confidence=0.85,
        area_fraction=0.12,
        visible=False,
        predicted=True,
        missed_frames=3,
    )
    est = EstimatedState(
        now_s=1.4,
        vehicle=vehicle,
        gate_track=gate,
        link_ready=True,
        vision_ready=True,
        race_started=True,
        gate_confidence=0.85,
        gate_bearing_x_rad=0.12,
        gate_bearing_y_rad=0.08,
        gate_range_m=6.0,
    )
    plan = fsm.update(est)
    assert plan.state in {"ALIGN_GATE", "APPROACH_GATE"}
    est.now_s = 1.9
    plan = fsm.update(est)
    assert plan.state in {"ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE"}
    assert plan.state != "RECOVER"


def test_fsm_vision_align_overspeed_skips_recover_when_gate_locked():
    """Reproduces log: high smoothed confidence + predicted dropout should not RECOVER."""
    fsm = RaceFSM(
        SIM_CONFIG
        | {
            "vision_primary_navigation": True,
            "use_map_first_gate": False,
            "vision_recover_speed_mps": 3.0,
            "vision_overspeed_min_time_s": 0.2,
            "gate_dropout_s": 1.0,
            "vision_gate_timeout_s": 0.65,
        }
    )
    fsm.state = "ALIGN_GATE"
    fsm._state_since_s = 1.0
    vehicle = VehicleState(
        heartbeat_wall_time_s=1.0,
        armed=True,
        velocity_ned_mps=(-5.77, -0.1, -0.02),
    )
    gate = GateTrack(
        frame_id=1,
        timestamp_s=1.0,
        center_px=(320.0, 280.0),
        bbox=(200, 100, 140, 140),
        confidence=0.9,
        area_fraction=0.12,
        visible=False,
        predicted=True,
        missed_frames=2,
    )
    est = EstimatedState(
        now_s=1.25,
        vehicle=vehicle,
        gate_track=gate,
        link_ready=True,
        vision_ready=True,
        race_started=True,
        gate_confidence=0.94,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.02,
        gate_range_m=6.0,
    )
    plan = fsm.update(est)
    assert plan.state != "RECOVER"
    assert plan.state in {"ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE"}


def test_fsm_vision_align_overspeed_enters_recover_without_gate_lock():
    fsm = RaceFSM(
        SIM_CONFIG
        | {
            "vision_primary_navigation": True,
            "use_map_first_gate": False,
            "vision_recover_speed_mps": 3.0,
            "vision_overspeed_min_time_s": 0.2,
            "detect_confidence": 0.22,
        }
    )
    fsm.state = "ALIGN_GATE"
    fsm._state_since_s = 1.0
    vehicle = VehicleState(
        heartbeat_wall_time_s=1.0,
        armed=True,
        velocity_ned_mps=(3.2, 0.0, -0.2),
    )
    est = EstimatedState(
        now_s=1.25,
        vehicle=vehicle,
        gate_track=GateTrack(
            frame_id=1,
            timestamp_s=1.25,
            center_px=(320.0, 280.0),
            bbox=(200, 100, 140, 140),
            confidence=0.05,
            area_fraction=0.002,
            visible=True,
        ),
        link_ready=True,
        vision_ready=True,
        race_started=True,
        gate_confidence=0.05,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.02,
        gate_range_m=6.0,
    )
    plan = fsm.update(est)
    assert plan.state == "RECOVER"


def test_fsm_recover_waits_for_speed_and_dwell_before_realign():
    fsm = RaceFSM(
        SIM_CONFIG
        | {
            "vision_primary_navigation": True,
            "use_map_first_gate": False,
            "stable_frames_required": 2,
            "vision_recover_min_time_s": 0.3,
            "vision_recover_align_max_speed_mps": 1.2,
        }
    )
    fsm.state = "RECOVER"
    fsm._state_since_s = 1.0
    fsm._stable_frames = 2
    vehicle = VehicleState(
        heartbeat_wall_time_s=1.0,
        armed=True,
        velocity_ned_mps=(1.4, 0.0, 0.0),
    )
    est = EstimatedState(
        now_s=1.2,
        vehicle=vehicle,
        gate_track=GateTrack(
            frame_id=1,
            timestamp_s=1.2,
            center_px=(320.0, 280.0),
            bbox=(200, 100, 140, 140),
            confidence=0.9,
            area_fraction=0.12,
            visible=True,
        ),
        link_ready=True,
        vision_ready=True,
        race_started=True,
        gate_confidence=0.9,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.02,
        gate_range_m=6.0,
    )
    plan = fsm.update(est)
    assert plan.state == "RECOVER"
    est.now_s = 1.35
    est.vehicle.velocity_ned_mps = (1.0, 0.0, 0.0)
    plan = fsm.update(est)
    assert plan.state == "ALIGN_GATE"


def test_fsm_recover_requires_attitude_and_vertical_stability_before_realign():
    fsm = RaceFSM(
        SIM_CONFIG
        | {
            "vision_primary_navigation": True,
            "use_map_first_gate": False,
            "stable_frames_required": 2,
            "vision_recover_min_time_s": 0.3,
            "vision_recover_align_max_speed_mps": 1.2,
            "stabilize_max_vz_mps": 0.3,
        }
    )
    fsm.state = "RECOVER"
    fsm._state_since_s = 1.0
    fsm._stable_frames = 2
    vehicle = VehicleState(
        heartbeat_wall_time_s=1.0,
        armed=True,
        pitch_rad=0.35,
        velocity_ned_mps=(0.8, 0.0, 0.4),
    )
    est = EstimatedState(
        now_s=1.4,
        vehicle=vehicle,
        gate_track=GateTrack(
            frame_id=1,
            timestamp_s=1.4,
            center_px=(320.0, 280.0),
            bbox=(200, 100, 140, 140),
            confidence=0.9,
            area_fraction=0.12,
            visible=True,
        ),
        link_ready=True,
        vision_ready=True,
        race_started=True,
        gate_confidence=0.9,
        gate_bearing_x_rad=0.02,
        gate_bearing_y_rad=0.02,
        gate_range_m=6.0,
    )
    plan = fsm.update(est)
    assert plan.state == "RECOVER"
    est.vehicle.pitch_rad = 0.0
    est.vehicle.velocity_ned_mps = (0.8, 0.0, 0.0)
    est.now_s = 1.5
    plan = fsm.update(est)
    assert plan.state == "ALIGN_GATE"


def test_fsm_vision_stabilize_requires_partial_alignment_before_align():
    fsm = RaceFSM(
        SIM_CONFIG
        | {
            "vision_primary_navigation": True,
            "use_map_first_gate": False,
            "vision_stabilize_time_s": 0.2,
        }
    )
    fsm.state = "STABILIZE"
    fsm._state_since_s = 1.0
    vehicle = VehicleState(
        heartbeat_wall_time_s=1.0,
        armed=True,
        velocity_ned_mps=(0.2, 0.0, 0.0),
    )
    est = EstimatedState(
        now_s=1.3,
        vehicle=vehicle,
        gate_track=GateTrack(
            frame_id=1,
            timestamp_s=1.3,
            center_px=(320.0, 280.0),
            bbox=(200, 100, 140, 140),
            confidence=0.9,
            area_fraction=0.12,
            visible=True,
        ),
        link_ready=True,
        vision_ready=True,
        race_started=True,
        gate_confidence=0.9,
        gate_bearing_x_rad=0.50,
        gate_bearing_y_rad=0.02,
    )
    plan = fsm.update(est)
    assert plan.state == "SEARCH_GATE"
