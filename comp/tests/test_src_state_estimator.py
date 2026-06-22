import time

from src.estimation.state_estimator import StateEstimator
from src.types import GateTrack, VehicleState


CAMERA = {
    "cx": 320.0,
    "desired_v_px": 180.0,
    "fx": 320.0,
    "fy": 320.0,
    "gate_inner_size_m": 1.5,
}


SIM = {
    "link_timeout_s": 1.0,
    "gate_timeout_s": 0.35,
    "detect_confidence": 0.5,
    "map_max_range_m": 80.0,
    "use_map_first_gate": True,
    "vision_primary_navigation": False,
}


def test_state_estimator_uses_track_index_fallback_for_first_gate():
    est = StateEstimator(CAMERA, SIM)
    vehicle = VehicleState(
        heartbeat_wall_time_s=9.9,
        armed=True,
        yaw_rad=0.0,
        position_ned_m=(0.0, 0.0, 0.0),
    )
    race_status = {
        "race_start_boot_time_ms": 100,
        "race_finish_time_ns": -1,
        "active_gate_index": 0,
        "updated_wall_time": time.time(),
    }
    track_gates = [
        {"gate_id": 7, "position_ned": [20.0, 0.0, 0.0], "width": 2.7, "height": 2.7},
        {"gate_id": 8, "position_ned": [40.0, 0.0, 0.0], "width": 2.7, "height": 2.7},
    ]

    state = est.estimate(
        10.0,
        vehicle,
        gate_track=None,
        vision_ready=True,
        race_status=race_status,
        track_gates=track_gates,
    )

    assert state.race_started is True
    assert state.has_track_map is True
    assert state.active_gate_index == 0
    assert state.gate_range_m is not None
    assert abs(state.gate_bearing_x_rad) < 1e-6
    assert state.map_commit_range_m is not None
    assert state.map_commit_range_m > state.gate_range_m


def test_state_estimator_ignores_visual_on_first_gate_when_map_available():
    est = StateEstimator(CAMERA, SIM)
    vehicle = VehicleState(
        heartbeat_wall_time_s=9.9,
        armed=True,
        yaw_rad=0.0,
        position_ned_m=(0.0, 0.0, 0.0),
    )
    race_status = {
        "race_start_boot_time_ms": 100,
        "race_finish_time_ns": -1,
        "active_gate_index": 0,
        "updated_wall_time": time.time(),
    }
    track_gates = [
        {"gate_id": 0, "position_ned": [20.0, 0.0, 0.0], "width": 2.7, "height": 2.7},
    ]
    track = GateTrack(
        frame_id=1,
        timestamp_s=10.0,
        center_px=(500.0, 180.0),
        bbox=(450, 150, 80, 80),
        confidence=0.99,
        area_fraction=0.05,
        visible=True,
    )

    state = est.estimate(
        10.0,
        vehicle,
        gate_track=track,
        vision_ready=True,
        race_status=race_status,
        track_gates=track_gates,
    )

    assert state.has_track_map is True
    assert state.active_gate_index == 0
    assert abs(state.gate_bearing_x_rad) < 1e-6


def test_state_estimator_ignores_visual_on_later_gates_when_map_available():
    est = StateEstimator(CAMERA, SIM)
    vehicle = VehicleState(
        heartbeat_wall_time_s=9.9,
        armed=True,
        yaw_rad=0.0,
        position_ned_m=(25.0, 0.0, 0.0),
    )
    race_status = {
        "race_start_boot_time_ms": 100,
        "race_finish_time_ns": -1,
        "active_gate_index": 1,
        "updated_wall_time": time.time(),
    }
    track_gates = [
        {"gate_id": 0, "position_ned": [20.0, 0.0, 0.0], "width": 2.7, "height": 2.7},
        {"gate_id": 1, "position_ned": [40.0, 0.0, 0.0], "width": 2.7, "height": 2.7},
    ]
    track = GateTrack(
        frame_id=1,
        timestamp_s=10.0,
        center_px=(500.0, 180.0),
        bbox=(450, 150, 80, 80),
        confidence=0.99,
        area_fraction=0.05,
        visible=True,
    )

    state = est.estimate(
        10.0,
        vehicle,
        gate_track=track,
        vision_ready=True,
        race_status=race_status,
        track_gates=track_gates,
    )

    assert state.has_track_map is True
    assert state.active_gate_index == 1
    # Gate 1 is straight ahead on +X; map bearing ~0, not vision offset ~0.5 rad.
    assert abs(state.gate_bearing_x_rad) < 0.1


def test_state_estimator_vision_primary_ignores_track_map():
    est = StateEstimator(CAMERA, SIM | {"vision_primary_navigation": True, "use_map_first_gate": False})
    vehicle = VehicleState(
        heartbeat_wall_time_s=9.9,
        armed=True,
        yaw_rad=0.0,
        position_ned_m=(0.0, 0.0, 0.0),
    )
    race_status = {
        "race_start_boot_time_ms": 100,
        "race_finish_time_ns": -1,
        "active_gate_index": 0,
        "updated_wall_time": time.time(),
    }
    track_gates = [
        {"gate_id": 0, "position_ned": [20.0, 0.0, 0.0], "width": 2.7, "height": 2.7},
    ]
    track = GateTrack(
        frame_id=1,
        timestamp_s=10.0,
        center_px=(500.0, 180.0),
        bbox=(450, 150, 80, 80),
        confidence=0.99,
        area_fraction=0.05,
        visible=True,
    )

    state = est.estimate(
        10.0,
        vehicle,
        gate_track=track,
        vision_ready=True,
        race_status=race_status,
        track_gates=track_gates,
    )

    assert state.active_gate_index == 0
    assert state.has_track_map is False
    assert state.map_gate_range_m is None
    assert state.gate_bearing_x_rad > 0.15


def test_state_estimator_reports_gate_plane_metrics():
    est = StateEstimator(CAMERA, SIM | {"map_through_distance_m": 3.0})
    vehicle = VehicleState(
        heartbeat_wall_time_s=9.9,
        armed=True,
        yaw_rad=0.0,
        position_ned_m=(18.5, 0.0, 0.0),
    )
    race_status = {
        "race_start_boot_time_ms": 100,
        "race_finish_time_ns": -1,
        "active_gate_index": 0,
        "updated_wall_time": time.time(),
    }
    track_gates = [
        {
            "gate_id": 0,
            "position_ned": [20.0, 0.0, 0.0],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "width": 2.7,
            "height": 2.7,
        },
        {"gate_id": 1, "position_ned": [40.0, 0.0, 0.0], "width": 2.7, "height": 2.7},
    ]

    state = est.estimate(
        10.0,
        vehicle,
        gate_track=None,
        vision_ready=True,
        race_status=race_status,
        track_gates=track_gates,
    )

    assert state.map_plane_signed_m is not None
    assert state.map_plane_signed_m < 0.0
    assert state.map_within_gate_bounds is True
    assert state.map_gate_commit_active is True
    assert state.map_gate_commit_strength >= 0.40


def test_state_estimator_clears_stale_visual_range():
    est = StateEstimator(CAMERA, SIM | {"vision_primary_navigation": True, "use_map_first_gate": False})
    vehicle = VehicleState(
        heartbeat_wall_time_s=9.9,
        armed=True,
        yaw_rad=0.0,
        position_ned_m=(0.0, 0.0, 0.0),
    )
    track = GateTrack(
        frame_id=1,
        timestamp_s=10.0,
        center_px=(420.0, 200.0),
        bbox=(360, 140, 120, 120),
        confidence=0.9,
        area_fraction=0.08,
        visible=True,
    )

    fresh = est.estimate(
        10.0,
        vehicle,
        gate_track=track,
        vision_ready=True,
    )
    stale = est.estimate(
        10.5,
        vehicle,
        gate_track=None,
        vision_ready=True,
    )

    assert fresh.gate_range_m is not None
    assert stale.gate_range_m is None
    assert stale.gate_range_x_m is None
    assert stale.gate_range_y_m is None
    assert stale.gate_confidence < fresh.gate_confidence


def test_state_estimator_reset_for_next_gate_clears_visual_filters():
    est = StateEstimator(CAMERA, SIM | {"vision_primary_navigation": True, "use_map_first_gate": False})
    vehicle = VehicleState(
        heartbeat_wall_time_s=9.9,
        armed=True,
        yaw_rad=0.0,
        position_ned_m=(0.0, 0.0, 0.0),
    )
    track = GateTrack(
        frame_id=1,
        timestamp_s=10.0,
        center_px=(500.0, 210.0),
        bbox=(430, 130, 140, 140),
        confidence=0.95,
        area_fraction=0.10,
        visible=True,
    )

    seeded = est.estimate(
        10.0,
        vehicle,
        gate_track=track,
        vision_ready=True,
    )
    est.reset_for_next_gate()
    cleared = est.estimate(
        10.1,
        vehicle,
        gate_track=None,
        vision_ready=True,
    )

    assert seeded.gate_confidence > 0.0
    assert cleared.gate_confidence == 0.0
    assert cleared.gate_bearing_x_rad == 0.0
    assert cleared.gate_bearing_y_rad == 0.0
    assert cleared.gate_range_m is None
