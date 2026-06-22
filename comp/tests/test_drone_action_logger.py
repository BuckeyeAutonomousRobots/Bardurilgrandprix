import json
import time

import pytest

from src.infra.drone_action_logger import DroneActionLogger
from src.types import AttitudeCommand, FSMTransition, RacePlan, VehicleState


def test_action_logger_records_before_and_after_position(tmp_path):
    log_path = tmp_path / "actions.jsonl"
    journal = DroneActionLogger(str(log_path), post_timeout_s=0.05)

    vehicle_before = VehicleState(
        position_ned_m=(1.0, 2.0, -3.0),
        velocity_ned_mps=(0.1, 0.0, -0.2),
        position_wall_time_s=10.0,
        armed=True,
    )
    journal.record("arm", vehicle_before, mono_time_s=100.0, details={"fsm_state": "WAIT_START"})

    vehicle_after = VehicleState(
        position_ned_m=(1.2, 2.1, -3.0),
        velocity_ned_mps=(0.2, 0.0, -0.1),
        position_wall_time_s=10.05,
        armed=True,
    )
    journal.update_vehicle(vehicle_after, mono_time_s=100.02)
    journal.close()

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["action_type"] == "arm"
    assert row["before"]["position_ned_m"] == [1.0, 2.0, -3.0]
    assert row["after"]["position_ned_m"] == [1.2, 2.1, -3.0]
    assert row["after"]["position_delta_ned_m"] == [pytest.approx(0.2), pytest.approx(0.1), pytest.approx(0.0)]


def test_action_logger_flushes_on_timeout_without_telemetry(tmp_path):
    log_path = tmp_path / "actions.jsonl"
    journal = DroneActionLogger(str(log_path), post_timeout_s=0.01)
    vehicle = VehicleState(armed=False)
    journal.record("sim_reset", vehicle, mono_time_s=1.0, details={"repeats": 3})
    time.sleep(0.02)
    journal.flush_pending()
    journal.close()

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["action_type"] == "sim_reset"
    assert rows[0]["after"] is None


def test_record_control_includes_command_fields(tmp_path):
    log_path = tmp_path / "actions.jsonl"
    journal = DroneActionLogger(str(log_path), post_timeout_s=0.01)
    vehicle = VehicleState(
        position_ned_m=(0.0, 0.0, -4.0),
        position_wall_time_s=1.0,
    )
    command = AttitudeCommand(
        roll_rad=0.0,
        pitch_rad=0.1,
        yaw_rad=3.14,
        thrust=0.5,
        roll_rate_rps=0.0,
        pitch_rate_rps=0.1,
        yaw_rate_rps=0.0,
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    plan = RacePlan("COMMIT_GATE", 2.0, commit=True)
    journal.record_control(
        vehicle,
        mono_time_s=5.0,
        plan=plan,
        command=command,
        active_gate_index=0,
        map_dist_center_m=3.0,
        map_plane_signed_m=-1.0,
    )
    journal.flush_pending(force=True)
    journal.close()

    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert row["action_type"] == "control_command"
    assert row["fsm_state"] == "COMMIT_GATE"
    assert row["command"]["thrust"] == 0.5
    assert row["map_dist_center_m"] == 3.0


def test_record_fsm_transition(tmp_path):
    log_path = tmp_path / "actions.jsonl"
    journal = DroneActionLogger(str(log_path), post_timeout_s=0.01)
    vehicle = VehicleState(position_ned_m=(0.0, 0.0, -4.0), position_wall_time_s=1.0)
    transition = FSMTransition("ALIGN_GATE", "COMMIT_GATE", "map_approach_range_ready", 10.0)
    journal.record_fsm_transition(
        vehicle,
        mono_time_s=10.0,
        transition=transition,
        active_gate_index=0,
        map_plane_signed_m=-2.0,
    )
    journal.flush_pending(force=True)
    journal.close()

    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert row["action_type"] == "fsm_transition"
    assert row["old_state"] == "ALIGN_GATE"
    assert row["new_state"] == "COMMIT_GATE"
