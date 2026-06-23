from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2
import yaml

from src.comms.mavlink_client import MavlinkClient
from src.control.attitude_controller import AttitudeController
from src.estimation.state_estimator import StateEstimator
from src.infra.drone_action_logger import DroneActionLogger
from src.infra.logger import StructuredLogger
from src.perception.gate_detector import GateDetector
from src.planning.race_fsm import RaceFSM
from src.tracking.gate_tracker import GateTracker
from src.types import SharedState
from src.vision.frame_preview import VisionOverlay, VisionPreview, annotate_vision_frame
from src.vision.gate_sight_log import gate_sight_record
from src.vision.udp_receiver import UdpVisionReceiver


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _position_jump_is_spawn_rebase(
    from_pos: tuple[float, float, float],
    to_pos: tuple[float, float, float],
    jump_m: float,
    *,
    far_m: float = 25.0,
    spawn_xy_m: float = 8.0,
    spawn_z_m: float = 3.0,
) -> bool:
    if jump_m < far_m:
        return False
    to_xy = math.hypot(float(to_pos[0]), float(to_pos[1]))
    to_near_spawn = to_xy <= spawn_xy_m and abs(float(to_pos[2])) <= spawn_z_m
    from_xy = math.hypot(float(from_pos[0]), float(from_pos[1]))
    return to_near_spawn and from_xy >= far_m


def _should_ignore_position_jump(
    plan_state: str,
    now_s: float,
    reset_sent_mono: float | None,
    from_pos: tuple[float, float, float],
    to_pos: tuple[float, float, float],
    jump_m: float,
    sim_config: dict,
) -> bool:
    if plan_state in {"WAIT_LINK", "WAIT_VISION", "WAIT_START", "TAKEOFF", "STABILIZE"}:
        return True
    if plan_state in {"ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE"} and bool(
        sim_config.get("vision_primary_navigation", False)
    ):
        return True
    if reset_sent_mono is not None:
        ignore_s = float(sim_config.get("position_jump_ignore_after_reset_s", 12.0))
        if (now_s - reset_sent_mono) <= ignore_s:
            return True
    if _position_jump_is_spawn_rebase(from_pos, to_pos, jump_m):
        return True
    return False


def _should_auto_arm(
    *,
    now_s: float,
    arm_deadline_s: float,
    last_arm_attempt_s: float,
    link_ready: bool,
    armed: bool,
    plan_state: str,
) -> bool:
    if not link_ready or armed:
        return False
    if (now_s - last_arm_attempt_s) < 1.0:
        return False
    if now_s < arm_deadline_s:
        return False
    return plan_state in {"WAIT_VISION", "WAIT_START", "STABILIZE", "TAKEOFF"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulator-only AI Grand Prix / DCL autonomy stack")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--log-path", default="")
    parser.add_argument(
        "--sim-config",
        default="config/sim.yaml",
        help="Sim/FSM config path relative to comp root (use config/sim_comp.yaml for vision-primary)",
    )
    parser.add_argument(
        "--show-vision",
        action="store_true",
        help="Open a live window showing the drone camera with gate overlays (press q to quit)",
    )
    parser.add_argument(
        "--show-vision-scale",
        type=float,
        default=1.5,
        help="Display scale for --show-vision window",
    )
    parser.add_argument(
        "--save-vision-captures",
        action="store_true",
        help="Save annotated camera frames under logs/captures/<run>/",
    )
    return parser.parse_args()


def build_camera_config(camera_raw: dict) -> dict:
    cfg = dict(camera_raw)
    offset_px = float(cfg.get("aim_vertical_offset_px", 0.0))
    cfg["desired_v_px"] = float(cfg["cy"]) + offset_px
    return cfg


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)) == 0:
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except (OSError, SystemError):
        return False
    return True


def _active_lock_path(root: Path) -> Path:
    return root / "logs" / "sim_stack.active.json"


def _check_active_lock(root: Path) -> None:
    lock_path = _active_lock_path(root)
    if not lock_path.exists():
        return
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception:
        return
    pid = int(payload.get("pid", 0))
    if not _pid_is_alive(pid):
        try:
            lock_path.unlink()
        except OSError:
            pass
        return
    log_path = str(payload.get("log_path", "unknown"))
    raise RuntimeError(
        f"Another sim stack instance is still active (pid={pid}, log={log_path}). "
        f"Stop that process, delete logs/sim_stack.active.json, or rerun with: "
        f".\\run_sim_stack.ps1 -VisionPrimary (auto-stops the previous stack)."
    )


def _write_active_lock(root: Path, log_path: str) -> None:
    lock_path = _active_lock_path(root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "log_path": log_path, "started_at": time.time()}),
        encoding="utf-8",
    )


def _clear_active_lock(root: Path) -> None:
    lock_path = _active_lock_path(root)
    if not lock_path.exists():
        return
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if int(payload.get("pid", -1)) != os.getpid():
        return
    try:
        lock_path.unlink()
    except OSError:
        pass


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    sim_config_path = root / args.sim_config
    sim_config = load_yaml(sim_config_path)
    gains = load_yaml(root / "config" / "gains.yaml")
    gains["takeoff_altitude_m"] = float(sim_config.get("takeoff_altitude_m", 5.0))
    gains["takeoff_climb_rate_mps"] = float(
        sim_config.get("takeoff_climb_rate_mps", gains.get("takeoff_climb_rate_mps", 2.0))
    )
    if "takeoff_thrust" in sim_config:
        gains["takeoff_thrust"] = float(sim_config["takeoff_thrust"])
    gains["vision_takeoff_altitude_m"] = float(
        sim_config.get("vision_takeoff_altitude_m", sim_config.get("takeoff_altitude_m", 5.0))
    )
    gains["vision_primary_navigation"] = bool(sim_config.get("vision_primary_navigation", False))
    gains["vision_search_steering_conf"] = float(sim_config.get("vision_search_steering_conf", 0.12))
    gains["vision_align_bearing_rad"] = float(sim_config.get("vision_align_bearing_rad", 0.24))
    gains["vision_search_forward_pitch_rad"] = float(sim_config.get("vision_search_forward_pitch_rad", 0.12))
    gains["vision_search_forward_scale"] = float(sim_config.get("vision_search_forward_scale", 0.20))
    gains["vision_align_forward_scale"] = float(sim_config.get("vision_align_forward_scale", 0.35))
    gains["vision_align_pitch_limit_rad"] = float(sim_config.get("vision_align_pitch_limit_rad", 0.22))
    gains["vision_approach_pitch_limit_rad"] = float(sim_config.get("vision_approach_pitch_limit_rad", 0.38))
    gains["vision_search_speed_mps"] = float(sim_config.get("vision_search_speed_mps", 0.6))
    gains["vision_align_speed_mps"] = float(sim_config.get("vision_align_speed_mps", 0.9))
    gains["vision_forward_max_speed_mps"] = float(sim_config.get("vision_forward_max_speed_mps", 1.8))
    gains["vision_yaw_hold_bearing_rad"] = float(sim_config.get("vision_yaw_hold_bearing_rad", 0.10))
    gains["vision_align_speed_brake_mps"] = float(sim_config.get("vision_align_speed_brake_mps", 1.8))
    gains["vision_commit_area_fraction"] = float(sim_config.get("vision_commit_area_fraction", 0.08))
    gains["vision_commit_pitch_trim_rad"] = float(sim_config.get("vision_commit_pitch_trim_rad", 0.26))
    gains["vision_commit_push_pitch_trim_rad"] = float(sim_config.get("vision_commit_push_pitch_trim_rad", 0.30))
    gains["vision_commit_max_speed_mps"] = float(sim_config.get("vision_commit_max_speed_mps", 2.5))
    gains["vision_approach_alt_drop_m"] = float(sim_config.get("vision_approach_alt_drop_m", 0.5))
    gains["vision_approach_close_range_m"] = float(sim_config.get("vision_approach_close_range_m", 4.0))
    gains["vision_approach_far_range_m"] = float(sim_config.get("vision_approach_far_range_m", 12.0))
    gains["vision_approach_min_alt_m"] = float(sim_config.get("vision_approach_min_alt_m", 0.8))
    gains["vision_commit_alt_drop_fraction"] = float(sim_config.get("vision_commit_alt_drop_fraction", 0.85))
    gains["vision_gate_center_alt_kp_m"] = float(sim_config.get("vision_gate_center_alt_kp_m", 1.8))
    gains["vision_gate_center_alt_limit_m"] = float(sim_config.get("vision_gate_center_alt_limit_m", 1.0))
    gains["vision_lower_gate_extra_alt_gain"] = float(sim_config.get("vision_lower_gate_extra_alt_gain", 0.9))
    gains["vision_lower_gate_extra_drop_per_rad_m"] = float(
        sim_config.get("vision_lower_gate_extra_drop_per_rad_m", 1.8)
    )
    gains["vision_lower_gate_extra_drop_max_m"] = float(
        sim_config.get("vision_lower_gate_extra_drop_max_m", 0.45)
    )
    gains["vision_lower_gate_align_drop_scale"] = float(
        sim_config.get("vision_lower_gate_align_drop_scale", 1.15)
    )
    gains["vision_lower_gate_extra_alt_slew_per_rad_mps"] = float(
        sim_config.get("vision_lower_gate_extra_alt_slew_per_rad_mps", 12.0)
    )
    gains["vision_lower_gate_extra_alt_slew_max_mps"] = float(
        sim_config.get("vision_lower_gate_extra_alt_slew_max_mps", 2.5)
    )
    gains["vision_gate_alt_slew_rate_mps"] = float(sim_config.get("vision_gate_alt_slew_rate_mps", 2.0))
    gains["vision_align_vertical_pitch_scale"] = float(sim_config.get("vision_align_vertical_pitch_scale", 0.75))
    gains["vision_align_alt_drop_fraction"] = float(sim_config.get("vision_align_alt_drop_fraction", 0.10))
    gains["vision_align_hover_altitude_m"] = float(sim_config["vision_align_hover_altitude_m"]) if sim_config.get("vision_align_hover_altitude_m") is not None else None
    gains["vision_align_proximity_floor"] = float(sim_config.get("vision_align_proximity_floor", 0.40))
    gains["vision_align_alt_slew_rate_mps"] = float(sim_config.get("vision_align_alt_slew_rate_mps", 4.0))
    gains["vision_approach_vertical_pitch_scale"] = float(sim_config.get("vision_approach_vertical_pitch_scale", 1.0))
    gains["vision_approach_roll_gain_scale"] = float(sim_config.get("vision_approach_roll_gain_scale", 1.0))
    gains["vision_approach_yaw_gain_scale"] = float(sim_config.get("vision_approach_yaw_gain_scale", 1.0))
    gains["vision_center_bearing_rad"] = float(sim_config.get("vision_center_bearing_rad", 0.06))
    gains["vision_center_vertical_rad"] = float(sim_config.get("vision_center_vertical_rad", 0.08))
    gains["vision_center_fixation_gain"] = float(sim_config.get("vision_center_fixation_gain", 1.8))
    gains["vision_center_locked_gain"] = float(sim_config.get("vision_center_locked_gain", 1.25))
    gains["vision_approach_center_tighten"] = float(sim_config.get("vision_approach_center_tighten", 0.55))
    gains["vision_straight_lateral_vy_kp"] = float(sim_config.get("vision_straight_lateral_vy_kp", 0.55))
    gains["vision_straight_roll_limit_rad"] = float(sim_config.get("vision_straight_roll_limit_rad", 0.14))
    gains["vision_yaw_hold_max_speed_mps"] = float(sim_config.get("vision_yaw_hold_max_speed_mps", 0.6))
    gains["vision_yaw_step_limit_rad"] = float(sim_config.get("vision_yaw_step_limit_rad", 0.14))
    gains["vision_yaw_speed_ref_mps"] = float(sim_config.get("vision_yaw_speed_ref_mps", 1.4))
    gains["vision_roll_yaw_speed_ref_mps"] = float(sim_config.get("vision_roll_yaw_speed_ref_mps", 1.0))
    gains["vision_yaw_coarse_bearing_rad"] = float(sim_config.get("vision_yaw_coarse_bearing_rad", 0.16))
    gains["vision_align_roll_gain_scale"] = float(sim_config.get("vision_align_roll_gain_scale", 0.55))
    gains["vision_align_yaw_gain_scale"] = float(sim_config.get("vision_align_yaw_gain_scale", 0.75))
    gains["vision_approach_roll_limit_rad"] = float(sim_config.get("vision_approach_roll_limit_rad", gains.get("roll_limit_rad", 0.18)))
    gains["vision_pitch_up_limit_rad"] = float(sim_config.get("vision_pitch_up_limit_rad", 0.06))
    gains["vision_min_nose_down_pitch_delta_rad"] = float(sim_config.get("vision_min_nose_down_pitch_delta_rad", 0.05))
    gains["vision_gate_below_bearing_rad"] = float(sim_config.get("vision_gate_below_bearing_rad", 0.08))
    gains["vision_gate_descent_thrust_cut"] = float(sim_config.get("vision_gate_descent_thrust_cut", 0.10))
    gains["vision_gate_descent_extra_thrust_cut_per_rad"] = float(
        sim_config.get("vision_gate_descent_extra_thrust_cut_per_rad", 0.18)
    )
    gains["vision_lower_gate_align_extra_thrust_cut_per_rad"] = float(
        sim_config.get("vision_lower_gate_align_extra_thrust_cut_per_rad", 0.10)
    )
    gains["vision_gate_descent_alt_error_cut_gain"] = float(
        sim_config.get("vision_gate_descent_alt_error_cut_gain", 0.04)
    )
    gains["vision_gate_descent_thrust_cut_max_extra"] = float(
        sim_config.get("vision_gate_descent_thrust_cut_max_extra", 0.08)
    )
    gains["vision_gate_descent_thrust_cut_max"] = float(
        sim_config.get("vision_gate_descent_thrust_cut_max", 0.18)
    )
    gains["vision_lower_gate_pitch_boost_per_rad"] = float(
        sim_config.get("vision_lower_gate_pitch_boost_per_rad", 0.55)
    )
    gains["vision_lower_gate_pitch_boost_max_rad"] = float(
        sim_config.get("vision_lower_gate_pitch_boost_max_rad", 0.12)
    )
    gains["vision_lower_gate_align_pitch_boost_scale"] = float(
        sim_config.get("vision_lower_gate_align_pitch_boost_scale", 1.20)
    )
    gains["vision_lower_gate_pitch_boost_close_taper"] = float(
        sim_config.get("vision_lower_gate_pitch_boost_close_taper", 0.0)
    )
    gains["vision_lower_gate_pitch_boost_close_floor"] = float(
        sim_config.get("vision_lower_gate_pitch_boost_close_floor", 0.25)
    )
    gains["vision_lower_gate_extra_pitch_limit_per_rad"] = float(
        sim_config.get("vision_lower_gate_extra_pitch_limit_per_rad", 0.90)
    )
    gains["vision_lower_gate_extra_pitch_limit_rad"] = float(
        sim_config.get("vision_lower_gate_extra_pitch_limit_rad", 0.10)
    )
    gains["vision_lower_gate_forward_scale_min"] = float(
        sim_config.get("vision_lower_gate_forward_scale_min", 0.25)
    )
    gains["vision_lower_gate_forward_scale_per_rad"] = float(
        sim_config.get("vision_lower_gate_forward_scale_per_rad", 2.2)
    )
    gains["vision_lower_gate_align_forward_scale_min"] = float(
        sim_config.get("vision_lower_gate_align_forward_scale_min", 0.18)
    )
    gains["vision_lower_gate_align_forward_scale_per_rad"] = float(
        sim_config.get("vision_lower_gate_align_forward_scale_per_rad", 2.8)
    )
    gains["vision_lower_gate_speed_cap_mps"] = float(
        sim_config.get("vision_lower_gate_speed_cap_mps", 0.55)
    )
    gains["vision_lower_gate_speed_cap_per_rad"] = float(
        sim_config.get("vision_lower_gate_speed_cap_per_rad", 0.9)
    )
    gains["vision_lower_gate_speed_cap_min_mps"] = float(
        sim_config.get("vision_lower_gate_speed_cap_min_mps", 0.12)
    )
    gains["vision_lower_gate_speed_cap_near_delta_mps"] = float(
        sim_config.get("vision_lower_gate_speed_cap_near_delta_mps", 0.20)
    )
    gains["vision_lower_gate_speed_cap_far_range_m"] = float(
        sim_config.get("vision_lower_gate_speed_cap_far_range_m", 20.0)
    )
    gains["vision_lower_gate_speed_decel_mps2"] = float(
        sim_config.get("vision_lower_gate_speed_decel_mps2", 2.8)
    )
    gains["vision_lower_gate_brake_speed_mps"] = float(
        sim_config.get("vision_lower_gate_brake_speed_mps", 0.9)
    )
    gains["vision_lower_gate_brake_mix_per_rad"] = float(
        sim_config.get("vision_lower_gate_brake_mix_per_rad", 1.2)
    )
    gains["vision_lower_gate_brake_mix_max"] = float(
        sim_config.get("vision_lower_gate_brake_mix_max", 0.90)
    )
    gains["vision_lower_gate_brake_blend_per_rad"] = float(
        sim_config.get("vision_lower_gate_brake_blend_per_rad", 0.8)
    )
    gains["vision_lower_gate_brake_blend_max"] = float(
        sim_config.get("vision_lower_gate_brake_blend_max", 0.90)
    )
    gains["vision_lower_gate_forward_pitch_cap_rad"] = float(
        sim_config.get("vision_lower_gate_forward_pitch_cap_rad", 0.09)
    )
    gains["vision_lower_gate_forward_pitch_cap_per_rad"] = float(
        sim_config.get("vision_lower_gate_forward_pitch_cap_per_rad", 0.25)
    )
    gains["vision_lower_gate_forward_pitch_cap_min_rad"] = float(
        sim_config.get("vision_lower_gate_forward_pitch_cap_min_rad", 0.02)
    )
    gains["vision_lower_gate_forward_pitch_cap_near_delta_rad"] = float(
        sim_config.get("vision_lower_gate_forward_pitch_cap_near_delta_rad", 0.0)
    )
    gains["vision_lower_gate_align_forward_pitch_cap_rad"] = float(
        sim_config.get("vision_lower_gate_align_forward_pitch_cap_rad", 0.07)
    )
    gains["vision_lower_gate_descent_close_cut_max"] = float(
        sim_config.get("vision_lower_gate_descent_close_cut_max", 0.0)
    )
    gains["capture_launch_pitch_trim"] = bool(sim_config.get("capture_launch_pitch_trim", True))
    gains["launch_pitch_trim_limit_rad"] = float(sim_config.get("launch_pitch_trim_limit_rad", 0.55))
    gains["commit_speed_mps"] = float(sim_config.get("commit_speed_mps", gains.get("commit_speed_mps", 2.2)))
    gains["search_yaw_scan_rps"] = float(sim_config.get("search_yaw_rate_rps", 0.35))
    gains["search_scan_max_speed_mps"] = float(sim_config.get("search_scan_max_speed_mps", 1.0))
    camera = build_camera_config(load_yaml(root / "config" / "camera.yaml"))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = args.log_path or str(root / "logs" / f"sim_stack_{timestamp}.jsonl")
    _check_active_lock(root)
    _write_active_lock(root, log_path)
    logger = StructuredLogger(log_path)
    action_journal: DroneActionLogger | None = None
    if bool(sim_config.get("action_log_enabled", True)):
        action_path = Path(log_path).with_name(f"{Path(log_path).stem}_actions{Path(log_path).suffix}")
        action_journal = DroneActionLogger(
            str(action_path),
            post_timeout_s=float(sim_config.get("action_log_post_timeout_s", 0.15)),
        )
    print(
        f"[SIM] starting pid={os.getpid()} log={log_path} "
        f"hover_thrust={gains['hover_thrust']} detect_conf={sim_config['detect_confidence']} "
        f"nav={'vision' if sim_config.get('vision_primary_navigation') else 'map'}"
        + (f" action_log={action_journal.log_path}" if action_journal is not None else ""),
        flush=True,
    )
    logger.log(
        "startup",
        {
            "pid": os.getpid(),
            "log_path": log_path,
            "sim_config": str(sim_config_path),
            "vision_primary_navigation": bool(sim_config.get("vision_primary_navigation", False)),
            "action_log_path": None if action_journal is None else action_journal.log_path,
            "hover_thrust": gains["hover_thrust"],
            "detect_confidence": sim_config["detect_confidence"],
        },
    )

    shared_state = SharedState()
    mavlink = MavlinkClient(sim_config["mavlink_connection"], shared_state, logger=logger)
    vision = UdpVisionReceiver(
        shared_state,
        host=str(sim_config["vision_host"]),
        port=int(sim_config["vision_port"]),
        logger=logger,
    )
    detector = GateDetector(min_area_fraction=float(sim_config["detector_min_area_fraction"]))
    tracker = GateTracker(
        alpha=float(sim_config["tracker_alpha"]),
        dropout_s=float(sim_config["gate_timeout_s"]),
    )
    estimator = StateEstimator(camera, sim_config)
    fsm = RaceFSM(sim_config)
    controller = AttitudeController(gains)
    last_race_start_ms: int | None = None
    reset_sent_mono: float | None = None
    race_latched = False

    mavlink.start()
    if bool(sim_config.get("reset_before_run", False)):
        print("[SIM] sending simulator reset", flush=True)
        logger.log("sim_reset", {"repeats": 3})
        if action_journal is not None:
            action_journal.record(
                "sim_reset",
                shared_state.get_vehicle(),
                mono_time_s=time.monotonic(),
                details={"repeats": 3},
            )
        mavlink.reset_sim()
        reset_sent_mono = time.monotonic()
        reset_pause = float(sim_config.get("reset_pause_s", 1.5))
        race_wait = float(sim_config.get("post_reset_race_wait_s", 3.0))
        time.sleep(reset_pause + race_wait)
    vision.start()
    arm_deadline = time.monotonic() + float(sim_config["arm_delay_s"])
    last_arm_attempt = 0.0

    started = time.monotonic()
    last_control = started
    last_plan_update = started
    last_perception_update = 0.0
    last_processed_frame = -1
    plan = fsm._build_plan()
    last_status_print = started
    last_status_frames = 0
    last_status_drops = 0
    last_command = None
    saturation_started_s = None
    saturation_reported = False
    recover_requested = False
    recover_reason = ""
    last_position_ned: tuple[float, float, float] | None = None
    last_map_dist_m: float | None = None
    planner_hz = float(sim_config.get("planner_hz", 15.0))
    perception_hz = float(sim_config.get("perception_hz", 12.0))
    vision_preview = (
        VisionPreview(scale=float(args.show_vision_scale))
        if bool(args.show_vision)
        else None
    )
    capture_dir: Path | None = None
    capture_stride = int(sim_config.get("vision_capture_stride", 20))
    if bool(args.save_vision_captures):
        capture_dir = Path(log_path).parent / "captures" / Path(log_path).stem
        capture_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SIM] saving vision captures to {capture_dir}", flush=True)
    last_detection = None
    last_detection_count = 0
    last_vision_show_s = 0.0
    last_target_gate_index = -1
    last_gate_sight_log_s = 0.0
    gate_sight_log_hz = float(sim_config.get("vision_sight_log_hz", 4.0))
    pending_gate_sight_frame: int | None = None
    pending_all_centers_px: list[tuple[float, float]] | None = None
    vision_show_hz = float(sim_config.get("vision_hz", 30.0))
    vision_primary_nav = bool(sim_config.get("vision_primary_navigation", False))
    vision_timeout_s = float(
        sim_config.get("vision_link_timeout_s", sim_config.get("link_timeout_s", 1.0))
    )

    try:
        while True:
            now_s = time.monotonic()
            if args.max_seconds > 0.0 and (now_s - started) >= args.max_seconds:
                break

            vehicle = shared_state.get_vehicle()
            if action_journal is not None:
                action_journal.update_vehicle(vehicle, now_s)

            race_status = shared_state.get_race_status()
            track_gates = shared_state.get_track_gates()
            race_start_ms = int(race_status.get("race_start_boot_time_ms", -1))
            if last_race_start_ms is None:
                last_race_start_ms = race_start_ms
                race_started = False
            else:
                race_started = race_start_ms >= 0 and (
                    (last_race_start_ms < 0 and race_start_ms >= 0)
                    or (race_start_ms >= 0 and race_start_ms != last_race_start_ms and last_race_start_ms >= 0)
                )
                if (
                    not race_started
                    and race_start_ms >= 0
                    and reset_sent_mono is not None
                    and (now_s - reset_sent_mono) <= float(sim_config.get("accept_race_after_reset_s", 45.0))
                ):
                    race_started = True
                if race_started:
                    race_latched = True
                if race_latched:
                    race_started = True
                last_race_start_ms = race_start_ms
            map_first_gate_active = (
                bool(sim_config.get("use_map_first_gate", True))
                and not bool(sim_config.get("vision_primary_navigation", False))
                and race_started
                and bool(track_gates)
                and int(race_status.get("active_gate_index", -1)) == 0
            )
            frame = shared_state.get_frame()
            if (
                frame is not None
                and frame.frame_id != last_processed_frame
                and (now_s - last_perception_update) >= (1.0 / max(perception_hz, 1.0))
            ):
                if bool(sim_config.get("skip_detection_on_map_first_gate", True)) and map_first_gate_active:
                    detection = None
                    track = None
                    last_detection_count = 0
                    pending_all_centers_px = None
                    shared_state.update_gate_track(None)
                else:
                    if vision_primary_nav:
                        all_dets = detector.detect_all(frame)
                        last_detection_count = len(all_dets)
                        detection = detector.detect_for_target(
                            frame,
                            target_gate_index=max(0, fsm.target_gate_index),
                            seeking_next_gate=(
                                fsm.state in {"SEARCH_GATE", "PASS_GATE"} and fsm.target_gate_index > 0
                            ),
                            aim_cx=float(camera["cx"]),
                            aim_cy=float(camera["desired_v_px"]),
                            max_acquire_area=float(sim_config.get("vision_max_acquire_area", 0.45)),
                            min_next_gate_area=float(sim_config.get("vision_min_next_gate_area", 0.006)),
                        )
                    else:
                        all_dets = []
                        detection = detector.detect(frame)
                    track = tracker.update(detection, frame.wall_time_s)
                    shared_state.update_gate_track(track)
                    pending_gate_sight_frame = frame.frame_id
                    pending_all_centers_px = [d.center_px for d in all_dets] if vision_primary_nav else None
                last_detection = detection
                last_processed_frame = frame.frame_id
                last_perception_update = now_s
                aim_cx = float(camera["cx"])
                aim_cy = float(camera["desired_v_px"])
                pixel_error_px = None
                if track is not None:
                    pixel_error_px = [
                        float(track.center_px[0]) - aim_cx,
                        float(track.center_px[1]) - aim_cy,
                    ]
                elif detection is not None:
                    pixel_error_px = [
                        float(detection.center_px[0]) - aim_cx,
                        float(detection.center_px[1]) - aim_cy,
                    ]
                logger.log(
                    "detection",
                    {
                        "frame_id": frame.frame_id,
                        "fsm_state": plan.state,
                        "detected": detection is not None,
                        "aim_px": [aim_cx, aim_cy],
                        "track_visible": track.visible if track is not None else False,
                        "track_confidence": None if track is None else track.confidence,
                        "track_center_px": None if track is None else list(track.center_px),
                        "pixel_error_px": pixel_error_px,
                        "track_bbox": None if track is None else list(track.bbox),
                        "detections_count": last_detection_count,
                    },
                )

            if vehicle.position_ned_m is not None and last_position_ned is not None:
                jump_m = math.hypot(
                    vehicle.position_ned_m[0] - last_position_ned[0],
                    vehicle.position_ned_m[1] - last_position_ned[1],
                )
                jump_threshold = float(sim_config.get("position_jump_threshold_m", 20.0))
                if jump_m > jump_threshold:
                    near_gate = (
                        last_map_dist_m is not None
                        and last_map_dist_m <= float(sim_config.get("position_jump_near_gate_m", 12.0))
                    )
                    in_gate_phase = plan.state in {"ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE"}
                    ignore_jump = _should_ignore_position_jump(
                        plan.state,
                        now_s,
                        reset_sent_mono,
                        last_position_ned,
                        vehicle.position_ned_m,
                        jump_m,
                        sim_config,
                    )
                    logger.log(
                        "position_jump",
                        {
                            "jump_m": jump_m,
                            "fsm_state": plan.state,
                            "near_gate": near_gate,
                            "ignored": ignore_jump,
                            "last_map_dist_m": last_map_dist_m,
                            "from_position_ned_m": list(last_position_ned),
                            "to_position_ned_m": list(vehicle.position_ned_m),
                        },
                    )
                    if ignore_jump:
                        last_position_ned = vehicle.position_ned_m
                    else:
                        if action_journal is not None:
                            action_journal.record(
                                "position_jump",
                                vehicle,
                                mono_time_s=now_s,
                                details={
                                    "jump_m": jump_m,
                                    "fsm_state": plan.state,
                                    "near_gate": near_gate,
                                    "last_map_dist_m": last_map_dist_m,
                                    "from_position_ned_m": list(last_position_ned),
                                    "to_position_ned_m": list(vehicle.position_ned_m),
                                },
                            )
                        controller.reset_transient()
                        saturation_started_s = None
                        saturation_reported = False
                        if near_gate and in_gate_phase:
                            plan = fsm.force_pass_gate(now_s, "near_gate_position_discontinuity")
                            if action_journal is not None:
                                action_journal.record(
                                    "force_pass_gate",
                                    vehicle,
                                    mono_time_s=now_s,
                                    details={
                                        "reason": "near_gate_position_discontinuity",
                                        "fsm_state": plan.state,
                                        "active_gate_index": int(race_status.get("active_gate_index", -1)),
                                    },
                                )
                            last_plan_update = now_s
                        else:
                            recover_requested = True
                            recover_reason = "position_discontinuity"
                            if action_journal is not None:
                                action_journal.record(
                                    "recover_request",
                                    vehicle,
                                    mono_time_s=now_s,
                                    details={
                                        "reason": recover_reason,
                                        "fsm_state": plan.state,
                                        "jump_m": jump_m,
                                    },
                                )
            if vehicle.position_ned_m is not None:
                last_position_ned = vehicle.position_ned_m

            gate_track = shared_state.get_gate_track()
            est = estimator.estimate(
                now_s,
                vehicle,
                gate_track,
                shared_state.vision_is_ready(now_s=now_s, timeout_s=vision_timeout_s),
                race_status=race_status,
                track_gates=track_gates,
                force_race_started=race_started,
                gate_corners_px=last_detection.corners_px if last_detection is not None else None,
            )
            if est.map_dist_center_m is not None:
                last_map_dist_m = est.map_dist_center_m
            if pending_gate_sight_frame is not None:
                logger.log(
                    "gate_sight",
                    gate_sight_record(
                        frame_id=pending_gate_sight_frame,
                        plan=plan,
                        est=est,
                        camera=camera,
                        detection=last_detection,
                        track=gate_track,
                        target_gate_index=fsm.target_gate_index,
                        detections_count=last_detection_count,
                        all_centers_px=pending_all_centers_px,
                    ),
                )
                last_gate_sight_log_s = now_s
                pending_gate_sight_frame = None
            elif (
                gate_track is not None
                and (gate_track.visible or gate_track.predicted)
                and (now_s - last_gate_sight_log_s) >= (1.0 / max(gate_sight_log_hz, 1.0))
            ):
                logger.log(
                    "gate_sight",
                    gate_sight_record(
                        frame_id=last_processed_frame,
                        plan=plan,
                        est=est,
                        camera=camera,
                        detection=last_detection,
                        track=gate_track,
                        target_gate_index=fsm.target_gate_index,
                        detections_count=last_detection_count,
                        all_centers_px=pending_all_centers_px,
                    ),
                )
                last_gate_sight_log_s = now_s
            if (now_s - last_plan_update) >= (1.0 / max(planner_hz, 1.0)):
                plan = fsm.update(est, recover_requested=recover_requested, recover_reason=recover_reason)
                recover_requested = False
                recover_reason = ""
                last_plan_update = now_s
                if fsm.last_transition is not None:
                    if (
                        fsm.target_gate_index > last_target_gate_index
                        and last_target_gate_index >= 0
                        and fsm.last_transition.reason in {
                            "active_gate_index_advanced",
                            "sim_gate_race_time",
                            "next_gate_search",
                        }
                    ):
                        tracker.reset()
                        estimator.reset_for_next_gate()
                        controller.reset_for_next_gate()
                    last_target_gate_index = fsm.target_gate_index
                    sat_flags = {} if last_command is None else last_command.saturation.as_dict()
                    logger.log(
                        "fsm_transition",
                        {
                            "old_state": fsm.last_transition.old_state,
                            "new_state": fsm.last_transition.new_state,
                            "reason": fsm.last_transition.reason,
                            "mono_time_s": fsm.last_transition.timestamp_s,
                            "gate_confidence": est.gate_confidence,
                            "map_ready": est.has_track_map,
                            "active_gate_index": est.active_gate_index,
                            "map_plane_signed_m": est.map_plane_signed_m,
                            "map_within_gate_bounds": est.map_within_gate_bounds,
                            "attitude_rad": [vehicle.roll_rad, vehicle.pitch_rad, vehicle.yaw_rad],
                            "velocity_ned_mps": vehicle.velocity_ned_mps,
                            "saturation": sat_flags,
                        },
                    )
                    if action_journal is not None:
                        action_journal.record_fsm_transition(
                            vehicle,
                            mono_time_s=now_s,
                            transition=fsm.last_transition,
                            active_gate_index=est.active_gate_index,
                            map_plane_signed_m=est.map_plane_signed_m,
                        )

            if _should_auto_arm(
                now_s=now_s,
                arm_deadline_s=arm_deadline,
                last_arm_attempt_s=last_arm_attempt,
                link_ready=est.link_ready,
                armed=vehicle.armed,
                plan_state=plan.state,
            ):
                mavlink.arm()
                last_arm_attempt = now_s
                logger.log("arm_command", {"fsm_state": plan.state})
                if action_journal is not None:
                    action_journal.record(
                        "arm",
                        vehicle,
                        mono_time_s=now_s,
                        details={"fsm_state": plan.state},
                    )

            dt = max(now_s - last_control, 1.0 / float(sim_config["control_hz"]))
            if now_s - last_control >= 1.0 / float(sim_config["control_hz"]):
                command = controller.update(est, plan, dt)
                last_command = command
                if command.saturation.any:
                    if saturation_started_s is None:
                        saturation_started_s = now_s
                        saturation_reported = False
                        logger.log("control_saturation_start", {"state": plan.state, "saturation": command.saturation.as_dict()})
                    elif not saturation_reported and (now_s - saturation_started_s) >= 0.5:
                        if plan.state in {
                            "SEARCH_GATE",
                            "ALIGN_GATE",
                            "APPROACH_GATE",
                            "COMMIT_GATE",
                            "PASS_GATE",
                        }:
                            vz_now = float((vehicle.velocity_ned_mps or (0.0, 0.0, 0.0))[2])
                            overspeed_climb = vz_now < -float(sim_config.get("takeoff_climb_rate_mps", 2.0)) * 2.0
                            sat = command.saturation
                            # thrust_min is normal when pitched forward; only recover on hard limits
                            recoverable_sat = (
                                sat.thrust_max
                                or sat.roll_rate_limit
                                or sat.pitch_rate_limit
                                or (
                                sat.yaw_rate_limit
                                and not (
                                    bool(sim_config.get("vision_primary_navigation", False))
                                    and plan.state in {"SEARCH_GATE", "ALIGN_GATE", "APPROACH_GATE"}
                                )
                                )
                            )
                            if not overspeed_climb and recoverable_sat:
                                gate_locked = (
                                    bool(sim_config.get("vision_primary_navigation", False))
                                    and plan.state in {"ALIGN_GATE", "APPROACH_GATE", "COMMIT_GATE"}
                                    and est.gate_confidence
                                    >= float(sim_config.get("detect_confidence", 0.22))
                                    and est.gate_track is not None
                                    and (
                                        est.gate_track.visible
                                        or est.gate_track.predicted
                                    )
                                )
                                if not gate_locked:
                                    recover_requested = True
                                    recover_reason = "persistent_control_saturation"
                                    if action_journal is not None:
                                        action_journal.record(
                                            "recover_request",
                                            vehicle,
                                            mono_time_s=now_s,
                                            details={
                                                "reason": recover_reason,
                                                "fsm_state": plan.state,
                                                "saturation": sat.as_dict(),
                                            },
                                        )
                            saturation_reported = True
                        logger.log(
                            "control_saturation_persist",
                            {
                                "state": plan.state,
                                "duration_s": now_s - saturation_started_s,
                                "saturation": command.saturation.as_dict(),
                            },
                        )
                elif saturation_started_s is not None:
                    logger.log("control_saturation_clear", {"state": plan.state, "duration_s": now_s - saturation_started_s})
                    saturation_started_s = None
                    saturation_reported = False
                mavlink.send_attitude_target(command)
                last_control = now_s
                if action_journal is not None:
                    action_journal.record_control(
                        vehicle,
                        mono_time_s=now_s,
                        plan=plan,
                        command=command,
                        active_gate_index=est.active_gate_index,
                        map_dist_center_m=est.map_dist_center_m,
                        map_plane_signed_m=est.map_plane_signed_m,
                    )
                logger.log(
                    "control",
                    {
                        "fsm_state": plan.state,
                        "link_ready": est.link_ready,
                        "vision_ready": est.vision_ready,
                        "gate_confidence": est.gate_confidence,
                        "map_ready": est.has_track_map,
                        "active_gate_index": est.active_gate_index,
                        "gate_bearing_rad": [est.gate_bearing_x_rad, est.gate_bearing_y_rad],
                        "gate_range_m": est.gate_range_m,
                        "map_gate_bearing_rad": [est.map_gate_bearing_x_rad, est.map_gate_bearing_y_rad],
                        "map_gate_range_m": est.map_gate_range_m,
                        "map_approach_bearing_rad": [est.map_approach_bearing_x_rad, est.map_approach_bearing_y_rad],
                        "map_approach_range_m": est.map_approach_range_m,
                        "map_commit_bearing_rad": [est.map_commit_bearing_x_rad, est.map_commit_bearing_y_rad],
                        "map_commit_range_m": est.map_commit_range_m,
                        "map_plane_signed_m": est.map_plane_signed_m,
                        "map_within_gate_bounds": est.map_within_gate_bounds,
                        "map_gate_commit_active": est.map_gate_commit_active,
                        "map_gate_commit_strength": est.map_gate_commit_strength,
                        "map_dist_center_m": est.map_dist_center_m,
                        "vehicle_position_ned_m": vehicle.position_ned_m,
                        "vehicle_velocity_ned_mps": vehicle.velocity_ned_mps,
                        "vehicle_attitude_rad": [vehicle.roll_rad, vehicle.pitch_rad, vehicle.yaw_rad],
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

            if now_s - last_status_print >= 0.5:
                stats = vision.get_stats()
                frame_delta = stats["frames_rx"] - last_status_frames
                drop_delta = stats["frames_dropped"] - last_status_drops
                last_status_frames = stats["frames_rx"]
                last_status_drops = stats["frames_dropped"]
                fps = frame_delta / max(now_s - last_status_print, 1e-6)
                att_txt = f"r={vehicle.roll_rad:+.2f} p={vehicle.pitch_rad:+.2f} y={vehicle.yaw_rad:+.2f}"
                vel = vehicle.velocity_ned_mps
                vel_txt = "vel=—" if vel is None else f"vel=[{vel[0]:+.2f},{vel[1]:+.2f},{vel[2]:+.2f}]"
                cmd_txt = "cmd=—"
                if last_command is not None:
                    cmd_txt = (
                        f"thr={last_command.thrust:.3f} "
                        f"rr={last_command.roll_rate_rps:+.2f} "
                        f"pr={last_command.pitch_rate_rps:+.2f} "
                        f"yr={last_command.yaw_rate_rps:+.2f}"
                    )
                print(
                    f"[SIM] link={'up' if est.link_ready else 'down'} armed={vehicle.armed} "
                    f"state={plan.state} map={'on' if est.has_track_map else 'off'} "
                    f"gate={est.active_gate_index} {att_txt} {vel_txt} {cmd_txt} "
                    f"gate_conf={est.gate_confidence:.2f} "
                    + (
                        f"range={est.gate_range_m:.1f}m "
                        if est.gate_range_m is not None
                        else ""
                    )
                    + f"fps={fps:.1f} dropped={drop_delta}",
                    flush=True,
                )
                logger.log(
                    "status",
                    {
                        "link_ready": est.link_ready,
                        "armed": vehicle.armed,
                        "fsm_state": plan.state,
                        "map_ready": est.has_track_map,
                        "active_gate_index": est.active_gate_index,
                        "map_plane_signed_m": est.map_plane_signed_m,
                        "map_within_gate_bounds": est.map_within_gate_bounds,
                        "attitude_rad": [vehicle.roll_rad, vehicle.pitch_rad, vehicle.yaw_rad],
                        "velocity_ned_mps": vehicle.velocity_ned_mps,
                        "gate_confidence": est.gate_confidence,
                        "vision_fps": fps,
                        "vision_frames_rx": stats["frames_rx"],
                        "vision_frames_dropped": stats["frames_dropped"],
                        "command": None if last_command is None else {
                            "thrust": last_command.thrust,
                            "roll_rate_rps": last_command.roll_rate_rps,
                            "pitch_rate_rps": last_command.pitch_rate_rps,
                            "yaw_rate_rps": last_command.yaw_rate_rps,
                        },
                    },
                )
                last_status_print = now_s

            frame_to_save = None
            if vision_preview is not None and (now_s - last_vision_show_s) >= (1.0 / max(vision_show_hz, 1.0)):
                show_frame = shared_state.get_frame()
                if show_frame is not None:
                    altitude_m = None
                    if vehicle.position_ned_m is not None:
                        altitude_m = -float(vehicle.position_ned_m[2])
                    overlay = VisionOverlay(
                        fsm_state=plan.state,
                        gate_confidence=est.gate_confidence,
                        active_gate_index=est.active_gate_index,
                        gate_bearing_x_rad=est.gate_bearing_x_rad,
                        gate_bearing_y_rad=est.gate_bearing_y_rad,
                        gate_range_m=est.gate_range_m,
                        gate_depth_confidence=est.gate_depth_confidence,
                        detections_visible=last_detection_count,
                        altitude_m=altitude_m,
                    )
                    viz = annotate_vision_frame(
                        show_frame.image_bgr,
                        gate_track=gate_track,
                        detection=last_detection,
                        overlay=overlay,
                        camera_cx=float(camera["cx"]),
                        camera_cy=float(camera["desired_v_px"]),
                    )
                    if not vision_preview.show(viz):
                        break
                    last_vision_show_s = now_s
                    frame_to_save = viz
                else:
                    frame_to_save = None
            elif capture_dir is not None:
                show_frame = shared_state.get_frame()
                if show_frame is not None and show_frame.frame_id % capture_stride == 0:
                    altitude_m = None
                    if vehicle.position_ned_m is not None:
                        altitude_m = -float(vehicle.position_ned_m[2])
                    overlay = VisionOverlay(
                        fsm_state=plan.state,
                        gate_confidence=est.gate_confidence,
                        active_gate_index=est.active_gate_index,
                        gate_bearing_x_rad=est.gate_bearing_x_rad,
                        gate_bearing_y_rad=est.gate_bearing_y_rad,
                        gate_range_m=est.gate_range_m,
                        gate_depth_confidence=est.gate_depth_confidence,
                        detections_visible=last_detection_count,
                        altitude_m=altitude_m,
                    )
                    frame_to_save = annotate_vision_frame(
                        show_frame.image_bgr,
                        gate_track=gate_track,
                        detection=last_detection,
                        overlay=overlay,
                        camera_cx=float(camera["cx"]),
                        camera_cy=float(camera["desired_v_px"]),
                    )

            if capture_dir is not None and frame_to_save is not None:
                show_frame = shared_state.get_frame()
                if show_frame is not None:
                    capture_path = capture_dir / f"frame_{show_frame.frame_id:06d}.jpg"
                    cv2.imwrite(str(capture_path), frame_to_save)

            time.sleep(0.001)
    finally:
        if vision_preview is not None:
            vision_preview.close()
        vision.stop()
        mavlink.stop()
        if action_journal is not None:
            action_journal.close()
        print("[SIM] stopped", flush=True)
        logger.log("shutdown", {})
        logger.close()
        _clear_active_lock(root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
