from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def newest_log() -> Path | None:
    logs = sorted(Path("logs").glob("sim_stack_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def has_startup_event(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as fh:
            for _idx, line in enumerate(fh):
                if _idx > 32:
                    break
                if '"event":"startup"' in line or '"event": "startup"' in line:
                    return True
    except OSError:
        return False
    return False


def fmt_stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"min={min(values):.3f} mean={mean(values):.3f} max={max(values):.3f}"


def likely_failure(summary: dict) -> tuple[str, list[str]]:
    suggestions: list[str] = []
    frames = summary["frames_received"]
    heartbeats = summary["heartbeat_count"]
    final_state = summary["final_state"]
    sat = summary["saturation_counts"]
    thrusts = summary["thrusts"]
    vz_values = summary["vz_values"]
    if heartbeats == 0:
        suggestions.append("No heartbeat seen: debug MAVLink endpoint or simulator race screen.")
        return "no_heartbeat", suggestions
    if frames == 0:
        suggestions.append("No frames arrived: debug UDP vision receiver on port 5600.")
        return "no_vision", suggestions
    if frames / max(summary["duration_s"], 1e-6) > 100.0:
        suggestions.append("Vision FPS is implausibly high: inspect UDP frame deduplication or header parsing.")
    if final_state in {"WAIT_LINK", "WAIT_VISION"}:
        suggestions.append("FSM never reached flight/search states: check heartbeat and vision readiness gating.")
    if sat["thrust_max"] > 10 and vz_values and mean(vz_values[-min(len(vz_values), 20):]) > 1.0:
        suggestions.append("Thrust is often max while vertical velocity diverges: lower hover thrust or vertical gain.")
    if sat["roll_rate_limit"] + sat["pitch_rate_limit"] > 20:
        suggestions.append("Roll/pitch rates saturate often: reduce lateral and pitch speed gains.")
    if sat["yaw_rate_limit"] > 20:
        suggestions.append("Yaw rate saturates often: reduce yaw gain or slow SEARCH/ALIGN aggressiveness.")
    if vz_values and mean(vz_values) < -0.25:
        suggestions.append("Sustained sink detected: raise hover_thrust and thrust_vz_d in config/gains.yaml.")
    if vz_values and mean(vz_values[-min(len(vz_values), 20):]) > 0.8:
        suggestions.append("Climbing too fast: lower hover_thrust or increase thrust_vz_d damping.")
    if summary["search_before_armed"]:
        suggestions.append("SEARCH_GATE occurred before armed/stable: tighten FSM gating.")
    if not suggestions:
        suggestions.append("No obvious hard failure from logs. Tune detector confidence, hover thrust, and lateral gains first.")
    return "mixed_or_tuning", suggestions


def main() -> int:
    path = newest_log()
    if path is None:
        print("No sim_stack_*.jsonl logs found in logs/")
        return 1

    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        print(f"Log {path} is empty")
        return 1

    start_ts = rows[0]["ts"]
    end_ts = rows[-1]["ts"]
    duration = max(end_ts - start_ts, 0.0)

    frames_received = 0
    frames_dropped = 0
    arm_attempts = 0
    heartbeat_count = 0
    control_count = 0
    state_durations = defaultdict(float)
    transition_counts = Counter()
    saturation_counts = Counter()
    thrusts: list[float] = []
    roll_rates: list[float] = []
    pitch_rates: list[float] = []
    yaw_rates: list[float] = []
    vz_values: list[float] = []
    final_state = "unknown"
    last_control_ts = None
    last_state = None
    search_before_armed = False
    armed_seen = False

    for row in rows:
        event = row.get("event")
        if event == "vision_frame":
            frames_received += 1
        elif event == "vision_drop":
            frames_dropped += int(row.get("count", 0))
        elif event == "arm_command":
            arm_attempts += 1
        elif event == "telemetry_rx" and row.get("msg_type") == "HEARTBEAT":
            heartbeat_count += 1
            armed_seen = armed_seen or bool(row.get("armed"))
        elif event == "fsm_transition":
            transition_counts[f"{row.get('old_state')}->{row.get('new_state')}"] += 1
        elif event == "control":
            control_count += 1
            ts = float(row["ts"])
            state = row.get("fsm_state", "unknown")
            final_state = state
            if last_control_ts is not None and last_state is not None:
                state_durations[last_state] += ts - last_control_ts
            last_control_ts = ts
            last_state = state
            cmd = row.get("command", {})
            thrusts.append(float(cmd.get("thrust", 0.0)))
            roll_rates.append(float(cmd.get("roll_rate_rps", 0.0)))
            pitch_rates.append(float(cmd.get("pitch_rate_rps", 0.0)))
            yaw_rates.append(float(cmd.get("yaw_rate_rps", 0.0)))
            vel = row.get("vehicle_velocity_ned_mps")
            if isinstance(vel, list) and len(vel) >= 3 and vel[2] is not None:
                vz_values.append(float(vel[2]))
            sat = row.get("saturation", {})
            for key, value in sat.items():
                if value:
                    saturation_counts[key] += 1
            if state == "SEARCH_GATE" and not armed_seen:
                search_before_armed = True

    if last_control_ts is not None and last_state is not None:
        state_durations[last_state] += end_ts - last_control_ts

    cmd_rate = control_count / duration if duration > 0 else 0.0
    avg_fps = frames_received / duration if duration > 0 else 0.0

    summary = {
        "frames_received": frames_received,
        "frames_dropped": frames_dropped,
        "heartbeat_count": heartbeat_count,
        "final_state": final_state,
        "saturation_counts": saturation_counts,
        "thrusts": thrusts,
        "vz_values": vz_values,
        "search_before_armed": search_before_armed,
        "duration_s": duration,
    }
    failure, suggestions = likely_failure(summary)

    print(f"log: {path}")
    print(f"startup event present: {'yes' if has_startup_event(path) else 'no'}")
    print(f"total run time: {duration:.2f}s")
    print("fsm state durations:")
    for state, secs in sorted(state_durations.items()):
        print(f"  {state}: {secs:.2f}s")
    print(f"frames received: {frames_received}")
    print(f"frames dropped: {frames_dropped}")
    print(f"average vision FPS: {avg_fps:.2f}")
    print(f"command rate: {cmd_rate:.2f} Hz")
    print(f"thrust: {fmt_stats(thrusts)}")
    print(f"roll_rate: {fmt_stats(roll_rates)}")
    print(f"pitch_rate: {fmt_stats(pitch_rates)}")
    print(f"yaw_rate: {fmt_stats(yaw_rates)}")
    print("saturation counts:")
    for key in ["thrust_min", "thrust_max", "roll_rate_limit", "pitch_rate_limit", "yaw_rate_limit"]:
        print(f"  {key}: {saturation_counts.get(key, 0)}")
    print(f"arming attempts: {arm_attempts}")
    print(f"final state: {final_state}")
    print(f"likely failure cause: {failure}")
    print("tuning recommendations:")
    if not has_startup_event(path):
        print("  - This log predates the current startup marker. You may still be reading an older stale-process run.")
    for suggestion in suggestions:
        print(f"  - {suggestion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
