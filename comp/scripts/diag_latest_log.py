"""Quick diagnostic dump of the newest sim_stack log."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean


def main() -> int:
    logs = sorted(Path("logs").glob("sim_stack_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        print("no logs")
        return 1
    path = logs[0]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"log: {path}")

    transitions = [r for r in rows if r.get("event") == "fsm_transition"]
    controls = [r for r in rows if r.get("event") == "control"]
    race = [r for r in rows if r.get("event") == "race_status"]

    print("\n=== FSM transitions ===")
    for t in transitions:
        print(
            f"  {t.get('old_state')} -> {t.get('new_state')} "
            f"({t.get('reason')}) roll={t.get('attitude_rad', [None])[0]}"
        )

    print(f"\n=== Arm attempts: {sum(1 for r in rows if r.get('event') == 'arm_command')} ===")

    print("\n=== Race status (first/last) ===")
    for r in (race[:3] + race[-3:]) if race else []:
        print(
            f"  gate={r.get('active_gate_index')} race_start_ms={r.get('race_start_boot_time_ms')} "
            f"map_gates={r.get('track_gate_count')}"
        )

    by_state = Counter(c.get("fsm_state") for c in controls)
    print("\n=== Control samples by state ===")
    for state, count in by_state.most_common():
        subset = [c for c in controls if c.get("fsm_state") == state]
        rolls = [c["attitude_rad"][0] for c in subset if c.get("attitude_rad")]
        pitches = [c["attitude_rad"][1] for c in subset if c.get("attitude_rad")]
        vz = [
            c["vehicle_velocity_ned_mps"][2]
            for c in subset
            if c.get("vehicle_velocity_ned_mps") and len(c["vehicle_velocity_ned_mps"]) >= 3
        ]
        thrusts = [c["command"]["thrust"] for c in subset if c.get("command")]
        yaw_sat = sum(1 for c in subset if c.get("saturation", {}).get("yaw_rate_limit"))
        conf = [c.get("gate_confidence", 0.0) for c in subset]
        print(f"  {state}: n={count}")
        if rolls:
            print(f"    roll  min/mean/max = {min(rolls):.3f} / {mean(rolls):.3f} / {max(rolls):.3f}")
        if pitches:
            print(f"    pitch min/mean/max = {min(pitches):.3f} / {mean(pitches):.3f} / {max(pitches):.3f}")
        if vz:
            print(f"    vz    min/mean/max = {min(vz):.3f} / {mean(vz):.3f} / {max(vz):.3f}")
        if thrusts:
            print(f"    thrust min/mean/max = {min(thrusts):.3f} / {mean(thrusts):.3f} / {max(thrusts):.3f}")
        if conf:
            print(f"    gate_conf min/mean/max = {min(conf):.3f} / {mean(conf):.3f} / {max(conf):.3f}")
        print(f"    yaw_rate_limit hits = {yaw_sat}")

    det = [r for r in rows if r.get("event") == "detection"]
    detected = sum(1 for r in det if r.get("detected"))
    visible = sum(1 for r in det if r.get("track_visible"))
    print(f"\n=== Detection: {detected}/{len(det)} detected, {visible}/{len(det)} visible ===")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
