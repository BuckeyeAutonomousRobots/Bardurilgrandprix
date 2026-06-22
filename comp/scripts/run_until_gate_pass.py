"""Run src.main repeatedly until simulator reports a gate pass."""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time


def latest_log(log_dir: str) -> str:
    paths = glob.glob(os.path.join(log_dir, "sim_stack_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no logs in {log_dir}")
    return max(paths, key=os.path.getmtime)


def log_gate_pass(log_path: str) -> tuple[bool, str]:
    race_started = False
    pass_transition = False
    gate_one_after_race = False
    flight_active = False
    prev_gate_idx: int | None = None
    details: list[str] = []

    with open(log_path, encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") == "fsm_transition":
                if event.get("new_state") == "TAKEOFF":
                    flight_active = True
                    prev_gate_idx = None
                if (
                    event.get("new_state") == "PASS_GATE"
                    and event.get("reason") == "active_gate_index_advanced"
                    and int(event.get("active_gate_index", 0)) >= 1
                ):
                    pass_transition = True
                    details.append(
                        f"PASS_GATE at plane={event.get('map_plane_signed_m')}"
                    )
            if event.get("event") == "race_status" and flight_active:
                start_ms = int(event.get("race_start_boot_time_ms", -1))
                if start_ms >= 0:
                    race_started = True
                gate_idx = int(event.get("active_gate_index", -1))
                if race_started and prev_gate_idx is not None and prev_gate_idx < 1 and gate_idx >= 1:
                    gate_one_after_race = True
                if race_started and gate_idx >= 0:
                    prev_gate_idx = gate_idx

    if pass_transition:
        return True, "; ".join(details) or "FSM PASS_GATE on gate advance"
    if gate_one_after_race:
        return True, "race_status active_gate_index advanced 0->1 after takeoff"
    return False, f"pass_fsm={pass_transition} gate1_after_race={gate_one_after_race}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=45.0)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument(
        "--sim-config",
        default="config/sim.yaml",
        help="Sim config path relative to comp root (config/sim_comp.yaml for vision-primary)",
    )
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    for attempt in range(1, args.max_attempts + 1):
        print(f"[attempt {attempt}/{args.max_attempts}] starting sim run...", flush=True)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.main",
                "--max-seconds",
                str(args.max_seconds),
                "--sim-config",
                args.sim_config,
                "--save-vision-captures",
            ],
            cwd=root,
        )
        if proc.returncode != 0:
            print(f"  run exited {proc.returncode}", flush=True)

        time.sleep(0.2)
        log_path = latest_log(os.path.join(root, args.log_dir))
        passed, detail = log_gate_pass(log_path)
        print(f"  log={log_path}", flush=True)
        print(f"  result={detail}", flush=True)
        if passed:
            print(f"GATE PASS VERIFIED: {detail}", flush=True)
            return 0

    print("FAILED: no gate pass within attempt limit", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
