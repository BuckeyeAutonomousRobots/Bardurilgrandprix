"""Print gate-in-view timeline from a sim_stack jsonl log."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def latest_log(log_dir: str) -> str:
    paths = glob.glob(os.path.join(log_dir, "sim_stack_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no logs in {log_dir}")
    return max(paths, key=os.path.getmtime)


def main() -> int:
    parser = argparse.ArgumentParser(description="Show where the drone saw the gate over time")
    parser.add_argument("--log", default="", help="Path to sim_stack jsonl (default: latest in logs/)")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--max-lines", type=int, default=80)
    parser.add_argument("--fsm", default="", help="Filter by FSM state substring")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    log_path = args.log or latest_log(os.path.join(root, args.log_dir))

    rows: list[dict] = []
    with open(log_path, encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") not in {"gate_sight", "detection"}:
                continue
            if args.fsm and args.fsm not in str(event.get("fsm_state", "")):
                continue
            rows.append(event)

    if not rows:
        print(f"No gate_sight/detection events in {log_path}", file=sys.stderr)
        return 1

    tail = rows[-args.max_lines :]
    print(f"log={log_path} events={len(rows)} showing_last={len(tail)}")
    print("frame  state          err_px[x,y]   bearing[x,y]  range  conf  alt   dets")
    for row in tail:
        if row.get("event") == "detection" and not row.get("track_center_px"):
            continue
        frame = row.get("frame_id", "-")
        state = str(row.get("fsm_state", "?"))[:14]
        err = row.get("pixel_error_px")
        if err is None and row.get("track_center_px") and row.get("aim_px"):
            cx, cy = row["track_center_px"]
            ax, ay = row["aim_px"]
            err = [cx - ax, cy - ay]
        err_txt = "   -   " if err is None else f"{err[0]:+5.0f},{err[1]:+5.0f}"
        bearing = row.get("gate_bearing_rad", [0.0, 0.0])
        br_txt = f"{bearing[0]:+.2f},{bearing[1]:+.2f}"
        rng = row.get("gate_range_m")
        rng_txt = f"{rng:5.1f}" if rng is not None else "  -  "
        conf = row.get("gate_confidence", row.get("track_confidence", 0.0))
        alt = row.get("altitude_m")
        alt_txt = f"{alt:4.1f}" if alt is not None else "  - "
        dets = row.get("detections_count", "-")
        print(
            f"{str(frame):>5}  {state:14}  {err_txt}  {br_txt}  {rng_txt}  {conf:.2f}  {alt_txt}  {dets}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
