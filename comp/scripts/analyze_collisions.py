#!/usr/bin/env python3
"""Summarize high-speed / misaligned samples from latest sim log."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    logs = sorted((root / "logs").glob("sim_stack_*.jsonl"), key=lambda p: p.stat().st_mtime)
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else logs[-1]
    rows: list[dict] = []
    fsm_rows: list[dict] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("event") == "fsm_transition":
            fsm_rows.append(d)
        if d.get("event") != "control":
            continue
        vel = d.get("vehicle_velocity_ned_mps") or [0.0, 0.0, 0.0]
        att = d.get("vehicle_attitude_rad") or [0.0, 0.0, 0.0]
        rows.append(
            {
                "fsm": d.get("fsm_state"),
                "gate": d.get("active_gate_index"),
                "dist": d.get("map_dist_center_m"),
                "plane": d.get("map_plane_signed_m"),
                "bounds": d.get("map_within_gate_bounds"),
                "lat": d.get("map_lateral_error_m"),
                "vert": d.get("map_vertical_error_m"),
                "conf": d.get("gate_confidence"),
                "sp": math.hypot(*vel),
                "vy": vel[1],
                "roll": att[0],
            }
        )

    print(f"log: {log}")
    print(f"control_samples: {len(rows)}")
    print(f"fsm_transitions: {len(fsm_rows)}")
    if not rows:
        return 0

    fast = sorted([r for r in rows if r["sp"] > 3.0], key=lambda r: -r["sp"])[:15]
    print("\n=== top speeds (>3 m/s) ===")
    for r in fast:
        print(
            f"  sp={r['sp']:.2f} fsm={r['fsm']} gate={r['gate']} dist={r['dist']} "
            f"plane={r['plane']} bounds={r['bounds']} lat={r['lat']} vy={r['vy']:.2f}"
        )

    risky = [
        r
        for r in rows
        if r["dist"] is not None
        and r["dist"] < 5.0
        and (not r["bounds"] or (r["lat"] is not None and abs(r["lat"]) > 0.5))
        and r["sp"] > 2.0
    ]
    print(f"\n=== near gate, fast, misaligned: {len(risky)} ===")
    for r in risky[:12]:
        print(
            f"  sp={r['sp']:.2f} dist={r['dist']:.2f} lat={r['lat']} bounds={r['bounds']} "
            f"plane={r['plane']} fsm={r['fsm']}"
        )

    collisions = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if '"event": "collision"' in line or '"event":"collision"' in line]
    print(f"\n=== mavlink collisions logged: {len(collisions)} ===")
    for c in collisions[:10]:
        print(f"  id={c.get('id')} threat={c.get('threat_level')} impact={c.get('impact')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
