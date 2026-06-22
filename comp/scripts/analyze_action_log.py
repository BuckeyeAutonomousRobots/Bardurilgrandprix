#!/usr/bin/env python3
"""Summarize the latest drone action journal (before/after position per action)."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def _latest_action_log(logs_dir: Path) -> Path | None:
    candidates = sorted(logs_dir.glob("sim_stack_*_actions.jsonl"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _resolve_log_path(argv: list[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1])
    root = Path(__file__).resolve().parents[1]
    path = _latest_action_log(root / "logs")
    if path is None:
        raise SystemExit("No action logs found in logs/sim_stack_*_actions.jsonl")
    return path


def main() -> int:
    path = _resolve_log_path(sys.argv)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        print(f"log: {path}")
        print("No actions recorded.")
        return 0

    by_type = Counter(row["action_type"] for row in rows)
    with_after = sum(1 for row in rows if row.get("after") is not None)
    with_position_after = sum(
        1 for row in rows if row.get("after") and row["after"].get("position_ned_m") is not None
    )

    print(f"log: {path}")
    print(f"total_actions: {len(rows)}")
    print(f"with_after_state: {with_after}")
    print(f"with_position_after: {with_position_after}")
    print("\n=== counts by action_type ===")
    for action_type, count in by_type.most_common():
        print(f"  {action_type}: {count}")

    print("\n=== last 15 non-control actions ===")
    significant = [row for row in rows if row["action_type"] != "control_command"]
    for row in significant[-15:]:
        before = row.get("before", {}).get("position_ned_m")
        after = (row.get("after") or {}).get("position_ned_m")
        delta = (row.get("after") or {}).get("position_delta_ned_m")
        print(
            f"  #{row['action_id']} {row['action_type']} "
            f"fsm={row.get('new_state') or row.get('fsm_state', '—')} "
            f"before={before} after={after} delta={delta}"
        )

    print("\n=== sample control actions (first 3, last 3) ===")
    controls = [row for row in rows if row["action_type"] == "control_command"]
    for row in (controls[:3] + controls[-3:]):
        after = row.get("after") or {}
        print(
            f"  #{row['action_id']} {row.get('fsm_state')} thr={row['command']['thrust']:.3f} "
            f"pos_before={row['before'].get('position_ned_m')} "
            f"pos_after={after.get('position_ned_m')} "
            f"delta={after.get('position_delta_ned_m')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
