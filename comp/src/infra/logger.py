from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(self, log_path: str) -> None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8", buffering=1)
        self._last_flush_s = time.monotonic()

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()

    def log(self, event: str, payload: dict[str, Any]) -> None:
        row = {"ts": time.time(), "event": event, **payload}
        self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        now_s = time.monotonic()
        if event in {"startup", "shutdown", "fsm_transition", "control_saturation_start", "control_saturation_persist"}:
            self._fh.flush()
            self._last_flush_s = now_s
        elif (now_s - self._last_flush_s) >= 0.25:
            self._fh.flush()
            self._last_flush_s = now_s
