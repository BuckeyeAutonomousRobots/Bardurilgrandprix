"""Pick the best gate detection when multiple gates are visible."""

from __future__ import annotations

import math
from typing import Optional

from src.types import GateDetection


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def score_gate_detection(
    detection: GateDetection,
    *,
    image_w: int,
    image_h: int,
    aim_cx: float,
    aim_cy: float,
    seeking_next_gate: bool,
    max_acquire_area: float,
    min_next_gate_area: float,
) -> float:
    """Higher score = better target for the current race leg."""
    dx = float(detection.center_px[0]) - aim_cx
    dy = float(detection.center_px[1]) - aim_cy
    norm = max(math.hypot(float(image_w), float(image_h)), 1.0)
    center_dist = math.hypot(dx, dy) / norm
    center_score = clamp(1.0 - center_dist * 2.8, 0.05, 1.0)

    area = float(detection.area_fraction)
    if seeking_next_gate:
        if area >= max_acquire_area:
            size_score = 0.08
        elif area < min_next_gate_area:
            size_score = clamp(area / max(min_next_gate_area, 1e-4), 0.2, 0.7)
        else:
            size_score = 1.0
    elif area >= max_acquire_area:
        size_score = 0.15
    else:
        size_score = clamp(area / 0.12, 0.35, 1.0)

    return float(detection.confidence) * center_score * size_score


def select_gate_detection(
    detections: list[GateDetection],
    *,
    image_w: int,
    image_h: int,
    aim_cx: float,
    aim_cy: float,
    target_gate_index: int,
    seeking_next_gate: bool = False,
    max_acquire_area: float = 0.45,
    min_next_gate_area: float = 0.006,
) -> Optional[GateDetection]:
    if not detections:
        return None
    if len(detections) == 1:
        return detections[0]

    scored = [
        (
            score_gate_detection(
                det,
                image_w=image_w,
                image_h=image_h,
                aim_cx=aim_cx,
                aim_cy=aim_cy,
                seeking_next_gate=seeking_next_gate or target_gate_index > 0,
                max_acquire_area=max_acquire_area,
                min_next_gate_area=min_next_gate_area,
            ),
            det,
        )
        for det in detections
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]
