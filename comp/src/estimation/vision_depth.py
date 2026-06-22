"""Monocular gate depth from bbox / corners."""

from __future__ import annotations

import math
from typing import Optional

from src.types import GateTrack


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _corner_span_px(corners: tuple[tuple[float, float], ...]) -> Optional[tuple[float, float]]:
    if len(corners) != 4:
        return None
    xs = [float(p[0]) for p in corners]
    ys = [float(p[1]) for p in corners]
    return max(max(xs) - min(xs), 1.0), max(max(ys) - min(ys), 1.0)


def estimate_gate_depth_m(
    gate_track: GateTrack,
    camera: dict,
    *,
    corners_px: Optional[tuple[tuple[float, float], ...]] = None,
) -> tuple[float, float, float, float]:
    """Return (range_m, range_x_m, range_y_m, confidence in 0..1)."""
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    gate_inner_m = float(camera.get("gate_inner_size_m", 1.5))
    bw = max(float(gate_track.bbox[2]), 1.0)
    bh = max(float(gate_track.bbox[3]), 1.0)
    range_x = gate_inner_m * fx / bw
    range_y = gate_inner_m * fy / bh

    if corners_px:
        span = _corner_span_px(corners_px)
        if span is not None:
            cw, ch = span
            range_x = gate_inner_m * fx / cw
            range_y = gate_inner_m * fy / ch

    area_fraction = max(float(gate_track.area_fraction), 1e-5)
    image_area_scale = float(camera.get("width", 640)) * float(camera.get("height", 360))
    _ = image_area_scale  # reserved for future calibrated area model
    range_area = gate_inner_m * math.sqrt(float(camera.get("vision_depth_area_gain", 9.0)) / area_fraction)

    spread = abs(range_x - range_y) / max(0.5 * (range_x + range_y), 1.0)
    bbox_conf = clamp(1.0 - spread, 0.15, 1.0)
    range_bbox = 0.5 * (range_x + range_y)
    range_m = 0.65 * range_bbox + 0.35 * range_area
    confidence = clamp(0.55 * bbox_conf + 0.45 * float(gate_track.confidence), 0.0, 1.0)
    return range_m, range_x, range_y, confidence
