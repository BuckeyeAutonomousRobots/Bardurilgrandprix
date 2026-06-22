"""Structured gate-in-view telemetry for flight logs."""

from __future__ import annotations

import math
from typing import Any, Optional

from src.types import EstimatedState, GateDetection, GateTrack, RacePlan


def gate_sight_record(
    *,
    frame_id: int,
    plan: RacePlan,
    est: EstimatedState,
    camera: dict,
    detection: Optional[GateDetection],
    track: Optional[GateTrack],
    target_gate_index: int,
    detections_count: int = 0,
    all_centers_px: Optional[list[tuple[float, float]]] = None,
) -> dict[str, Any]:
    aim_cx = float(camera["cx"])
    aim_cy = float(camera["desired_v_px"])
    center_px: Optional[tuple[float, float]] = None
    bbox = None
    area_fraction = None
    predicted = False
    visible = False

    if track is not None:
        center_px = (float(track.center_px[0]), float(track.center_px[1]))
        bbox = list(track.bbox)
        area_fraction = float(track.area_fraction)
        predicted = bool(track.predicted)
        visible = bool(track.visible)
    elif detection is not None:
        center_px = (float(detection.center_px[0]), float(detection.center_px[1]))
        bbox = list(detection.bbox)
        area_fraction = float(detection.area_fraction)
        visible = True

    pixel_error_px = None
    if center_px is not None:
        pixel_error_px = [center_px[0] - aim_cx, center_px[1] - aim_cy]

    record: dict[str, Any] = {
        "frame_id": frame_id,
        "fsm_state": plan.state,
        "target_gate_index": target_gate_index,
        "active_gate_index": est.active_gate_index,
        "aim_px": [aim_cx, aim_cy],
        "gate_center_px": None if center_px is None else list(center_px),
        "pixel_error_px": pixel_error_px,
        "gate_bbox": bbox,
        "area_fraction": area_fraction,
        "track_visible": visible,
        "track_predicted": predicted,
        "gate_bearing_rad": [est.gate_bearing_x_rad, est.gate_bearing_y_rad],
        "gate_confidence": est.gate_confidence,
        "gate_range_m": est.gate_range_m,
        "gate_depth_confidence": est.gate_depth_confidence,
        "detections_count": detections_count,
        "all_gate_centers_px": None if not all_centers_px else [list(c) for c in all_centers_px],
    }
    if pixel_error_px is not None:
        record["pixel_error_norm"] = math.hypot(pixel_error_px[0], pixel_error_px[1]) / max(
            float(camera.get("width", 640)),
            1.0,
        )
    if est.vehicle.velocity_ned_mps is not None:
        vx, vy, vz = est.vehicle.velocity_ned_mps
        record["velocity_ned_mps"] = [vx, vy, vz]
        record["horizontal_speed_mps"] = math.hypot(float(vx), float(vy))
    if est.vehicle.position_ned_m is not None:
        record["altitude_m"] = -float(est.vehicle.position_ned_m[2])
    return record
