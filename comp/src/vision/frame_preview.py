from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.types import GateDetection, GateTrack


@dataclass
class VisionOverlay:
    fsm_state: str = ""
    gate_confidence: float = 0.0
    active_gate_index: int = -1
    gate_bearing_x_rad: float = 0.0
    gate_bearing_y_rad: float = 0.0
    gate_range_m: Optional[float] = None
    gate_depth_confidence: float = 0.0
    detections_visible: int = 0
    altitude_m: Optional[float] = None


def annotate_vision_frame(
    image_bgr: np.ndarray,
    *,
    gate_track: Optional[GateTrack],
    detection: Optional[GateDetection] = None,
    overlay: Optional[VisionOverlay] = None,
    camera_cx: Optional[float] = None,
    camera_cy: Optional[float] = None,
) -> np.ndarray:
    viz = image_bgr.copy()
    h, w = viz.shape[:2]
    cx = float(camera_cx if camera_cx is not None else w * 0.5)
    cy = float(camera_cy if camera_cy is not None else h * 0.5)
    cv2.drawMarker(
        viz,
        (int(cx), int(cy)),
        (200, 200, 200),
        markerType=cv2.MARKER_CROSS,
        markerSize=14,
        thickness=1,
        line_type=cv2.LINE_AA,
    )

    if detection is not None and detection.corners_px:
        corners = np.array(detection.corners_px, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(viz, [corners], isClosed=True, color=(255, 180, 0), thickness=2, lineType=cv2.LINE_AA)

    if gate_track is not None and gate_track.visible and gate_track.bbox[2] > 0 and gate_track.bbox[3] > 0:
        x, y, bw, bh = gate_track.bbox
        color = (0, 220, 255) if gate_track.predicted else (0, 255, 80)
        cv2.rectangle(viz, (x, y), (x + bw, y + bh), color, 2, lineType=cv2.LINE_AA)
        tcx, tcy = int(gate_track.center_px[0]), int(gate_track.center_px[1])
        cv2.circle(viz, (tcx, tcy), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.line(viz, (int(cx), int(cy)), (tcx, tcy), (80, 80, 255), 1, lineType=cv2.LINE_AA)
        label = f"{'pred' if gate_track.predicted else 'track'} {gate_track.confidence:.2f} a={gate_track.area_fraction:.3f}"
        cv2.putText(
            viz,
            label,
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    if overlay is not None:
        lines = [
            f"FSM: {overlay.fsm_state}",
            f"gate={overlay.active_gate_index} conf={overlay.gate_confidence:.2f}",
            f"bearing x={overlay.gate_bearing_x_rad:+.2f} y={overlay.gate_bearing_y_rad:+.2f}",
        ]
        if overlay.gate_range_m is not None:
            lines.append(f"depth={overlay.gate_range_m:.1f}m conf={overlay.gate_depth_confidence:.2f}")
        if overlay.detections_visible > 1:
            lines.append(f"detections={overlay.detections_visible}")
        if gate_track is not None and gate_track.visible:
            err_x = float(gate_track.center_px[0]) - cx
            err_y = float(gate_track.center_px[1]) - cy
            lines.append(f"err_px={err_x:+.0f},{err_y:+.0f}")
        if overlay.altitude_m is not None:
            lines.append(f"alt={overlay.altitude_m:.1f} m")
        y0 = 22
        for line in lines:
            cv2.putText(viz, line, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
            y0 += 18
    return viz


class VisionPreview:
    """OpenCV window for live drone camera feed."""

    def __init__(self, window_name: str = "Drone camera", scale: float = 1.0) -> None:
        self.window_name = window_name
        self.scale = max(0.25, float(scale))
        self._opened = False

    def show(self, image_bgr: np.ndarray) -> bool:
        display = image_bgr
        if self.scale != 1.0:
            display = cv2.resize(
                image_bgr,
                None,
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_LINEAR,
            )
        if not self._opened:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._opened = True
        cv2.imshow(self.window_name, display)
        key = cv2.waitKey(1) & 0xFF
        return key not in (27, ord("q"))

    def close(self) -> None:
        if self._opened:
            cv2.destroyWindow(self.window_name)
            self._opened = False
