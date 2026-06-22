from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.perception.gate_selector import select_gate_detection
from src.types import GateDetection, VisionFrame


def _order_box_points(points: np.ndarray) -> tuple[tuple[float, float], ...]:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(4)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    return tuple((float(p[0]), float(p[1])) for p in ordered)


class GateDetector:
    """Simulator gate detector: GateNet pipeline when available, else HSV fallback."""

    def __init__(self, min_area_fraction: float = 0.0025) -> None:
        self.min_area_fraction = min_area_fraction
        self._sim_detector = None
        try:
            from src.perception.sim_gate_detector import SimGateDetector

            self._sim_detector = SimGateDetector(backend="auto", min_confidence=0.35)
        except Exception:
            self._sim_detector = None

    def detect(self, frame: VisionFrame) -> Optional[GateDetection]:
        return self.detect_for_target(frame, target_gate_index=0, seeking_next_gate=False)

    def detect_all(self, frame: VisionFrame) -> list[GateDetection]:
        if self._sim_detector is not None:
            hits = self._sim_detector.detect_all(frame.image_bgr)
            dets = []
            for hit in hits:
                det = self._detection_from_sim_hit(frame, hit)
                if det is not None:
                    dets.append(det)
            if dets:
                return dets
        single = self._detect_hsv(frame)
        return [single] if single is not None else []

    def detect_for_target(
        self,
        frame: VisionFrame,
        *,
        target_gate_index: int,
        seeking_next_gate: bool,
        aim_cx: float,
        aim_cy: float,
        max_acquire_area: float = 0.45,
        min_next_gate_area: float = 0.006,
    ) -> Optional[GateDetection]:
        detections = self.detect_all(frame)
        if not detections:
            return None
        h, w = frame.image_bgr.shape[:2]
        return select_gate_detection(
            detections,
            image_w=w,
            image_h=h,
            aim_cx=aim_cx,
            aim_cy=aim_cy,
            target_gate_index=target_gate_index,
            seeking_next_gate=seeking_next_gate,
            max_acquire_area=max_acquire_area,
            min_next_gate_area=min_next_gate_area,
        )

    def _detection_from_sim_hit(self, frame: VisionFrame, hit: dict) -> Optional[GateDetection]:
        bbox = hit.get("bbox")
        corners = hit.get("corners")
        if not bbox or len(bbox) != 4 or not corners or len(corners) != 4:
            return None
        h, w = frame.image_bgr.shape[:2]
        x, y, bw, bh = [int(v) for v in bbox]
        area_fraction = float((bw * bh) / max(float(w * h), 1.0))
        border_margin = max(4, int(round(min(h, w) * 0.02)))
        wide_fraction = float(bw) / max(float(w), 1.0)
        tall_fraction = float(bh) / max(float(h), 1.0)
        touches_border = (
            x <= border_margin
            or y <= border_margin
            or (x + bw) >= (w - border_margin)
            or (y + bh) >= (h - border_margin)
        )
        aspect = min(float(bw), float(bh)) / max(float(max(bw, bh)), 1.0)
        if area_fraction >= 0.75:
            return None
        if wide_fraction >= 0.72 or tall_fraction >= 0.85:
            return None
        if touches_border and area_fraction >= 0.60:
            return None
        if y <= max(12, border_margin * 2) and tall_fraction >= 0.75:
            return None
        if aspect < 0.45:
            return None
        return GateDetection(
            frame_id=frame.frame_id,
            timestamp_s=frame.wall_time_s,
            center_px=(float(x + bw * 0.5), float(y + bh * 0.5)),
            bbox=(x, y, bw, bh),
            corners_px=tuple((float(pt[0]), float(pt[1])) for pt in corners),
            confidence=float(hit.get("confidence", 0.0)),
            area_fraction=area_fraction,
            source=str(hit.get("source", "sim_detector")),
        )

    def _detect_with_sim_detector(self, frame: VisionFrame) -> Optional[GateDetection]:
        hit = self._sim_detector.detect(frame.image_bgr)
        if not hit:
            return None
        return self._detection_from_sim_hit(frame, hit)

    def _detect_hsv(self, frame: VisionFrame) -> Optional[GateDetection]:
        image = frame.image_bgr
        h, w = image.shape[:2]

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (5, 80, 80), (25, 255, 255))
        mask2 = cv2.inRange(hsv, (0, 40, 120), (35, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        image_area = float(w * h)
        border_margin = int(round(min(h, w) * 0.03))

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= image_area * self.min_area_fraction:
                continue
            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), _angle = rect
            if rw < 5.0 or rh < 5.0:
                continue
            aspect = min(rw, rh) / max(rw, rh)
            if aspect < 0.45:
                continue
            box = cv2.boxPoints(rect)
            box_area = max(float(rw * rh), 1.0)
            fill_ratio = area / box_area
            x, y, bw, bh = cv2.boundingRect(contour)
            if x <= border_margin or y <= border_margin or (x + bw) >= (w - border_margin) or (y + bh) >= (h - border_margin):
                continue
            if fill_ratio < 0.18 or fill_ratio > 0.88:
                continue
            score = area * aspect * min(fill_ratio, 1.2)
            if score > best_score:
                best = {
                    "center_px": (float(cx), float(cy)),
                    "bbox": (int(x), int(y), int(bw), int(bh)),
                    "corners_px": _order_box_points(box),
                    "area_fraction": box_area / image_area,
                    "confidence": float(np.clip(0.65 * aspect + 0.35 * (1.0 - abs(fill_ratio - 0.45)), 0.0, 1.0)),
                }
                best_score = score

        if best is None:
            return None

        return GateDetection(
            frame_id=frame.frame_id,
            timestamp_s=frame.wall_time_s,
            center_px=best["center_px"],
            bbox=best["bbox"],
            corners_px=best["corners_px"],
            confidence=best["confidence"],
            area_fraction=best["area_fraction"],
        )
