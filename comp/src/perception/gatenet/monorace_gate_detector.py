"""Classical CV detector for AI-GP / MonoRace-style racing gates."""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class MonoRaceGateDetection:
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    confidence: float
    area: float
    source: str


class MonoRaceGateDetector:
    """Detect racing gate frames by HSV color profile."""

    def __init__(
        self,
        min_area: int = 80,
        min_confidence: float = 0.18,
        color_profile: str = "monorace",
    ):
        self.min_area = min_area
        self.min_confidence = min_confidence
        self.color_profile = color_profile

    def _mask(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.color_profile == "aigp_orange":
            return self._mask_aigp_orange(image)
        return self._mask_monorace(image)

    def _mask_monorace(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([14, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([168, 70, 70]), np.array([179, 255, 255]))
        orange = cv2.inRange(hsv, np.array([5, 70, 80]), np.array([30, 255, 255]))
        cyan = cv2.inRange(hsv, np.array([78, 55, 60]), np.array([108, 255, 255]))

        warm = cv2.bitwise_or(cv2.bitwise_or(red1, red2), orange)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, kernel, iterations=1)
        warm = cv2.dilate(warm, kernel, iterations=1)
        cyan = cv2.morphologyEx(cyan, cv2.MORPH_CLOSE, kernel, iterations=1)
        gate = warm
        return warm, cyan, gate

    def _mask_aigp_orange(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """AI-GP qualifier gates are orange frames on desaturated backgrounds."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Wide orange band; lower S/V mins for desaturated qualifier lighting.
        orange_lo = cv2.inRange(hsv, np.array([6, 35, 55]), np.array([28, 255, 255]))
        orange_hi = cv2.inRange(hsv, np.array([28, 35, 55]), np.array([38, 255, 255]))
        orange = cv2.bitwise_or(orange_lo, orange_hi)
        # BGR fallback catches vivid orange that drifts in HSV under compression.
        b, g, r = cv2.split(image)
        bgr_orange = ((r > 120) & (g > 55) & (g < 210) & (b < 95)).astype(np.uint8) * 255

        gate = cv2.bitwise_or(orange, bgr_orange)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gate = cv2.morphologyEx(gate, cv2.MORPH_CLOSE, kernel, iterations=2)
        gate = cv2.dilate(gate, kernel, iterations=1)
        cyan = np.zeros_like(gate)
        return gate, cyan, gate

    def detect(self, image: np.ndarray) -> List[MonoRaceGateDetection]:
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        warm, cyan, gate = self._mask(image)
        contours, _ = cv2.findContours(gate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[MonoRaceGateDetection] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 5 or bh < 5:
                continue
            if bw > w * 0.8 or bh > h * 0.8:
                continue

            aspect = bw / max(float(bh), 1.0)
            if aspect < 0.35 or aspect > 2.4:
                continue

            roi_warm = warm[y:y + bh, x:x + bw]
            roi_cyan = cyan[y:y + bh, x:x + bw]
            warm_px = int(np.count_nonzero(roi_warm))
            cyan_px = int(np.count_nonzero(roi_cyan))
            min_warm_px = 12 if self.color_profile == "aigp_orange" else 20
            if warm_px < min_warm_px:
                continue

            fill = (warm_px + cyan_px) / float(max(bw * bh, 1))
            center_bias = 1.0 - 0.25 * abs((x + bw * 0.5) - w * 0.5) / max(w * 0.5, 1.0)
            size_score = min(1.0, (bw * bh) / float(w * h) * 24.0)
            if self.color_profile == "aigp_orange":
                color_score = min(1.0, warm_px / 140.0)
            else:
                color_score = min(1.0, warm_px / 180.0) * 0.65 + min(1.0, cyan_px / 600.0) * 0.35
            shape_score = min(aspect, 1.0 / max(aspect, 0.01))
            compactness = min(1.0, area / max(float(bw * bh), 1.0) * 3.0)
            confidence = float(max(0.0, min(1.0, color_score * 0.55 + size_score * 0.25 + shape_score * 0.15 + fill * 0.05)) * center_bias)
            confidence *= compactness

            if confidence < self.min_confidence:
                continue

            if self.color_profile == "aigp_orange":
                source = "orange"
            elif cyan_px:
                source = "warm+cyan"
            else:
                source = "warm"

            detections.append(MonoRaceGateDetection(
                bbox=(x, y, bw, bh),
                center=(x + bw * 0.5, y + bh * 0.5),
                confidence=confidence,
                area=area,
                source=source,
            ))

        detections.sort(key=lambda det: det.confidence, reverse=True)
        return detections

    def detect_with_viz(self, image: np.ndarray) -> Tuple[List[MonoRaceGateDetection], np.ndarray]:
        detections = self.detect(image)
        viz = image.copy()
        for idx, det in enumerate(detections):
            x, y, w, h = det.bbox
            color = (0, 255, 0) if idx == 0 else (0, 220, 255)
            cv2.rectangle(viz, (x, y), (x + w, y + h), color, 2)
            cx, cy = int(det.center[0]), int(det.center[1])
            cv2.circle(viz, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(
                viz,
                f"gate {det.confidence:.2f}",
                (x, max(14, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        return detections, viz
