"""MonoRace-style perception: crop → GateNet → QuAdGate → detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from src.perception.gatenet.base import GateDetection
from src.perception.gatenet.gate_net import GateNetDetector


@dataclass
class MonoRacePerceptionConfig:
    crop_size: int = 384
    mask_threshold: float = 0.5
    use_center_crop_fallback: bool = True


class MonoRacePerception:
    """Full MonoRace perception stack for live camera frames."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[MonoRacePerceptionConfig] = None,
        device: str = "auto",
    ):
        self.cfg = config or MonoRacePerceptionConfig()
        self.detector = GateNetDetector(
            model_path=model_path,
            input_size=self.cfg.crop_size,
            mask_threshold=self.cfg.mask_threshold,
            device=device,
            use_quad_gate=True,
        )
        self._last_priors: Optional[np.ndarray] = None

    def _adaptive_crop(
        self,
        image: np.ndarray,
        priors: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w = image.shape[:2]
        cs = self.cfg.crop_size

        if priors is not None and len(priors) == 4:
            cx = float(np.mean(priors[:, 0]))
            cy = float(np.mean(priors[:, 1]))
        else:
            cx, cy = w * 0.5, h * 0.5

        x1 = int(np.clip(cx - cs * 0.5, 0, max(w - cs, 0)))
        y1 = int(np.clip(cy - cs * 0.5, 0, max(h - cs, 0)))
        x2 = min(x1 + cs, w)
        y2 = min(y1 + cs, h)
        crop = image[y1:y2, x1:x2]
        if crop.shape[0] != cs or crop.shape[1] != cs:
            crop = cv2.resize(crop, (cs, cs))
        return crop, (x1, y1)

    def _center_crop(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w = image.shape[:2]
        cs = self.cfg.crop_size
        x1 = max((w - cs) // 2, 0)
        y1 = max((h - cs) // 2, 0)
        crop = image[y1 : y1 + cs, x1 : x1 + cs]
        if crop.shape[0] != cs or crop.shape[1] != cs:
            crop = cv2.resize(crop, (cs, cs))
        return crop, (x1, y1)

    def detect(self, image: np.ndarray) -> List[GateDetection]:
        if not CV2_AVAILABLE or image is None or image.size == 0:
            return []

        crop, (ox, oy) = self._adaptive_crop(image, self._last_priors)
        dets = self.detector.detect(crop, priors=None)
        if not dets and self.cfg.use_center_crop_fallback:
            crop, (ox, oy) = self._center_crop(image)
            dets = self.detector.detect(crop, priors=None)

        out: List[GateDetection] = []
        for det in dets:
            if det.corners_pixel is not None:
                corners = det.corners_pixel.copy()
                corners[:, 0] += ox
                corners[:, 1] += oy
                center = corners.mean(axis=0)
                self._last_priors = corners
            else:
                center = det.center_pixel.copy()
                center[0] += ox
                center[1] += oy
                corners = None

            out.append(
                GateDetection(
                    center_pixel=center,
                    corners_pixel=corners,
                    confidence=det.confidence,
                )
            )
        return out

    def reset(self) -> None:
        self._last_priors = None
