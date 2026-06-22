"""Gate detector for the AI-GP simulator camera stream (GateNet + orange CV)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.perception.gatenet.monorace_gate_detector import MonoRaceGateDetector
from src.perception.gatenet.monorace_perception import MonoRacePerception

COMP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = COMP_ROOT / "models" / "gate_net.pth"


class SimGateDetector:
    """MonoRace stack: GateNet+QuAdGate when available, else color CV fallback."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = "auto",
        min_confidence: float = 0.18,
    ):
        self.backend = backend
        self.min_confidence = min_confidence
        self.monorace = MonoRaceGateDetector(
            min_confidence=min_confidence,
            color_profile="aigp_orange",
        )
        print("[CV] Gate color profile: aigp_orange", flush=True)

        self.monorace_pipeline = None
        self._last_source = "none"

        gate_net_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        want_gatenet = backend in ("auto", "gatenet", "monorace", "monorace_pipeline")
        if want_gatenet and gate_net_path.exists():
            try:
                self.monorace_pipeline = MonoRacePerception(model_path=str(gate_net_path))
                print(f"[CV] MonoRace GateNet pipeline loaded: {gate_net_path}", flush=True)
            except Exception as exc:
                print(f"[CV] GateNet unavailable ({exc}); falling back", flush=True)

        if self.monorace_pipeline is None:
            print("[CV] Using HSV color detector (train GateNet: scripts/train_gatenet.ps1)", flush=True)

    def detect(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        hits = self.detect_all(image)
        return hits[0] if hits else None

    def detect_all(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Return gate detections sorted by apparent size (largest first, up to 3)."""
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        hits: List[Dict[str, Any]] = []

        if self.monorace_pipeline is not None and self.backend != "color":
            for best in self.monorace_pipeline.detect(image)[:3]:
                if best.confidence < self.min_confidence or best.corners_pixel is None:
                    continue
                corners = best.corners_pixel
                x1 = float(corners[:, 0].min())
                y1 = float(corners[:, 1].min())
                x2 = float(corners[:, 0].max())
                y2 = float(corners[:, 1].max())
                packed = self._pack_detection(
                    w, h, x1, y1, x2 - x1, y2 - y1, best.confidence, "gatenet", corners
                )
                refined = self._refine_orange_box(image, packed)
                hits.append(refined if refined is not None else packed)
            if hits:
                self._last_source = str(hits[0].get("source", "gatenet"))
                hits.sort(key=lambda d: -float(d.get("area_fraction", 0.0)))
                return hits

        mono = self.monorace.detect(image)
        if mono and self.backend in ("auto", "color", "monorace"):
            for best in mono[:3]:
                if best.confidence < self.min_confidence:
                    continue
                x, y, bw, bh = best.bbox
                packed = self._pack_detection(w, h, x, y, bw, bh, best.confidence, best.source)
                refined = self._refine_orange_box(image, packed)
                hits.append(refined if refined is not None else packed)
            if hits:
                self._last_source = str(hits[0].get("source", "orange"))
                hits.sort(key=lambda d: -float(d.get("area_fraction", 0.0)))
                return hits

        self._last_source = "none"
        return []

    def _refine_orange_box(
        self,
        image: np.ndarray,
        prior: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        h, w = image.shape[:2]
        bbox = prior.get("bbox")
        if not bbox or len(bbox) != 4:
            return None

        x, y, bw, bh = (float(v) for v in bbox)
        cx = x + bw * 0.5
        cy = y + bh * 0.5
        margin = max(18.0, 0.45 * max(bw, bh))
        x1 = int(np.clip(x - margin, 0, w - 1))
        y1 = int(np.clip(y - margin, 0, h - 1))
        x2 = int(np.clip(x + bw + margin, x1 + 1, w))
        y2 = int(np.clip(y + bh + margin, y1 + 1, h))

        try:
            gate_mask = self.monorace._mask(image)[2]
        except Exception:
            return None
        roi = gate_mask[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, tuple[int, int, int, int], np.ndarray]] = []
        prior_area = max(bw * bh, 1.0)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 12.0:
                continue
            rx, ry, rw, rh = cv2.boundingRect(contour)
            if rw < 4 or rh < 4:
                continue
            gx = x1 + rx + rw * 0.5
            gy = y1 + ry + rh * 0.5
            dist = ((gx - cx) / max(bw, 1.0)) ** 2 + ((gy - cy) / max(bh, 1.0)) ** 2
            box_area = float(rw * rh)
            area_ratio = box_area / prior_area
            if area_ratio > 1.35 or area_ratio < 0.015:
                continue
            aspect = rw / max(float(rh), 1.0)
            if aspect < 0.35 or aspect > 2.6:
                continue
            score = area * (1.0 / (1.0 + dist))
            candidates.append((score, (x1 + rx, y1 + ry, rw, rh), contour))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        _score, (tx, ty, tw, th), _contour = candidates[0]

        pad = max(1, int(round(0.04 * max(tw, th))))
        tx = int(np.clip(tx - pad, 0, w - 1))
        ty = int(np.clip(ty - pad, 0, h - 1))
        tw = int(np.clip(tw + 2 * pad, 1, w - tx))
        th = int(np.clip(th + 2 * pad, 1, h - ty))

        prior_source = str(prior.get("source", "det"))
        source = (
            prior_source
            if prior_source == "orange" or prior_source.endswith("+orange")
            else f"{prior_source}+orange"
        )
        conf = float(max(prior.get("confidence", 0.0), self.min_confidence))
        return self._pack_detection(w, h, tx, ty, tw, th, conf, source)

    @staticmethod
    def _pack_detection(
        w: int,
        h: int,
        x: float,
        y: float,
        bw: float,
        bh: float,
        confidence: float,
        source: str,
        corners: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        cx = x + bw * 0.5
        cy = y + bh * 0.5
        out: Dict[str, Any] = {
            "center_x": (cx - w * 0.5) / max(w * 0.5, 1.0),
            "center_y": (cy - h * 0.5) / max(h * 0.5, 1.0),
            "area_fraction": (bw * bh) / float(max(w * h, 1)),
            "confidence": float(confidence),
            "bbox": [int(x), int(y), int(bw), int(bh)],
            "source": source,
        }
        if corners is not None:
            out["corners"] = corners.astype(float).tolist()
        else:
            x2 = x + bw
            y2 = y + bh
            out["corners"] = [
                [float(x), float(y)],
                [float(x2), float(y)],
                [float(x2), float(y2)],
                [float(x), float(y2)],
            ]
        return out
