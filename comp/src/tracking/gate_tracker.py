from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.types import GateDetection, GateTrack


@dataclass
class GateTracker:
    alpha: float = 0.35
    dropout_s: float = 0.30
    confidence_decay: float = 0.85
    _track: Optional[GateTrack] = None

    def reset(self) -> None:
        self._track = None

    def update(self, detection: Optional[GateDetection], now_s: float) -> Optional[GateTrack]:
        if detection is None:
            if self._track is None:
                return None
            if now_s - self._track.timestamp_s > self.dropout_s:
                self._track = None
                return None
            self._track = GateTrack(
                frame_id=self._track.frame_id,
                timestamp_s=self._track.timestamp_s,
                center_px=self._track.center_px,
                bbox=self._track.bbox,
                confidence=self._track.confidence * self.confidence_decay,
                area_fraction=self._track.area_fraction,
                source=self._track.source,
                visible=False,
                predicted=True,
                missed_frames=self._track.missed_frames + 1,
            )
            return self._track

        if self._track is None:
            self._track = GateTrack(
                frame_id=detection.frame_id,
                timestamp_s=detection.timestamp_s,
                center_px=detection.center_px,
                bbox=detection.bbox,
                confidence=detection.confidence,
                area_fraction=detection.area_fraction,
                source=detection.source,
                visible=True,
                predicted=False,
                missed_frames=0,
            )
            return self._track

        a = self.alpha
        cx = (1.0 - a) * self._track.center_px[0] + a * detection.center_px[0]
        cy = (1.0 - a) * self._track.center_px[1] + a * detection.center_px[1]
        bbox = tuple(
            int(round((1.0 - a) * old + a * new))
            for old, new in zip(self._track.bbox, detection.bbox)
        )
        self._track = GateTrack(
            frame_id=detection.frame_id,
            timestamp_s=detection.timestamp_s,
            center_px=(cx, cy),
            bbox=bbox,
            confidence=(1.0 - a) * self._track.confidence + a * detection.confidence,
            area_fraction=(1.0 - a) * self._track.area_fraction + a * detection.area_fraction,
            source=detection.source,
            visible=True,
            predicted=False,
            missed_frames=0,
        )
        return self._track
