"""Base class for gate detectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class GateDetection:
    """A detected gate in camera frame."""

    gate_id: Optional[int] = None
    center_pixel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    corners_pixel: Optional[np.ndarray] = None
    position_camera: Optional[np.ndarray] = None
    position_world: Optional[np.ndarray] = None
    confidence: float = 0.0
    distance_estimate: float = 0.0


class GateDetector(ABC):
    """Abstract base for gate detection approaches."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[GateDetection]:
        """Detect gates in an image. Returns list of detections."""
        ...

    @abstractmethod
    def estimate_pose(
        self, detection: GateDetection, camera_matrix: np.ndarray, gate_size: float
    ) -> Optional[np.ndarray]:
        """Estimate 3D pose of a detected gate relative to camera."""
        ...
