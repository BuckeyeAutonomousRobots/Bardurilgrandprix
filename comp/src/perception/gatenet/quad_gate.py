"""QuAdGate: precise gate corners from segmentation masks (MonoRace paper)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Corner order: TL, TR, BR, BL (MonoRaceGate dataset convention)
CORNER_NAMES = ("TL", "TR", "BR", "BL")
PRIOR_DESCRIPTORS = {
    "TL": np.array([0, 0, 0, 1], dtype=np.float32),
    "TR": np.array([0, 0, 1, 1], dtype=np.float32),
    "BR": np.array([0, 1, 1, 0], dtype=np.float32),
    "BL": np.array([1, 1, 0, 0], dtype=np.float32),
}


class QuAdGate:
    """Extract gate corners from a binary mask using line intersections."""

    def __init__(
        self,
        lsd_scale: float = 0.8,
        lsd_sigma_scale: float = 0.8,
        lsd_quant: float = 25.0,
        lsd_ang_th: float = 30.0,
        line_extend: float = 5.0 / 3.0,
        max_prior_dist_px: float = 100.0,
        max_translation_px: float = 150.0,
        sample_dist_px: int = 5,
    ):
        self.lsd_scale = lsd_scale
        self.lsd_sigma_scale = lsd_sigma_scale
        self.lsd_quant = lsd_quant
        self.lsd_ang_th = lsd_ang_th
        self.line_extend = line_extend
        self.max_prior_dist_px = max_prior_dist_px
        self.max_translation_px = max_translation_px
        self.sample_dist_px = sample_dist_px

    def detect_corners(
        self,
        mask: np.ndarray,
        priors: Optional[np.ndarray] = None,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Return (4,2) corners [TL, TR, BR, BL] or None."""
        if not CV2_AVAILABLE or mask is None or mask.size == 0:
            return None

        work = self._derotate_mask(mask, roll_rad, pitch_rad)
        binary = (work > 127).astype(np.uint8) * 255
        if binary.max() == 0:
            return None

        lines = self._detect_lines(binary)
        if len(lines) < 4:
            return self._bbox_fallback(binary)

        candidates = self._line_intersections(lines)
        if len(candidates) < 4:
            return self._bbox_fallback(binary)

        if priors is not None and len(priors) == 4:
            matched = self._match_priors(binary, candidates, priors)
            if matched is not None:
                return matched

        return self._pick_best_quad(candidates, binary)

    def _derotate_mask(self, mask: np.ndarray, roll_rad: float, pitch_rad: float) -> np.ndarray:
        if abs(roll_rad) < 1e-4 and abs(pitch_rad) < 1e-4:
            return mask
        h, w = mask.shape[:2]
        center = (w * 0.5, h * 0.5)
        angle_deg = float(np.degrees(roll_rad + 0.5 * pitch_rad))
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

    def _detect_lines(self, binary: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        lsd = cv2.createLineSegmentDetector(
            cv2.LSD_REFINE_STD,
            self.lsd_scale,
            self.lsd_sigma_scale,
            self.lsd_quant,
            self.lsd_ang_th,
        )
        segments, _, _, _ = lsd.detect(binary)
        if segments is None:
            return []

        lines: List[Tuple[np.ndarray, np.ndarray]] = []
        for seg in segments:
            x1, y1, x2, y2 = seg[0]
            p1 = np.array([x1, y1], dtype=np.float32)
            p2 = np.array([x2, y2], dtype=np.float32)
            center = 0.5 * (p1 + p2)
            half = 0.5 * (p2 - p1) * self.line_extend
            lines.append((center - half, center + half))
        return lines

    @staticmethod
    def _intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> Optional[np.ndarray]:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return np.array([px, py], dtype=np.float32)

    def _line_intersections(self, lines: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        candidates: List[np.ndarray] = []
        h, w = 0, 0
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                pt = self._intersect(lines[i][0], lines[i][1], lines[j][0], lines[j][1])
                if pt is None:
                    continue
                candidates.append(pt)
        return candidates

    def _descriptor(self, mask: np.ndarray, point: np.ndarray, line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]
        d = self.sample_dist_px

        def sample(dx: float, dy: float) -> float:
            x = int(np.clip(point[0] + dx, 0, w - 1))
            y = int(np.clip(point[1] + dy, 0, h - 1))
            return 1.0 if mask[y, x] > 127 else 0.0

        dir_a = line_a - point
        dir_b = line_b - point
        norm_a = np.linalg.norm(dir_a)
        norm_b = np.linalg.norm(dir_b)
        if norm_a > 1e-3:
            dir_a = dir_a / norm_a * d
        if norm_b > 1e-3:
            dir_b = dir_b / norm_b * d

        tl = sample(-abs(dir_a[0]) - abs(dir_b[0]), -abs(dir_a[1]) - abs(dir_b[1]))
        tr = sample(abs(dir_a[0]) - abs(dir_b[0]), -abs(dir_a[1]) + abs(dir_b[1]))
        br = sample(abs(dir_a[0]) + abs(dir_b[0]), abs(dir_a[1]) + abs(dir_b[1]))
        bl = sample(-abs(dir_a[0]) + abs(dir_b[0]), abs(dir_a[1]) - abs(dir_b[1]))
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _match_priors(
        self,
        mask: np.ndarray,
        candidates: List[np.ndarray],
        priors: np.ndarray,
    ) -> Optional[np.ndarray]:
        h, w = mask.shape[:2]
        filtered: List[np.ndarray] = []
        for c in candidates:
            if 0 <= c[0] < w and 0 <= c[1] < h:
                filtered.append(c)
        if len(filtered) < 4:
            return None

        src = priors.astype(np.float32)
        dst_list = []
        for prior in priors:
            dists = [np.linalg.norm(c - prior) for c in filtered]
            idx = int(np.argmin(dists))
            if dists[idx] > self.max_prior_dist_px:
                return None
            dst_list.append(filtered[idx])
        dst = np.array(dst_list, dtype=np.float32)

        M, inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )
        if M is None:
            return None

        tx, ty = float(M[0, 2]), float(M[1, 2])
        if abs(tx) > self.max_translation_px or abs(ty) > self.max_translation_px:
            return None

        aligned = cv2.transform(priors.reshape(1, 4, 2).astype(np.float32), M)[0]
        return aligned.astype(np.float32)

    def _pick_best_quad(self, candidates: List[np.ndarray], mask: np.ndarray) -> Optional[np.ndarray]:
        h, w = mask.shape[:2]
        in_bounds = [c for c in candidates if 0 <= c[0] < w and 0 <= c[1] < h]
        if len(in_bounds) < 4:
            return self._bbox_fallback(mask)

        pts = np.array(in_bounds, dtype=np.float32)
        hull = cv2.convexHull(pts)
        if len(hull) < 4:
            return self._bbox_fallback(mask)

        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            return self._order_corners(corners)
        return self._bbox_fallback(mask)

    @staticmethod
    def _order_corners(corners: np.ndarray) -> np.ndarray:
        """Order as TL, TR, BR, BL."""
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1).reshape(-1)
        tl = corners[np.argmin(s)]
        br = corners[np.argmax(s)]
        tr = corners[np.argmin(diff)]
        bl = corners[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _bbox_fallback(self, binary: np.ndarray) -> Optional[np.ndarray]:
        ys, xs = np.where(binary > 0)
        if len(xs) < 4:
            return None
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
