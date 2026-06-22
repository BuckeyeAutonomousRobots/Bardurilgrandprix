"""GateNet: U-Net gate segmentation (MonoRace paper architecture)."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from src.perception.gatenet.base import GateDetection, GateDetector

INPUT_SIZE = 384


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = _DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class GateNet(nn.Module):
    """U-Net with multi-scale deep supervision (MonoRace GateNet, f=4)."""

    def __init__(self, factor: int = 4):
        super().__init__()
        base = max(64 // factor, 16)

        self.inc = _DoubleConv(3, base)
        self.down1 = _Down(base, base * 2)
        self.down2 = _Down(base * 2, base * 4)
        self.down3 = _Down(base * 4, base * 8)
        self.down4 = _Down(base * 8, base * 8)

        self.up1 = _Up(base * 8, base * 8, base * 4)
        self.up2 = _Up(base * 4, base * 4, base * 2)
        self.up3 = _Up(base * 2, base * 2, base)
        self.up4 = _Up(base, base, base)

        self.out0 = nn.Sequential(nn.Conv2d(base * 8, 1, 1), nn.Sigmoid())
        self.out1 = nn.Sequential(nn.Conv2d(base * 4, 1, 1), nn.Sigmoid())
        self.out2 = nn.Sequential(nn.Conv2d(base * 2, 1, 1), nn.Sigmoid())
        self.out3 = nn.Sequential(nn.Conv2d(base, 1, 1), nn.Sigmoid())
        self.out4 = nn.Sequential(nn.Conv2d(base, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        u3 = self.up1(x4, x3)
        u2 = self.up2(u3, x2)
        u1 = self.up3(u2, x1)
        u0 = self.up4(u1, x0)

        y0 = self.out4(u0)
        y1 = self.out3(u1)
        y2 = self.out2(u2)
        y3 = self.out1(u3)
        y4 = self.out0(x4)
        return [y0, y1, y2, y3, y4]

    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def gate_net_loss(preds: List[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Multi-scale Dice + BCE (MonoRace Eq. in paper)."""
    weights = [4.0, 2.0, 1.0, 1.0, 1.0]
    total = torch.tensor(0.0, device=target.device)
    for pred, w in zip(preds, weights):
        tgt = target
        if pred.shape[-2:] != target.shape[-2:]:
            tgt = F.interpolate(target, size=pred.shape[-2:], mode="nearest")
        li = dice_loss(pred, tgt) + 2.0 * F.binary_cross_entropy(pred, tgt)
        total = total + w * li
    return total


class GateNetDetector(GateDetector):
    """Segmentation-based gate detector using trained GateNet + optional QuAdGate."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: int = INPUT_SIZE,
        mask_threshold: float = 0.5,
        device: str = "auto",
        use_quad_gate: bool = True,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for GateNet. pip install torch")
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for GateNet detector")

        self.input_size = int(input_size)
        self.mask_threshold = float(mask_threshold)
        self.use_quad_gate = use_quad_gate

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = GateNet(factor=4).to(self.device)
        self.model.eval()

        if model_path and os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state)
            print(f"[GateNet] Loaded {model_path}")
        else:
            print("[GateNet] No weights loaded — segmentation will be untrained")

        self._quad = None
        if use_quad_gate:
            from src.perception.gatenet.quad_gate import QuAdGate

            self._quad = QuAdGate()

    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, float, float, int, int]:
        h, w = image.shape[:2]
        resized = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        sx = w / float(self.input_size)
        sy = h / float(self.input_size)
        return tensor, sx, sy, w, h

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Return uint8 mask (0/255) at original image resolution."""
        tensor, _sx, _sy, w, h = self._preprocess(image)
        with torch.no_grad():
            mask = self.model.predict_mask(tensor)[0, 0].cpu().numpy()
        mask = (mask >= self.mask_threshold).astype(np.uint8) * 255
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    def detect(
        self,
        image: np.ndarray,
        priors: Optional[np.ndarray] = None,
    ) -> List[GateDetection]:
        if image is None or image.size == 0:
            return []

        mask = self.segment(image)
        if mask.max() == 0:
            return []

        h, w = image.shape[:2]
        if self._quad is not None:
            corners = self._quad.detect_corners(mask, priors=priors)
            if corners is not None and len(corners) == 4:
                center = corners.mean(axis=0)
                area = float(cv2.contourArea(corners.astype(np.float32)))
                conf = float(np.clip(area / max(w * h * 0.02, 1.0), 0.2, 0.99))
                return [
                    GateDetection(
                        center_pixel=center,
                        corners_pixel=corners,
                        confidence=conf,
                    )
                ]

        ys, xs = np.where(mask > 0)
        if len(xs) < 20:
            return []
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        center = corners.mean(axis=0)
        area = float((x2 - x1) * (y2 - y1))
        conf = float(np.clip(area / max(w * h * 0.02, 1.0), 0.15, 0.85))
        return [GateDetection(center_pixel=center, corners_pixel=corners, confidence=conf)]

    def estimate_pose(
        self,
        detection: GateDetection,
        camera_matrix: np.ndarray,
        gate_size: float,
    ) -> Optional[np.ndarray]:
        if detection.corners_pixel is None or len(detection.corners_pixel) != 4:
            return None
        half = gate_size / 2.0
        obj = np.array(
            [
                [-half, -half, 0],
                [half, -half, 0],
                [half, half, 0],
                [-half, half, 0],
            ],
            dtype=np.float64,
        )
        img = detection.corners_pixel.astype(np.float64)
        ok, _, tvec = cv2.solvePnP(obj, img, camera_matrix, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            return tvec.flatten()
        return None
