"""Dataset loaders for GateNet training."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.perception.gatenet.synthetic_gate_data import generate_sample

try:
    import torch
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _parse_corners(row: List[str]) -> Optional[np.ndarray]:
    if len(row) < 9:
        return None
    pts = []
    for i in range(1, 9, 2):
        pts.append([float(row[i]), float(row[i + 1])])
    return np.array(pts, dtype=np.float32)


class SyntheticGateDataset:
    """Infinite synthetic dataset for GateNet pre-training."""

    def __init__(self, size: int = 384, length: int = 2000):
        self.size = size
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        del idx
        img, mask = generate_sample(self.size)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = rgb.transpose(2, 0, 1)
        return chw, mask[np.newaxis, ...]


if TORCH_AVAILABLE:

    class GateNetTorchDataset(Dataset):
        def __init__(self, base_dataset: SyntheticGateDataset):
            self.base = base_dataset

        def __len__(self) -> int:
            return len(self.base)

        def __getitem__(self, idx: int):
            x, y = self.base[idx]
            return torch.from_numpy(x), torch.from_numpy(y)


class MonoRaceGateFolderDataset:
    """Load MonoRaceGate images + corners.csv when images are downloaded."""

    def __init__(self, data_root: str, size: int = 384):
        self.size = size
        self.samples: List[Tuple[str, np.ndarray]] = []
        root = Path(data_root)
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            csv_path = folder / "corners.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 9:
                        continue
                    img_name = row[0]
                    corners = _parse_corners(row)
                    if corners is None:
                        continue
                    img_path = folder / img_name
                    if img_path.exists():
                        self.samples.append((str(img_path), corners))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        path, corners = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            return generate_sample(self.size)[0], np.zeros((1, self.size, self.size), np.float32)

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [corners.astype(np.int32)], 255)

        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb.transpose(2, 0, 1), (mask.astype(np.float32) / 255.0)[np.newaxis, ...]


class LiveCaptureFolderDataset:
    """GateNet training samples saved during sim flight (frame_XXXXX.jpg + _mask.png)."""

    def __init__(self, capture_dir: str, size: int = 384):
        self.size = size
        self.samples: List[str] = []
        root = Path(capture_dir)
        for img_path in sorted(root.glob("frame_*.jpg")):
            mask_path = img_path.with_name(img_path.stem + "_mask.png")
            if mask_path.exists():
                self.samples.append(str(img_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img_path = self.samples[idx]
        mask_path = Path(img_path).with_name(Path(img_path).stem + "_mask.png")
        img = cv2.imread(img_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            return generate_sample(self.size)[0], np.zeros((1, self.size, self.size), np.float32)
        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        m = (mask.astype(np.float32) / 255.0)[np.newaxis, ...]
        return rgb.transpose(2, 0, 1), m


def default_monorace_gate_root() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "MonoRaceGate" / "MonoRaceGate-main" / "data"
