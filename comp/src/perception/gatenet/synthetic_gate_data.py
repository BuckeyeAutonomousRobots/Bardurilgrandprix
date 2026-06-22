"""Synthetic gate images + masks for GateNet training (MonoRace-style)."""

from __future__ import annotations

import random
from typing import Iterator, Tuple

import cv2
import numpy as np


def _random_background(size: int) -> np.ndarray:
    bg = np.random.randint(15, 90, (size, size, 3), dtype=np.uint8)
    if random.random() < 0.5:
        c1 = np.random.randint(0, 180, 3)
        c2 = np.random.randint(0, 180, 3)
        grad = np.linspace(0, 1, size, dtype=np.float32).reshape(size, 1, 1)
        bg = (c1 * (1 - grad) + c2 * grad).astype(np.uint8)
        bg = np.tile(bg, (1, size, 1))
    if random.random() < 0.4:
        noise = np.random.normal(0, 12, bg.shape).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return bg


def _gate_colors_bgr() -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    profiles = [
        ((0, 0, 220), (255, 220, 0)),      # red frame + cyan center (MonoRace)
        ((0, 140, 255), (40, 40, 40)),       # orange (AI-GP)
        ((0, 200, 255), (30, 30, 30)),
        ((0, 0, 200), (200, 200, 200)),
    ]
    return random.choice(profiles)


def _warp_gate(
    size: int,
    frame_color: Tuple[int, int, int],
    inner_color: Tuple[int, int, int],
    border_frac: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return BGR image tile and binary mask for inner opening."""
    tile = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    margin = random.randint(int(size * 0.08), int(size * 0.22))
    inner = size - 2 * margin
    src = np.array(
        [
            [margin, margin],
            [margin + inner, margin],
            [margin + inner, margin + inner],
            [margin, margin + inner],
        ],
        dtype=np.float32,
    )

    jitter = random.uniform(-0.15, 0.15) * inner
    dst = np.array(
        [
            [margin + jitter, margin - jitter * 0.5],
            [margin + inner - jitter * 0.3, margin + jitter],
            [margin + inner + jitter * 0.2, margin + inner - jitter],
            [margin - jitter, margin + inner + jitter * 0.4],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)

    gate_img = np.full((size, size, 3), inner_color, dtype=np.uint8)
    bw = max(int(inner * border_frac), 3)
    cv2.rectangle(gate_img, (margin, margin), (margin + inner, margin + inner), frame_color, bw)
    inner_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(inner_mask, [dst.astype(np.int32)], 255)

    warped = cv2.warpPerspective(gate_img, M, (size, size), borderValue=inner_color)
    warped_mask = cv2.warpPerspective(inner_mask, M, (size, size), borderValue=0)

    # Frame mask = outer bbox minus inner opening
    frame_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(frame_mask, [dst.astype(np.int32)], 255)
    border_mask = cv2.subtract(frame_mask, warped_mask)
    tile[warped_mask > 0] = warped[warped_mask > 0]
    tile[border_mask > 0] = frame_color

    return tile, border_mask


def _augment_pair(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    if random.random() < 0.6:
        angle = random.uniform(-18, 18)
        M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    if random.random() < 0.5:
        k = random.choice([3, 5, 7, 9])
        angle = random.uniform(0, 180)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        rot = cv2.getRotationMatrix2D((k * 0.5, k * 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot, (k, k))
        image = cv2.filter2D(image, -1, kernel)

    if random.random() < 0.35:
        image = cv2.GaussianBlur(image, (random.choice([3, 5]),) * 2, 0)

    if random.random() < 0.4:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-12, 12)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(-30, 30), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.randint(-35, 35), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return image, mask


def generate_sample(size: int = 384) -> Tuple[np.ndarray, np.ndarray]:
    """One synthetic (BGR image, float mask in [0,1])."""
    bg = _random_background(size)
    frame_c, inner_c = _gate_colors_bgr()
    gate, mask = _warp_gate(size, frame_c, inner_c, border_frac=random.uniform(0.08, 0.16))

    ox = random.randint(0, max(size // 4, 1))
    oy = random.randint(0, max(size // 4, 1))
    scale = random.uniform(0.45, 0.95)
    gw = int(size * scale)
    gate_rs = cv2.resize(gate, (gw, gw))
    mask_rs = cv2.resize(mask, (gw, gw), interpolation=cv2.INTER_NEAREST)

    x2 = min(ox + gw, size)
    y2 = min(oy + gw, size)
    gw_eff = x2 - ox
    gh_eff = y2 - oy
    if gw_eff < 20 or gh_eff < 20:
        return generate_sample(size)

    roi = bg[oy:y2, ox:x2]
    g_crop = gate_rs[:gh_eff, :gw_eff]
    m_crop = mask_rs[:gh_eff, :gw_eff]
    gate_pixels = m_crop > 127
    roi[gate_pixels] = g_crop[gate_pixels]
    bg[oy:y2, ox:x2] = roi

    full_mask = np.zeros((size, size), dtype=np.uint8)
    full_mask[oy:y2, ox:x2] = np.maximum(full_mask[oy:y2, ox:x2], m_crop)

    image, full_mask = _augment_pair(bg, full_mask)
    return image, (full_mask.astype(np.float32) / 255.0)


def synthetic_batch(batch_size: int, size: int = 384) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    for _ in range(batch_size):
        yield generate_sample(size)
