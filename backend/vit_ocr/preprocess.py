"""Chunk normalisation shared by training-data synthesis and inference.

Every text chunk (a phrase or table cell cut out of a report) is converted to the same
canvas the Vision Transformer was trained on: dark text on a light background, cropped to the
ink bounding box plus a margin, scaled to a fixed height and left-aligned on a fixed width.
"""
from __future__ import annotations

import cv2
import numpy as np

H = 32
W = 512
MARGIN = 0.12


def ink_mask(g: np.ndarray) -> np.ndarray:
    """Binary ink mask (True = ink) for a dark-on-light grayscale chunk, specks removed."""
    blur = cv2.GaussianBlur(g, (3, 3), 0)
    thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (blur < thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return ink.astype(bool)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 3
    return keep[lab]


def to_dark_on_light(g: np.ndarray) -> np.ndarray:
    return 255 - g if np.median(g) < 110 else g


def scaled_width(w: int, h: int) -> int:
    pad = int(round(MARGIN * h))
    return int(round((w + 2 * pad) * H / (h + 2 * pad)))


def normalize_chunk(g: np.ndarray, return_bbox: bool = False):
    """g: 2-D uint8 grayscale chunk. Returns a (H, W) uint8 array, or None if there is no ink
    or the chunk is too wide for the canvas (caller should split it first)."""
    g = to_dark_on_light(g)
    ink = ink_mask(g)
    if not ink.any():
        return None
    ys, xs = np.where(ink)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    if h < 3 or w < 2:
        return None
    pad = int(round(MARGIN * h))
    bg = int(np.median(g[~ink])) if (~ink).any() else 255
    crop = cv2.copyMakeBorder(g[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad],
                              max(0, pad - y0), max(0, y1 + pad - g.shape[0]),
                              max(0, pad - x0), max(0, x1 + pad - g.shape[1]),
                              cv2.BORDER_CONSTANT, value=bg)
    lo, hi = np.percentile(crop, 1), np.percentile(crop, 99)
    if hi - lo < 20:
        return None
    crop = np.clip((crop.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    scale = H / crop.shape[0]
    new_w = max(4, int(round(crop.shape[1] * scale)))
    if new_w > W:
        return None
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    crop = cv2.resize(crop, (new_w, H), interpolation=interp)
    canvas = np.full((H, W), 255, np.uint8)
    canvas[:, :new_w] = crop
    if return_bbox:
        return canvas, (int(x0), int(y0), int(x1), int(y1))
    return canvas
