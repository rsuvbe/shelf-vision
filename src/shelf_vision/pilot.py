"""Tools for the shop pilot: pre-label real photos, then score the model on them.

1. `shelf prelabel <photos>` writes a YOLO-format folder (images/, labels/,
   classes.txt) with the model's boxes, so labelling is fixing boxes rather
   than drawing ~150 of them per photo.
2. Correct the labels in any YOLO-capable tool (X-AnyLabeling, labelImg, ...).
3. `shelf pilot-eval <folder>` compares the model with the corrected labels:
   product-count error per photo and box precision/recall at IoU 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(src: Path) -> list[Path]:
    """Image files under `src` (recursively), sorted."""
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def to_yolo(boxes: np.ndarray, w: int, h: int) -> str:
    """xyxy pixel boxes -> YOLO label text (class 0, normalised centre/size)."""
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def from_yolo(text: str, w: int, h: int) -> np.ndarray:
    """YOLO label text -> xyxy pixel boxes (any class)."""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if not rows:
        return np.zeros((0, 4), np.float32)
    a = np.array([[float(v) for v in r[1:5]] for r in rows], np.float32)
    cx, cy, bw, bh = a[:, 0] * w, a[:, 1] * h, a[:, 2] * w, a[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def match_count(pred: np.ndarray, gt: np.ndarray, iou: float = 0.5) -> int:
    """True positives: greedy one-to-one matching of predictions (best first) to labels."""
    if not len(pred) or not len(gt):
        return 0
    x1 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y1 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x2 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y2 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ious = inter / (area(pred)[:, None] + area(gt)[None, :] - inter + 1e-9)
    taken = np.zeros(len(gt), bool)
    tp = 0
    for row in ious:
        cand = np.where(taken, -1, row)
        j = int(cand.argmax())
        if cand[j] >= iou:
            taken[j] = True
            tp += 1
    return tp


@dataclass
class PhotoScore:
    name: str
    labelled: int
    predicted: int
    matched: int


def summarise(scores: list[PhotoScore]) -> str:
    err = np.array([s.predicted - s.labelled for s in scores], float)
    lab = np.array([s.labelled for s in scores], float)
    tp = sum(s.matched for s in scores)
    n_pred = sum(s.predicted for s in scores)
    n_lab = int(lab.sum())
    rel = np.abs(err) / np.maximum(lab, 1)
    return (
        f"{len(scores)} photos, {n_lab} labelled products ({lab.mean():.0f} per photo)\n"
        f"count error per photo: {np.abs(err).mean():.1f} products ({rel.mean() * 100:.1f}%), "
        f"bias {err.mean():+.1f}\n"
        f"boxes at IoU 0.5: precision {tp / max(n_pred, 1):.3f}, recall {tp / max(n_lab, 1):.3f}"
    )
