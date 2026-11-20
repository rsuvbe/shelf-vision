"""Empty spots on a shelf, from product boxes alone (no training).

Products stand on shelves, so boxes whose bottom edges line up form a row.
Inside a row, a horizontal gap at least `min_gap` typical product widths wide
that no other box covers, standing on the row's shelf line and about as tall
as a product, is reported as an empty spot. Row ends are not checked: there
the photo edge or the end of the shelf is just as likely. Rows are found from
bottom edges, so the photo should face the shelf; on photos taken at a steep
angle the rows, and so the gaps, are unreliable.
"""

from __future__ import annotations

import numpy as np


def rows(boxes: np.ndarray, tol: float = 0.35) -> list[np.ndarray]:
    """Indices of boxes grouped into shelf rows by bottom edge, top to bottom.

    A box joins a row when its bottom is within `tol` x the row's median box
    height of the row's median bottom.
    """
    if not len(boxes):
        return []
    order = np.argsort(boxes[:, 3])
    groups: list[list[int]] = [[order[0]]]
    for i in order[1:]:
        g = boxes[groups[-1]]
        if abs(boxes[i, 3] - np.median(g[:, 3])) <= tol * np.median(g[:, 3] - g[:, 1]):
            groups[-1].append(i)
        else:
            groups.append([i])
    return [np.array(g) for g in groups]


def _covered(rect: np.ndarray, boxes: np.ndarray) -> float:
    """Share of `rect` covered by the union of `boxes` (on a coarse grid)."""
    x1, y1, x2, y2 = rect
    xs = np.linspace(x1, x2, 12)[:-1] + (x2 - x1) / 22
    ys = np.linspace(y1, y2, 8)[:-1] + (y2 - y1) / 14
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    inside = np.zeros(len(pts), bool)
    for b in boxes:
        inside |= (
            (pts[:, 0] >= b[0]) & (pts[:, 0] <= b[2]) & (pts[:, 1] >= b[1]) & (pts[:, 1] <= b[3])
        )
    return float(inside.mean())


def find_gaps(
    boxes: np.ndarray,
    min_gap: float = 0.7,
    min_height: float = 0.6,
    min_row: int = 3,
    max_covered: float = 0.15,
) -> np.ndarray:
    """Empty spots as (n, 4) xyxy rectangles.

    min_gap: gap width as a fraction of the row's median product width.
    min_height: gap height as a fraction of the row's median product height; thin
        strips between stacked boxes are not empty spots.
    min_row: rows with fewer boxes are skipped (too little to judge spacing).
    max_covered: a gap more than this share covered by other boxes is not empty.
    """
    out = []
    for idx in rows(boxes):
        if len(idx) < min_row:
            continue
        r = boxes[idx[np.argsort(boxes[idx, 0])]]
        width = np.median(r[:, 2] - r[:, 0])
        height = np.median(r[:, 3] - r[:, 1])
        shelf = np.median(r[:, 3])  # where the row's products stand
        right = np.maximum.accumulate(r[:, 2])  # rightmost edge so far, for overlaps
        for k in range(len(r) - 1):
            x1, x2 = right[k], r[k + 1, 0]
            if x2 - x1 >= min_gap * width:
                # height of the two neighbours: rows mix tall and short products
                top = min(r[k, 1], r[k + 1, 1])
                bottom = max(r[k, 3], r[k + 1, 3])
                rect = np.array([x1, top, x2, bottom])
                on_shelf = abs(bottom - shelf) <= 0.35 * height
                tall = bottom - top >= min_height * height
                if on_shelf and tall and _covered(rect, boxes) <= max_covered:
                    out.append(rect)
    return np.array(out, dtype=np.float32).reshape(-1, 4)
