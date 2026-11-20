"""Product counting with the exported ONNX detector: numpy + onnxruntime, no torch.

Same pipeline as the browser demo (space/app.js): square letterbox to 1280,
raw model output, greedy NMS. On SKU-110K val it agrees with ultralytics'
own predict on 98% of boxes and counts as accurately.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ONNX_URL = (
    "https://github.com/rsuvbe/shelf-vision/releases/download/detector-v1/yolo11m-sku110k.onnx"
)


@dataclass(frozen=True)
class Letterbox:
    ratio: float
    pad_x: int
    pad_y: int
    width: int
    height: int


def letterbox(bgr: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    """Model input (1, 3, size, size) float32 RGB in [0, 1], and how to map back."""
    import cv2

    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[py : py + nh, px : px + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    return x, Letterbox(r, px, py, w, h)


def decode_image(data: bytes) -> np.ndarray:
    """BGR array from image bytes, with EXIF rotation applied (photos sent as files keep it)."""
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float, max_det: int) -> np.ndarray:
    """Indices kept by greedy NMS, highest score first."""
    order = np.argsort(-scores, kind="stable")
    boxes = boxes[order]
    alive = np.ones(len(order), bool)
    keep = []
    for i in range(len(order)):
        if not alive[i]:
            continue
        keep.append(i)
        if len(keep) >= max_det:
            break
        alive[i + 1 :] &= _iou(boxes[i], boxes[i + 1 :]) <= iou
    return order[np.array(keep, dtype=int)]


def decode(out: np.ndarray, lb: Letterbox, conf: float) -> tuple[np.ndarray, np.ndarray]:
    """Raw output (5, anchors) = cx, cy, w, h, score -> xyxy boxes in original pixels."""
    cx, cy, bw, bh, score = out[:, out[4] >= conf]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - lb.pad_x) / lb.ratio).clip(0, lb.width)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - lb.pad_y) / lb.ratio).clip(0, lb.height)
    return boxes, score


def draw(bgr: np.ndarray, boxes: np.ndarray, gaps: np.ndarray | None = None) -> np.ndarray:
    """Copy of the photo: green rectangle per product, thicker red one per empty spot."""
    import cv2

    out = bgr.copy()
    t = max(2, round(max(bgr.shape[:2]) / 500))
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), (118, 230, 0), t)
    for x1, y1, x2, y2 in (gaps if gaps is not None else np.zeros((0, 4))).astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 40, 235), 2 * t)
    return out


class ShelfCounter:
    """Counts products on shelf photos. Settings default to the config's `infer` section."""

    def __init__(
        self,
        model: Path,
        imgsz: int = 1280,
        iou: float = 0.8,
        max_det: int = 1000,
        threads: int = 0,
    ) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads  # 0 = onnxruntime's default (all cores)
        self.session = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.imgsz, self.iou, self.max_det = imgsz, iou, max_det

    def detect(self, bgr: np.ndarray, conf: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
        """Boxes (n, 4) xyxy in original pixels and their scores, best first."""
        x, lb = letterbox(bgr, self.imgsz)
        out = self.session.run(None, {self.input: x})[0][0]
        boxes, scores = decode(out, lb, conf)
        keep = nms(boxes, scores, self.iou, self.max_det)
        return boxes[keep], scores[keep]


def ensure_model(path: Path) -> Path:
    """Download the ONNX detector from the GitHub release if it is not there yet."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(ONNX_URL, tmp)
        tmp.rename(path)
    return path
