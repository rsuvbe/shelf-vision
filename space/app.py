"""Gradio demo: count products on a shelf photo with the shelf-vision detector.

Runs on a free Hugging Face Space (CPU). Settings mirror the `infer` section of
configs/detect/yolo11m-sku110k.yaml in https://github.com/rsuvbe/shelf-vision.
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import gradio as gr
import numpy as np
from ultralytics import YOLO

WEIGHTS_URL = (
    "https://github.com/rsuvbe/shelf-vision/releases/download/detector-v1/yolo11m-sku110k.pt"
)
WEIGHTS = Path(__file__).with_name("yolo11m-sku110k.pt")

IMGSZ = 1280  # trained at 960; 1280 scores higher on dense shelves
IOU = 0.8  # neighbouring products overlap, the default 0.7 drops real boxes
CONF = 0.4  # lowest product-count error on SKU-110K val (9.4% per photo)
MAX_DET = 1000  # a shelf photo can hold several hundred products

if not WEIGHTS.exists():
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
model = YOLO(str(WEIGHTS))


def detect(image: np.ndarray | None, conf: float) -> tuple[np.ndarray | None, str]:
    if image is None:
        return None, "Upload a shelf photo."
    t = time.perf_counter()
    # Gradio gives RGB; ultralytics expects BGR for numpy input
    r = model.predict(
        image[:, :, ::-1], imgsz=IMGSZ, iou=IOU, conf=conf, max_det=MAX_DET, verbose=False
    )[0]
    dt = time.perf_counter() - t
    plotted = r.plot(labels=False, conf=False)[:, :, ::-1]
    return plotted, f"## {len(r.boxes)} products\n\nconfidence ≥ {conf:.2f} · {dt:.1f} s on CPU"


with gr.Blocks(title="Shelf product counter") as demo:
    gr.Markdown(
        "# Shelf product counter\n"
        "Upload a photo of a store shelf: the model finds every product and counts them. "
        "YOLO11m trained on [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) "
        "(test mAP50-95 ≈ 0.616; product count within ~9% per photo on its val split). "
        "It detects products, it does not tell which product is which yet. "
        "Code: [rsuvbe/shelf-vision](https://github.com/rsuvbe/shelf-vision)."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Shelf photo")
            conf = gr.Slider(
                0.1,
                0.9,
                value=CONF,
                step=0.05,
                label="Confidence threshold",
                info="Lower finds more products but adds false boxes; 0.4 counts best.",
            )
            btn = gr.Button("Count products", variant="primary")
        with gr.Column():
            out = gr.Image(type="numpy", label="Detected products")
            count = gr.Markdown()
    btn.click(detect, [inp, conf], [out, count])
    inp.upload(detect, [inp, conf], [out, count])

if __name__ == "__main__":
    demo.launch()
