---
title: Shelf Product Counter
emoji: 🛒
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
short_description: Count products on a shelf photo, in the browser
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# Shelf product counter

Demo of the product detector from [shelf-vision](https://github.com/rsuvbe/shelf-vision).
Upload a shelf photo; the page draws a box around every product and counts them.

Everything runs in the browser with [onnxruntime-web](https://onnxruntime.ai/):
WebGPU when the device has it, WebAssembly otherwise. The photo is never uploaded.

- Model: YOLO11m trained on SKU-110K (release `detector-v1`), exported to ONNX at 1280.
- Inference: NMS IoU 0.8, confidence 0.4 by default (adjustable).
- First visit downloads ~110 MB (model + runtime); the browser caches it.

The cross-origin headers let WebAssembly use several threads when the page is
opened directly, which makes the CPU fallback several times faster.
