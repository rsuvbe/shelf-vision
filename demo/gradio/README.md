---
title: Shelf Product Counter
emoji: 🛒
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Count products on a store shelf photo (YOLO11m, SKU-110K)
---

# Shelf product counter

Demo of the product detector from [shelf-vision](https://github.com/rsuvbe/shelf-vision).
Upload a shelf photo; the app draws a box around every product and counts them.

- Model: YOLO11m trained on SKU-110K, weights from the
  [detector-v1 release](https://github.com/rsuvbe/shelf-vision/releases/tag/detector-v1)
  (downloaded on first start).
- Inference: imgsz 1280, NMS IoU 0.8, confidence 0.4 by default.
- Runs on CPU, a few seconds per photo.

This app lives in `demo/gradio/` of the GitHub repo. Hosting a Gradio Space
on Hugging Face needs a PRO account; the public demo is the in-browser version
in `space/`. Locally: `pip install gradio -r requirements.txt && python app.py`.
