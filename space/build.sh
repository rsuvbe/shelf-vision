#!/usr/bin/env bash
# Fill space/ with the files that are too big for git, then upload with:
#   hf upload <user>/shelf-product-counter space . --repo-type space
# - yolo11m-sku110k.onnx: the detector-v1 weights exported at 1280 (81 MB)
# - onnxruntime-web: the WebGPU bundle and the WASM runtime it loads
set -euo pipefail
cd "$(dirname "$0")"
ORT_VERSION=1.31.1
WEIGHTS_URL=https://github.com/rsuvbe/shelf-vision/releases/download/detector-v1/yolo11m-sku110k.pt
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

if [ ! -f yolo11m-sku110k.onnx ]; then
  curl -sSL -o "$tmp/yolo11m-sku110k.pt" "$WEIGHTS_URL"
  # static 1280x1280 input; NMS runs in app.js
  (cd "$tmp" && uv run --no-project --python 3.12 \
    --with ultralytics==8.4.216 --with torch --with torchvision --with onnx --with onnxslim \
    --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    python -c "from ultralytics import YOLO; YOLO('yolo11m-sku110k.pt').export(format='onnx', imgsz=1280, simplify=True, device='cpu')")
  mv "$tmp/yolo11m-sku110k.onnx" .
fi

(cd "$tmp" && npm pack "onnxruntime-web@$ORT_VERSION" --silent >/dev/null && tar -xzf "onnxruntime-web-$ORT_VERSION.tgz")
for f in ort.webgpu.min.mjs ort-wasm-simd-threaded.asyncify.mjs ort-wasm-simd-threaded.asyncify.wasm; do
  cp "$tmp/package/dist/$f" .
done
ls -la
