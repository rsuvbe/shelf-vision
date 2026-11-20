// Shelf product counter, in the browser: YOLO11m (ONNX) via onnxruntime-web.
// Settings mirror the `infer` section of configs/detect/yolo11m-sku110k.yaml.
import * as ort from "./ort.webgpu.min.mjs";

const SIZE = 1280; // model input, square letterbox
const IOU = 0.8; // NMS: neighbouring products overlap, 0.7 drops real boxes
const MIN_CONF = 0.1; // keep candidates down to the slider's minimum
const MAX_DET = 1000;
const MODEL_URL = "yolo11m-sku110k.onnx";

const $ = (id) => document.getElementById(id);
const status = (msg) => ($("status").textContent = msg);

let session = null;
let backend = "";
let last = null; // {bitmap, boxes: Float32Array (n*4, original pixels), scores, ms}

async function fetchWithProgress(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    status(
      total
        ? `Downloading the model… ${Math.round((100 * got) / total)}% of ${(total / 1e6).toFixed(0)} MB`
        : `Downloading the model… ${(got / 1e6).toFixed(0)} MB`,
    );
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  return buf;
}

async function hasWebGPU() {
  try {
    return !!(navigator.gpu && (await navigator.gpu.requestAdapter()));
  } catch {
    return false;
  }
}

async function init() {
  ort.env.wasm.wasmPaths = new URL("./", import.meta.url).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(navigator.hardwareConcurrency || 4, 8)
    : 1;
  const model = await fetchWithProgress(MODEL_URL);
  status("Preparing the model…");
  const tryBackends = (await hasWebGPU()) ? ["webgpu", "wasm"] : ["wasm"];
  let lastErr;
  for (const ep of tryBackends) {
    try {
      session = await ort.InferenceSession.create(model, { executionProviders: [ep] });
      backend = ep === "webgpu" ? "GPU (WebGPU)" : `CPU (WebAssembly, ${ort.env.wasm.numThreads} thread${ort.env.wasm.numThreads > 1 ? "s" : ""})`;
      break;
    } catch (e) {
      lastErr = e;
    }
  }
  if (!session) throw lastErr;
  $("pick").classList.remove("disabled");
  status(`Ready, running on ${backend}. Choose a photo.`);
}

function letterbox(bitmap) {
  const w = bitmap.width, h = bitmap.height;
  const r = Math.min(SIZE / w, SIZE / h);
  const nw = Math.round(w * r), nh = Math.round(h * r);
  const px = Math.floor((SIZE - nw) / 2), py = Math.floor((SIZE - nh) / 2);
  const c = new OffscreenCanvas(SIZE, SIZE);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, px, py, nw, nh);
  const rgba = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const n = SIZE * SIZE;
  const x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    x[i] = rgba[4 * i] / 255;
    x[n + i] = rgba[4 * i + 1] / 255;
    x[2 * n + i] = rgba[4 * i + 2] / 255;
  }
  return { x, r, px, py, w, h };
}

// output0: [1, 5, A] = cx, cy, w, h, score for A anchors (one class)
function candidates(out, lb) {
  const A = out.length / 5;
  const idx = [];
  for (let i = 0; i < A; i++) if (out[4 * A + i] >= MIN_CONF) idx.push(i);
  idx.sort((a, b) => out[4 * A + b] - out[4 * A + a]);
  const boxes = new Float32Array(idx.length * 4);
  const scores = new Float32Array(idx.length);
  idx.forEach((i, k) => {
    const cx = out[i], cy = out[A + i], bw = out[2 * A + i], bh = out[3 * A + i];
    const clip = (v, m) => Math.min(Math.max(v, 0), m);
    boxes[4 * k] = clip((cx - bw / 2 - lb.px) / lb.r, lb.w);
    boxes[4 * k + 1] = clip((cy - bh / 2 - lb.py) / lb.r, lb.h);
    boxes[4 * k + 2] = clip((cx + bw / 2 - lb.px) / lb.r, lb.w);
    boxes[4 * k + 3] = clip((cy + bh / 2 - lb.py) / lb.r, lb.h);
    scores[k] = out[4 * A + i];
  });
  return { boxes, scores };
}

// Greedy NMS over score-sorted candidates; returns kept indices.
function nms(boxes, scores, conf) {
  const keep = [];
  const n = scores.length;
  const dead = new Uint8Array(n);
  for (let i = 0; i < n && scores[i] >= conf; i++) {
    if (dead[i]) continue;
    keep.push(i);
    if (keep.length >= MAX_DET) break;
    const x1 = boxes[4 * i], y1 = boxes[4 * i + 1], x2 = boxes[4 * i + 2], y2 = boxes[4 * i + 3];
    const a = (x2 - x1) * (y2 - y1);
    for (let j = i + 1; j < n && scores[j] >= conf; j++) {
      if (dead[j]) continue;
      const ix = Math.min(x2, boxes[4 * j + 2]) - Math.max(x1, boxes[4 * j]);
      const iy = Math.min(y2, boxes[4 * j + 3]) - Math.max(y1, boxes[4 * j + 1]);
      if (ix <= 0 || iy <= 0) continue;
      const inter = ix * iy;
      const b = (boxes[4 * j + 2] - boxes[4 * j]) * (boxes[4 * j + 3] - boxes[4 * j + 1]);
      if (inter / (a + b - inter) > IOU) dead[j] = 1;
    }
  }
  return keep;
}

function draw() {
  if (!last) return;
  const conf = Number($("conf").value);
  const keep = nms(last.boxes, last.scores, conf);
  const { bitmap } = last;
  const cv = $("view");
  cv.width = bitmap.width;
  cv.height = bitmap.height;
  const ctx = cv.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  ctx.lineWidth = Math.max(2, Math.round(Math.max(bitmap.width, bitmap.height) / 500));
  ctx.strokeStyle = "#00e676";
  for (const i of keep) {
    const b = last.boxes.subarray(4 * i, 4 * i + 4);
    ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
  }
  $("count").innerHTML = `${keep.length} products <small>confidence ≥ ${conf.toFixed(2)}</small>`;
  $("result").classList.remove("hidden");
  status(`Done in ${(last.ms / 1000).toFixed(1)} s on ${backend}.`);
}

async function onFile(file) {
  if (!file || !session) return;
  $("pick").classList.add("disabled");
  status("Counting…");
  try {
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    await new Promise((r) => setTimeout(r, 0)); // let the status repaint
    const t = performance.now();
    const lb = letterbox(bitmap);
    const input = new ort.Tensor("float32", lb.x, [1, 3, SIZE, SIZE]);
    const res = await session.run({ images: input });
    const out = res.output0.data;
    const { boxes, scores } = candidates(out, lb);
    last = { bitmap, boxes, scores, ms: performance.now() - t };
    draw();
  } catch (e) {
    status(`Something went wrong: ${e.message || e}`);
    console.error(e);
  } finally {
    $("pick").classList.remove("disabled");
    $("file").value = "";
  }
}

$("file").addEventListener("change", (e) => onFile(e.target.files[0]));
$("conf").addEventListener("input", (e) => {
  $("confv").textContent = Number(e.target.value).toFixed(2);
  draw();
});

// exposed for automated checks
window.__shelf = { ready: () => !!session, run: onFile, result: () => last && nms(last.boxes, last.scores, Number($("conf").value)).length };

init().catch((e) => {
  status(`Could not load the model: ${e.message || e}`);
  console.error(e);
});
