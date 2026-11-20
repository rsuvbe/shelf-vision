# shelf-vision

Retail shelf monitoring: detect every product on a shelf photo, recognise which
SKU it is, and report what is missing or misplaced.

Status: product detection is done (YOLO11m, test mAP50-95 ~0.616, product
count within ~9% per shelf photo), with an experimental empty-spot marker, a
web demo and a Telegram bot. Next: a pilot on photos from real shops, then
SKU recognition.

## Detector

Weights: [`yolo11m-sku110k.pt`](https://github.com/rsuvbe/shelf-vision/releases/download/detector-v1/yolo11m-sku110k.pt)
(GitHub release `detector-v1`, 40 MB). Inference settings live in the `infer`
section of `configs/detect/yolo11m-sku110k.yaml`: imgsz 1280, NMS IoU 0.8,
conf 0.4.

```bash
mkdir -p runs/weights
gh release download detector-v1 -p yolo11m-sku110k.pt -D runs/weights
uv run shelf predict runs/weights/yolo11m-sku110k.pt configs/detect/yolo11m-sku110k.yaml <photo, folder or .txt list>
# -> runs/predict/yolo11m-sku110k/: images with boxes + counts.json
uv run shelf evaluate runs/weights/yolo11m-sku110k.pt configs/detect/yolo11m-sku110k.yaml --split test
```

Live demo: [Shelf product counter](https://huggingface.co/spaces/rsuvbe/shelf-product-counter),
a static Hugging Face Space in `space/`. The model runs in the browser
(onnxruntime-web: WebGPU, or multi-threaded WebAssembly), so the photo never
leaves the device; about 2 s per photo on a laptop CPU. To update it:
`space/build.sh` (exports the ONNX model, fetches the runtime), then
`hf upload rsuvbe/shelf-product-counter space . --repo-type space`.
`demo/gradio/` is the same demo as a server-side Gradio app, for running
locally (`python demo/gradio/app.py`); Gradio Spaces need HF PRO.

### Telegram bot

Send the bot a shelf photo, get it back with every product boxed and the count.
It uses the ONNX export (`shelf_vision.counter`: numpy + onnxruntime, no torch),
downloaded from the release on first start; on val it counts within 13.4
products per photo (8.7%), 0.6 s per photo on a laptop CPU.

```bash
uv sync --extra bot
echo "SHELF_BOT_TOKEN=<token from @BotFather>" >> .env     # .env is git-ignored
uv run shelf bot                                           # long polling, no public URL
uv run shelf bot --allowed 123,456 --save                  # only these users; keep photos
```

The bot also marks possible empty spots in red (`/gaps` turns that off for a
chat); see *Empty spots* below.

`--save` stores every received photo with its count in `runs/bot/photos/<date>/`,
which turns the bot into a data collection tool for shop pilots. Without
`--allowed` anyone who finds the bot can use it; a stranger gets their Telegram
id in the reply, so it is easy to add them. In chat, `/conf 0.3` changes the
confidence threshold for that chat.

On the laptop the bot runs as a systemd user service that starts at boot and
restarts on failure (`deploy/shelf-bot.service`):

```bash
cp deploy/shelf-bot.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now shelf-bot
journalctl --user -u shelf-bot -f        # log
```

### Empty spots (experimental)

`shelf_vision.gaps` finds empty spots from the product boxes alone: boxes whose
bottom edges line up form a shelf row, and a gap inside a row at least 0.7
product widths wide and 0.6 heights tall, standing on the row's shelf line and
not covered by other boxes, is marked. Row ends are skipped. The browser demo
runs the same rules (`space/app.js`).

SKU-110K has no empty-spot labels, so it was checked two ways on val:

- Synthetic: removing one or two neighbouring products from a row of labelled
  boxes, the removed spot is found 79-80% of the time.
- By eye on five photos: about three in four marked spots are really empty;
  most false ones are products the detector missed. Photos taken at a steep
  angle break the row grouping, so shelves should be photographed head-on.

Real numbers need labelled shop photos (next section).

### Shop pilot

```bash
uv run shelf prelabel runs/bot/photos/2026-11-21 runs/pilot/day1   # model boxes as YOLO labels
# fix the boxes in a YOLO-capable tool (X-AnyLabeling, labelImg, ...)
uv run shelf pilot-eval runs/pilot/day1                            # count error, precision/recall
```

Pre-labels use conf 0.3 because deleting a wrong box is quicker than drawing a
missed one. Corrected pre-labels lean towards the model's boxes, so treat the
first numbers as optimistic; on SKU-110K val with its own labels `pilot-eval`
reports 8.9% count error, precision 0.93, recall 0.93.

### How it got here

SKU-110K test split (2,936 photos, ~147 products each). mAP50-95 from
`model.val()` unless marked.

| Run | Test mAP50-95 | Notes |
|---|---|---|
| YOLO11s, 960, 11.3 h | 0.604 | plateaued by epoch ~60 |
| + low-LR fine-tune | 0.601 | overfit |
| YOLO11m, 960, 11.1 h | 0.612 | `yolo11m-sku110k.yaml` |
| same weights evaluated at 1280 | 0.614 | no training |
| + pseudo-label fine-tune at 1280 | 0.610 | `yolo11m-sku110k-pl.yaml`, did not help |
| same weights, 1280, NMS IoU 0.8 | ~0.616 | own evaluator*, +0.2 over IoU 0.7 |
| + TTA | ~0.617 | own evaluator*, not worth 2-3x the time |
| WBF ensembles of the above | lower | -0.5 on val, dense shelves merge neighbours |

\* `notebooks/kaggle_tune.ipynb` scores raw predictions with ultralytics'
matching and AP code (matches `model.val()` to 1e-4 on the same predictions).
Its absolute test numbers are ~0.005 lower than `model.val()`, which repairs
damaged JPEGs while `predict` reads them as they are; the table shifts them by
that offset. Settings were chosen on val and test was scored once.

Product counting (val, 588 photos, 155 labelled products on average): at conf
0.4 the count is off by 14.7 products per photo (9.4%), with almost no bias.
Lower thresholds overcount (+15 at 0.25), higher ones undercount. Some of the
"extra" products are real ones the annotators skipped.

## Layout

```
configs/            experiment configs (one file = one reproducible run)
src/shelf_vision/   package: env detection, CLI, later data/retrieval/serving
data/               datasets (not in git)
runs/               checkpoints, logs, metrics (not in git)
notebooks/          Kaggle launchers and analysis
tests/
```

## Where things run

The same code and configs run in both places; `shelf_vision.env` resolves
paths, GPU count and batch size.

| Laptop (RTX 5060, 8 GB)                         | Kaggle (free GPUs)                   |
|-------------------------------------------------|--------------------------------------|
| development, smoke runs, error analysis         | full training runs                   |
| embedding extraction, retrieval experiments     | hyperparameter sweeps                |
| ONNX / TensorRT export and latency benchmarks   | anything longer than about two hours |

## Quick start (laptop)

```bash
uv sync                      # Python 3.14, PyTorch built for CUDA 12.8
uv run shelf doctor          # shows runtime, GPU, paths
uv run shelf train configs/detect/yolo11s-sku110k.yaml --smoke
uv run shelf train configs/detect/yolo11s-sku110k.yaml --epochs 5   # short real run
uv run pytest
```

## Running on Kaggle

Full training runs from `notebooks/kaggle_train.ipynb`: import it into Kaggle
(File → Import Notebook), set the accelerator to GPU T4 x2 and turn Internet
on, then Save & Run All. The notebook downloads the dataset, trains with
`shelf train --hours` for the rest of the 12-hour session (the epoch count and
LR schedule are fitted to that time) and evaluates on the test split.

By hand, in a notebook with GPU and internet enabled:

```python
!git clone https://github.com/rsuvbe/shelf-vision && pip install -q -e shelf-vision
%env SHELF_DATA_DIR=/tmp/shelf-data
!shelf doctor
!bash shelf-vision/scripts/fetch_sku110k.sh /tmp/shelf-data
!shelf train shelf-vision/configs/detect/yolo11s-sku110k.yaml --hours 11
```

Checkpoints land in `/kaggle/working/runs`; download `best.pt` into `runs/`
locally for evaluation and export.

Early stopping stays on with `--hours`. The first 11.32h run stopped at epoch
83 of 146 with the LR still high, but that was a real plateau: a 12-epoch
annealing fine-tune from its `best.pt` (low LR, last 6 epochs without mosaic)
overfit and scored 0.601 test mAP50-95 against 0.604 for the base weights.

To fine-tune a finished run (for example on new data), use
`notebooks/kaggle_finetune.ipynb` with the training notebook's output attached
as an input. It runs `configs/detect/yolo11s-sku110k-ft.yaml` from that run's
`best.pt` and evaluates both on the test split. Locally:
`shelf train configs/detect/yolo11s-sku110k-ft.yaml --weights <run>/weights/best.pt`.

A Kaggle session is limited to 12 hours. To continue an interrupted run, put its
`last.pt` back at `runs/detect/<run name>/weights/last.pt` and add `--resume`.

## Data notes

SKU-110K ships with some broken JPEGs. Decoding all 11,743 files with libjpeg
shows that every file opens, but 35 have damaged pixel data (truncated file or
broken entropy stream): 25 in train, 1 in val, 9 in test. The 25 training files
are listed in `configs/data/sku110k-damaged-train.txt` and left out of training
through the config's `exclude` key. Val and test are untouched, so metrics stay
comparable with published results.

The package supports Python 3.11+ so it installs on Kaggle's interpreter; the
local environment is pinned to 3.14 in `.python-version`.
