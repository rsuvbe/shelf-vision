# shelf-vision

Retail shelf monitoring: detect every product on a shelf photo, recognise which
SKU it is, and report what is missing or misplaced.

Status: week 1, detection baseline.

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

`--hours` turns early stopping off: the LR schedule is stretched over the whole
budget, and stopping early leaves it unfinished (the first 11.32h run stopped at
epoch 83 of 146 with the LR still high and no mosaic-free epochs).

To fine-tune a finished run, use `notebooks/kaggle_finetune.ipynb` with the
training notebook's output attached as an input. It runs
`configs/detect/yolo11s-sku110k-ft.yaml` (12 epochs, low LR, last 6 without
mosaic) from that run's `best.pt` and evaluates both on the test split. Locally:
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
