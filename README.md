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
uv run pytest
```

## Running on Kaggle

In a notebook with GPU and internet enabled:

```python
!git clone https://github.com/rsuvbe/shelf-vision && pip install -q -e shelf-vision
!shelf doctor
!shelf train shelf-vision/configs/detect/yolo11s-sku110k.yaml
```

Checkpoints land in `/kaggle/working/runs`; download `best.pt` into `runs/`
locally for evaluation and export.

The package supports Python 3.11+ so it installs on Kaggle's interpreter; the
local environment is pinned to 3.14 in `.python-version`.
