"""Runtime environment: the same code runs on the laptop and on Kaggle.

Everything that differs between the two (paths, default batch size, workers)
is resolved here, so configs and training code stay identical.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def on_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").is_dir()


@dataclass(frozen=True)
class Runtime:
    name: str  # "kaggle" | "local"
    data_dir: Path  # datasets live here
    runs_dir: Path  # checkpoints, logs, metrics
    device: str  # "0", "0,1", or "cpu" (ultralytics format)
    gpu_name: str
    vram_gb: float
    workers: int

    def batch_for(self, per_8gb: int) -> int:
        """Scale a batch size tuned for 8 GB of VRAM to the GPUs available here."""
        if self.device == "cpu":
            return max(1, per_8gb // 4)
        n_gpus = len(self.device.split(","))
        scaled = int(per_8gb * self.vram_gb / 8) * n_gpus
        return max(n_gpus, scaled - scaled % n_gpus)


def detect() -> Runtime:
    kaggle = on_kaggle()
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus:
        props = torch.cuda.get_device_properties(0)
        device = ",".join(str(i) for i in range(n_gpus))
        gpu_name, vram_gb = props.name, props.total_memory / 2**30
    else:
        device, gpu_name, vram_gb = "cpu", "none", 0.0

    if kaggle:
        data_dir = Path(os.environ.get("SHELF_DATA_DIR", "/kaggle/working/data"))
        runs_dir = Path(os.environ.get("SHELF_RUNS_DIR", "/kaggle/working/runs"))
    else:
        data_dir = Path(os.environ.get("SHELF_DATA_DIR", REPO_ROOT / "data"))
        runs_dir = Path(os.environ.get("SHELF_RUNS_DIR", REPO_ROOT / "runs"))

    return Runtime(
        name="kaggle" if kaggle else "local",
        data_dir=data_dir,
        runs_dir=runs_dir,
        device=device,
        gpu_name=gpu_name,
        vram_gb=round(vram_gb, 1),
        workers=min(8, os.cpu_count() or 2),
    )
