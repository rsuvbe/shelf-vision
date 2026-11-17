"""Single entry point: `shelf <command>` works the same locally and on Kaggle."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml

from shelf_vision import env

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _load(config: Path) -> dict:
    with open(config) as f:
        return yaml.safe_load(f)


def _use_data_dir(rt: env.Runtime) -> None:
    """Make ultralytics download and look up datasets in our data directory."""
    from ultralytics import settings

    rt.data_dir.mkdir(parents=True, exist_ok=True)
    if settings.get("datasets_dir") != str(rt.data_dir):
        settings.update(datasets_dir=str(rt.data_dir))


@app.command()
def doctor() -> None:
    """Show where we are running and what hardware is available."""
    rt = env.detect()
    for key, value in vars(rt).items():
        typer.echo(f"{key:10} {value}")


def _dataset(cfg: dict, config: Path) -> str:
    """Dataset YAML for a config, with the config's `exclude` list applied."""
    if "exclude" not in cfg:
        return cfg["data"]
    from shelf_vision.data import without_excluded

    return without_excluded(cfg["data"], (config.parent / cfg["exclude"]).resolve())


@app.command()
def train(
    config: Annotated[Path, typer.Argument(exists=True, help="configs/detect/*.yaml")],
    smoke: Annotated[
        bool, typer.Option(help="1 epoch on a small fraction: checks the pipeline")
    ] = False,
    epochs: Annotated[
        int | None, typer.Option(help="override the config; the run is named <config>-eN")
    ] = None,
    resume: Annotated[bool, typer.Option(help="continue the run from its weights/last.pt")] = False,
) -> None:
    """Train a detector from a config."""
    from ultralytics import YOLO

    cfg = _load(config)
    rt = env.detect()
    _use_data_dir(rt)
    data = _dataset(cfg, config)
    name = config.stem + ("-smoke" if smoke else f"-e{epochs}" if epochs else "")
    project = rt.runs_dir / "detect"

    if resume:
        last = project / name / "weights" / "last.pt"
        if not last.exists():
            raise typer.BadParameter(f"nothing to resume: {last} does not exist")
        # epochs, batch and the rest come from the checkpoint
        YOLO(str(last)).train(resume=True)
        return

    args = dict(cfg["train"])
    args["batch"] = rt.batch_for(args.pop("batch_per_8gb"))
    if epochs:
        args["epochs"] = epochs
    if smoke:
        args.update(epochs=1, fraction=0.02)

    model = YOLO(cfg["model"])
    model.train(
        data=data,
        project=str(project),
        name=name,
        device=rt.device,
        workers=rt.workers,
        exist_ok=smoke,
        **args,
    )


@app.command()
def evaluate(
    weights: Annotated[Path, typer.Argument(exists=True)],
    config: Annotated[Path, typer.Argument(exists=True)],
    split: str = "test",
) -> None:
    """Evaluate weights on a held-out split."""
    from ultralytics import YOLO

    cfg = _load(config)
    rt = env.detect()
    _use_data_dir(rt)
    metrics = YOLO(str(weights)).val(
        data=_dataset(cfg, config),
        split=split,
        imgsz=cfg["train"]["imgsz"],
        device=rt.device.split(",")[0],
        max_det=cfg["train"].get("max_det", 300),
        project=str(rt.runs_dir / "eval"),
        name=f"{config.stem}-{split}",
    )
    typer.echo(f"mAP50-95 {metrics.box.map:.4f}  mAP50 {metrics.box.map50:.4f}")


if __name__ == "__main__":
    app()
