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


def _dataset(cfg: dict, config: Path, rt: env.Runtime) -> str:
    """Dataset YAML for a config, rooted in our data directory.

    Downloads and converts the dataset on first use, then applies the config's
    `exclude` list.
    """
    from shelf_vision.data import in_data_dir, without_excluded

    data = in_data_dir(cfg["data"], rt.data_dir)
    if "exclude" not in cfg:
        from ultralytics.data.utils import check_det_dataset

        check_det_dataset(data)
        return data
    return without_excluded(data, (config.parent / cfg["exclude"]).resolve())


@app.command()
def prepare(
    config: Annotated[Path, typer.Argument(exists=True, help="configs/detect/*.yaml")],
) -> None:
    """Download and convert the config's dataset without training."""
    rt = env.detect()
    _use_data_dir(rt)
    typer.echo(_dataset(_load(config), config, rt))


@app.command()
def train(
    config: Annotated[Path, typer.Argument(exists=True, help="configs/detect/*.yaml")],
    smoke: Annotated[
        bool, typer.Option(help="1 epoch on a small fraction: checks the pipeline")
    ] = False,
    epochs: Annotated[
        int | None, typer.Option(help="override the config; the run is named <config>-eN")
    ] = None,
    hours: Annotated[
        float | None,
        typer.Option(
            help="train for this many hours instead of a fixed number of epochs; "
            "the epoch count and LR schedule are fitted after the first epoch "
            "(run named <config>-<H>h)"
        ),
    ] = None,
    resume: Annotated[bool, typer.Option(help="continue the run from its weights/last.pt")] = False,
    weights: Annotated[
        Path | None,
        typer.Option(exists=True, help="start from these weights instead of the config's model"),
    ] = None,
) -> None:
    """Train a detector from a config."""
    from ultralytics import YOLO

    cfg = _load(config)
    rt = env.detect()
    _use_data_dir(rt)
    data = _dataset(cfg, config, rt)
    if epochs and hours:
        raise typer.BadParameter("use either --epochs or --hours")
    suffix = "-smoke" if smoke else f"-e{epochs}" if epochs else f"-{hours:g}h" if hours else ""
    name = config.stem + suffix
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
    if hours:
        args["time"] = hours
        # The schedule is stretched over the whole time budget, so early stopping
        # would cut it off before the LR decays and close_mosaic kicks in.
        args["patience"] = 0
    if smoke:
        args.update(epochs=1, fraction=0.02)

    model = YOLO(str(weights) if weights else cfg["model"])
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
        data=_dataset(cfg, config, rt),
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
