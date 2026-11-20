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


def infer_args(cfg: dict, imgsz: int | None = None, iou: float | None = None) -> dict:
    """Inference settings for a config: its `infer` section, else the training ones.

    Explicit `imgsz` / `iou` arguments win over both.
    """
    train, infer = cfg.get("train", {}), cfg.get("infer", {})
    return {
        "imgsz": imgsz or infer.get("imgsz") or train["imgsz"],
        "iou": iou or infer.get("iou", 0.7),
        "max_det": infer.get("max_det") or train.get("max_det", 300),
    }


@app.command()
def evaluate(
    weights: Annotated[Path, typer.Argument(exists=True)],
    config: Annotated[Path, typer.Argument(exists=True)],
    split: str = "test",
    imgsz: Annotated[int | None, typer.Option(help="override the config's inference size")] = None,
    iou: Annotated[float | None, typer.Option(help="override the config's NMS IoU")] = None,
) -> None:
    """Evaluate weights on a held-out split with the config's inference settings."""
    from ultralytics import YOLO

    cfg = _load(config)
    rt = env.detect()
    _use_data_dir(rt)
    args = infer_args(cfg, imgsz, iou)
    metrics = YOLO(str(weights)).val(
        data=_dataset(cfg, config, rt),
        split=split,
        device=rt.device.split(",")[0],
        project=str(rt.runs_dir / "eval"),
        name=f"{config.stem}-{split}",
        **args,
    )
    typer.echo(
        f"mAP50-95 {metrics.box.map:.4f}  mAP50 {metrics.box.map50:.4f}  "
        f"(imgsz {args['imgsz']}, iou {args['iou']})"
    )


@app.command()
def predict(
    weights: Annotated[Path, typer.Argument(exists=True)],
    config: Annotated[Path, typer.Argument(exists=True)],
    source: Annotated[Path, typer.Argument(exists=True, help="image, folder, or .txt list")],
    conf: Annotated[
        float | None, typer.Option(help="minimum box confidence (default: the config's infer.conf)")
    ] = None,
    save: Annotated[bool, typer.Option(help="save images with boxes drawn")] = True,
) -> None:
    """Detect products on shelf photos; writes counts.json and annotated images."""
    import json

    from ultralytics import YOLO

    cfg = _load(config)
    rt = env.detect()
    conf = conf or cfg.get("infer", {}).get("conf", 0.25)
    out = rt.runs_dir / "predict" / config.stem
    counts = {}
    # stream=True with a path source reads images one at a time; a Python list of
    # paths would be loaded into memory all at once
    for r in YOLO(str(weights)).predict(
        str(source),
        stream=True,
        conf=conf,
        device=rt.device.split(",")[0],
        save=save,
        project=str(out.parent),
        name=out.name,
        exist_ok=True,
        verbose=False,
        **infer_args(cfg),
    ):
        counts[Path(r.path).name] = len(r.boxes)
    out.mkdir(parents=True, exist_ok=True)
    (out / "counts.json").write_text(json.dumps(counts, indent=2))
    typer.echo(f"{len(counts)} images, {sum(counts.values())} products -> {out}")


ONNX_DEFAULT = env.REPO_ROOT / "runs" / "weights" / "yolo11m-sku110k.onnx"


@app.command()
def prelabel(
    photos: Annotated[Path, typer.Argument(exists=True, help="folder with shelf photos")],
    out: Annotated[Path, typer.Argument(help="new YOLO-format folder: images/, labels/")],
    conf: Annotated[
        float,
        typer.Option(help="lower than for counting: deleting a box is quicker than drawing one"),
    ] = 0.3,
    model: Annotated[Path, typer.Option(help="ONNX detector")] = ONNX_DEFAULT,
) -> None:
    """Pre-label real shelf photos with the model, for correction in a labelling tool."""
    import cv2

    from shelf_vision.counter import ShelfCounter, decode_image, ensure_model
    from shelf_vision.pilot import list_images, to_yolo

    counter = ShelfCounter(ensure_model(model))
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("product\n")
    images = list_images(photos)
    for i, p in enumerate(images):
        # write the photo upright: labelling tools disagree on EXIF rotation
        bgr = decode_image(p.read_bytes())
        name = f"{i:04d}_{p.stem}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        boxes, _ = counter.detect(bgr, conf)
        (out / "labels" / f"{name}.txt").write_text(to_yolo(boxes, bgr.shape[1], bgr.shape[0]))
    typer.echo(f"{len(images)} photos pre-labelled at conf {conf} -> {out}")


@app.command("pilot-eval")
def pilot_eval(
    folder: Annotated[Path, typer.Argument(exists=True, help="YOLO folder with corrected labels")],
    conf: Annotated[float, typer.Option(help="threshold to score; the bot uses 0.4")] = 0.4,
    model: Annotated[Path, typer.Option(help="ONNX detector")] = ONNX_DEFAULT,
) -> None:
    """Score the model on corrected real-shop labels: count error, box precision/recall."""
    import cv2

    from shelf_vision.counter import ShelfCounter, ensure_model
    from shelf_vision.pilot import PhotoScore, from_yolo, list_images, match_count, summarise

    counter = ShelfCounter(ensure_model(model))
    scores = []
    for p in list_images(folder / "images"):
        label = folder / "labels" / f"{p.stem}.txt"
        if not label.exists():
            continue
        bgr = cv2.imread(str(p))
        h, w = bgr.shape[:2]
        gt = from_yolo(label.read_text(), w, h)
        pred, _ = counter.detect(bgr, conf)
        scores.append(PhotoScore(p.name, len(gt), len(pred), match_count(pred, gt)))
    if not scores:
        raise typer.BadParameter(f"no labelled photos in {folder}/images")
    typer.echo(summarise(scores))


@app.command()
def bot(
    model: Annotated[
        Path, typer.Option(help="ONNX detector; downloaded from the detector-v1 release if missing")
    ] = ONNX_DEFAULT,
    allowed: Annotated[
        str,
        typer.Option(
            envvar="SHELF_BOT_ALLOWED",
            help="comma-separated Telegram user ids allowed to use the bot; empty = everyone",
        ),
    ] = "",
    save: Annotated[
        bool, typer.Option(help="keep received photos and counts in runs/bot/photos/<date>/")
    ] = False,
    threads: Annotated[int, typer.Option(help="CPU threads for inference; 0 = all")] = 0,
) -> None:
    """Run the Telegram bot (token: SHELF_BOT_TOKEN in the environment or .env)."""
    import logging

    from shelf_vision import bot as tgbot
    from shelf_vision.counter import ensure_model

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per poll otherwise
    tgbot.run(
        ensure_model(model),
        tgbot.parse_allowed(allowed),
        env.REPO_ROOT / "runs" / "bot" / "photos" if save else None,
        threads,
    )


if __name__ == "__main__":
    app()
