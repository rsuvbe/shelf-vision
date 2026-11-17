"""Dataset preparation shared by training and evaluation."""

from __future__ import annotations

from pathlib import Path

import yaml


def read_names(path: Path) -> set[str]:
    """File names from a list file; blank lines and `#` comments are skipped."""
    lines = (line.strip() for line in path.read_text().splitlines())
    return {line for line in lines if line and not line.startswith("#")}


def drop_from_list(lines: list[str], names: set[str]) -> list[str]:
    """Image list entries whose file name is not in `names`."""
    return [line for line in lines if line.strip() and Path(line.strip()).name not in names]


def pin_dataset_root(spec: dict, data_dir: Path) -> dict:
    """Dataset spec whose relative `path` is resolved against `data_dir`."""
    root = Path(spec.get("path") or "")
    if not root.is_absolute():
        root = data_dir / root
    return {**spec, "path": str(root)}


def in_data_dir(data: str, data_dir: Path) -> str:
    """Copy of a dataset YAML with its root pinned under `data_dir`.

    ultralytics resolves relative dataset paths against `datasets_dir`, which it
    reads once at import time; changing the setting later has no effect, so on
    a fresh machine (Kaggle) data would land in the default directory instead.
    An absolute root makes the location independent of ultralytics settings.
    """
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    spec = pin_dataset_root(YAML.load(check_yaml(data)), data_dir)
    out = data_dir / f"{Path(data).stem}.yaml"
    data_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
    return str(out)


def without_excluded(data: str, exclude: Path) -> str:
    """Dataset YAML equal to `data`, minus the listed training images.

    Val and test are passed through unchanged. The filtered list and the YAML
    are written next to the dataset, so the result is the same on every machine.
    """
    from ultralytics.data.utils import check_det_dataset

    ds = check_det_dataset(data)  # downloads on first use and resolves paths
    train_list = Path(ds["train"])
    if train_list.suffix != ".txt":
        raise ValueError(f"exclude needs a train image list (.txt), got {train_list}")

    names = read_names(exclude)
    lines = train_list.read_text().splitlines()
    kept = drop_from_list(lines, names)
    clean_list = train_list.with_name(f"{train_list.stem}.clean.txt")
    clean_list.write_text("\n".join(kept) + "\n")

    out = Path(ds["path"]) / f"{Path(data).stem}.clean.yaml"
    spec = {
        "path": str(ds["path"]),
        "train": str(clean_list),
        "val": ds["val"],
        "test": ds.get("test"),
        "names": ds["names"],
    }
    out.write_text(yaml.safe_dump(spec, sort_keys=False))
    print(f"excluded {len(lines) - len(kept)} of {len(lines)} training images ({exclude.name})")
    return str(out)
