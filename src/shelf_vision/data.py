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
