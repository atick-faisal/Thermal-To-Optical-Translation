"""Refine a stage-1 export through the cascade's frozen stage 2, keeping the export's layout.

cascade-gate plan Q1: stage 2 runs over an export that already exists, not inside a
`Translator`. The refined tree copies the export's layout (`<split>/images`,
`<split>/labels`, `data.yaml`; `engine/export.py`), so `t2o faithfulness` and
`t2o completion` score it with no change. Both match frames to their references by stem
(`metrics/faithfulness.py::_translated_counterparts`), so every frame keeps its stem.

Frames are refined one at a time. The refiner works per image anyway (AdaIN statistics, VAE
and UNet normalisation), and a batch would only add a size-matching rule for no measured
gain on 153 frames.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn

from t2o.config import RefineConfig
from t2o.data.dataset import IMAGE_SUFFIXES
from t2o.engine.export import DATA_FILENAME, EXPORT_SUFFIX
from t2o.imaging import to_uint8

logger = logging.getLogger(__name__)


class RefineError(Exception):
    """The refine's inputs or output directory are not what `t2o refine` needs."""


def check_refine_paths(config: RefineConfig) -> list[Path]:
    """Fail on any missing input or a non-empty output, and return the frames to refine.

    Called before SD-2.1 and the checkpoint are loaded, so a typo fails in a second rather
    than after minutes of loading. The output must be empty because the refined tree must
    never land on top of an earlier refine, or on the stage-1 export it was made from.
    """
    export = config.source.export
    images = export / config.source.split / "images"
    out = config.runtime.path
    problems = []
    if not (export / DATA_FILENAME).is_file():
        problems.append(f"no {DATA_FILENAME} in {export}; is it a `t2o export` root?")
    frames = (
        sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if images.is_dir()
        else []
    )
    if not frames:
        problems.append(f"no images in {images}")
    if out.exists() and any(out.iterdir()):
        problems.append(f"{out} is not empty; refine into a fresh directory")
    if not config.refiner.checkpoint.is_file():
        problems.append(f"no PiSA-SR checkpoint at {config.refiner.checkpoint}")
    if problems:
        raise RefineError("cannot refine:\n  " + "\n  ".join(problems))
    return frames


def refine_export(refiner: nn.Module, config: RefineConfig) -> Path:
    """Write every frame of the source split, refined, into `config.runtime.path`.

    `refiner` maps RGB `(1, 3, H, W)` in [0, 1] to the same shape in [0, 1]
    (`translators/pisa_sr.py::PiSASRRefiner`), on whatever device and dtype it was moved to.
    Labels are copied byte for byte. The config is snapshotted first, so even a refine that
    dies halfway leaves a record of what it was. Returns the refined tree's `data.yaml`.
    """
    frames = check_refine_paths(config)
    split = config.source.split
    out = config.runtime.path
    images_out = out / split / "images"
    images_out.mkdir(parents=True)
    config.snapshot(out)

    changes = []
    for frame in frames:
        with Image.open(frame) as image:
            before = torch.from_numpy(np.asarray(image.convert("RGB")).copy())
        source = before.permute(2, 0, 1).float().div(255.0)
        after = to_uint8(refiner(source[None])[0])
        Image.fromarray(after.permute(1, 2, 0).numpy(), mode="RGB").save(
            images_out / f"{frame.stem}{EXPORT_SUFFIX}"
        )
        # cascade-gate ADR-02: the server's first step reads these. A mean change of zero
        # means the LoRA weights did nothing, and the run stops there.
        change = (after.float() - before.permute(2, 0, 1).float()).abs().mean().item() / 255.0
        changes.append(change)
        logger.info(
            "refined %s  %dx%d  mean |change| %.4f",
            frame.name,
            after.shape[-1],
            after.shape[-2],
            change,
        )

    _copy_labels(config.source.export / split / "labels", out / split / "labels")
    data_yaml = _write_data_yaml(config.source.export / DATA_FILENAME, out, split)
    logger.info(
        "refined %d frames into %s; mean |change| %.4f over the split",
        len(frames),
        out,
        sum(changes) / len(changes),
    )
    return data_yaml


def _copy_labels(source: Path, destination: Path) -> None:
    """Copy the `.txt` labels verbatim, and never ultralytics' `labels.cache`.

    That cache is keyed on file paths and sizes, never contents (CLAUDE.md), so a copied
    cache describing the stage-1 tree could be trusted for the refined one.
    """
    destination.mkdir(parents=True)
    for label in sorted(source.glob("*.txt")):
        shutil.copyfile(label, destination / label.name)


def _write_data_yaml(source: Path, out: Path, split: str) -> Path:
    """The source export's `data.yaml`, rewritten to point at the refined split.

    `path:` is absolute, because ultralytics honours it literally. The refined split is named
    as both `train` and `val`, as `t2o erase` does (`data/erasure.py::write_erased_split`):
    only one split exists in the refined tree, and it is a scoring input, never a training set.
    """
    data = yaml.safe_load(source.read_text())
    data.update(path=str(out.resolve()), train=f"{split}/images", val=f"{split}/images")
    destination = out / DATA_FILENAME
    destination.write_text(yaml.safe_dump(data, sort_keys=False))
    return destination
