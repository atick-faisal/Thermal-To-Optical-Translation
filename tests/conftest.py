"""Synthetic thermal/visible dataset fixtures. No real data, no network access.

`dataset/` is never tracked (PLAN.md §9) -- a fresh clone (including the server's) has no
images, so every test that needs a dataset builds one here at test time instead of relying
on the untracked local sample. Train and val use disjoint filenames deliberately: the real
local 9-pair sample has train == val, which would hide any bug that conflates the splits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

NAMES = ["Fuse", "Pole", "Switch", "Transformer"]
IMG_SIZE = (480, 640)  # (height, width) -- matches the real FLIR pairs, stride-32 divisible
N_TRAIN = 5
N_VAL = 3

# Deliberately not alphabetical-by-index (catches sorted-index pairing bugs) and
# deliberately disjoint between splits (catches train/val conflation).
TRAIN_STEMS = [f"{(7 * i) % 11:02d}_train_{i:03d}" for i in range(N_TRAIN)]
VAL_STEMS = [f"{(5 * i) % 11:02d}_val_{i:03d}" for i in range(N_VAL)]


def _write_split(root: Path, split: str, stems: list[str], rng: np.random.Generator) -> None:
    for modality in ("visible", "infrared"):
        (root / split / modality / "images").mkdir(parents=True)
        (root / split / modality / "labels").mkdir(parents=True)

    h, w = IMG_SIZE
    for i, stem in enumerate(stems):
        visible = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        infrared = rng.integers(0, 255, (h, w), dtype=np.uint8)
        Image.fromarray(visible, mode="RGB").save(
            root / split / "visible" / "images" / f"{stem}.jpg"
        )
        Image.fromarray(infrared, mode="L").save(
            root / split / "infrared" / "images" / f"{stem}.jpg"
        )

        # One image per split is left unlabelled, to exercise the zero-instance path.
        boxes = (
            ""
            if i == 0
            else f"{i % len(NAMES)} 0.5 0.5 0.4 0.4\n{(i + 1) % len(NAMES)} 0.25 0.75 0.2 0.2\n"
        )
        (root / split / "visible" / "labels" / f"{stem}.txt").write_text(boxes)


@pytest.fixture(scope="session")
def dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A two-split synthetic dataset with disjoint train/val filenames."""
    root = tmp_path_factory.mktemp("yolo_rgbt")
    rng = np.random.default_rng(0)
    _write_split(root, "train", TRAIN_STEMS, rng)
    _write_split(root, "val", VAL_STEMS, rng)
    (root / "data.yaml").write_text(
        f"path: {root}\n"
        "train: train/visible/images\n"
        "val: val/visible/images\n"
        f"nc: {len(NAMES)}\n"
        "names:\n" + "".join(f" - {n}\n" for n in NAMES) + "rgbt:\n"
        "  visible_token: visible\n"
        "  infrared_token: infrared\n"
    )
    return root


@pytest.fixture(scope="session")
def data_yaml(dataset_root: Path) -> Path:
    """The dataset manifest -- what `DatasetManifest.load` points at."""
    return dataset_root / "data.yaml"


@pytest.fixture(scope="session")
def detector_weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A randomly-initialised yolo11n in ultralytics checkpoint format.

    Built locally rather than downloaded: the test suite must not touch the network. The
    saved `train_args` is a plain dict (not ultralytics' own namespace type) deliberately --
    that is exactly the shape `FrozenDetector`'s `_normalize_args` has to repair, and a
    freshly-trained checkpoint loaded via `YOLO(path).model.args` really does come back this
    way (verified against the installed ultralytics 8.4.117).
    """
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(0)
    model = DetectionModel(cfg="yolo11n.yaml", ch=3, nc=len(NAMES), verbose=False)
    model.names = dict(enumerate(NAMES))  # type: ignore[assignment]
    path = tmp_path_factory.mktemp("detector") / "yolo11n_random.pt"
    torch.save(
        {"model": model, "epoch": -1, "best_fitness": None, "train_args": {"data": "synthetic"}},
        path,
    )
    return path
