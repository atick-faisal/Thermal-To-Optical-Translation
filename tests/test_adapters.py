"""`data/adapters` -- MSRS raw layout -> the internal representation.

Everything here builds a tiny MSRS-*shaped* raw tree at test time (PLAN.md §9's
synthetic-fixture discipline): same directory names and label format as the real
`github.com/Linfeng-Tang/MSRS` clone, at fixture scale. One test additionally exercises the
real local `dataset/raw/msrs` when present, `skipif`-guarded, and is never the sole coverage
of any code path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from t2o.data.adapters import adapt_msrs
from t2o.data.labels import load_yolo_labels
from t2o.data.manifest import DatasetManifest
from t2o.data.pairing import Pairing

TRAIN_STEMS = ["00001D", "00002N", "00003D"]
TEST_STEMS = ["00901D", "00902N"]
DETECTION_STEMS = ["1", "42"]
CLASSES = ["person", "bicycle", "car"]


def _write_pair(vi_dir: Path, ir_dir: Path, stem: str) -> None:
    vi_dir.mkdir(parents=True, exist_ok=True)
    ir_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(stem)) % (2**32))
    Image.fromarray(rng.integers(0, 255, (4, 4, 3), dtype=np.uint8), mode="RGB").save(
        vi_dir / f"{stem}.png"
    )
    Image.fromarray(rng.integers(0, 255, (4, 4), dtype=np.uint8), mode="L").save(
        ir_dir / f"{stem}.png"
    )


def _build_msrs_raw(root: Path) -> Path:
    for stem in TRAIN_STEMS:
        _write_pair(root / "train" / "vi", root / "train" / "ir", stem)
    for stem in TEST_STEMS:
        _write_pair(root / "test" / "vi", root / "test" / "ir", stem)
    for stem in DETECTION_STEMS:
        _write_pair(root / "detection" / "vi", root / "detection" / "ir", stem)

    labels_dir = root / "detection" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    for i, stem in enumerate(DETECTION_STEMS):
        (labels_dir / f"{stem}.txt").write_text(f"{i % len(CLASSES)} 0.5 0.5 0.2 0.2\n")
    return root


@pytest.fixture
def msrs_raw_root(tmp_path: Path) -> Path:
    return _build_msrs_raw(tmp_path / "raw")


def test_adapt_msrs_produces_expected_layout_and_counts(
    msrs_raw_root: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "processed"

    adapt_msrs(msrs_raw_root, dest)

    train_visible = list((dest / "train" / "visible" / "images").iterdir())
    train_infrared = list((dest / "train" / "infrared" / "images").iterdir())
    val_visible = list((dest / "val" / "visible" / "images").iterdir())
    assert len(train_visible) == len(TRAIN_STEMS)
    assert len(train_infrared) == len(TRAIN_STEMS)
    assert len(val_visible) == len(TEST_STEMS)
    assert (dest / "data.yaml").is_file()


def test_written_manifest_loads_and_matches_classes(msrs_raw_root: Path, tmp_path: Path) -> None:
    dest = tmp_path / "processed"

    data_yaml = adapt_msrs(msrs_raw_root, dest)
    manifest = DatasetManifest.load(data_yaml)

    assert manifest.class_names == CLASSES
    assert manifest.nc == len(CLASSES)
    assert manifest.pairing == Pairing()


def test_unlabelled_train_image_has_no_label_file(msrs_raw_root: Path, tmp_path: Path) -> None:
    dest = tmp_path / "processed"

    adapt_msrs(msrs_raw_root, dest)

    label_path = dest / "train" / "visible" / "labels" / f"{TRAIN_STEMS[0]}.txt"
    assert not label_path.exists()
    cls, bboxes = load_yolo_labels(label_path)
    assert cls.shape == (0, 1)
    assert bboxes.shape == (0, 4)


def test_detection_stems_never_reach_the_tree(msrs_raw_root: Path, tmp_path: Path) -> None:
    # 71 of the 80 real ``detection/`` pairs are pixel-identical to MSRS frames (23 of them val),
    # so reading the folder leaks val into train. See the adapter's module docstring.
    dest = tmp_path / "processed"

    adapt_msrs(msrs_raw_root, dest)

    written = {
        p.stem
        for split in ("train", "val")
        for modality in ("visible", "infrared")
        for kind in ("images", "labels")
        if (dest / split / modality / kind).is_dir()
        for p in (dest / split / modality / kind).iterdir()
    }
    assert written.isdisjoint(DETECTION_STEMS)


def test_adapt_msrs_is_idempotent_against_a_populated_dest(
    msrs_raw_root: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "processed"
    dest.mkdir()
    (dest / "sentinel.txt").write_text("already adapted")

    result = adapt_msrs(msrs_raw_root, dest)

    assert result == dest / "data.yaml"
    assert not (dest / "train").exists()
    assert (dest / "sentinel.txt").read_text() == "already adapted"


REAL_MSRS_RAW = Path("dataset/raw/msrs")


@pytest.mark.skipif(not REAL_MSRS_RAW.is_dir(), reason="real dataset/raw/msrs not present")
def test_real_msrs_adapts_and_loads(tmp_path: Path) -> None:
    from t2o.data.dataset import TranslationPairDataset

    data_yaml = adapt_msrs(REAL_MSRS_RAW, tmp_path / "processed")
    manifest = DatasetManifest.load(data_yaml)

    train = TranslationPairDataset(
        manifest.train_images, pairing=manifest.pairing, num_classes=manifest.nc
    )
    val = TranslationPairDataset(
        manifest.val_images, pairing=manifest.pairing, num_classes=manifest.nc
    )
    assert len(train) >= 1083
    assert len(val) >= 361
