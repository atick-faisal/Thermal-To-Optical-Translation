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

from t2o.data.adapters import AdapterError, adapt_msrs
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


def _build_msrs_raw(
    root: Path,
    train_stems: list[str] = TRAIN_STEMS,
    detection_stems: list[str] = DETECTION_STEMS,
) -> Path:
    for stem in train_stems:
        _write_pair(root / "train" / "vi", root / "train" / "ir", stem)
    for stem in TEST_STEMS:
        _write_pair(root / "test" / "vi", root / "test" / "ir", stem)
    for stem in detection_stems:
        _write_pair(root / "detection" / "vi", root / "detection" / "ir", stem)

    labels_dir = root / "detection" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    for i, stem in enumerate(detection_stems):
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
    assert len(train_visible) == len(TRAIN_STEMS) + len(DETECTION_STEMS)
    assert len(train_infrared) == len(TRAIN_STEMS) + len(DETECTION_STEMS)
    assert len(val_visible) == len(TEST_STEMS)
    assert (dest / "data.yaml").is_file()


def test_written_manifest_loads_and_matches_classes(msrs_raw_root: Path, tmp_path: Path) -> None:
    dest = tmp_path / "processed"

    data_yaml = adapt_msrs(msrs_raw_root, dest)
    manifest = DatasetManifest.load(data_yaml)

    assert manifest.class_names == CLASSES
    assert manifest.nc == len(CLASSES)
    assert manifest.pairing == Pairing()


def test_detection_label_is_copied_verbatim(msrs_raw_root: Path, tmp_path: Path) -> None:
    dest = tmp_path / "processed"

    adapt_msrs(msrs_raw_root, dest)

    stem = DETECTION_STEMS[0]
    source = (msrs_raw_root / "detection" / "labels" / f"{stem}.txt").read_text()
    copied = (dest / "train" / "visible" / "labels" / f"{stem}.txt").read_text()
    assert copied == source


def test_unlabelled_train_image_has_no_label_file(msrs_raw_root: Path, tmp_path: Path) -> None:
    dest = tmp_path / "processed"

    adapt_msrs(msrs_raw_root, dest)

    label_path = dest / "train" / "visible" / "labels" / f"{TRAIN_STEMS[0]}.txt"
    assert not label_path.exists()
    cls, bboxes = load_yolo_labels(label_path)
    assert cls.shape == (0, 1)
    assert bboxes.shape == (0, 4)


def test_colliding_stems_between_train_and_detection_raises(tmp_path: Path) -> None:
    raw_root = _build_msrs_raw(
        tmp_path / "raw", train_stems=[*TRAIN_STEMS, "1"], detection_stems=DETECTION_STEMS
    )

    with pytest.raises(AdapterError, match="both train/ and detection/"):
        adapt_msrs(raw_root, tmp_path / "processed")


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
