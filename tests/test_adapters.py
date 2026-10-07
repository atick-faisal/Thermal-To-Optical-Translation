"""`data/adapters` -- MSRS raw layout -> the internal representation.

Everything here builds a tiny MSRS-*shaped* raw tree at test time (PLAN.md §9's
synthetic-fixture discipline): same directory names and mask format as the real
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
from t2o.data.adapters.msrs import KEPT_CLASSES
from t2o.data.labels import load_yolo_labels
from t2o.data.manifest import DatasetManifest
from t2o.data.pairing import Pairing

TRAIN_STEMS = ["00001D", "00002N", "00003D"]
TEST_STEMS = ["00901D", "00902N"]
DETECTION_STEMS = ["1", "42"]
DETECTION_CLASSES = ["person", "bicycle", "car"]
SIZE = 16  # pixels per side; large enough for a blob above MIN_BLOB_AREA

# Segmentation_labels pixel values (dataset/raw/msrs/visualize.py): 2 person, 4 curve (dropped),
# 5 car_stop. Each fixture frame holds one 4x4 blob of its value; 0 is an empty mask.
MASK_VALUES = {"00001D": 5, "00002N": 4, "00003D": 2, "00901D": 2, "00902N": 0}
# Output ids under KEPT_CLASSES: car_stop -> 3, person -> 1. Curve and empty frames get no file.
EXPECTED_CLASS_IDS = {"00001D": 3, "00003D": 1, "00901D": 1}
CURVE_ONLY_STEM = "00002N"


def _write_pair(vi_dir: Path, ir_dir: Path, stem: str) -> None:
    vi_dir.mkdir(parents=True, exist_ok=True)
    ir_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(stem)) % (2**32))
    Image.fromarray(rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8), mode="RGB").save(
        vi_dir / f"{stem}.png"
    )
    Image.fromarray(rng.integers(0, 255, (SIZE, SIZE), dtype=np.uint8), mode="L").save(
        ir_dir / f"{stem}.png"
    )


def _write_mask(masks_dir: Path, stem: str) -> None:
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[4:8, 4:8] = MASK_VALUES[stem]
    Image.fromarray(mask, mode="L").save(masks_dir / f"{stem}.png")


def _build_msrs_raw(root: Path) -> Path:
    for split, stems in (("train", TRAIN_STEMS), ("test", TEST_STEMS)):
        for stem in stems:
            _write_pair(root / split / "vi", root / split / "ir", stem)
            _write_mask(root / split / "Segmentation_labels", stem)
    for stem in DETECTION_STEMS:
        _write_pair(root / "detection" / "vi", root / "detection" / "ir", stem)

    # detection/ and its classes.txt stay in the fixture to prove the adapter ignores them.
    labels_dir = root / "detection" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "classes.txt").write_text("\n".join(DETECTION_CLASSES) + "\n")
    for i, stem in enumerate(DETECTION_STEMS):
        (labels_dir / f"{stem}.txt").write_text(f"{i % len(DETECTION_CLASSES)} 0.5 0.5 0.2 0.2\n")
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


def test_written_manifest_names_the_kept_classes(msrs_raw_root: Path, tmp_path: Path) -> None:
    dest = tmp_path / "processed"

    data_yaml = adapt_msrs(msrs_raw_root, dest)
    manifest = DatasetManifest.load(data_yaml)

    # Not DETECTION_CLASSES: detection/labels/classes.txt is no longer read.
    assert manifest.class_names == list(KEPT_CLASSES)
    assert manifest.nc == len(KEPT_CLASSES)
    assert manifest.pairing == Pairing()


def test_frames_are_labelled_from_their_masks(msrs_raw_root: Path, tmp_path: Path) -> None:
    dest = tmp_path / "processed"

    adapt_msrs(msrs_raw_root, dest)

    written = {
        p.stem: [int(line.split()[0]) for line in p.read_text().splitlines()]
        for split in ("train", "val")
        for p in (dest / split / "visible" / "labels").iterdir()
    }
    assert written == {stem: [class_id] for stem, class_id in EXPECTED_CLASS_IDS.items()}


def test_frame_with_only_dropped_classes_has_no_label_file(
    msrs_raw_root: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "processed"

    adapt_msrs(msrs_raw_root, dest)

    label_path = dest / "train" / "visible" / "labels" / f"{CURVE_ONLY_STEM}.txt"
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


def test_missing_mask_raises_before_anything_is_written(
    msrs_raw_root: Path, tmp_path: Path
) -> None:
    (msrs_raw_root / "test" / "Segmentation_labels" / f"{TEST_STEMS[0]}.png").unlink()
    dest = tmp_path / "processed"

    with pytest.raises(AdapterError, match=TEST_STEMS[0]):
        adapt_msrs(msrs_raw_root, dest)

    # A half-written tree would be skipped as already populated on the rerun.
    assert not dest.exists()


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
    assert len(train) == 1083
    assert len(val) == 361
    for raw_split in ("train", "test"):
        frames = {p.stem for p in (REAL_MSRS_RAW / raw_split / "vi").iterdir()}
        masks = {p.stem for p in (REAL_MSRS_RAW / raw_split / "Segmentation_labels").iterdir()}
        assert frames <= masks
    # The other 44 train / 13 val frames have no kept-class blob of at least MIN_BLOB_AREA px.
    assert len(list((tmp_path / "processed" / "train" / "visible" / "labels").iterdir())) == 1039
    assert len(list((tmp_path / "processed" / "val" / "visible" / "labels").iterdir())) == 348
