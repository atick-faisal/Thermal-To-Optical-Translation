from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import IMG_SIZE, N_TRAIN, TRAIN_STEMS
from t2o.data.dataset import (
    TranslationPairDataset,
    TranslationSample,
    _crop_bboxes,
    collate_translation_batch,
)
from t2o.data.labels import load_yolo_labels
from t2o.data.pairing import Pairing

H, W = IMG_SIZE


def _images(root: Path, split: str = "train") -> Path:
    return root / split / "visible" / "images"


@pytest.fixture
def dataset(dataset_root: Path) -> TranslationPairDataset:
    return TranslationPairDataset(_images(dataset_root))


# --------------------------------------------------------------------------- basics


def test_length_matches_the_split(dataset: TranslationPairDataset) -> None:
    assert len(dataset) == N_TRAIN


def test_sample_shapes_dtypes_and_ranges(dataset: TranslationPairDataset) -> None:
    sample = dataset[1]
    assert sample.visible.shape == (3, H, W)
    assert sample.infrared.shape == (1, H, W)
    assert sample.visible.dtype == torch.float32
    assert sample.infrared.dtype == torch.float32
    assert sample.visible.min() >= 0.0 and sample.visible.max() <= 1.0
    assert sample.infrared.min() >= 0.0 and sample.infrared.max() <= 1.0
    assert sample.cls.shape == (2, 1)
    assert sample.bboxes.shape == (2, 4)


def test_unlabelled_image_yields_zero_instances(dataset: TranslationPairDataset) -> None:
    index = [p.stem for p in dataset.visible_paths].index(TRAIN_STEMS[0])
    sample = dataset[index]
    assert sample.cls.shape == (0, 1)
    assert sample.bboxes.shape == (0, 4)


def test_pairs_by_filename_not_sorted_index(
    dataset: TranslationPairDataset, dataset_root: Path
) -> None:
    # TRAIN_STEMS is deliberately not ordered by index, so the dataset's sorted listing
    # visits files in a different order than they were written. Loading the infrared plane
    # and comparing it against the file of the *same name* catches any index-based pairing.
    stem = sorted(TRAIN_STEMS)[2]
    index = [p.stem for p in dataset.visible_paths].index(stem)
    sample = dataset[index]

    on_disk = np.asarray(Image.open(dataset_root / "train" / "infrared" / "images" / f"{stem}.jpg"))
    loaded = (sample.infrared[0] * 255).round().to(torch.uint8).numpy()
    assert np.array_equal(loaded, on_disk)
    assert sample.name == f"{stem}.jpg"


def test_missing_infrared_counterpart_fails_fast(dataset_root: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(dataset_root, broken)
    next((broken / "train" / "infrared" / "images").iterdir()).unlink()

    with pytest.raises(FileNotFoundError, match="infrared counterparts missing"):
        TranslationPairDataset(_images(broken))


def test_missing_split_directory_fails_fast(dataset_root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="visible image directory not found"):
        TranslationPairDataset(_images(dataset_root, "test"))


# --------------------------------------------------------------------------- augmentation


def test_hflip_flips_image_and_bboxes_consistently(dataset_root: Path) -> None:
    plain = TranslationPairDataset(_images(dataset_root))
    flipped = TranslationPairDataset(_images(dataset_root), hflip=1.0)

    torch.manual_seed(0)
    original = plain[1]
    torch.manual_seed(0)
    augmented = flipped[1]

    assert torch.equal(augmented.visible, torch.flip(original.visible, dims=(2,)))
    assert torch.equal(augmented.infrared, torch.flip(original.infrared, dims=(2,)))
    assert torch.allclose(augmented.bboxes[:, 0], 1.0 - original.bboxes[:, 0])
    # Only cx moves; cy and the extents are untouched.
    assert torch.equal(augmented.bboxes[:, 1:], original.bboxes[:, 1:])


def test_hflip_probability_zero_is_identity(dataset_root: Path) -> None:
    plain = TranslationPairDataset(_images(dataset_root))
    never = TranslationPairDataset(_images(dataset_root), hflip=0.0)
    assert torch.equal(plain[2].visible, never[2].visible)


def test_crop_resizes_images_and_renormalises_bboxes(dataset_root: Path) -> None:
    crop = (64, 96)
    dataset = TranslationPairDataset(_images(dataset_root), crop=crop)
    sample = dataset[1]

    assert sample.visible.shape == (3, *crop)
    assert sample.infrared.shape == (1, *crop)
    assert sample.cls.shape[0] == sample.bboxes.shape[0]
    if sample.bboxes.numel():
        assert sample.bboxes.min() >= 0.0
        assert sample.bboxes.max() <= 1.0


def test_crop_drops_boxes_that_fall_outside_the_window() -> None:
    # One centred box, one in the far bottom-right corner; a top-left crop keeps only the
    # first, and cls must be filtered by the same mask.
    bboxes = torch.tensor([[0.25, 0.25, 0.2, 0.2], [0.95, 0.95, 0.05, 0.05]])
    kept, keep = _crop_bboxes(bboxes, (64, 64), (0, 0, 32, 32))

    assert keep.tolist() == [True, False]
    assert kept.shape == (1, 4)
    assert kept[0, 0].item() == pytest.approx(0.5, abs=1e-6)
    assert kept[0, 2].item() == pytest.approx(0.4, abs=1e-6)


def test_crop_clips_a_box_straddling_the_window_edge() -> None:
    # Box spans x in [0.4, 0.6] of a 100px image, i.e. pixels 40..60. A crop of the left
    # half keeps pixels 40..50, so the box halves in width.
    bboxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    kept, keep = _crop_bboxes(bboxes, (100, 100), (0, 0, 100, 50))

    assert keep.tolist() == [True]
    assert kept[0, 2].item() == pytest.approx(10 / 50, abs=1e-6)
    assert kept[0, 0].item() == pytest.approx(45 / 50, abs=1e-6)


def test_crop_on_zero_instances_is_a_noop() -> None:
    kept, keep = _crop_bboxes(torch.zeros(0, 4), (64, 64), (0, 0, 32, 32))
    assert kept.shape == (0, 4)
    assert keep.shape == (0,)


# --------------------------------------------------------------------------- collate


def test_collate_matches_the_ultralytics_contract(dataset: TranslationPairDataset) -> None:
    samples = [dataset[i] for i in range(4)]
    batch = collate_translation_batch(samples)
    total = sum(s.cls.shape[0] for s in samples)

    assert batch["visible"].shape == (4, 3, H, W)
    assert batch["infrared"].shape == (4, 1, H, W)
    assert batch["batch_idx"].shape == (total,)
    assert batch["cls"].shape == (total, 1)
    assert batch["bboxes"].shape == (total, 4)
    assert batch["names"] == [s.name for s in samples]

    # batch_idx must map each row back to its image, and only images with instances appear.
    for i, sample in enumerate(samples):
        assert int((batch["batch_idx"] == i).sum()) == sample.cls.shape[0]

    # The concatenation v8DetectionLoss performs must not fail on this layout.
    targets = torch.cat(
        (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
    )
    assert targets.shape == (total, 6)


def test_collate_survives_an_all_unlabelled_batch() -> None:
    empty = [
        TranslationSample(
            visible=torch.rand(3, 8, 8),
            infrared=torch.rand(1, 8, 8),
            cls=torch.zeros(0, 1),
            bboxes=torch.zeros(0, 4),
            name=f"{i}.jpg",
        )
        for i in range(2)
    ]
    batch = collate_translation_batch(empty)
    assert batch["batch_idx"].shape == (0,)
    assert batch["cls"].shape == (0, 1)
    assert batch["bboxes"].shape == (0, 4)


def test_works_with_a_torch_dataloader(dataset: TranslationPairDataset) -> None:
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=3, collate_fn=collate_translation_batch)
    batch = next(iter(loader))
    assert batch["visible"].shape[0] == 3


# --------------------------------------------------------------------------- labels


def test_labels_tolerate_missing_and_empty_files(tmp_path: Path) -> None:
    cls, bboxes = load_yolo_labels(tmp_path / "absent.txt")
    assert cls.shape == (0, 1)
    assert bboxes.shape == (0, 4)

    empty = tmp_path / "empty.txt"
    empty.write_text("\n  \n")
    cls, bboxes = load_yolo_labels(empty)
    assert cls.shape == (0, 1)


def test_labels_reject_malformed_lines(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("0 0.5 0.5 0.4\n")
    with pytest.raises(ValueError, match="expected 5 fields"):
        load_yolo_labels(bad)

    worse = tmp_path / "worse.txt"
    worse.write_text("0 0.5 0.5 0.4 abc\n")
    with pytest.raises(ValueError, match="non-numeric"):
        load_yolo_labels(worse)


def test_custom_pairing_is_used(dataset_root: Path) -> None:
    dataset = TranslationPairDataset(
        _images(dataset_root, "val"), pairing=Pairing("visible", "infrared")
    )
    assert len(dataset) > 0


# --------------------------------------------------------------------------- class-id precheck


def test_class_ids_beyond_nc_fail_fast(dataset_root: Path, tmp_path: Path) -> None:
    """A 5-class label set against a 4-class manifest must stop at startup.

    Unchecked, the out-of-range id reaches v8DetectionLoss and indexes a class logit that
    does not exist -- an opaque CUDA device-side assert far into a run.
    """
    broken = tmp_path / "broken"
    shutil.copytree(dataset_root / "train", broken)
    label = next(p for p in (broken / "visible" / "labels").iterdir() if p.read_text().strip())
    label.write_text("4 0.5 0.5 0.2 0.2\n")

    with pytest.raises(ValueError, match=r"class ids outside the manifest's nc=4") as exc:
        TranslationPairDataset(broken / "visible" / "images", num_classes=4)
    assert "class 4" in str(exc.value)
    assert label.name.replace(".txt", "") in str(exc.value)


def test_class_ids_within_nc_are_accepted(dataset_root: Path) -> None:
    dataset = TranslationPairDataset(_images(dataset_root), num_classes=4)
    assert len(dataset) == N_TRAIN


def test_validation_is_skipped_when_num_classes_is_unset(dataset_root: Path) -> None:
    assert len(TranslationPairDataset(_images(dataset_root))) == N_TRAIN


# --------------------------------------------------------------------------- annotation fraction


def test_full_annotation_fraction_keeps_every_label(dataset_root: Path) -> None:
    full = TranslationPairDataset(_images(dataset_root), annotation_fraction=1.0)
    plain = TranslationPairDataset(_images(dataset_root))
    for i in range(N_TRAIN):
        assert torch.equal(full[i].cls, plain[i].cls)
        assert torch.equal(full[i].bboxes, plain[i].bboxes)


def test_annotation_fraction_is_deterministic_for_a_fixed_seed(dataset_root: Path) -> None:
    a = TranslationPairDataset(_images(dataset_root), annotation_fraction=0.5, annotation_seed=1)
    b = TranslationPairDataset(_images(dataset_root), annotation_fraction=0.5, annotation_seed=1)
    assert a._annotated == b._annotated


def test_annotation_fraction_changes_with_seed(dataset_root: Path) -> None:
    a = TranslationPairDataset(_images(dataset_root), annotation_fraction=0.5, annotation_seed=1)
    b = TranslationPairDataset(_images(dataset_root), annotation_fraction=0.5, annotation_seed=2)
    assert a._annotated != b._annotated


def test_annotation_fraction_only_zeroes_labels_not_images(dataset_root: Path) -> None:
    """Excluded images still yield real image tensors -- only cls/bboxes are dropped.

    The translator's reconstruction losses (L2/LPIPS) always have a visible target, so a
    low-annotation run must not shrink the dataset itself, only the detection supervision.
    """
    subsampled = TranslationPairDataset(
        _images(dataset_root), annotation_fraction=0.5, annotation_seed=1
    )
    full = TranslationPairDataset(_images(dataset_root))
    assert len(subsampled) == len(full)

    excluded = [p for p in subsampled.visible_paths if p not in subsampled._annotated]
    assert excluded  # the split is non-trivial: at least one image is excluded
    index = subsampled.visible_paths.index(excluded[0])
    assert torch.equal(subsampled[index].visible, full[index].visible)
    assert subsampled[index].cls.shape == (0, 1)
    assert subsampled[index].bboxes.shape == (0, 4)


def test_annotation_fraction_out_of_range_is_rejected(dataset_root: Path) -> None:
    with pytest.raises(ValueError, match="annotation_fraction"):
        TranslationPairDataset(_images(dataset_root), annotation_fraction=0.0)
    with pytest.raises(ValueError, match="annotation_fraction"):
        TranslationPairDataset(_images(dataset_root), annotation_fraction=1.5)
