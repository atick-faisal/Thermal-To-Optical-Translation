"""Counterfactual erasure (`t2o.data.erasure`) and its `t2o erase` command.

The pure function is tested on a hand-built frame: a bright object on a smooth gradient, whose
label box is deliberately 2 px tighter than the object, the way a registration-shifted box is.
Random noise would not do: Telea fills about 1 in 200 masked noise pixels with their original
value by chance, which makes "every pixel inside changed" flaky for no real reason.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from t2o.cli import main
from t2o.config import Config
from t2o.data.erasure import ErasureError, box_mask, erase_boxes, write_erased_split
from t2o.data.labels import load_yolo_labels
from t2o.data.manifest import DatasetManifest
from t2o.engine.export import export_translated
from t2o.translators import StubTranslator

HEIGHT, WIDTH = 120, 160
# The object spans x 60..100, y 40..80; its label box is 2 px tighter on every side.
OBJECT = (slice(40, 80), slice(60, 100))
TIGHT_BOX = np.array([[80 / WIDTH, 60 / HEIGHT, 36 / WIDTH, 36 / HEIGHT]])


def _frame(channels: int) -> np.ndarray:
    gradient = np.tile(np.linspace(20, 120, WIDTH, dtype=np.float64), (HEIGHT, 1))
    frame = gradient.astype(np.uint8)
    frame[OBJECT] = 255
    return frame if channels == 1 else np.stack([frame] * channels, axis=-1)


@pytest.mark.parametrize("channels", [1, 3])
def test_erases_inside_the_dilated_box_and_nothing_outside(channels: int) -> None:
    frame = _frame(channels)
    erased = erase_boxes(frame, TIGHT_BOX, dilate_px=4)

    outside = box_mask((HEIGHT, WIDTH), TIGHT_BOX, dilate_px=4) == 0
    assert np.array_equal(erased[outside], frame[outside])
    # The whole object is gone, including the 2 px rim the tight box missed.
    assert (erased[OBJECT] < 200).all()


def test_without_dilation_the_rim_outside_a_tight_box_survives() -> None:
    """Why the boxes are grown at all: a registration-shifted box leaves a warm rim behind."""
    erased = erase_boxes(_frame(1), TIGHT_BOX, dilate_px=0)

    assert (erased[OBJECT] == 255).any()


def test_is_deterministic() -> None:
    frame = _frame(1)

    assert np.array_equal(erase_boxes(frame, TIGHT_BOX), erase_boxes(frame, TIGHT_BOX))


def test_a_frame_with_no_boxes_is_returned_unchanged() -> None:
    frame = _frame(1)
    erased = erase_boxes(frame, np.zeros((0, 4)))

    assert np.array_equal(erased, frame)
    assert erased is not frame


def test_box_mask_clips_a_dilated_box_at_the_frame_edge() -> None:
    corner = np.array([[0.01, 0.01, 0.02, 0.02]])
    mask = box_mask((HEIGHT, WIDTH), corner, dilate_px=4)

    assert mask[0, 0] == 255
    assert mask.shape == (HEIGHT, WIDTH)


@pytest.fixture
def erased_yaml(data_yaml: Path, tmp_path: Path) -> Path:
    return write_erased_split(DatasetManifest.load(data_yaml), tmp_path / "erased")


def test_split_copy_is_loadable_and_points_at_itself(erased_yaml: Path) -> None:
    erased = DatasetManifest.load(erased_yaml)
    declared = yaml.safe_load(erased_yaml.read_text())

    # Absolute and correct, because ultralytics honours `path:` literally.
    assert declared["path"] == str(erased_yaml.parent.resolve())
    assert erased.root == erased_yaml.parent.resolve()
    assert erased.train_images == erased.val_images == erased.root / "val" / "visible" / "images"
    assert declared["erasure"]["split"] == "val"


def test_split_copy_keeps_labels_and_visible_frames_and_erases_thermal(
    erased_yaml: Path, dataset_root: Path
) -> None:
    source = DatasetManifest.load(dataset_root / "data.yaml")
    erased = DatasetManifest.load(erased_yaml)
    pairing = source.pairing

    labelled = 0
    for visible_path in sorted(source.val_images.glob("*.jpg")):
        name = f"{visible_path.stem}.png"
        copied_label = pairing.label_path(erased.val_images / name)
        assert copied_label.read_bytes() == pairing.label_path(visible_path).read_bytes()

        with Image.open(visible_path) as original, Image.open(erased.val_images / name) as copy:
            assert np.array_equal(np.asarray(original), np.asarray(copy))

        _, bboxes = load_yolo_labels(pairing.label_path(visible_path))
        with (
            Image.open(pairing.infrared_path(visible_path)) as original,
            Image.open(pairing.infrared_path(erased.val_images / name)) as copy,
        ):
            before, after = np.asarray(original), np.asarray(copy)
        inside = box_mask(before.shape, bboxes.numpy(), dilate_px=4) > 0
        assert np.array_equal(before[~inside], after[~inside])
        if len(bboxes):
            labelled += 1
            assert not np.array_equal(before[inside], after[inside])
    assert labelled, "the fixture's val split should carry boxes to erase"


def test_split_copy_never_carries_a_label_cache(data_yaml: Path, tmp_path: Path) -> None:
    source = DatasetManifest.load(data_yaml)
    stale = source.val_images.parent / "labels.cache"
    stale.write_bytes(b"stale")
    try:
        out = write_erased_split(source, tmp_path / "erased").parent
    finally:
        stale.unlink()

    assert not list(out.rglob("*.cache"))


def test_split_copy_is_byte_deterministic(data_yaml: Path, tmp_path: Path) -> None:
    source = DatasetManifest.load(data_yaml)
    first = write_erased_split(source, tmp_path / "a").parent
    second = write_erased_split(source, tmp_path / "b").parent

    frames = sorted(p.relative_to(first) for p in first.rglob("*.png"))
    assert frames
    for frame in frames:
        assert (first / frame).read_bytes() == (second / frame).read_bytes()


def test_refuses_a_non_empty_output(data_yaml: Path, tmp_path: Path) -> None:
    out = tmp_path / "erased"
    out.mkdir()
    (out / "leftover.png").write_bytes(b"")

    with pytest.raises(ErasureError, match="not empty"):
        write_erased_split(DatasetManifest.load(data_yaml), out)


def test_export_reads_the_erased_split(erased_yaml: Path, tmp_path: Path) -> None:
    from ultralytics.data.utils import check_det_dataset

    config = Config.load(
        overrides={
            "data": {"manifest": str(erased_yaml)},
            "runtime": {"workers": 0},
            "detector": {"evaluation": {"batch": 2}},
        }
    )
    torch.manual_seed(0)
    exported = export_translated(StubTranslator(hidden_channels=4).eval(), config, tmp_path / "x")

    n_val = len(list(DatasetManifest.load(erased_yaml).val_images.glob("*.png")))
    assert len(list((tmp_path / "x" / "val" / "images").glob("*.png"))) == n_val
    assert check_det_dataset(str(exported))["nc"] == 4
    assert check_det_dataset(str(erased_yaml))["nc"] == 4


def test_main_erase_writes_the_split(data_yaml: Path, tmp_path: Path) -> None:
    out = tmp_path / "erased"
    exit_code = main(
        [
            "erase",
            "--data",
            str(data_yaml),
            "--out",
            str(out),
            "--split",
            "train",
            "--dilate-px",
            "2",
        ]
    )

    assert exit_code == 0
    declared = yaml.safe_load((out / "data.yaml").read_text())
    assert declared["train"] == "train/visible/images"
    assert declared["erasure"]["dilate_px"] == 2
    assert declared["erasure"]["inpaint_radius"] == 3.0
