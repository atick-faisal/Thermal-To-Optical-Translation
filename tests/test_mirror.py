"""Mirroring visible-side labels onto the infrared side (`t2o.data.mirror`)."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

from conftest import N_TRAIN, N_VAL
from t2o.data.budget import BudgetError, write_budget_manifest
from t2o.data.calibration import CalibrationError
from t2o.data.manifest import DatasetManifest
from t2o.data.mirror import PROVENANCE_FILENAME, MirrorError, mirror_labels_onto_infrared
from test_calibration import make_calibration


@pytest.fixture
def mirrorable(dataset_root: Path, tmp_path: Path) -> DatasetManifest:
    """A throwaway copy of the shared fixture, unmirrored.

    `dataset_root` is session-scoped and these tests write into the tree (and one deletes from
    it), so they cannot touch it -- same reason `test_budget.py`'s `thermal_labelled` copies.
    Unlike that fixture this one does *not* mirror: that is the code under test.
    """
    root = tmp_path / "mirrorable"
    shutil.copytree(dataset_root, root)
    # The fixture's `path:` is absolute, so a copied manifest resolves back to the original
    # tree and everything written below would go unseen (`manifest.py::_resolve_root`).
    copied = root / "data.yaml"
    copied.write_text(copied.read_text().replace(f"path: {dataset_root}", f"path: {root}"))
    return DatasetManifest.load(copied)


def _labels_dir(manifest: DatasetManifest, split: str, modality: str) -> Path:
    images = manifest.train_images if split == "train" else manifest.val_images
    if modality == "infrared":
        images = manifest.pairing.infrared_path(images)
    return images.parent / "labels"


def test_mirrors_every_split_byte_for_byte(mirrorable: DatasetManifest) -> None:
    written = mirror_labels_onto_infrared(mirrorable)

    assert written == {"train": N_TRAIN, "val": N_VAL}
    for split in ("train", "val"):
        sources = sorted(_labels_dir(mirrorable, split, "visible").glob("*.txt"))
        assert sources, f"{split} fixture has no visible labels to compare against"
        for source in sources:
            mirrored = _labels_dir(mirrorable, split, "infrared") / source.name
            assert mirrored.read_bytes() == source.read_bytes()


def test_unblocks_the_thermal_budget_arm(mirrorable: DatasetManifest, tmp_path: Path) -> None:
    """The whole point: `write_budget_manifest(infrared=True)` refuses an unmirrored tree.

    That refusal is `budget.py::_require_labels_beside`, whose message tells the caller to
    mirror first. Asserting both sides of it is what ties this operation to the reason it
    exists, rather than testing a file copy for its own sake.
    """
    with pytest.raises(BudgetError, match="has no labels beside it"):
        write_budget_manifest(mirrorable, tmp_path / "before", 0.4, infrared=True)

    mirror_labels_onto_infrared(mirrorable)

    assert write_budget_manifest(mirrorable, tmp_path / "after", 0.4, infrared=True).is_file()


def test_is_idempotent_under_force(mirrorable: DatasetManifest) -> None:
    first = mirror_labels_onto_infrared(mirrorable)
    infrared = _labels_dir(mirrorable, "train", "infrared")
    before = {p.name: p.read_bytes() for p in infrared.glob("*.txt")}

    assert mirror_labels_onto_infrared(mirrorable, force=True) == first
    assert {p.name: p.read_bytes() for p in infrared.glob("*.txt")} == before


def test_refuses_to_overwrite_existing_thermal_labels(mirrorable: DatasetManifest) -> None:
    """The custom dataset carries real labels on both sides; those must not be clobbered."""
    mirror_labels_onto_infrared(mirrorable)

    with pytest.raises(MirrorError, match="already holds"):
        mirror_labels_onto_infrared(mirrorable)


def test_rejects_a_tree_with_no_visible_labels(mirrorable: DatasetManifest) -> None:
    for label in _labels_dir(mirrorable, "train", "visible").glob("*.txt"):
        label.unlink()

    with pytest.raises(MirrorError, match="nothing to mirror"):
        mirror_labels_onto_infrared(mirrorable)


# --- de-rolling -------------------------------------------------------------------------------
#
# A plain mirror inherits the dataset's cross-modal residual, which pushes the raw-thermal floor
# down and so *flatters* translation (module docstring). These cover the correction.

ONE_BOX = "1 0.500000 0.500000 0.125000 0.125000\n"  # 80x60 px centred in a 640x480 frame
# FLIR's own corner field in miniature -- a roll, so corner-mapping re-bounds and centre-mapping
# does not. A uniform field would make the two modes identical and test nothing.
ROLL = ((0.0, 6.0), (-12.0, 1.0), (0.0, -6.0), (13.0, 0.0))


def _constant(manifest: DatasetManifest, corner_shift: tuple[tuple[float, float], ...]):
    """A constant `validate_for` accepts on this tree.

    The tree's own root directory names the dataset and its frames are 480x640, and both are
    checked -- so a fixture-bound factory is what every de-roll test has to go through.
    """
    return make_calibration(dataset=manifest.root.name, corner_shift=corner_shift)


def _shift(manifest: DatasetManifest, dx: float, dy: float):
    """A pure translation: all four corners move identically."""
    return _constant(manifest, ((dx, dy),) * 4)


@pytest.fixture
def one_known_box(mirrorable: DatasetManifest) -> Path:
    """Replace one visible label with a single box well clear of every edge.

    The fixture's own boxes are random, so an exact assertion about where a box lands needs a
    box whose starting position is known and whose corrected position cannot be clamped.
    """
    label = sorted(_labels_dir(mirrorable, "train", "visible").glob("*.txt"))[0]
    label.write_text(ONE_BOX)
    return label


def _mirrored(manifest: DatasetManifest, label: Path) -> Path:
    return _labels_dir(manifest, "train", "infrared") / label.name


def test_a_calibration_transforms_boxes_instead_of_copying(
    mirrorable: DatasetManifest, one_known_box: Path
) -> None:
    mirror_labels_onto_infrared(mirrorable, calibration=_shift(mirrorable, 4.0, 3.0))

    assert _mirrored(mirrorable, one_known_box).read_text() != ONE_BOX


def test_a_pure_translation_moves_a_box_by_exactly_that_many_pixels(
    mirrorable: DatasetManifest, one_known_box: Path
) -> None:
    """A uniform corner shift of (+40, +30) means the visible -> thermal map is its inverse, so
    the box moves *back* by (40, 30) px -- 0.0625 of the width and 0.0625 of the height."""
    mirror_labels_onto_infrared(mirrorable, calibration=_shift(mirrorable, 40.0, 30.0))

    assert _mirrored(mirrorable, one_known_box).read_text() == (
        "1 0.437500 0.437500 0.125000 0.125000\n"
    )


def test_the_centre_transform_preserves_the_box_extent(
    mirrorable: DatasetManifest, one_known_box: Path
) -> None:
    """Corner-mapping re-bounds a rotated rectangle and so grows it; centre-mapping cannot.

    Both ship, because the two agreeing on the measured floor is the cheapest evidence that the
    choice does not carry the result.
    """
    roll = _constant(mirrorable, ROLL)

    mirror_labels_onto_infrared(mirrorable, calibration=roll, box_transform="centre")
    centre = _mirrored(mirrorable, one_known_box).read_text().split()
    mirror_labels_onto_infrared(mirrorable, calibration=roll, force=True)
    corners = _mirrored(mirrorable, one_known_box).read_text().split()

    assert (centre[3], centre[4]) == ("0.125000", "0.125000")
    assert float(corners[3]) > 0.125
    assert float(corners[4]) > 0.125


def test_a_box_pushed_out_of_frame_is_dropped_and_counted(
    mirrorable: DatasetManifest, one_known_box: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Clamp-then-drop, the rule `adapters/common.py::voc_to_yolo_lines` already follows: a box
    with no area left is a degenerate detection target, not a negative. A ~6 px residual cannot
    do this to a real box, which is why a nonzero count is logged at WARNING."""
    with caplog.at_level(logging.INFO, logger="t2o.data.mirror"):
        mirror_labels_onto_infrared(mirrorable, calibration=_shift(mirrorable, 700.0, 0.0))

    assert _mirrored(mirrorable, one_known_box).read_text() == ""
    assert "box(es) dropped" in caplog.text
    assert "0 box(es) dropped" not in caplog.text


def test_the_provenance_marker_records_the_constant_and_a_plain_rerun_removes_it(
    mirrorable: DatasetManifest,
) -> None:
    """Both states are the same filenames in the same directory, so the tree has to say which
    one it holds -- and the marker must not outlive the correction it describes."""
    calibration = _shift(mirrorable, 4.0, 3.0)
    mirror_labels_onto_infrared(mirrorable, calibration=calibration)
    marker = _labels_dir(mirrorable, "val", "infrared").parent / PROVENANCE_FILENAME

    recorded = json.loads(marker.read_text())
    assert recorded["calibration_digest"] == calibration.digest()
    assert recorded["box_transform"] == "corners"
    assert recorded["files"] == N_VAL

    mirror_labels_onto_infrared(mirrorable, force=True)

    assert not marker.exists()


def test_a_constant_for_another_dataset_is_refused(mirrorable: DatasetManifest) -> None:
    """The tree's own directory name is what `validate_for` compares against, so a constant
    measured on a different dataset cannot be applied by accident."""
    with pytest.raises(CalibrationError, match="is for 'synthetic'"):
        mirror_labels_onto_infrared(mirrorable, calibration=make_calibration())


def test_a_constant_for_another_resolution_is_refused(mirrorable: DatasetManifest) -> None:
    """The shape comes from reading an actual image, not from a caller's assertion: a 640x512
    corner field applied at 1280x1024 is silently half the offset it claims to be."""
    wrong_shape = make_calibration(dataset=mirrorable.root.name, height=1024, width=1280)

    with pytest.raises(CalibrationError, match="does not rescale"):
        mirror_labels_onto_infrared(mirrorable, calibration=wrong_shape)
