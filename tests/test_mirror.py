"""Mirroring visible-side labels onto the infrared side (`t2o.data.mirror`)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from conftest import N_TRAIN, N_VAL
from t2o.data.budget import BudgetError, write_budget_manifest
from t2o.data.manifest import DatasetManifest
from t2o.data.mirror import MirrorError, mirror_labels_onto_infrared


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
