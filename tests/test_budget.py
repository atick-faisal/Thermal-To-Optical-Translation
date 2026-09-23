"""E8's annotation-budget manifests (`t2o.data.budget`)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from conftest import N_TRAIN, N_VAL
from t2o.data.budget import BudgetError, write_budget_manifest
from t2o.data.dataset import annotated_subset
from t2o.data.manifest import DatasetManifest, ManifestError


@pytest.fixture
def manifest(data_yaml: Path) -> DatasetManifest:
    return DatasetManifest.load(data_yaml)


@pytest.fixture
def thermal_labelled(dataset_root: Path, tmp_path: Path) -> DatasetManifest:
    """A copy of the fixture with labels mirrored onto the infrared side.

    The shared fixture leaves `infrared/labels` empty, like every adapted public dataset
    (`data/adapters/common.py`). The *custom* pairs carry both sides, and E8's thermal arm
    is only buildable on a dataset that does -- so the two cases need two fixtures.
    """
    root = tmp_path / "mirrored"
    shutil.copytree(dataset_root, root)
    # The fixture's `path:` is absolute, so a copied manifest resolves back to the original
    # tree and the mirrored labels below would never be seen (`manifest.py::_resolve_root`).
    copied = root / "data.yaml"
    copied.write_text(copied.read_text().replace(f"path: {dataset_root}", f"path: {root}"))
    for split in ("train", "val"):
        for label in (root / split / "visible" / "labels").glob("*.txt"):
            shutil.copy(label, root / split / "infrared" / "labels" / label.name)
    return DatasetManifest.load(root / "data.yaml")


def _listed(data_yaml: Path) -> list[Path]:
    declared = yaml.safe_load(data_yaml.read_text())
    return [Path(line) for line in Path(declared["train"]).read_text().splitlines()]


def test_a_budget_selects_exactly_what_the_translator_side_selects(
    manifest: DatasetManifest, tmp_path: Path
) -> None:
    """The invariant the whole sweep rests on.

    If the detector arm and the translator arm chose different images at the same
    (fraction, seed), "annotation budget" would silently mean two different things on one
    x-axis and no output would say so. One selector, shared, is what prevents that.
    """
    written = write_budget_manifest(manifest, tmp_path / "b", 0.4, seed=3)

    train_images = sorted(manifest.train_images.iterdir())
    expected = annotated_subset(train_images, 0.4, 3)
    assert {p.resolve() for p in _listed(written)} == {p.resolve() for p in expected}


def test_the_full_budget_keeps_every_train_image(manifest: DatasetManifest, tmp_path: Path) -> None:
    written = write_budget_manifest(manifest, tmp_path / "b", 1.0)
    assert len(_listed(written)) == N_TRAIN


def test_the_val_split_is_never_subsampled(manifest: DatasetManifest, tmp_path: Path) -> None:
    """Every point on the curve has to be scored against the same frames to be a curve."""
    written = write_budget_manifest(manifest, tmp_path / "b", 0.4)
    val = Path(yaml.safe_load(written.read_text())["val"])

    assert val == manifest.val_images.resolve()
    assert len(list(val.glob("*.jpg"))) == N_VAL


def test_a_budget_is_byte_identical_for_a_fixed_seed(
    manifest: DatasetManifest, tmp_path: Path
) -> None:
    a = write_budget_manifest(manifest, tmp_path / "a", 0.4, seed=1)
    b = write_budget_manifest(manifest, tmp_path / "b", 0.4, seed=1)
    assert (a.parent / "train.txt").read_text() == (b.parent / "train.txt").read_text()


def test_a_different_seed_selects_a_different_subset(
    manifest: DatasetManifest, tmp_path: Path
) -> None:
    a = write_budget_manifest(manifest, tmp_path / "a", 0.4, seed=1)
    b = write_budget_manifest(manifest, tmp_path / "b", 0.4, seed=2)
    assert _listed(a) != _listed(b)


def test_a_budget_that_selects_nothing_is_refused(
    manifest: DatasetManifest, tmp_path: Path
) -> None:
    """E8's N=0 point is the untrained detector -- an evaluation, never a fine-tune."""
    with pytest.raises(BudgetError, match="selects nothing"):
        write_budget_manifest(manifest, tmp_path / "b", 0.01)


def test_the_thermal_arm_switches_modality_on_both_splits(
    thermal_labelled: DatasetManifest, tmp_path: Path
) -> None:
    written = write_budget_manifest(thermal_labelled, tmp_path / "b", 0.4, infrared=True)
    declared = yaml.safe_load(written.read_text())

    assert all(p.parent.parent.name == "infrared" for p in _listed(written))
    assert all(p.exists() for p in _listed(written))
    assert Path(declared["val"]).parent.name == "infrared"


def test_the_thermal_arm_picks_the_same_scenes_as_the_translated_arm(
    thermal_labelled: DatasetManifest, tmp_path: Path
) -> None:
    """Arms A and B differ in pixels and in nothing else -- that is what pairs them."""
    thermal = write_budget_manifest(thermal_labelled, tmp_path / "a", 0.4, seed=5, infrared=True)
    visible = write_budget_manifest(thermal_labelled, tmp_path / "b", 0.4, seed=5)

    assert [p.name for p in _listed(thermal)] == [p.name for p in _listed(visible)]


def test_a_thermal_arm_with_no_labels_on_that_modality_is_refused(
    manifest: DatasetManifest, tmp_path: Path
) -> None:
    """The one way E8 could return a confident, meaningless number.

    Every adapted public dataset labels the visible side only, so a detector pointed at
    `infrared/images` there reads the split as entirely unlabelled: it trains, it reports,
    and the mAP describes nothing.
    """
    with pytest.raises(BudgetError, match="has no labels beside it"):
        write_budget_manifest(manifest, tmp_path / "b", 0.4, infrared=True)


def test_a_budget_manifest_is_rejected_by_our_own_data_layer(
    manifest: DatasetManifest, tmp_path: Path
) -> None:
    """It is an ultralytics manifest, and only that.

    `train` is a text file, so `DatasetManifest.load`'s split-directory requirement refuses
    it. That rejection is the guard keeping a detector-side budget out of the translator's
    data layer, where it would mean something quite different.
    """
    written = write_budget_manifest(manifest, tmp_path / "b", 0.4)
    with pytest.raises(ManifestError):
        DatasetManifest.load(written)
