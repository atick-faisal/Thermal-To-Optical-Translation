"""`t2o.data.inner_split` -- ModTr's calibration split (modtr-baseline ADR-04).

Built on the synthetic `data_yaml` fixture: 5 train / 3 val pairs with disjoint stems, so a val
image leaking into the inner split shows up by name.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from conftest import TRAIN_STEMS, VAL_STEMS
from t2o.data.dataset import TranslationPairDataset, annotated_subset
from t2o.data.inner_split import MARKER_KEY, InnerSplitError, write_inner_split
from t2o.data.manifest import DatasetManifest


def _stems(images_dir: Path) -> set[str]:
    return {p.stem for p in images_dir.iterdir()}


def test_the_inner_split_partitions_train_and_never_reads_val(
    data_yaml: Path, tmp_path: Path
) -> None:
    inner = DatasetManifest.load(
        write_inner_split(DatasetManifest.load(data_yaml), tmp_path / "inner", fraction=0.4)
    )

    train, val = _stems(inner.train_images), _stems(inner.val_images)
    assert len(val) == 2
    assert train.isdisjoint(val)
    assert train | val == set(TRAIN_STEMS)
    assert not (train | val) & set(VAL_STEMS)


def test_the_held_out_images_are_the_shared_selectors_draw(data_yaml: Path, tmp_path: Path) -> None:
    source = DatasetManifest.load(data_yaml)
    inner = DatasetManifest.load(write_inner_split(source, tmp_path / "inner", 0.4, seed=3))

    images = sorted(source.train_images.iterdir())
    assert _stems(inner.val_images) == {p.stem for p in annotated_subset(images, 0.4, 3)}


def test_pairs_travel_with_their_thermal_image_and_labels(data_yaml: Path, tmp_path: Path) -> None:
    source = DatasetManifest.load(data_yaml)
    inner = DatasetManifest.load(write_inner_split(source, tmp_path / "inner", 0.4))

    for images_dir in (inner.train_images, inner.val_images):
        # Validates every pair's thermal counterpart and every label's class ids.
        dataset = TranslationPairDataset(images_dir, pairing=inner.pairing, num_classes=inner.nc)
        for image in dataset.visible_paths:
            original = source.train_images / image.name
            label = source.pairing.label_path(original)
            assert inner.pairing.label_path(image).read_text() == label.read_text()


def test_thermal_labels_are_copied_when_the_source_has_them(
    dataset_root: Path, tmp_path: Path
) -> None:
    # The custom set carries labels on both modalities; the fixture carries visible ones only.
    root = tmp_path / "source"
    shutil.copytree(dataset_root, root)
    (root / "data.yaml").write_text(
        (root / "data.yaml").read_text().replace(str(dataset_root), str(root))
    )
    for label in (root / "train" / "visible" / "labels").iterdir():
        shutil.copyfile(label, root / "train" / "infrared" / "labels" / label.name)

    write_inner_split(DatasetManifest.load(root / "data.yaml"), tmp_path / "inner", 0.4)

    for split in ("train", "val"):
        visible = {p.name for p in (tmp_path / "inner" / split / "visible" / "labels").iterdir()}
        infrared = {p.name for p in (tmp_path / "inner" / split / "infrared" / "labels").iterdir()}
        assert visible == infrared


def test_the_split_is_deterministic(data_yaml: Path, tmp_path: Path) -> None:
    source = DatasetManifest.load(data_yaml)
    first = DatasetManifest.load(write_inner_split(source, tmp_path / "a", 0.4, seed=7))
    second = DatasetManifest.load(write_inner_split(source, tmp_path / "b", 0.4, seed=7))

    assert _stems(first.val_images) == _stems(second.val_images)


def test_the_manifest_is_absolute_and_marked(data_yaml: Path, tmp_path: Path) -> None:
    source = DatasetManifest.load(data_yaml)
    declared = yaml.safe_load(write_inner_split(source, tmp_path / "inner", 0.4, 5).read_text())

    # ultralytics honours `path:` literally (CLAUDE.md), so a relative one breaks off the Mac.
    assert Path(declared["path"]).is_absolute()
    assert declared[MARKER_KEY] == {"source": str(source.path), "fraction": 0.4, "seed": 5}


def test_a_non_empty_output_directory_is_refused(data_yaml: Path, tmp_path: Path) -> None:
    out = tmp_path / "inner"
    out.mkdir()
    (out / "leftover.txt").write_text("")

    with pytest.raises(InnerSplitError, match="not empty"):
        write_inner_split(DatasetManifest.load(data_yaml), out)


@pytest.mark.parametrize("fraction", [0.05, 0.95])
def test_a_fraction_that_empties_either_side_is_refused(
    data_yaml: Path, tmp_path: Path, fraction: float
) -> None:
    with pytest.raises(InnerSplitError, match="both must be non-empty"):
        write_inner_split(DatasetManifest.load(data_yaml), tmp_path / "inner", fraction)


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_a_fraction_outside_the_open_unit_interval_is_refused(
    data_yaml: Path, tmp_path: Path, fraction: float
) -> None:
    with pytest.raises(InnerSplitError, match="must be in"):
        write_inner_split(DatasetManifest.load(data_yaml), tmp_path / "inner", fraction)
