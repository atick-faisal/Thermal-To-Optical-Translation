"""The dataset's data.yaml as the single source of truth."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from conftest import NAMES
from t2o.data.manifest import DatasetManifest, ManifestError

_REAL_DATASET = Path(__file__).resolve().parent.parent / "dataset" / "yolo_rgbt" / "data.yaml"


def test_reads_classes_from_the_dataset_not_from_code(data_yaml: Path) -> None:
    manifest = DatasetManifest.load(data_yaml)
    assert manifest.nc == len(NAMES)
    assert manifest.class_names == NAMES


def test_a_five_class_dataset_needs_no_config_change(tmp_path: Path, data_yaml: Path) -> None:
    """The same pipeline must run on a 4-class and a 5-class dataset unchanged.

    Nothing about the classes lives in this project's config, so pointing at a different
    manifest is the whole of the difference.
    """
    declared = yaml.safe_load(data_yaml.read_text())
    declared["names"] = [*NAMES, "Insulator"]
    declared["nc"] = 5
    declared["path"] = str(data_yaml.parent)

    five = tmp_path / "five.yaml"
    five.write_text(yaml.safe_dump(declared))

    manifest = DatasetManifest.load(five)
    assert manifest.nc == 5
    assert manifest.class_names[-1] == "Insulator"


def test_splits_and_tokens_come_from_the_manifest(data_yaml: Path) -> None:
    manifest = DatasetManifest.load(data_yaml)
    assert manifest.train_images.name == "images"
    assert manifest.train_images.parent.name == "visible"
    assert manifest.val_images.parent.parent.name == "val"
    assert manifest.pairing.visible_token == "visible"
    assert manifest.pairing.infrared_token == "infrared"


def test_custom_tokens_are_honoured(tmp_path: Path, data_yaml: Path) -> None:
    declared = yaml.safe_load(data_yaml.read_text())
    declared["rgbt"] = {"visible_token": "rgb", "infrared_token": "thermal"}
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.safe_dump(declared))

    manifest = DatasetManifest.load(custom)
    assert manifest.pairing.infrared_token == "thermal"


def test_names_as_an_index_mapping(tmp_path: Path, data_yaml: Path) -> None:
    # ultralytics also writes names as {0: "a", 1: "b"}; order must follow the index.
    declared = yaml.safe_load(data_yaml.read_text())
    declared["names"] = {1: "second", 0: "first"}
    declared["nc"] = 2
    mapped = tmp_path / "mapped.yaml"
    mapped.write_text(yaml.safe_dump(declared))

    assert DatasetManifest.load(mapped).class_names == ["first", "second"]


def test_nc_is_derived_when_absent(tmp_path: Path, data_yaml: Path) -> None:
    declared = yaml.safe_load(data_yaml.read_text())
    del declared["nc"]
    derived = tmp_path / "derived.yaml"
    derived.write_text(yaml.safe_dump(declared))

    assert DatasetManifest.load(derived).nc == len(NAMES)


def test_internally_inconsistent_manifest_is_rejected(tmp_path: Path, data_yaml: Path) -> None:
    declared = yaml.safe_load(data_yaml.read_text())
    declared["nc"] = 9  # names still has 4 entries
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(declared))

    with pytest.raises(ManifestError, match="does not match"):
        DatasetManifest.load(bad)


def test_a_directory_resolves_to_its_manifest(dataset_root: Path) -> None:
    assert DatasetManifest.load(dataset_root) == DatasetManifest.load(dataset_root / "data.yaml")


def test_missing_manifest_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="dataset manifest not found"):
        DatasetManifest.load(tmp_path / "nope.yaml")


def test_missing_split_fails_loudly(tmp_path: Path, data_yaml: Path) -> None:
    declared = yaml.safe_load(data_yaml.read_text())
    del declared["val"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(declared))

    with pytest.raises(ManifestError, match="missing required 'val' split"):
        DatasetManifest.load(bad)


def test_root_resolves_when_path_is_relative_to_the_project(
    tmp_path: Path, data_yaml: Path
) -> None:
    """A `path:` written relative to the project root, not the manifest.

    A real `data.yaml` can be written with `path: dataset/yolo_rgbt` while itself sitting
    inside `dataset/yolo_rgbt` -- resolving naively against the manifest's own directory
    would then look for `dataset/yolo_rgbt/dataset/yolo_rgbt`. This checks the final
    fallback (the manifest's own directory) catches a stale/bogus `path:` instead.
    """
    declared = yaml.safe_load(data_yaml.read_text())
    declared["path"] = "some/stale/prefix"
    stale = data_yaml.parent / "stale.yaml"
    stale.write_text(yaml.safe_dump(declared))

    manifest = DatasetManifest.load(stale)
    assert manifest.train_images.is_dir()
    stale.unlink()


@pytest.mark.skipif(not _REAL_DATASET.exists(), reason="local dataset/ fixture not present")
def test_real_local_dataset_resolves() -> None:
    """Sanity check against the real (untracked) 9-pair local sample, when present.

    Additive only -- PLAN.md §9 requires every code path stay covered by the synthetic
    fixture regardless, which the tests above already do.
    """
    manifest = DatasetManifest.load(_REAL_DATASET)
    assert manifest.nc == 4
    assert manifest.train_images.is_dir()
    assert manifest.val_images.is_dir()
