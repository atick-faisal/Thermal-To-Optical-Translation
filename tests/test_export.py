"""`engine/export.py` -- translated images + mirrored labels -> detector-ready dataset.

Built on the synthetic `dataset_root`/`data_yaml` fixtures (`tests/conftest.py`): 5 train /
3 val images, disjoint stems. Expected counts/sizes are read from the fixtures themselves
rather than importing conftest's constants -- a `tests` package shipped by a dependency
shadows the local one on the pyright module search path, so a `from tests.conftest import
...` resolves to the wrong module under type checking.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from t2o.config import Config
from t2o.data.calibration import CalibrationError
from t2o.data.dataset import TranslationPairDataset
from t2o.data.manifest import DatasetManifest
from t2o.data.mirror import mirror_labels_onto_infrared
from t2o.data.misalignment import Misalignment, misalign_label_lines
from t2o.engine.export import ExportError, export_split, export_translated
from t2o.imaging import Normalize
from t2o.translators import StubTranslator


@pytest.fixture(scope="module")
def translator() -> StubTranslator:
    torch.manual_seed(0)
    return StubTranslator(hidden_channels=4).eval()


@pytest.fixture
def dataset(dataset_root: Path) -> TranslationPairDataset:
    return TranslationPairDataset(dataset_root / "train" / "visible" / "images")


@pytest.fixture
def val_dataset(dataset_root: Path) -> TranslationPairDataset:
    return TranslationPairDataset(dataset_root / "val" / "visible" / "images")


def _config(dataset_root: Path) -> Config:
    return Config.load(
        overrides={
            "data": {"manifest": str(dataset_root / "data.yaml")},
            "runtime": {"workers": 0},
            "detector": {"evaluation": {"batch": 4}},
        }
    )


def test_export_writes_one_image_and_label_per_pair(
    translator: StubTranslator, dataset: TranslationPairDataset, tmp_path: Path
) -> None:
    written = export_split(translator, dataset, tmp_path, workers=0)

    assert written == len(dataset)
    assert len(list((tmp_path / "images").glob("*.png"))) == len(dataset)
    assert len(list((tmp_path / "labels").glob("*.txt"))) == len(dataset)


def test_exported_images_have_source_geometry(
    translator: StubTranslator, dataset: TranslationPairDataset, tmp_path: Path
) -> None:
    export_split(translator, dataset, tmp_path, workers=0)
    image = Image.open(next((tmp_path / "images").glob("*.png")))
    source = Image.open(dataset.visible_paths[0])

    assert image.size == source.size  # translation is fully convolutional
    assert image.mode == "RGB"


def test_label_tree_is_mirrored_verbatim(
    translator: StubTranslator,
    dataset: TranslationPairDataset,
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    export_split(translator, dataset, tmp_path, workers=0)

    for exported in (tmp_path / "labels").glob("*.txt"):
        source = dataset_root / "train" / "visible" / "labels" / exported.name
        assert exported.read_text() == source.read_text()


def test_export_is_independent_of_batch_size(
    translator: StubTranslator, dataset: TranslationPairDataset, tmp_path: Path
) -> None:
    """The regression test for batch-wise normalisation.

    The original min-max normalised across the whole batch (Clean-SeAFusion's
    `train.py:369`), so an exported pixel depended on which other images shared its batch
    -- and the detector's training data silently changed with batch size and shuffle order.
    """
    for normalize in (Normalize.CLAMP, Normalize.PER_IMAGE):
        one = tmp_path / f"{normalize}_b1"
        many = tmp_path / f"{normalize}_b5"
        export_split(translator, dataset, one, normalize=normalize, batch_size=1, workers=0)
        export_split(translator, dataset, many, normalize=normalize, batch_size=5, workers=0)

        for a in (one / "images").glob("*.png"):
            b = many / "images" / a.name
            assert np.array_equal(np.asarray(Image.open(a)), np.asarray(Image.open(b))), (
                f"{normalize}: {a.name} changed with batch size"
            )


def test_clamp_and_per_image_agree_on_shape_and_dtype(
    translator: StubTranslator, dataset: TranslationPairDataset, tmp_path: Path
) -> None:
    clamp, stretched = tmp_path / "clamp", tmp_path / "per_image"
    export_split(translator, dataset, clamp, normalize=Normalize.CLAMP, workers=0)
    export_split(translator, dataset, stretched, normalize=Normalize.PER_IMAGE, workers=0)

    name = next((clamp / "images").glob("*.png")).name
    a = np.asarray(Image.open(clamp / "images" / name))
    b = np.asarray(Image.open(stretched / "images" / name))

    assert a.shape == b.shape
    assert a.dtype == b.dtype == np.uint8


def test_export_does_not_leave_the_translator_in_eval(
    translator: StubTranslator, dataset: TranslationPairDataset, tmp_path: Path
) -> None:
    translator.train()
    export_split(translator, dataset, tmp_path, workers=0)
    assert translator.training
    translator.eval()


def test_translator_must_implement_the_protocol(
    dataset: TranslationPairDataset, tmp_path: Path
) -> None:
    class NotATranslator(nn.Module):
        pass

    with pytest.raises(TypeError, match="Translator protocol"):
        export_split(NotATranslator(), dataset, tmp_path, workers=0)


def test_export_translated_writes_both_splits_and_data_yaml(
    translator: StubTranslator,
    dataset: TranslationPairDataset,
    val_dataset: TranslationPairDataset,
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    config = _config(dataset_root)
    data_yaml_path = export_translated(translator, config, tmp_path)

    assert data_yaml_path == tmp_path / "data.yaml"
    for split, count in (("train", len(dataset)), ("val", len(val_dataset))):
        assert len(list((tmp_path / split / "images").glob("*.png"))) == count
        assert len(list((tmp_path / split / "labels").glob("*.txt"))) == count


def test_export_caps_train_and_leaves_val_whole(
    translator: StubTranslator,
    dataset: TranslationPairDataset,
    val_dataset: TranslationPairDataset,
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    """`max_train_images` reaches the export, and `val_loss_images` deliberately does not.

    Both halves have teeth. Capping train matters because the exported train split is what
    the evaluation detector fine-tunes on -- exporting the whole split while the translator
    trained on a cut would hand the run a larger detector budget than the experiment it is
    matched against. Leaving val whole matters more: every reported number (zero-shot mAP,
    FID, the detector's own val) is read off this split, so shrinking it here would quietly
    change the denominator of every result while looking like tidying.
    """
    config = Config.load(
        overrides={
            "data": {
                "manifest": str(dataset_root / "data.yaml"),
                "max_train_images": 3,
                "val_loss_images": 1,
            },
            "runtime": {"workers": 0},
            "detector": {"evaluation": {"batch": 4}},
        }
    )
    export_translated(translator, config, tmp_path)

    assert len(list((tmp_path / "train" / "images").glob("*.png"))) == 3
    assert len(list((tmp_path / "train" / "images").glob("*.png"))) < len(dataset)
    assert len(list((tmp_path / "val" / "images").glob("*.png"))) == len(val_dataset)


def test_export_misaligns_train_labels_and_leaves_val_verbatim(
    translator: StubTranslator, dataset_root: Path, tmp_path: Path
) -> None:
    """The evaluation detector fine-tunes on the boxes the misaligned translator trained on.

    Val is the scoring split: its labels stay the dataset's own (design.md §5, invariant 2).
    """
    config = Config.load(
        overrides={
            "data": {"manifest": str(dataset_root / "data.yaml"), "misalignment_px": 12.0},
            "runtime": {"workers": 0},
            "detector": {"evaluation": {"batch": 4}},
        }
    )
    export_translated(translator, config, tmp_path)

    height, width = np.asarray(Image.open(next((tmp_path / "train" / "images").iterdir()))).shape[
        :2
    ]
    misalignment = Misalignment(height, width, 12.0)
    moved = 0
    for exported in sorted((tmp_path / "train" / "labels").glob("*.txt")):
        source = dataset_root / "train" / "visible" / "labels" / exported.name
        lines, _ = misalign_label_lines(source, misalignment)
        assert exported.read_text() == ("\n".join(lines) + "\n" if lines else "")
        moved += exported.read_text() != source.read_text()
    assert moved > 0

    for exported in (tmp_path / "val" / "labels").glob("*.txt"):
        source = dataset_root / "val" / "visible" / "labels" / exported.name
        assert exported.read_text() == source.read_text()


def _derolled_config(data: Path, calibration: Path, level: float) -> Config:
    return Config.load(
        overrides={
            "data": {
                "manifest": str(data),
                "calibration": str(calibration),
                "misalignment_px": level,
            },
            "runtime": {"workers": 0},
            "detector": {"evaluation": {"batch": 4}},
        }
    )


@pytest.mark.parametrize("level", [0.0, 12.0])
def test_a_de_rolled_export_writes_thermal_side_labels(
    translator: StubTranslator, derolled_tree: tuple[Path, Path], tmp_path: Path, level: float
) -> None:
    """ADR-02: train boxes are the de-rolled thermal ones moved by the level; val is scored on
    the de-rolled thermal ones verbatim, never on the rolled visible-side boxes."""
    data, calibration = derolled_tree
    root = data.parent
    export_translated(translator, _derolled_config(data, calibration, level), tmp_path / "out")

    height, width = np.asarray(Image.open(next(root.rglob("*.jpg")))).shape[:2]
    misalignment = Misalignment(height, width, level)
    for exported in sorted((tmp_path / "out" / "train" / "labels").glob("*.txt")):
        thermal = root / "train" / "infrared" / "labels" / exported.name
        if level == 0:
            assert exported.read_bytes() == thermal.read_bytes()
        else:
            lines, _ = misalign_label_lines(thermal, misalignment)
            assert exported.read_text() == ("\n".join(lines) + "\n" if lines else "")

    val = sorted((tmp_path / "out" / "val" / "labels").glob("*.txt"))
    assert val
    differs_from_visible = 0
    for exported in val:
        thermal = root / "val" / "infrared" / "labels" / exported.name
        visible = root / "val" / "visible" / "labels" / exported.name
        assert exported.read_bytes() == thermal.read_bytes()
        differs_from_visible += exported.read_bytes() != visible.read_bytes()
    assert differs_from_visible > 0, "the fixture would not catch visible-side val labels"


def test_a_de_rolled_export_refuses_a_tree_the_constant_did_not_de_roll(
    translator: StubTranslator, derolled_tree: tuple[Path, Path], tmp_path: Path
) -> None:
    data, calibration = derolled_tree
    mirror_labels_onto_infrared(DatasetManifest.load(data), force=True)

    with pytest.raises(CalibrationError, match="a plain mirror"):
        export_translated(translator, _derolled_config(data, calibration, 0.0), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_generated_data_yaml_is_loadable_by_ultralytics(
    translator: StubTranslator, dataset_root: Path, tmp_path: Path
) -> None:
    from ultralytics.data.utils import check_det_dataset

    data_yaml_path = export_translated(translator, _config(dataset_root), tmp_path)

    contents = yaml.safe_load(data_yaml_path.read_text())
    assert contents["nc"] == 4
    assert contents["train"] == "train/images"

    checked = check_det_dataset(str(data_yaml_path))
    assert checked["nc"] == 4
    assert Path(checked["train"]).exists()


# --- paired export: the cyclegan-cascade's stage-1 input tree (plan Q8) ---------------------


def _paired_export(translator: StubTranslator, source: Path, out: Path) -> Path:
    return export_translated(translator, _config(source), out, paired=True).parent


@pytest.mark.parametrize("split", ["train", "val"])
def test_a_paired_export_loads_as_pairs_with_the_translation_as_infrared(
    translator: StubTranslator, dataset_root: Path, tmp_path: Path, split: str
) -> None:
    tree = _paired_export(translator, dataset_root, tmp_path / "paired")
    source = TranslationPairDataset(dataset_root / split / "visible" / "images")
    export_split(translator, source, tmp_path / "plain", workers=0)

    manifest = DatasetManifest.load(tree / "data.yaml")
    images = manifest.train_images if split == "train" else manifest.val_images
    pairs = TranslationPairDataset(images, pairing=manifest.pairing, num_classes=manifest.nc)

    assert [Path(p).stem for p in pairs.visible_paths] == [
        Path(p).stem for p in source.visible_paths
    ]
    for visible in pairs.visible_paths:
        infrared = manifest.pairing.infrared_path(visible)
        translated = tmp_path / "plain" / "images" / visible.name
        assert np.array_equal(np.asarray(Image.open(infrared)), np.asarray(Image.open(translated)))


def test_a_paired_export_keeps_the_visible_pixels_and_the_label_bytes(
    translator: StubTranslator, dataset_root: Path, tmp_path: Path
) -> None:
    tree = _paired_export(translator, dataset_root, tmp_path / "paired")

    compared = 0
    for split in ("train", "val"):
        for source in sorted((dataset_root / split / "visible" / "images").glob("*.jpg")):
            copied = tree / split / "visible" / "images" / f"{source.stem}.png"
            assert np.array_equal(
                np.asarray(Image.open(copied)), np.asarray(Image.open(source).convert("RGB"))
            )
            label = f"{source.stem}.txt"
            assert (tree / split / "visible" / "labels" / label).read_bytes() == (
                dataset_root / split / "visible" / "labels" / label
            ).read_bytes()
            compared += 1
    assert compared == len(list(dataset_root.rglob("visible/images/*.jpg")))


def test_a_paired_tree_still_loads_after_a_move(
    translator: StubTranslator, dataset_root: Path, tmp_path: Path
) -> None:
    """Both readers, not only t2o's: ultralytics honours `path:` literally (CLAUDE.md)."""
    from ultralytics.data.utils import check_det_dataset

    tree = _paired_export(translator, dataset_root, tmp_path / "built")
    moved = Path(shutil.move(tree, tmp_path / "moved"))

    assert "path" not in yaml.safe_load((moved / "data.yaml").read_text())
    manifest = DatasetManifest.load(moved / "data.yaml")
    assert manifest.root == moved.resolve()
    TranslationPairDataset(manifest.train_images, pairing=manifest.pairing)
    checked = check_det_dataset(str(moved / "data.yaml"))
    assert Path(checked["train"]) == (moved / "train" / "visible" / "images").resolve()


def test_a_paired_export_carries_no_labels_cache(
    translator: StubTranslator, dataset_root: Path, tmp_path: Path
) -> None:
    """ultralytics keys `labels.cache` on sizes and paths, so a stale one would be trusted."""
    source = tmp_path / "source"
    shutil.copytree(dataset_root, source)
    (source / "data.yaml").write_text(
        (dataset_root / "data.yaml").read_text().replace(str(dataset_root), str(source))
    )
    for split in ("train", "val"):
        (source / split / "visible" / "labels.cache").write_bytes(b"stale")

    tree = _paired_export(translator, source, tmp_path / "paired")

    assert not list(tree.rglob("*.cache"))


def test_a_paired_export_refuses_moved_labels(
    translator: StubTranslator, dataset_root: Path, tmp_path: Path
) -> None:
    config = Config.load(
        overrides={
            "data": {"manifest": str(dataset_root / "data.yaml"), "misalignment_px": 12.0},
            "runtime": {"workers": 0},
        }
    )

    with pytest.raises(ExportError, match="misalignment_px"):
        export_translated(translator, config, tmp_path / "out", paired=True)
    assert not (tmp_path / "out").exists()
