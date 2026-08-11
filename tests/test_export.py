"""`engine/export.py` -- translated images + mirrored labels -> detector-ready dataset.

Built on the synthetic `dataset_root`/`data_yaml` fixtures (`tests/conftest.py`): 5 train /
3 val images, disjoint stems. Expected counts/sizes are read from the fixtures themselves
rather than importing conftest's constants -- a `tests` package shipped by a dependency
shadows the local one on the pyright module search path, so a `from tests.conftest import
...` resolves to the wrong module under type checking.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from t2o.config import Config
from t2o.data.dataset import TranslationPairDataset
from t2o.engine.export import export_split, export_translated
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
            "train": {"workers": 0},
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
