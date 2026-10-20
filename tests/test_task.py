"""Detection-task metrics: mAP@50, mAP@50:95, per-class AP.

`_extract_metrics`'s tests are ported verbatim from
``../Clean-SeAFusion/tests/test_detector_stage.py`` -- they moved here, not changed, when
the function itself moved to become the canonical implementation shared with
``t2o.engine.detector_stage.train_detector`` (see that module's docstring). Per-class
extraction and the end-to-end `evaluate_detector` path are new in M0.5 step 2.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from conftest import NAMES, VAL_STEMS
from t2o.metrics.task import (
    DuplicateImageError,
    ImageStats,
    _extract_metrics,
    _extract_per_class_ap,
    collect_image_stats,
    evaluate_detector,
    map50_from_stats,
)

# --------------------------------------------------------------------------- P/R/mAP


def test_metrics_from_results_dict() -> None:
    results = SimpleNamespace(
        results_dict={
            "metrics/precision(B)": 0.7,
            "metrics/recall(B)": 0.6,
            "metrics/mAP50(B)": 0.5,
            "metrics/mAP50-95(B)": 0.4,
        }
    )
    assert _extract_metrics(results) == {
        "precision": 0.7,
        "recall": 0.6,
        "map50": 0.5,
        "map50_95": 0.4,
    }


def test_metrics_fall_back_to_the_box_object() -> None:
    results = SimpleNamespace(
        results_dict={}, box=SimpleNamespace(mp=0.3, mr=0.2, map50=0.1, map=0.05)
    )
    assert _extract_metrics(results)["precision"] == 0.3
    assert _extract_metrics(results)["map50_95"] == 0.05


def test_metrics_degrade_to_zero_rather_than_raising() -> None:
    # A shape change in either surface must not take down a run mid-loop.
    assert _extract_metrics(SimpleNamespace()) == {
        "precision": 0.0,
        "recall": 0.0,
        "map50": 0.0,
        "map50_95": 0.0,
    }


# --------------------------------------------------------------------------- per-class AP


def _box(class_index: list[int], ap50: list[float], ap: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        ap_class_index=np.array(class_index), ap50=np.array(ap50), ap=np.array(ap)
    )


def test_per_class_ap_keyed_by_name() -> None:
    results = SimpleNamespace(
        box=_box([0, 2], [0.9, 0.4], [0.8, 0.3]),
        names={0: "Fuse", 1: "Pole", 2: "Switch"},
    )
    ap50, ap = _extract_per_class_ap(results)
    assert ap50 == {"Fuse": pytest.approx(0.9), "Switch": pytest.approx(0.4)}
    assert ap == {"Fuse": pytest.approx(0.8), "Switch": pytest.approx(0.3)}


def test_per_class_ap_omits_classes_with_no_ground_truth() -> None:
    # "Pole" (class 1) never appears in ap_class_index -- it must not show up as 0.0,
    # which would conflate "missed every instance" with "there was nothing to detect".
    results = SimpleNamespace(box=_box([0], [0.9], [0.8]), names={0: "Fuse", 1: "Pole"})
    ap50, _ = _extract_per_class_ap(results)
    assert "Pole" not in ap50


def test_per_class_ap_degrades_to_empty_rather_than_raising() -> None:
    assert _extract_per_class_ap(SimpleNamespace()) == ({}, {})
    assert _extract_per_class_ap(SimpleNamespace(box=_box([], [], []))) == ({}, {})


# --------------------------------------------------------------------------- end-to-end


@pytest.mark.slow
def test_evaluate_detector_end_to_end(data_yaml: Path, detector_weights: Path) -> None:
    metrics = evaluate_detector(
        weights=detector_weights, data_yaml=data_yaml, imgsz=640, batch=4, device="cpu"
    )

    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.map50 <= 1.0
    assert 0.0 <= metrics.map50_95 <= metrics.map50 + 1e-6
    # A randomly-initialised detector still runs the full val pipeline; whichever classes
    # had ground truth in the val split get an entry, nothing more.
    assert set(metrics.per_class_ap50) <= {"Fuse", "Pole", "Switch", "Transformer"}


# --------------------------------------------------------------------------- per-image stats

# Why the AP math and the capture are tested apart: the random `detector_weights` cannot score
# a true positive on any labels. Its activations shrink ~4x per layer to ~1e-9 at the head, so
# it outputs only biases -- one confidence for every anchor, every box past the image edge. Its
# mAP@50 is 0.0, and an equality against it holds for any capture. So the math is checked on
# hand-built records with real true positives, and the capture on what each image carries.

# The smallest size RT-DETR runs at; `test_rtdetr_judge.py` says why.
RTDETR_IMGSZ = 128


def _record(
    stem: str, tp50: list[bool], conf: list[float], pred_cls: list[int], target_cls: list[int]
) -> ImageStats:
    """A hand-built record whose match at IoU 0.50 holds at every stricter threshold too."""
    return ImageStats(
        stem=stem,
        tp=np.repeat(np.array(tp50, dtype=bool).reshape(-1, 1), 10, axis=1),
        conf=np.array(conf, dtype=float),
        pred_cls=np.array(pred_cls, dtype=float),
        target_cls=np.array(target_cls, dtype=float),
    )


def test_map50_from_stats_matches_ultralytics_own_det_metrics() -> None:
    from ultralytics.utils.metrics import DetMetrics

    records = [
        _record("a", [True, False, False], [0.9, 0.8, 0.3], [0, 1, 0], [0, 1]),
        _record("b", [True, False], [0.7, 0.2], [1, 2], [1]),
        _record("c", [False], [0.6], [0], []),  # a background image with a false positive
    ]
    metrics = DetMetrics(names=dict(enumerate(NAMES)))
    for record in records:
        metrics.update_stats(
            {
                "tp": record.tp,
                "conf": record.conf,
                "pred_cls": record.pred_cls,
                "target_cls": record.target_cls,
                "target_img": np.unique(record.target_cls),
                "im_name": f"{record.stem}.png",
            }
        )
    metrics.process()

    assert metrics.box.map50 > 0.0
    assert map50_from_stats(records) == pytest.approx(metrics.box.map50)


def _copy_dataset(dataset_root: Path, dest: Path) -> Path:
    """Copy the synthetic dataset without its `labels.cache`, pointing `data.yaml` at the copy.

    ultralytics honours `path:` literally, so a copy that kept the old one would score the
    original images.
    """
    shutil.copytree(dataset_root, dest, ignore=shutil.ignore_patterns("*.cache"))
    data_yaml = dest / "data.yaml"
    data_yaml.write_text(
        data_yaml.read_text().replace(f"path: {dataset_root}\n", f"path: {dest}\n")
    )
    return data_yaml


def test_image_stats_carry_each_images_own_labels(
    data_yaml: Path, dataset_root: Path, detector_weights: Path
) -> None:
    records = collect_image_stats(detector_weights, data_yaml, imgsz=640, batch=2, device="cpu")
    metrics = evaluate_detector(detector_weights, data_yaml, imgsz=640, batch=2, device="cpu")

    assert set(records) == set(VAL_STEMS)
    labels = dataset_root / "val" / "visible" / "labels"
    for stem, record in records.items():
        # The fixture's val images carry different labels, so a record filed under the wrong
        # stem shows up here -- where a mAP over all records, concatenated, cannot see it.
        lines = (labels / f"{stem}.txt").read_text().splitlines()
        classes = sorted(float(line.split()[0]) for line in lines)
        assert sorted(record.target_cls.tolist()) == classes
        assert len(record.tp) == len(record.conf) == len(record.pred_cls)
    assert map50_from_stats(list(records.values())) == pytest.approx(metrics.map50)


def test_image_stats_refuse_two_images_with_one_stem(
    dataset_root: Path, detector_weights: Path, tmp_path: Path
) -> None:
    data_yaml = _copy_dataset(dataset_root, tmp_path / "data")
    first = sorted((tmp_path / "data" / "val" / "visible" / "images").iterdir())[0]
    Image.open(first).save(first.with_suffix(".png"))

    with pytest.raises(DuplicateImageError, match=first.stem):
        collect_image_stats(detector_weights, data_yaml, imgsz=640, batch=2, device="cpu")


@pytest.mark.slow
def test_image_stats_capture_under_the_rtdetr_validator(
    rtdetr_weights: Path, data_yaml: Path
) -> None:
    records = collect_image_stats(
        rtdetr_weights, data_yaml, imgsz=RTDETR_IMGSZ, batch=2, device="cpu"
    )
    metrics = evaluate_detector(
        rtdetr_weights, data_yaml, imgsz=RTDETR_IMGSZ, batch=2, device="cpu"
    )

    assert set(records) == set(VAL_STEMS)
    assert map50_from_stats(list(records.values())) == pytest.approx(metrics.map50)
