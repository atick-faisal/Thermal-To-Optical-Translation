"""Detection-task metrics: mAP@50, mAP@50:95, per-class AP.

`_extract_metrics`'s tests are ported verbatim from
``../Clean-SeAFusion/tests/test_detector_stage.py`` -- they moved here, not changed, when
the function itself moved to become the canonical implementation shared with
``t2o.engine.detector_stage.train_detector`` (see that module's docstring). Per-class
extraction and the end-to-end `evaluate_detector` path are new in M0.5 step 2.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from t2o.metrics.task import _extract_metrics, _extract_per_class_ap, evaluate_detector

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
