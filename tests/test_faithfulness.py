"""Faithfulness (C2): false-object rate, missed-object rate, detection-consistency.

Built on synthetic detections with known answers throughout -- box coordinates are chosen
so the correct match/no-match outcome is obvious by construction, per house style
(conftest.py's "build the pathology explicitly rather than hoping real data contains one").
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml

from t2o.config import Config, ConfigError
from t2o.metrics.faithfulness import (
    Detections,
    FaithfulnessEvaluator,
    _cxcywh_to_xyxy,
    _greedy_match,
    detections_from_labels,
    detections_from_result,
)

# --------------------------------------------------------------------------- conversions


def test_cxcywh_to_xyxy() -> None:
    bboxes = torch.tensor([[0.5, 0.5, 0.4, 0.2]])
    xyxy = _cxcywh_to_xyxy(bboxes)
    assert torch.allclose(xyxy, torch.tensor([[0.3, 0.4, 0.7, 0.6]]))


def test_cxcywh_to_xyxy_empty() -> None:
    assert _cxcywh_to_xyxy(torch.zeros(0, 4)).shape == (0, 4)


def test_detections_from_labels_matches_dataset_convention() -> None:
    cls = torch.tensor([[1.0], [0.0]])
    bboxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.4, 0.4]])
    detections = detections_from_labels(cls, bboxes)

    assert detections.cls.tolist() == [1, 0]
    assert detections.conf is None
    assert torch.allclose(detections.bboxes[0], torch.tensor([0.4, 0.4, 0.6, 0.6]))


def test_detections_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="boxes but"):
        Detections(cls=torch.zeros(2, dtype=torch.long), bboxes=torch.zeros(1, 4))


def test_detections_from_result_applies_confidence_threshold() -> None:
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            conf=torch.tensor([0.9, 0.1]),
            cls=torch.tensor([0.0, 1.0]),
            xyxyn=torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]),
        )
    )
    detections = detections_from_result(result, conf_threshold=0.25)
    assert detections.cls.tolist() == [0]
    assert detections.conf is not None and detections.conf.tolist() == pytest.approx([0.9])


# --------------------------------------------------------------------------- matching


def _det(cls: list[int], boxes: list[list[float]], conf: list[float] | None = None) -> Detections:
    conf_t = torch.tensor(conf) if conf is not None else None
    return Detections(cls=torch.tensor(cls), bboxes=torch.tensor(boxes), conf=conf_t)


def test_identical_box_and_class_matches() -> None:
    box = [[0.1, 0.1, 0.3, 0.3]]
    pred = _det([0], box, conf=[0.9])
    ref = _det([0], box)
    pred_matched, ref_matched = _greedy_match(pred, ref, iou_threshold=0.5)
    assert pred_matched.tolist() == [True]
    assert ref_matched.tolist() == [True]


def test_disjoint_boxes_do_not_match() -> None:
    pred = _det([0], [[0.0, 0.0, 0.1, 0.1]], conf=[0.9])
    ref = _det([0], [[0.8, 0.8, 0.9, 0.9]])
    pred_matched, ref_matched = _greedy_match(pred, ref, iou_threshold=0.5)
    assert pred_matched.tolist() == [False]
    assert ref_matched.tolist() == [False]


def test_same_location_different_class_does_not_match() -> None:
    box = [[0.1, 0.1, 0.3, 0.3]]
    pred = _det([0], box, conf=[0.9])
    ref = _det([1], box)
    pred_matched, ref_matched = _greedy_match(pred, ref, iou_threshold=0.5)
    assert not pred_matched.any()
    assert not ref_matched.any()


def test_higher_confidence_prediction_claims_the_match_first() -> None:
    box = [[0.1, 0.1, 0.3, 0.3]]
    ref = _det([0], box)
    pred = _det([0, 0], box + box, conf=[0.5, 0.9])  # index 1 has higher confidence

    pred_matched, ref_matched = _greedy_match(pred, ref, iou_threshold=0.5)
    assert pred_matched.tolist() == [False, True]
    assert ref_matched.tolist() == [True]


def test_empty_pred_or_ref_matches_nothing() -> None:
    box = [[0.1, 0.1, 0.3, 0.3]]
    empty = Detections(cls=torch.zeros(0, dtype=torch.long), bboxes=torch.zeros(0, 4))
    full = _det([0], box, conf=[0.9])

    pred_matched, ref_matched = _greedy_match(empty, full, iou_threshold=0.5)
    assert pred_matched.tolist() == [] and ref_matched.tolist() == [False]

    pred_matched, ref_matched = _greedy_match(full, empty, iou_threshold=0.5)
    assert pred_matched.tolist() == [False] and ref_matched.tolist() == []


# --------------------------------------------------------------------------- evaluator

_BOX_A = [0.1, 0.1, 0.3, 0.3]
_BOX_B = [0.6, 0.6, 0.8, 0.8]


def test_perfect_translation_has_no_failures() -> None:
    gt = _det([0], [_BOX_A])
    visible = _det([0], [_BOX_A], conf=[1.0])
    translated = _det([0], [_BOX_A], conf=[0.9])

    evaluator = FaithfulnessEvaluator()
    evaluator.update(translated, visible, gt)
    metrics = evaluator.compute()

    assert metrics.false_object_rate == pytest.approx(0.0)
    assert metrics.missed_object_rate == pytest.approx(0.0)
    assert metrics.detection_consistency == pytest.approx(1.0)


def test_hallucinated_detection_raises_false_object_rate() -> None:
    gt = _det([0], [_BOX_A])
    visible = _det([0], [_BOX_A], conf=[1.0])
    # Translated invents a second, unlabelled component.
    translated = _det([0, 0], [_BOX_A, _BOX_B], conf=[0.9, 0.8])

    evaluator = FaithfulnessEvaluator()
    evaluator.update(translated, visible, gt)
    metrics = evaluator.compute()

    assert metrics.false_object_rate == pytest.approx(0.5)
    assert metrics.missed_object_rate == pytest.approx(0.0)


def test_erased_component_raises_missed_object_rate() -> None:
    gt = _det([0, 0], [_BOX_A, _BOX_B])
    visible = _det([0, 0], [_BOX_A, _BOX_B], conf=[1.0, 1.0])
    # Translation only reproduces one of the two real components.
    translated = _det([0], [_BOX_A], conf=[0.9])

    evaluator = FaithfulnessEvaluator()
    evaluator.update(translated, visible, gt)
    metrics = evaluator.compute()

    assert metrics.missed_object_rate == pytest.approx(0.5)
    assert metrics.false_object_rate == pytest.approx(0.0)


def test_inconsistent_with_the_real_photo_lowers_consistency() -> None:
    gt = _det([0, 0], [_BOX_A, _BOX_B])
    visible = _det([0, 0], [_BOX_A, _BOX_B], conf=[1.0, 1.0])
    translated = _det([0], [_BOX_A], conf=[0.9])

    evaluator = FaithfulnessEvaluator()
    evaluator.update(translated, visible, gt)
    assert evaluator.compute().detection_consistency == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("gt_boxes", "visible_boxes", "translated_boxes"),
    [([], [], []), ([], [_BOX_A], []), ([_BOX_A], [], [])],
)
def test_empty_denominators_default_to_the_no_failure_value(
    gt_boxes: list[list[float]],
    visible_boxes: list[list[float]],
    translated_boxes: list[list[float]],
) -> None:
    def _maybe_det(boxes: list[list[float]], with_conf: bool) -> Detections:
        if not boxes:
            cls = torch.zeros(0, dtype=torch.long)
            bboxes = torch.zeros(0, 4)
            conf = torch.zeros(0) if with_conf else None
            return Detections(cls=cls, bboxes=bboxes, conf=conf)
        return _det([0] * len(boxes), boxes, conf=[0.9] * len(boxes) if with_conf else None)

    evaluator = FaithfulnessEvaluator()
    evaluator.update(
        _maybe_det(translated_boxes, with_conf=True),
        _maybe_det(visible_boxes, with_conf=True),
        _maybe_det(gt_boxes, with_conf=False),
    )
    metrics = evaluator.compute()

    if not gt_boxes:
        assert metrics.missed_object_rate == pytest.approx(0.0)
    if not visible_boxes:
        assert metrics.detection_consistency == pytest.approx(1.0)
    if not translated_boxes:
        assert metrics.false_object_rate == pytest.approx(0.0)


def test_reset_clears_accumulated_state() -> None:
    gt = _det([0], [_BOX_A])
    visible = _det([0], [_BOX_A], conf=[1.0])
    translated = _det([0, 0], [_BOX_A, _BOX_B], conf=[0.9, 0.8])

    evaluator = FaithfulnessEvaluator()
    evaluator.update(translated, visible, gt)
    evaluator.reset()

    perfect = _det([0], [_BOX_A], conf=[0.9])
    evaluator.update(perfect, visible, gt)
    assert evaluator.compute().false_object_rate == pytest.approx(0.0)


# --------------------------------------------------------------------------- config


def test_metrics_config_faithfulness_defaults() -> None:
    config = Config.load()
    assert config.metrics.iou_threshold == pytest.approx(0.5)
    assert config.metrics.conf_threshold == pytest.approx(0.25)


def test_metrics_config_rejects_out_of_range_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"metrics": {"iou_threshold": 1.5}}))
    with pytest.raises(ConfigError):
        Config.load(path)


@pytest.mark.parametrize("override", [{"iou_threshold": 0.7}, {"conf_threshold": 0.1}])
def test_metrics_config_thresholds_are_part_of_the_hash(override: dict[str, Any]) -> None:
    baseline = Config.load().config_hash()
    assert Config.load(overrides={"metrics": override}).config_hash() != baseline
