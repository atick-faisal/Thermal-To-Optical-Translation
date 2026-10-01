"""Detection-task metrics: mAP@50, mAP@50:95, per-class AP.

Nothing here is a from-scratch port: mAP is ultralytics' own `DetMetrics`, reached through
its own `model.val()` -- the same NMS/IoU-matching/AP machinery every method's number goes
through, which is what makes the comparison table defensible (PLAN.md invariant 1).

`_extract_metrics` is not merely similar to
`t2o.engine.detector_stage.train_detector`'s extraction -- it is the same function,
imported from here. Verified against ultralytics 8.4.117 source: `Model.train()` returns
`self.trainer.validator.metrics` and `Model.val()` returns `validator.metrics` directly,
both the same `DetMetrics` instance type, so one extraction function genuinely serves both
the training-time and standalone-validation call sites rather than two call sites happening
to agree by convention.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TaskMetrics:
    """Detection performance for one detector checkpoint on one dataset split."""

    precision: float
    recall: float
    map50: float
    map50_95: float
    # Per-class AP, keyed by class name. A class absent from these dicts had zero
    # ground-truth instances in the evaluated split -- reporting it as 0.0 would conflate
    # "the detector missed every instance" with "there was nothing to detect".
    per_class_ap50: dict[str, float]
    per_class_ap50_95: dict[str, float]


# ultralytics' own naming, via DetMetrics.results_dict.
_RESULT_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}
# The equivalent attributes on the Metric object, used when results_dict is unavailable.
_BOX_ATTRS = {"precision": "mp", "recall": "mr", "map50": "map50", "map50_95": "map"}


class PrimaryClassError(ValueError):
    """Raised when none of the requested primary classes was scored in a split."""


def primary_mean(per_class_ap: Mapping[str, float], primary: Sequence[str]) -> float:
    """Mean AP over `primary`, skipping classes with no ground truth in the split.

    `_extract_per_class_ap` omits a zero-instance class entirely rather than reporting 0.0,
    because "the detector missed every instance" and "there was nothing to detect" are
    different facts. Intersecting keeps this a mean over what was actually scored instead of
    one diluted by an absence -- and it divides by the number of classes *scored*, not by
    `len(primary)`, so an absent class neither contributes 0.0 nor shrinks the denominator.

    This exists because a 4-class mean is the wrong headline on FLIR: `dog` has 13 instances
    in the val split against `car`'s 4,124, yet carries 25% of a 4-class mean, and an AP50
    swing of 0.4 on 13 instances -- entirely ordinary -- moves that mean by 0.1 (TASKS.md
    M3 E9). Lives here rather than in a script because both the gate table and the campaign
    aggregator read it, and `scripts/` imports `t2o`, never the other way round.
    """
    scored = [per_class_ap[name] for name in primary if name in per_class_ap]
    if not scored:
        raise PrimaryClassError(
            f"none of the primary classes {list(primary)} were scored -- the split has no "
            f"instances of any of them (it scored {sorted(per_class_ap)})"
        )
    return sum(scored) / len(scored)


def _extract_metrics(results: Any) -> dict[str, float]:
    """Pull P/R/mAP out of ultralytics' results, tolerating its shape changing."""
    results_dict = getattr(results, "results_dict", None) or {}
    if results_dict:
        return {name: float(results_dict.get(key, 0.0)) for name, key in _RESULT_KEYS.items()}

    box = getattr(results, "box", None)
    return {name: float(getattr(box, attr, 0.0) or 0.0) for name, attr in _BOX_ATTRS.items()}


def _extract_per_class_ap(results: Any) -> tuple[dict[str, float], dict[str, float]]:
    """Per-class AP50 and AP50:95, keyed by class name.

    ultralytics only populates `box.ap_class_index` for classes with at least one
    ground-truth instance in the evaluated split; classes with none are simply absent from
    the returned dicts rather than backfilled with a misleading 0.0.
    """
    box = getattr(results, "box", None)
    class_index = getattr(box, "ap_class_index", None)
    if box is None or class_index is None or len(class_index) == 0:
        return {}, {}

    names: dict[int, str] = getattr(results, "names", None) or {}
    ap50, ap = list(box.ap50), list(box.ap)
    ap50_by_class: dict[str, float] = {}
    ap_by_class: dict[str, float] = {}
    for i, class_id in enumerate(class_index):
        name = names.get(int(class_id), str(int(class_id)))
        ap50_by_class[name] = float(ap50[i])
        ap_by_class[name] = float(ap[i])
    return ap50_by_class, ap_by_class


def evaluate_detector(
    weights: Path | str,
    data_yaml: Path | str,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
) -> TaskMetrics:
    """Validate a detector checkpoint against a dataset split via ultralytics' `model.val()`.

    This is the E1/E2 entry point: point `weights` at the in-loop detector's own checkpoint
    and `data_yaml` at either the raw-thermal split or a translated-image export to compare
    them through the identical evaluation path. Distinct from
    `t2o.engine.detector_stage.train_detector`, which trains the *evaluation* detector on
    exported translations -- this function only ever validates, never trains, and so never
    touches a translator's autograd graph.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False,
        plots=False,
    )

    scores = _extract_metrics(results)
    per_class_ap50, per_class_ap50_95 = _extract_per_class_ap(results)
    logger.info(
        "eval %s on %s: P %.4f  R %.4f  mAP50 %.4f  mAP50-95 %.4f",
        Path(weights).name,
        Path(data_yaml).name,
        scores["precision"],
        scores["recall"],
        scores["map50"],
        scores["map50_95"],
    )
    return TaskMetrics(
        precision=scores["precision"],
        recall=scores["recall"],
        map50=scores["map50"],
        map50_95=scores["map50_95"],
        per_class_ap50=per_class_ap50,
        per_class_ap50_95=per_class_ap50_95,
    )
