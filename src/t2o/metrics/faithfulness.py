"""Faithfulness (C2): false-object rate, missed-object rate, detection-consistency.

The safety-critical co-contribution (PLAN.md §1, §12): can the translator's failures be
bounded and reported, not just its average quality? All three rates are built on the same
greedy, confidence-ordered, single-IoU-threshold matcher -- the same principle
COCOeval/ultralytics use per class per image, minus the multi-threshold AP integration
that follows it there (which belongs to `metrics/task.py`, a different question).
Predictions are visited in descending confidence order; each is matched to the
highest-IoU unclaimed reference box of the same class at or above `iou_threshold`, or
counted unmatched otherwise. This is what makes the three rates comparable to each other
and across translators (PLAN.md invariant 1).

**Hallucination Index deferred** (TASKS.md M0.5). The MICCAI 2024 metric it's adapted from
(arXiv:2407.12780) is a Hellinger distance between the *distribution* of a generative
model's reconstructions -- mean/variance/noise-power-spectrum across many repeated samples
per input -- and a zero-hallucination reference. It assumes a stochastic model sampled N
times per input; none of our translators (pix2pix, pix2pix-turbo one-step, LBBDM without
repeated sampling) are, so the literal metric does not apply yet.

**Detection-consistency's exact formula is a judgement call**, not a normative citation:
RESEARCH_FINDINGS.md §9 names it without a formula. Defined here as the fraction of
whatever the detector finds on the real visible image that it also finds (same class, same
place) on the translated image. This is deliberately a different failure mode from
missed-object rate: that compares against hand-labelled ground truth (annotation-quality
dependent); this compares the translator against the detector's own behaviour on the
untranslated photo, which needs no labels at all and so stays well-defined at any
`annotation_fraction` (PLAN.md §9, E8).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torchvision.ops import box_iou

from t2o.data.dataset import IMAGE_SUFFIXES
from t2o.data.labels import load_yolo_labels
from t2o.data.pairing import Pairing

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Detections:
    """One image's boxes: class ids, normalised xyxy in ``[0, 1]``, optional confidence.

    ``conf=None`` marks ground truth: every entry counts as certain, and the confidence
    threshold used to filter predictions does not apply to it.
    """

    cls: Tensor  # (N,) int64
    bboxes: Tensor  # (N, 4) normalised xyxy
    conf: Tensor | None = None  # (N,) float in [0, 1]; None for ground truth

    def __post_init__(self) -> None:
        if self.bboxes.shape[0] != self.cls.shape[0]:
            raise ValueError(f"{self.bboxes.shape[0]} boxes but {self.cls.shape[0]} class ids")
        if self.conf is not None and self.conf.shape[0] != self.cls.shape[0]:
            raise ValueError(f"{self.conf.shape[0]} confidences but {self.cls.shape[0]} class ids")


def _cxcywh_to_xyxy(bboxes: Tensor) -> Tensor:
    if bboxes.numel() == 0:
        return bboxes.reshape(0, 4)
    cx, cy, w, h = bboxes.unbind(dim=1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


def detections_from_labels(cls: Tensor, bboxes: Tensor) -> Detections:
    """Ground truth from ``TranslationSample``'s convention: ``(M, 1)`` cls, ``(M, 4)``
    normalised cxcywh (``t2o.data.dataset``).
    """
    return Detections(cls=cls.reshape(-1).long(), bboxes=_cxcywh_to_xyxy(bboxes))


def detections_from_result(result: Any, conf_threshold: float) -> Detections:
    """Predictions from an ultralytics ``Results`` object, already NMS'd.

    Confidence filtering happens here rather than inside the matcher, so every caller
    applies `conf_threshold` identically.
    """
    boxes = result.boxes
    keep = boxes.conf >= conf_threshold
    return Detections(cls=boxes.cls[keep].long(), bboxes=boxes.xyxyn[keep], conf=boxes.conf[keep])


def _greedy_match(pred: Detections, ref: Detections, iou_threshold: float) -> tuple[Tensor, Tensor]:
    """Confidence-ordered greedy matching, one IoU threshold, same-class only.

    Returns ``(pred_matched, ref_matched)`` boolean masks, aligned to each input's order.
    """
    n_pred, n_ref = pred.cls.shape[0], ref.cls.shape[0]
    pred_matched = torch.zeros(n_pred, dtype=torch.bool)
    ref_matched = torch.zeros(n_ref, dtype=torch.bool)
    if n_pred == 0 or n_ref == 0:
        return pred_matched, ref_matched

    order = (
        torch.argsort(pred.conf, descending=True) if pred.conf is not None else torch.arange(n_pred)
    )
    iou = box_iou(pred.bboxes, ref.bboxes)

    for i in order.tolist():
        candidates = (ref.cls == pred.cls[i]) & ~ref_matched & (iou[i] >= iou_threshold)
        if not bool(candidates.any()):
            continue
        j = int(torch.where(candidates, iou[i], torch.full_like(iou[i], -1.0)).argmax())
        pred_matched[i] = True
        ref_matched[j] = True

    return pred_matched, ref_matched


@dataclass(frozen=True, slots=True)
class FaithfulnessMetrics:
    false_object_rate: float
    missed_object_rate: float
    detection_consistency: float


class FaithfulnessEvaluator:
    """Accumulates false-object / missed-object / detection-consistency counts.

    ``update`` is called once per image with three :class:`Detections`: the detector's
    predictions on the translated image, its predictions on the paired real-visible image,
    and the hand-labelled ground truth. All three comparisons go through the same
    `_greedy_match` at the same `iou_threshold`, which is what makes the three rates
    comparable to each other and to every other translator's numbers run through this
    class (PLAN.md invariant 1).
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self._iou_threshold = iou_threshold
        self.reset()

    def update(self, translated: Detections, visible: Detections, gt: Detections) -> None:
        translated_matched_gt, gt_matched = _greedy_match(translated, gt, self._iou_threshold)
        self._false_objects += int((~translated_matched_gt).sum())
        self._total_translated += translated.cls.shape[0]
        self._missed_objects += int((~gt_matched).sum())
        self._total_gt += gt.cls.shape[0]

        _, visible_matched = _greedy_match(translated, visible, self._iou_threshold)
        self._consistent += int(visible_matched.sum())
        self._total_visible += visible.cls.shape[0]

    def compute(self) -> FaithfulnessMetrics:
        return FaithfulnessMetrics(
            false_object_rate=(
                self._false_objects / self._total_translated if self._total_translated else 0.0
            ),
            # Nothing to miss is not a failure -- defaults to the best-case value like
            # false_object_rate's empty case above, not to an undefined 0/0.
            missed_object_rate=(self._missed_objects / self._total_gt if self._total_gt else 0.0),
            # Nothing on the real photo to be inconsistent with -- vacuously consistent.
            detection_consistency=(
                self._consistent / self._total_visible if self._total_visible else 1.0
            ),
        )

    def reset(self) -> None:
        self._false_objects = 0
        self._total_translated = 0
        self._missed_objects = 0
        self._total_gt = 0
        self._consistent = 0
        self._total_visible = 0


def evaluate_faithfulness(
    translated_images: Path,
    visible_images: Path,
    weights: Path | str,
    pairing: Pairing,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
) -> FaithfulnessMetrics:
    """Score one translated export against the real visible frames it was translated from.

    The C2 entry point, and the discriminator PLAN.md §8's reward-hacking guardrails are
    written against: LPIPS falling while mAP rises is the *signature*, but only a count of
    invented and erased objects says which it was. Runs post hoc over a finished export --
    ``runs/<name>/stage<N>/translated`` -- so a campaign never has to be repeated to get it.

    ``weights`` must be the **reference** detector (``detector.reference.weights``), never the
    in-loop one. Counting hallucinations with the checkpoint that supplied the training
    gradient measures how well the translator learned to please that checkpoint, which is the
    confound E3's independent judge exists to remove (TASKS.md M1.2 step 1).

    Ground truth is read from the *source* labels via ``pairing``, not from the export's
    mirrored copy: ``export.py::_mirror_label`` copies them verbatim, so the two agree, and
    reading the source keeps one source of truth for what the boxes are.
    """
    from ultralytics import YOLO

    visible_paths = sorted(
        p for p in Path(visible_images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not visible_paths:
        raise FileNotFoundError(f"no images to score in {visible_images}")

    # Matched on stem, not filename: export.py writes every image as .png regardless of the
    # source suffix, so a name-for-name lookup silently finds nothing on a .jpg dataset.
    by_stem = {
        p.stem: p for p in Path(translated_images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    }
    missing = [p.stem for p in visible_paths if p.stem not in by_stem]
    if missing:
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f", ... (+{len(missing) - 5} more)"
        raise FileNotFoundError(
            f"{len(missing)} of {len(visible_paths)} visible frames have no translated "
            f"counterpart in {translated_images}: {preview}{more}"
        )
    translated_paths = [by_stem[p.stem] for p in visible_paths]

    model = YOLO(str(weights))

    def detect(paths: list[Path]) -> list[Detections]:
        """One streamed pass, converted to `Detections` as it goes.

        Converting inside the loop rather than collecting `Results` is what keeps this
        bounded: a `Results` object retains the decoded source image, so materialising a
        600-image split would hold hundreds of MB to read a handful of boxes out of each.

        The two passes are also deliberately **sequential, not zipped**. `YOLO.predict`
        reuses one `self.predictor` across calls, so starting a second stream while the
        first is still being consumed resets state underneath it.
        """
        return [
            detections_from_result(result, conf_threshold)
            for result in model.predict(
                paths,
                imgsz=imgsz,
                device=device,
                verbose=False,
                stream=True,
                # Filtered here too, so ultralytics never pays NMS on boxes that
                # `detections_from_result` is about to discard anyway.
                conf=conf_threshold,
            )
        ]

    translated_detections = detect(translated_paths)
    visible_detections = detect(visible_paths)

    evaluator = FaithfulnessEvaluator(iou_threshold=iou_threshold)
    for visible_path, translated, visible in zip(
        visible_paths, translated_detections, visible_detections, strict=True
    ):
        cls, bboxes = load_yolo_labels(pairing.label_path(visible_path))
        evaluator.update(
            translated=translated,
            visible=visible,
            gt=detections_from_labels(cls, bboxes),
        )

    metrics = evaluator.compute()
    logger.info(
        "faithfulness %s on %d pairs: false-object %.4f  missed-object %.4f  consistency %.4f",
        Path(translated_images).parent.name,
        len(visible_paths),
        metrics.false_object_rate,
        metrics.missed_object_rate,
        metrics.detection_consistency,
    )
    return metrics
