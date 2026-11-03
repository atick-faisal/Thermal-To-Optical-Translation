"""Faithfulness (C2): false-object rate, missed-object rate, detection-consistency, and the
cascade gate's completion rate.

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

**Completion rate** is the counterfactual-erasure probe's number
(``docs/features/cascade-gate/plan.md``, Tech Stack). The target boxes -- fuses and switches,
cascade-gate ADR-03 -- are inpainted out of the thermal input first (``data/erasure.py``), so a
box the judge finds again, with the same
class at IoU >= 0.5, on the translation of that erased input is an object the pipeline put back
with no evidence for it. The matcher is the same `_greedy_match`, so a completion and a missed
object are the same match read from opposite sides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
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

    Tensors are expected on **CPU**. Ground truth is read there by ``load_yolo_labels`` and
    predictions are moved there by ``detections_from_result``, so the matcher never has to
    reconcile two devices; ``_greedy_match`` says so out loud if a caller mixes them.
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
    # Pulled to CPU here, at the one place device-resident tensors enter this module. Ground
    # truth arrives from `load_yolo_labels` on CPU, so leaving predictions on the accelerator
    # makes the very first `box_iou(translated, gt)` a cross-device call and raises. Matching
    # is a Python-level loop over a handful of boxes -- every `.tolist()`/`int()`/`bool()` in
    # `_greedy_match` would be a device sync -- so CPU is both the correct and the faster side.
    return Detections(
        cls=boxes.cls[keep].long().cpu(),
        bboxes=boxes.xyxyn[keep].cpu(),
        conf=boxes.conf[keep].cpu(),
    )


def _greedy_match(pred: Detections, ref: Detections, iou_threshold: float) -> tuple[Tensor, Tensor]:
    """Confidence-ordered greedy matching, one IoU threshold, same-class only.

    Returns ``(pred_matched, ref_matched)`` boolean masks, aligned to each input's order.
    """
    if pred.bboxes.device != ref.bboxes.device:
        raise ValueError(
            f"predictions are on {pred.bboxes.device} but references are on "
            f"{ref.bboxes.device}; both sides of a match must share a device"
        )

    n_pred, n_ref = pred.cls.shape[0], ref.cls.shape[0]
    # On the inputs' device, not the default one: the masks are combined with `ref.cls` and
    # `iou` below, and a CPU mask against accelerator tensors raises.
    pred_matched = torch.zeros(n_pred, dtype=torch.bool, device=pred.cls.device)
    ref_matched = torch.zeros(n_ref, dtype=torch.bool, device=ref.cls.device)
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


@dataclass(frozen=True, slots=True)
class CompletionMetrics:
    completion_rate: float


class CompletionEvaluator:
    """Accumulates how many erased ground-truth boxes the judge finds again.

    ``update`` is called once per image with the judge's predictions on the translation of the
    *erased* thermal frame and the boxes that were erased from it -- the erased split's labels,
    which ``t2o erase`` writes with only those boxes. A predicted box only counts on an erased box's
    footprint, with its class: a detection anywhere else is the false-object rate's business,
    not this one's.
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self._iou_threshold = iou_threshold
        self.reset()

    def update(self, translated: Detections, erased: Detections) -> None:
        _, erased_matched = _greedy_match(translated, erased, self._iou_threshold)
        self._completed += int(erased_matched.sum())
        self._total_erased += erased.cls.shape[0]

    def compute(self) -> CompletionMetrics:
        # Nothing erased means nothing could be put back -- the no-failure value, the same
        # convention as missed_object_rate's empty case.
        rate = self._completed / self._total_erased if self._total_erased else 0.0
        return CompletionMetrics(completion_rate=rate)

    def reset(self) -> None:
        self._completed = 0
        self._total_erased = 0


def _list_images(images: Path) -> list[Path]:
    return sorted(p for p in Path(images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def _translated_counterparts(visible_paths: list[Path], translated_images: Path) -> list[Path]:
    """The translated frame for every visible frame, in the same order, or a loud failure.

    Matched on stem, not filename: export.py writes every image as .png regardless of the
    source suffix, so a name-for-name lookup silently finds nothing on a .jpg dataset.
    """
    by_stem = {p.stem: p for p in _list_images(translated_images)}
    missing = [p.stem for p in visible_paths if p.stem not in by_stem]
    if missing:
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f", ... (+{len(missing) - 5} more)"
        raise FileNotFoundError(
            f"{len(missing)} of {len(visible_paths)} visible frames have no translated "
            f"counterpart in {translated_images}: {preview}{more}"
        )
    return [by_stem[p.stem] for p in visible_paths]


def _detect(
    model: Any,
    paths: list[Path],
    *,
    conf_threshold: float,
    imgsz: int,
    batch: int,
    device: str | None,
) -> list[Detections]:
    """One judge pass over `paths` in chunks of `batch`, converted to `Detections` as it goes.

    **Chunked here because a list source is never streamed.** A `list` reaches
    ultralytics' `check_source` (`data/build.py`), which hands it to `autocast_list` --
    decoding every element up front -- and then to `LoadPilAndNumpy`, whose
    `bs = len(im0)` makes the whole split *one* batch. `batch=` is passed only to the
    other branch, `LoadImagesAndVideos`, so it is silently ignored on this one, and
    `stream=True` yields a generator over a single iteration. The custom set's 153-image
    val split fit in one batch on a 40 GiB card, which is why both earlier C2 campaigns
    passed; FLIR's 1,013 tried to allocate 9.89 GiB in one backbone concat and raised
    (TASKS.md M3 E9 step 5).

    Converting inside the loop still matters for a second, independent reason: a
    `Results` object retains the decoded source image, so collecting them would hold
    hundreds of MB to read a handful of boxes out of each.

    `rect=False` is ultralytics' own default, pinned because it is what makes a chunk
    boundary invisible. It leaves `pre_transform`'s `auto` false, so every image is
    letterboxed to a fixed square `imgsz` independently of its neighbours and NMS is
    per-image -- the batching below cannot move a number, and these rates stay comparable
    to the ones e3b and e3t were scored with (PLAN.md invariant 1).

    Two passes on one model must be **sequential, not zipped**. `YOLO.predict`
    reuses one `self.predictor` across calls, so starting a second stream while the
    first is still being consumed resets state underneath it -- which is also why each
    chunk's generator is drained before the next one starts.
    """
    detections: list[Detections] = []
    for start in range(0, len(paths), batch):
        detections.extend(
            detections_from_result(result, conf_threshold)
            for result in model.predict(
                paths[start : start + batch],
                imgsz=imgsz,
                device=device,
                rect=False,
                verbose=False,
                stream=True,
                # Filtered here too, so ultralytics never pays NMS on boxes that
                # `detections_from_result` is about to discard anyway.
                conf=conf_threshold,
            )
        )
    return detections


def evaluate_faithfulness(
    translated_images: Path,
    visible_images: Path,
    weights: Path | str,
    pairing: Pairing,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
    batch: int = 16,
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

    visible_paths = _list_images(visible_images)
    if not visible_paths:
        raise FileNotFoundError(f"no images to score in {visible_images}")
    translated_paths = _translated_counterparts(visible_paths, translated_images)

    detect = partial(
        _detect,
        YOLO(str(weights)),
        conf_threshold=conf_threshold,
        imgsz=imgsz,
        batch=batch,
        device=device,
    )
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


def evaluate_completion(
    translated_images: Path,
    erased_images: Path,
    weights: Path | str,
    pairing: Pairing,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
) -> CompletionMetrics:
    """Score the translation of an erased split for objects put back where none remain.

    ``erased_images`` is the visible side of the split ``t2o erase`` wrote, and
    ``translated_images`` is what a pipeline made from its erased thermal side. Only the
    translation is run through the judge; the erased tree supplies the frame list and,
    through ``pairing``, the erased boxes. The judge, thresholds and chunked pass are
    `evaluate_faithfulness`'s, so this rate sits beside its three on one footing.
    """
    from ultralytics import YOLO

    erased_paths = _list_images(erased_images)
    if not erased_paths:
        raise FileNotFoundError(f"no images to score in {erased_images}")
    translated_paths = _translated_counterparts(erased_paths, translated_images)

    translated_detections = _detect(
        YOLO(str(weights)),
        translated_paths,
        conf_threshold=conf_threshold,
        imgsz=imgsz,
        batch=batch,
        device=device,
    )

    evaluator = CompletionEvaluator(iou_threshold=iou_threshold)
    for erased_path, translated in zip(erased_paths, translated_detections, strict=True):
        cls, bboxes = load_yolo_labels(pairing.label_path(erased_path))
        evaluator.update(translated=translated, erased=detections_from_labels(cls, bboxes))

    metrics = evaluator.compute()
    logger.info(
        "completion %s on %d erased frames: completion rate %.4f",
        Path(translated_images).parent.name,
        len(erased_paths),
        metrics.completion_rate,
    )
    return metrics
