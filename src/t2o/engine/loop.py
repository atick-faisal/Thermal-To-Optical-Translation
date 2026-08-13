"""Staged alternating loop: translator epochs -> export -> detector fine-tune -> repeat.

Ported from ``../Clean-SeAFusion/src/seafusion/engine/loop.py``'s staging skeleton
(``task_weights: [0, 1, 2, 3]``, one stage per entry), restructured around two ownership
decisions t2o already made:

* **Translator warm-start needs no code here.** ``engine/trainer.py`` already accepts an
  already-constructed translator; holding the same instance across successive
  ``Trainer(...)`` calls *is* the warm start (unlike Clean-SeAFusion's ``train.py:203``,
  which re-instantiates the generator every stage and makes the loop one-directional).
* **The evaluation detector warm-starts across stages; the in-loop detector does not.**
  Clean-SeAFusion threads a single ``detector_weights`` variable through both roles: the
  frozen detector the coupling loss grades against, and ``train_detector``'s init weights.
  t2o's config already splits these two roles structurally
  (``DetectorConfig.in_loop``/``.evaluation``, M0.2, PLAN.md invariant 7 -- "two detectors,
  never conflated"). This loop keeps that split real at runtime too: ``eval_weights`` is the
  only variable reassigned across stages; ``config.detector.in_loop.weights`` is read fresh
  every stage but never written. Otherwise the frozen guiding detector would end each stage
  as literally the just-fine-tuned evaluation detector -- graded against its own prior,
  weaker translations -- which is a materially different (and less defensible) experiment
  than a fixed, independently-trained reference detector. "Detector warm-started across
  stages" in PLAN.md §8 is read as describing the evaluation detector's own accumulation,
  the same way the translator accumulates -- not the in-loop detector's identity.

Detector fine-tuning is unconditional every stage (including stage 0, matching
Clean-SeAFusion's table: stage 0 trains no coupling term but still bootstraps a detector
from the translation-only output). Only ``build_detection_loss``'s own weight-vs-zero check
(M0.7) decides whether a stage's coupling term exists at all.

**Every stage reports its task metric in two arms, and they answer different questions.**
``StageResult.detector`` is the *adapted* arm: the evaluation detector fine-tuned on this
stage's translated images, then measured. ``StageResult.zero_shot`` is the *unadapted* arm:
a fixed reference detector (``config.detector.reference``), never trained on anything this
project produced, run straight at the translated images. Only the second one answers M1's
gate. The first saturates -- M0.10's E1 bracket put a same-domain-trained detector above 0.9
mAP50 on raw thermal too, so ~0.9 on translated images says only that a YOLO fine-tune
converges on this dataset, whichever domain it is shown. It also presupposes exactly the
thermal-domain annotations the low-annotation framing (E8) exists to avoid needing. Both are
recorded because the adapted arm is still the right number for E7 (detector identity); the
gate and E3 are decided on the zero-shot arm.

``resume=True`` (M0.8's own "Tests" item, deferred here until now) picks up an interrupted
multi-stage run: stages already recorded in ``run_dir/metrics.json`` are not re-run at all,
and the first not-yet-recorded stage resumes its `Trainer` from a checkpoint if one exists,
so a crash loses at most the epoch in progress. Detector fine-tuning for a stage always
restarts fresh -- it is one bounded ``model.train()`` call, not something this loop tracks
epoch-by-epoch the way translator training is, so there is nothing to resume it *from*.
Warm-start continuity across a resume needs one extra step beyond re-reading
``metrics.json``: the caller's ``translator`` instance is whatever a fresh process just
constructed (random init, or seeded via ``translators.build_translator``), so if the stage
about to run has no checkpoint of its own yet, its weights are restored from the last
*completed* stage's checkpoint before training starts -- otherwise a resumed run would
silently train the next stage from scratch atop a translator that only looks warm-started.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from t2o.config import Config
from t2o.coupling.schedule import build_detection_loss, weight_for_stage
from t2o.data.manifest import DatasetManifest
from t2o.engine.detector_stage import DetectorResult, train_detector
from t2o.engine.export import export_translated
from t2o.engine.trainer import LAST_CHECKPOINT, EpochStats, Trainer
from t2o.metrics.fidelity import FidelityMetrics, evaluate_fidelity
from t2o.metrics.task import TaskMetrics, evaluate_detector
from t2o.tracking import RunTracker

logger = logging.getLogger(__name__)

METRICS_FILENAME = "metrics.json"


@dataclass(frozen=True, slots=True)
class StageResult:
    """One stage's outcome: translator epochs, zero-shot score, and its detector fine-tune."""

    stage: int
    task_weight: float
    epochs: list[EpochStats]
    detector: DetectorResult | None
    # The gate metric: a fixed reference detector on this stage's translated images, with no
    # fine-tuning of any kind. `detector` above is the adapted arm and saturates; see the
    # module docstring for why only this one answers M1's gate.
    zero_shot: TaskMetrics | None = None
    # PSNR/SSIM/LPIPS/FID/KID for the same export, against the real visible frames. Read
    # *together* with zero_shot: a detection metric rising while these fall is the signature
    # of reward hacking, and neither number diagnoses it alone (PLAN.md §8).
    fidelity: FidelityMetrics | None = None


def run_loop(
    config: Config,
    translator: nn.Module,
    run_dir: Path | None = None,
    tracker: RunTracker | None = None,
    train_detector_stages: bool = True,
    resume: bool = False,
) -> list[StageResult]:
    """Run every stage in ``config.coupling.task_weights`` and return their results.

    ``train_detector_stages=False`` skips export and detector fine-tuning entirely,
    reproducing a translation-only run -- useful as a fast smoke path and as an
    independently meaningful ablation of its own (Clean-SeAFusion carries the same flag).

    ``resume=True`` re-reads ``run_dir/metrics.json`` and skips every stage already recorded
    there; on a fresh ``run_dir`` (no file yet) it is a silent no-op, identical to
    ``resume=False``. See the module docstring for how warm-start continuity survives the
    resumed stage having no checkpoint of its own yet.
    """
    manifest = DatasetManifest.load(config.data.manifest)
    run_dir = Path(run_dir) if run_dir is not None else config.runtime.path
    run_dir.mkdir(parents=True, exist_ok=True)
    config.snapshot(run_dir)

    reference_weights = _resolve_reference_weights(config)

    results: list[StageResult] = _load_existing_results(run_dir) if resume else []
    start_stage = len(results)
    eval_weights = (
        results[-1].detector.weights
        if results and results[-1].detector is not None
        else config.detector.evaluation.init_weights
    )

    if resume and start_stage > 0 and start_stage < len(config.coupling.task_weights):
        current_checkpoint = run_dir / f"stage{start_stage}" / LAST_CHECKPOINT
        previous_checkpoint = run_dir / f"stage{start_stage - 1}" / LAST_CHECKPOINT
        if not current_checkpoint.is_file() and previous_checkpoint.is_file():
            state = torch.load(previous_checkpoint, map_location="cpu", weights_only=False)
            translator.load_state_dict(state["translator"])

    for stage in range(start_stage, len(config.coupling.task_weights)):
        stage_dir = run_dir / f"stage{stage}"
        task_weight = weight_for_stage(config.coupling, stage)
        task_loss = build_detection_loss(
            config.coupling, stage, config.detector.in_loop.weights, manifest.nc
        )

        trainer = Trainer(
            config,
            manifest,
            translator,
            run_dir=stage_dir,
            tracker=tracker,
            task_loss=task_loss,
            task_weight=task_weight,
        )
        if resume and stage == start_stage and (stage_dir / LAST_CHECKPOINT).is_file():
            trainer.resume()
        epochs = trainer.train()

        detector_result: DetectorResult | None = None
        zero_shot: TaskMetrics | None = None
        fidelity: FidelityMetrics | None = None
        if train_detector_stages:
            data_yaml = export_translated(
                translator,
                config,
                stage_dir / "translated",
                device=trainer.device,
                manifest=manifest,
            )
            # Before train_detector, deliberately: this is the gate metric, and it must
            # survive a detector fine-tune that OOMs or diverges. It also costs one val pass.
            zero_shot = evaluate_detector(
                reference_weights,
                data_yaml,
                imgsz=config.detector.reference.imgsz,
                batch=config.detector.reference.batch,
                device=config.runtime.device,
            )
            fidelity = evaluate_fidelity(
                data_yaml.parent / "val" / "images",
                manifest.val_images,
                lpips_net=config.metrics.lpips_net,
                kid_subset_size=config.metrics.kid_subset_size,
                device=config.runtime.device,
                batch_size=config.detector.evaluation.batch,
            )
            if tracker is not None:
                tracker.log(_zero_shot_metrics(zero_shot, f"stage{stage}/zero_shot"))
                tracker.log(_fidelity_metrics(fidelity, f"stage{stage}/fidelity"))
            detector_result = train_detector(
                data_yaml=data_yaml,
                init_weights=eval_weights,
                config=config,
                project=stage_dir / "detector",
                name=f"stage{stage}",
                tracker=tracker,
                metric_prefix=f"stage{stage}/detector",
            )
            eval_weights = detector_result.weights

        results.append(
            StageResult(
                stage=stage,
                task_weight=task_weight,
                epochs=epochs,
                detector=detector_result,
                zero_shot=zero_shot,
                fidelity=fidelity,
            )
        )
        _write_metrics(run_dir, results)
        logger.info("stage %d done: task_weight=%s epochs=%d", stage, task_weight, len(epochs))

    return results


def _resolve_reference_weights(config: Config) -> Path:
    """Pick the detector that scores translations zero-shot, and warn if it is compromised.

    Falling back to ``detector.evaluation.init_weights`` is the honest default: it is the
    un-fine-tuned bootstrap checkpoint, i.e. exactly "the detector you already have because
    you could label visible data for it", which is the detector M1's gate is about.

    The warning matters because that fallback frequently resolves to the same file as
    ``detector.in_loop.weights`` -- the frozen detector whose gradient trained the translator
    in every ``lambda_det > 0`` stage. Scoring those stages with it grades the translator
    against the objective it was optimised for, so a gain there is not separable from reward
    hacking. This warns rather than raising: the ``lambda_det = 0`` arm never touches the
    in-loop detector at all, so its number stays clean regardless, and that arm alone is
    enough to decide the gate.
    """
    reference = config.detector.reference.weights or config.detector.evaluation.init_weights
    couples = any(weight > 0.0 for weight in config.coupling.task_weights)
    if couples and Path(reference) == Path(config.detector.in_loop.weights):
        logger.warning(
            "zero-shot reference detector (%s) is the same checkpoint as the in-loop "
            "detector, so every task_weight > 0 stage is graded against the objective it "
            "was trained on. The task_weight == 0 stages remain uncontaminated. Set "
            "detector.reference.weights to an independently trained detector before "
            "reporting the coupled arm.",
            reference,
        )
    return Path(reference)


def _zero_shot_metrics(metrics: TaskMetrics, prefix: str) -> dict[str, float]:
    """Flatten TaskMetrics for the tracker, per-class AP included."""
    flat = {
        f"{prefix}/precision": metrics.precision,
        f"{prefix}/recall": metrics.recall,
        f"{prefix}/mAP50": metrics.map50,
        f"{prefix}/mAP50-95": metrics.map50_95,
    }
    flat.update({f"{prefix}/AP50/{name}": ap for name, ap in metrics.per_class_ap50.items()})
    return flat


def _fidelity_metrics(metrics: FidelityMetrics, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}/psnr": metrics.psnr,
        f"{prefix}/ssim": metrics.ssim,
        f"{prefix}/lpips": metrics.lpips,
        f"{prefix}/fid": metrics.fid,
        f"{prefix}/kid_mean": metrics.kid_mean,
        f"{prefix}/kid_std": metrics.kid_std,
    }


def _write_metrics(run_dir: Path, results: list[StageResult]) -> None:
    payload = [_stage_result_to_json(result) for result in results]
    (run_dir / METRICS_FILENAME).write_text(json.dumps(payload, indent=2))


def _stage_result_to_json(result: StageResult) -> dict[str, Any]:
    data = asdict(result)
    if data["detector"] is not None:
        data["detector"]["weights"] = str(data["detector"]["weights"])
    return data


def _load_existing_results(run_dir: Path) -> list[StageResult]:
    """Reconstruct completed stages from a prior run's ``metrics.json``, if any exists."""
    path = run_dir / METRICS_FILENAME
    if not path.is_file():
        return []
    return [_stage_result_from_json(entry) for entry in json.loads(path.read_text())]


def _stage_result_from_json(data: dict[str, Any]) -> StageResult:
    detector = data["detector"]
    # `.get`, not `[...]`: metrics.json files written before the zero-shot arm existed must
    # still resume rather than raising KeyError halfway through a long server run.
    zero_shot = data.get("zero_shot")
    fidelity = data.get("fidelity")
    return StageResult(
        stage=data["stage"],
        task_weight=data["task_weight"],
        epochs=[EpochStats(**epoch) for epoch in data["epochs"]],
        detector=DetectorResult(**{**detector, "weights": Path(detector["weights"])})
        if detector is not None
        else None,
        zero_shot=TaskMetrics(**zero_shot) if zero_shot is not None else None,
        fidelity=FidelityMetrics(**fidelity) if fidelity is not None else None,
    )
