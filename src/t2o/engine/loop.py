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
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from torch import nn

from t2o.config import Config
from t2o.coupling.schedule import build_detection_loss, weight_for_stage
from t2o.data.manifest import DatasetManifest
from t2o.engine.detector_stage import DetectorResult, train_detector
from t2o.engine.export import export_translated
from t2o.engine.trainer import EpochStats, Trainer
from t2o.tracking import RunTracker

logger = logging.getLogger(__name__)

METRICS_FILENAME = "metrics.json"


@dataclass(frozen=True, slots=True)
class StageResult:
    """One stage's outcome: the translator epochs it ran, and its detector fine-tune."""

    stage: int
    task_weight: float
    epochs: list[EpochStats]
    detector: DetectorResult | None


def run_loop(
    config: Config,
    translator: nn.Module,
    run_dir: Path | None = None,
    tracker: RunTracker | None = None,
    train_detector_stages: bool = True,
) -> list[StageResult]:
    """Run every stage in ``config.coupling.task_weights`` and return their results.

    ``train_detector_stages=False`` skips export and detector fine-tuning entirely,
    reproducing a translation-only run -- useful as a fast smoke path and as an
    independently meaningful ablation of its own (Clean-SeAFusion carries the same flag).
    """
    manifest = DatasetManifest.load(config.data.manifest)
    run_dir = Path(run_dir) if run_dir is not None else config.runtime.path
    run_dir.mkdir(parents=True, exist_ok=True)
    config.snapshot(run_dir)

    eval_weights = config.detector.evaluation.init_weights
    results: list[StageResult] = []

    for stage in range(len(config.coupling.task_weights)):
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
        epochs = trainer.train()

        detector_result: DetectorResult | None = None
        if train_detector_stages:
            data_yaml = export_translated(
                translator,
                config,
                stage_dir / "translated",
                device=trainer.device,
                manifest=manifest,
            )
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
                stage=stage, task_weight=task_weight, epochs=epochs, detector=detector_result
            )
        )
        _write_metrics(run_dir, results)
        logger.info("stage %d done: task_weight=%s epochs=%d", stage, task_weight, len(epochs))

    return results


def _write_metrics(run_dir: Path, results: list[StageResult]) -> None:
    payload = [_stage_result_to_json(result) for result in results]
    (run_dir / METRICS_FILENAME).write_text(json.dumps(payload, indent=2))


def _stage_result_to_json(result: StageResult) -> dict[str, Any]:
    data = asdict(result)
    if data["detector"] is not None:
        data["detector"]["weights"] = str(data["detector"]["weights"])
    return data
