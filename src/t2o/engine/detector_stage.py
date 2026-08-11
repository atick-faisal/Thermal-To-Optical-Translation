"""Thin wrapper over ultralytics' own trainer, run on the exported translated images.

Delegating rather than reimplementing means the detector stage inherits mosaic
augmentation, EMA, mAP validation, checkpoint selection, and ultralytics' own DDP -- all
of which would be a large and pointless surface to rebuild here.

Ported from ``../Clean-SeAFusion/src/seafusion/engine/detector_stage.py`` with two
deliberate deviations, both driven by how ``t2o.config.Config`` differs from
``seafusion.config.Config``:

* ``config.detector`` there is flat (``epochs_per_stage``/``imgsz``/``batch``); here it is
  split into ``in_loop``/``evaluation`` (PLAN.md invariant 7, M0.2). ``train_detector``
  always trains the *evaluation* detector -- the in-loop one is never trained -- so field
  access below goes through ``config.detector.evaluation``.
* ``seed`` there lives on ``config.runtime.seed``; here it lives on ``config.train.seed``
  (M0.2 decision: a seed is scientific and belongs in the hashed part of the config).

``_extract_metrics`` is imported from :mod:`t2o.metrics.task` rather than defined here
(M0.5 step 2): it is the same P/R/mAP extraction ``evaluate_detector`` uses for standalone
validation, not a merely similar copy -- ``model.train()`` and ``model.val()`` both return
the same ``DetMetrics`` type, so one function genuinely serves both call sites.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from t2o.config import Config
from t2o.metrics.task import _extract_metrics

logger = logging.getLogger(__name__)


@contextmanager
def wandb_integration_disabled() -> Iterator[None]:
    """Stop ultralytics registering its own wandb callbacks for the duration.

    Why this is needed: ultralytics' integration adopts whatever run is already open
    (``on_pretrain_routine_start`` begins with ``if not wb.run:``) and then unconditionally
    calls ``wb.run.finish()`` when training ends -- "required or run continues on
    dashboard". That is right for a standalone ``YOLO.train()`` and wrong here, because the
    stage loop (M0.8) keeps one run open across every stage. The first detector stage would
    kill it and every later log would raise "Run ... is finished".

    Why *here* and not on the model: ``BaseTrainer.__init__`` calls
    ``add_integration_callbacks(self)`` (``engine/trainer.py:180``) which re-adds every
    integration after taking the model's callbacks, so filtering ``model.callbacks`` is
    silently undone. That function does ``from .wb import callbacks as wb_cb`` at call
    time, so emptying the module attribute is what actually prevents registration.
    Mutating ``model.callbacks`` is also not free: ``engine/model.py:840`` compares it
    against the defaults to decide which trainer path to take.

    Doing this in code rather than through ultralytics' settings also makes behaviour
    identical on every machine -- whether the integration is active at all otherwise
    depends on the machine-global ``settings.json``, which any other project can write.

    Per-stage detector mAP is logged by the engine loop through t2o's own tracker (M0.8).
    """
    from ultralytics.utils.callbacks import wb

    original = wb.callbacks
    wb.callbacks = {}
    logger.debug("ultralytics wandb integration disabled; t2o owns the run")
    try:
        yield
    finally:
        wb.callbacks = original


@dataclass(frozen=True, slots=True)
class DetectorResult:
    """Where the trained weights landed, and how good they were."""

    weights: Path
    precision: float
    recall: float
    map50: float
    map50_95: float


def _forward_epoch_metrics(tracker: Any, prefix: str) -> Any:
    """Relay ultralytics' per-epoch validation metrics into t2o's own run.

    Suppressing ultralytics' wandb integration (see above) would otherwise cost the
    per-epoch precision/recall/mAP curves entirely, since t2o only records a single summary
    per stage. Forwarding them keeps one coherent run with everything in it.
    """

    def on_fit_epoch_end(trainer: Any) -> None:
        metrics = {
            f"{prefix}/{key.removeprefix('metrics/')}": float(value)
            for key, value in (getattr(trainer, "metrics", None) or {}).items()
            if isinstance(value, (int, float))
        }
        if metrics:
            metrics[f"{prefix}/epoch"] = float(trainer.epoch)
            tracker.log(metrics)

    return on_fit_epoch_end


def train_detector(
    data_yaml: Path,
    init_weights: Path,
    config: Config,
    project: Path,
    name: str,
    epochs: int | None = None,
    tracker: Any = None,
    metric_prefix: str = "detector",
) -> DetectorResult:
    """Fine-tune the evaluation detector on translated images and report its validation mAP.

    ``init_weights`` warm-starts from the previous stage's ``best.pt`` so the evaluation
    detector accumulates across stages, the same way the translator does. This never
    touches :class:`t2o.detection.frozen.FrozenDetector` -- ultralytics' own ``model.train()``
    has no connection to the translator's autograd graph, which is what keeps the reported
    mAP an honest, un-gamed number (PLAN.md invariant 7).
    """
    from ultralytics import YOLO

    model = YOLO(str(init_weights))
    if tracker is not None:
        model.add_callback("on_fit_epoch_end", _forward_epoch_metrics(tracker, metric_prefix))
    with wandb_integration_disabled():
        results = model.train(
            data=str(data_yaml),
            epochs=epochs if epochs is not None else config.detector.evaluation.epochs,
            imgsz=config.detector.evaluation.imgsz,
            batch=config.detector.evaluation.batch,
            workers=config.train.workers,
            seed=config.train.seed,
            device=config.runtime.device,
            project=str(project),
            name=name,
            exist_ok=True,
            verbose=False,
            plots=False,
        )

    weights = _resolve_weights(model, project, name)
    scores = _extract_metrics(results)
    logger.info(
        "detector %s: P %.4f  R %.4f  mAP50 %.4f  mAP50-95 %.4f",
        name,
        scores["precision"],
        scores["recall"],
        scores["map50"],
        scores["map50_95"],
    )
    return DetectorResult(weights=weights, **scores)


def _resolve_weights(model: Any, project: Path, name: str) -> Path:
    """Return the checkpoint ultralytics actually wrote, rather than a guessed path.

    ``best.pt`` is not guaranteed to exist. The trainer writes ``last.pt`` every epoch but
    only writes ``best.pt`` when ``best_fitness == fitness`` (engine/trainer.py:767), and
    ``best_fitness`` is assigned from the same value it is compared against
    (engine/trainer.py:873). If a validation pass ever yields a **NaN** fitness -- easy on
    a small or degenerate split -- then ``NaN == NaN`` is False on every epoch and
    ``best.pt`` is never written at all, for the whole run.

    Falling back to ``last.pt`` matters: it is a fully trained detector, and discarding an
    hour of GPU time over a missing symlink-of-convenience would be absurd.
    """
    trainer = getattr(model, "trainer", None)
    best = getattr(trainer, "best", None)
    last = getattr(trainer, "last", None)
    # Fall back to the documented layout only if the trainer is not reachable at all.
    save_dir = Path(getattr(trainer, "save_dir", Path(project) / name))
    if best is None or last is None:
        best, last = save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"

    if Path(best).is_file():
        return Path(best)
    if Path(last).is_file():
        logger.warning(
            "%s has no best.pt; falling back to last.pt. ultralytics only writes best.pt "
            "when best_fitness == fitness, which never holds if validation fitness was "
            "NaN. Treat this stage's reported metrics with suspicion.",
            save_dir,
        )
        return Path(last)

    weights_dir = save_dir / "weights"
    present = (
        ", ".join(sorted(p.name for p in weights_dir.iterdir()))
        if weights_dir.is_dir()
        else "the weights directory does not exist"
    )
    raise FileNotFoundError(
        f"detector training produced no usable checkpoint in {weights_dir} ({present}). "
        f"Training may have been interrupted before the first epoch completed."
    )
