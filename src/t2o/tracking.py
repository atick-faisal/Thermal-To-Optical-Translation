"""Optional Weights & Biases logging. A no-op unless ``runtime.wandb`` is on.

Ported from ``../Clean-SeAFusion/src/seafusion/tracking.py``, dropping the
``DistributedContext`` rank-0 gate: t2o has no DDP (PLAN.md §3), so there is no rank to
gate on. ``enabled`` is exactly ``config.runtime.wandb``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
from typing import Any

from t2o.config import Config

logger = logging.getLogger(__name__)


def run_tags(config: Config) -> list[str]:
    """The W&B tags for a run, *derived* from the resolved config rather than authored.

    E3's campaign (TASKS.md M1.2 step 5) launches 12 runs off two YAML files, overriding
    ``--seed`` and ``--name`` on every one of them. A tag written into either file would
    therefore be wrong on most launches; deriving them here means a tag cannot disagree with
    the run it labels. Kept to the three facts a campaign is actually filtered on -- which
    backbone, which seed, and whether the coupling term exists at all.
    """
    coupled = any(weight > 0.0 for weight in config.coupling.task_weights)
    return [
        f"backbone:{config.translator.backbone.value}",
        f"seed:{config.train.seed}",
        f"lambda_det:{'on' if coupled else 'off'}",
    ]


class RunTracker:
    """Logs to wandb; silently does nothing if ``runtime.wandb`` is off."""

    def __init__(self, config: Config) -> None:
        self.enabled = config.runtime.wandb
        self._run: Any = None
        if not self.enabled:
            return

        try:
            import wandb
        except ImportError:  # pragma: no cover - wandb is a declared dependency
            logger.warning("runtime.wandb is set but wandb is not installed; skipping")
            self.enabled = False
            return

        # wandb writes into `dir`, which the trainer has not necessarily created yet.
        run_dir = Path(config.runtime.path)
        run_dir.mkdir(parents=True, exist_ok=True)

        self._run = wandb.init(
            project=config.runtime.wandb_project,
            name=config.runtime.name,
            group=config.runtime.group,
            tags=run_tags(config),
            config=config.to_dict(),
            dir=str(run_dir),
        )
        logger.info(
            "wandb run '%s' initialised under project '%s' (group %s)",
            config.runtime.name,
            config.runtime.wandb_project,
            config.runtime.group or "none",
        )

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a metric dict. Never raises: telemetry must not take down a training run."""
        if self._run is None:
            return
        try:
            self._run.log(metrics, step=step)
        except Exception:
            logger.warning(
                "wandb logging failed; disabling it for the rest of this run", exc_info=True
            )
            self._run = None
            self.enabled = False

    def finish(self) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:
                logger.warning("wandb finish failed", exc_info=True)
            self._run = None

    def __enter__(self) -> RunTracker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.finish()
