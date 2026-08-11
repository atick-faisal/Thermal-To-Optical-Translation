"""Single-stage training loop over a `Translator`.

Ported from ``../Clean-SeAFusion/src/seafusion/engine/fusion_trainer.py``, but reshaped
around a different ownership boundary. That trainer owns the model's ``Adam`` optimizer,
``ExponentialLR`` scheduler, and ``GradScaler`` directly, and calls ``.backward()``/
``optimizer.step()`` itself. t2o's ``Translator.fit()`` (M0.6) is already a complete,
self-contained optimisation step -- ``StubTranslator.fit()`` does its own
``zero_grad()``/``backward()``/``optimizer.step()`` -- so there is nothing left here for a
trainer to wrap. AMP/GradScaler becomes each backbone's own concern if one ever needs it
(``StubTranslator`` is a CPU stand-in that never will); ``TrainConfig.lr``/``lr_gamma``/
``amp``/``amp_dtype`` stay in the schema for a future translator's own constructor to read,
not consumed here.

Also drops ``DistributedContext`` entirely -- t2o has no DDP (PLAN.md §3).

Checkpoints hold ``translator.state_dict()`` rather than separate
``optimizer``/``scheduler``/``scaler`` entries, so a translator's optimizer momentum does
not survive resume (``nn.Module.state_dict()`` does not see ``StubTranslator``'s private
``Adam`` attribute). Accepted rather than worked around: ``StubTranslator`` is dev/test-only,
and a real backbone that needs this can expose it itself via ``get_extra_state()``/
``set_extra_state()``.

Validation is a single generic pixel-L2 pass through ``translate()`` alone, under
``no_grad`` -- the one method every backbone keeps grad-connected and comparable
(``translators/protocol.py``). ``fit()``'s own loss composition is backbone-specific and,
per Clean-SeAFusion's own reasoning for excluding its task term from validation, the wrong
thing to select checkpoints on.

Warm-starting a translator across stages needs no ``model=None`` constructor branch (unlike
Clean-SeAFusion's): the caller always passes an already-constructed translator, and holding
the same instance across successive ``Trainer(...)`` calls *is* the warm start.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from t2o.config import Config
from t2o.data.dataset import TranslationPairDataset, collate_translation_batch
from t2o.data.manifest import DatasetManifest
from t2o.tracking import RunTracker
from t2o.translators.protocol import Translator

logger = logging.getLogger(__name__)

LAST_CHECKPOINT = "translator_last.pt"
BEST_CHECKPOINT = "translator_best.pt"


@dataclass(frozen=True, slots=True)
class EpochStats:
    """One epoch's summary. ``train_losses`` keys are whatever `translator.fit()` returns."""

    epoch: int
    train_losses: dict[str, float]
    val_loss: float


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """Runs `config.train.epochs_per_stage` epochs of `translator.fit()`."""

    def __init__(
        self,
        config: Config,
        manifest: DatasetManifest,
        translator: nn.Module,
        run_dir: Path,
        tracker: RunTracker | None = None,
    ) -> None:
        if not isinstance(translator, Translator):
            raise TypeError("translator must implement fit()/translate() (the Translator protocol)")

        self.config = config
        self.translator = translator
        self.run_dir = run_dir
        self.tracker = tracker

        self.device = _resolve_device(config.runtime.device)
        self.translator.to(self.device)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.train_dataset = TranslationPairDataset(
            manifest.train_images,
            pairing=manifest.pairing,
            hflip=config.train.hflip,
            crop=config.train.crop,
            num_classes=manifest.nc,
            annotation_fraction=config.data.annotation_fraction,
            annotation_seed=config.data.annotation_seed,
        )
        # No augmentation on val -- matches Clean-SeAFusion. Val loss never touches
        # cls/bboxes, so annotation_fraction is irrelevant here and left at its default.
        self.val_dataset = TranslationPairDataset(
            manifest.val_images,
            pairing=manifest.pairing,
            num_classes=manifest.nc,
        )

        # Reseeded once per epoch in `train()`, so a given epoch index always shuffles the
        # same way regardless of how many epochs ran before it -- required for resume to
        # reproduce training exactly.
        self._train_generator = torch.Generator()
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=config.train.workers,
            collate_fn=collate_translation_batch,
            generator=self._train_generator,
            pin_memory=self.device.type == "cuda",
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.workers,
            collate_fn=collate_translation_batch,
            pin_memory=self.device.type == "cuda",
        )

        self.start_epoch = 0
        self.best_val_loss = float("inf")

    def train(self) -> list[EpochStats]:
        history: list[EpochStats] = []
        for epoch in range(self.start_epoch, self.config.train.epochs_per_stage):
            self._train_generator.manual_seed(self.config.train.seed + epoch)
            train_losses = self._train_epoch()
            val_loss = self._validate()
            history.append(EpochStats(epoch=epoch, train_losses=train_losses, val_loss=val_loss))

            if self.tracker is not None:
                metrics = {f"train/{k}": v for k, v in train_losses.items()}
                metrics["val/pixel_l2"] = val_loss
                self.tracker.log(metrics, step=epoch)

            self._checkpoint(epoch, val_loss)
        return history

    def _train_epoch(self) -> dict[str, float]:
        translator = cast(Translator, self.translator)
        self.translator.train()
        totals: dict[str, float] = {}
        n = 0
        for batch in self.train_loader:
            losses = translator.fit(batch)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + value
            n += 1
        return {key: value / n for key, value in totals.items()} if n else {}

    @torch.no_grad()
    def _validate(self) -> float:
        translator = cast(Translator, self.translator)
        self.translator.eval()
        total, n = 0.0, 0
        for batch in self.val_loader:
            pred = translator.translate(batch)
            target = batch["visible"].to(pred.device)
            total += F.mse_loss(pred, target).item()
            n += 1
        return total / n if n else float("nan")

    def _checkpoint(self, epoch: int, val_loss: float) -> None:
        improved = val_loss < self.best_val_loss
        if improved:
            self.best_val_loss = val_loss
        state = {
            "translator": self.translator.state_dict(),
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "config_hash": self.config.config_hash(),
            "config": self.config.to_dict(),
        }
        torch.save(state, self.run_dir / LAST_CHECKPOINT)
        if improved:
            torch.save(state, self.run_dir / BEST_CHECKPOINT)

    def resume(self, path: Path | None = None) -> None:
        path = path if path is not None else self.run_dir / LAST_CHECKPOINT
        if not path.is_file():
            raise FileNotFoundError(f"no checkpoint to resume from: {path}")

        state = torch.load(path, map_location=self.device, weights_only=False)
        current_hash = self.config.config_hash()
        if state.get("config_hash") != current_hash:
            logger.warning(
                "resuming %s: checkpoint config_hash %s does not match the current config %s",
                path,
                state.get("config_hash"),
                current_hash,
            )

        self.translator.load_state_dict(state["translator"])
        self.start_epoch = state["epoch"] + 1
        self.best_val_loss = state["best_val_loss"]
