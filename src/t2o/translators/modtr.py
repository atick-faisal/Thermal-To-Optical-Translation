"""ModTr as a `Translator`: the method around the network (modtr-baseline ADR-01, ADR-04).

`modtr_unet.py` is ModTr's transformation network h(.). This module is what ModTr does with it,
reimplemented from upstream `heitorrapela/ModTr@cace11e`:

- **Input** -- 1-channel thermal in [0, 1], repeated to 3 channels with no ImageNet
  normalisation (`src/engine/utils/utils.py:96-97`).
- **Fusion** -- the output is `clip(x * h(x), 0, 1)`, the paper's Eq. 3 (ModTr-dot), which is
  the README's `--fuse-data cross` (`utils.py:507-516`). Both factors are in [0, 1], so the
  clip never binds and never cuts a gradient. It also means a pixel can only darken: a cold
  (black) thermal pixel stays black whatever the network learns.
- **Objective** -- the frozen detector's loss alone (paper Sec. 3). Here that is `task_loss`,
  the in-loop yolo11n's `v8DetectionLoss` (margin-gate ADR-01), not upstream's torchvision
  detectors. There is no reconstruction term, so `fit` refuses to run without one.
- **Optimiser** -- AdamW at `train.lr` (upstream 1e-4, `main.py:53`) with torch's default
  weight decay (`src/engine/config/config.py:233-239`), gradient accumulation 2 (`main.py:413`),
  and a cosine schedule stepped once per optimiser step with `T_max` counted in *batches*
  (`main.py:310-311, :370`). That last mismatch is upstream's, kept on purpose: a full run
  ends at about half the starting LR, not near zero (ADR-04).

Deviations from upstream:

1. Accumulation lives inside `fit`, so `fit` steps the optimiser on every second call. That
   bends `Translator.fit`'s "one full optimisation step"; nothing in `engine/` depends on it.
2. Lightning applies a half-accumulated gradient at the end of every epoch. `fit` cannot see
   epoch ends, so a leftover batch's gradient carries into the next epoch's first step instead.
   With the custom set's 75 batches per epoch that is about 1% fewer optimiser steps.
3. Not a deviation, but easy to drop by accident: Lightning divides the loss by the
   accumulation count before `backward`, and so does this, so the applied gradient is the mean
   over the two batches at the same scale as upstream's.
4. `T_max` is counted by `build_translator` from the training corpus (modtr-baseline Q8), not
   from a Lightning dataloader.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from t2o.data.dataset import TranslationBatch
from t2o.translators.modtr_unet import ModTrUNet

# Upstream `main.py:413`: `accumulate_grad_batches=2`.
_ACCUMULATION = 2
# torch's AdamW default, which upstream never overrides (`config.py:233-239`).
_WEIGHT_DECAY = 1.0e-2


class ModTrObjectiveError(ValueError):
    """`fit` was called without a detection term, which is ModTr's whole objective."""


class ModTrTranslator(nn.Module):
    """Implements the `Translator` protocol over `ModTrUNet`, with ModTr's fusion and optimiser.

    Owns its optimiser and scheduler, as every backbone does (`engine/trainer.py` never touches
    them). `total_batches` is the cosine's `T_max`: every training batch this translator will
    see, across every stage.
    """

    def __init__(self, total_batches: int, lr: float = 1.0e-4, pretrained: bool = True) -> None:
        super().__init__()
        self.net = ModTrUNet(pretrained=pretrained)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=_WEIGHT_DECAY)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_batches)
        # Batches whose gradient is summed in `.grad` but not yet applied.
        self._accumulated = 0

    def translate(self, batch: TranslationBatch) -> Tensor:
        device = next(self.net.parameters()).device
        return self._fuse(batch["infrared"].to(device))

    def fit(
        self,
        batch: TranslationBatch,
        task_loss: Callable[[Tensor, TranslationBatch], Tensor] | None = None,
        task_weight: float = 0.0,
    ) -> dict[str, float]:
        if task_loss is None or task_weight <= 0.0:
            raise ModTrObjectiveError(
                "ModTr trains on the frozen detector's loss alone, but fit() got "
                f"task_loss={'None' if task_loss is None else 'set'} and "
                f"task_weight={task_weight}. Every coupling.task_weights entry must be > 0."
            )
        device = next(self.net.parameters()).device
        fake = self._fuse(batch["infrared"].to(device))
        detection = task_loss(fake, batch)
        total = task_weight * detection
        (total / _ACCUMULATION).backward()

        self._accumulated += 1
        if self._accumulated == _ACCUMULATION:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self._accumulated = 0

        return {"loss_det": float(detection.detach()), "loss_total": float(total.detach())}

    def get_extra_state(self) -> dict[str, Any]:
        """AdamW and the cosine's position, so a resume continues the schedule.

        Without the scheduler, a resumed 400-epoch run would restart its cosine at the full LR.
        The half-accumulated gradient is not saved, so `set_extra_state` drops it.
        """
        return {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict()}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.optimizer.zero_grad()
        self._accumulated = 0

    def _fuse(self, infrared: Tensor) -> Tensor:
        x = infrared.repeat(1, 3, 1, 1)
        return torch.clamp(x * self.net(x), 0.0, 1.0)
