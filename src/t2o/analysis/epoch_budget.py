"""ModTr's campaign budget from its calibration run (modtr-baseline ADR-04 and ADR-03).

Both rules were fixed before the calibration run exists, and changing either afterwards needs a
superseding ADR. This module only states them as functions.

**The budget trigger (ADR-04).** Take the 10-epoch trailing mean of the inner-val mAP@50. If its
final value is more than 2.0 points below its maximum, the budget is the last epoch of the best
10-epoch window; otherwise it stays at the default. The trailing mean, not the raw curve,
because the maximum of 400 noisy readings on 120 images sits above the true curve and would
fire on noise alone. Read here as:

- **points**, not fractions: ultralytics reports mAP@50 in [0, 1], ADR-04's 2.0 is in points;
- **full windows only**: a mean over the first few epochs is not a 10-epoch mean;
- **more than** 2.0, strictly;
- **ties to the earliest** window, the shorter budget;
- **a budget is an epoch count**: epochs are numbered from 0, so a best window ending at
  epoch 249 gives a budget of 250.

**The loss rule (ADR-03).** The calibration run doubles as ModTr's smoke run, which stands in
for `scripts/loss_share.py`: the mean training `loss_det` over the last 5 epochs must be below
the mean over the first 5.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

TRAILING_WINDOW = 10
DROP_POINTS = 2.0
LOSS_EPOCHS = 5


class CurveTooShortError(ValueError):
    """Raised when a curve has too few epochs for the rule applied to it."""


@dataclass(frozen=True, slots=True)
class BudgetVerdict:
    """What the trigger read from the curve, and the budget it gives."""

    budget: int  # epochs
    fired: bool
    best_mean: float  # mAP@50 points, the trailing mean's maximum
    final_mean: float  # mAP@50 points, the trailing mean's last value
    best_last_epoch: int  # 0-indexed last epoch of the best window


def trailing_means(values: Sequence[float], window: int = TRAILING_WINDOW) -> np.ndarray:
    """The mean of every full ``window``-epoch run; entry ``i`` ends at epoch ``i + window - 1``."""
    if len(values) < window:
        raise CurveTooShortError(f"{len(values)} epoch(s) cannot fill one {window}-epoch window")
    # Sum, then divide: convolving with weights of 0.1 turns a flat 52.0 into 52.00000000000001,
    # which moves a drop of exactly 2.0 across the strict threshold.
    curve = np.asarray(values, dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(curve, window).mean(axis=1)


def epoch_budget(
    map50_points: Sequence[float],
    default: int,
    window: int = TRAILING_WINDOW,
    drop: float = DROP_POINTS,
) -> BudgetVerdict:
    """ADR-04's trigger over a per-epoch mAP@50 curve in points."""
    means = trailing_means(map50_points, window)
    best = int(np.argmax(means))  # argmax returns the first maximum: ties go to the earliest
    best_last_epoch = best + window - 1
    fired = bool(means[best] - means[-1] > drop)
    return BudgetVerdict(
        budget=best_last_epoch + 1 if fired else default,
        fired=fired,
        best_mean=float(means[best]),
        final_mean=float(means[-1]),
        best_last_epoch=best_last_epoch,
    )


def loss_falls(loss_det: Sequence[float], epochs: int = LOSS_EPOCHS) -> bool:
    """ADR-03's smoke rule: the last ``epochs`` mean below the first ``epochs`` mean."""
    if len(loss_det) < 2 * epochs:
        raise CurveTooShortError(
            f"{len(loss_det)} epoch(s) cannot hold a first and a last {epochs} that do not overlap"
        )
    return float(np.mean(loss_det[-epochs:])) < float(np.mean(loss_det[:epochs]))
