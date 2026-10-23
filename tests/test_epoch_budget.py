"""`t2o.analysis.epoch_budget` -- ADR-04's trigger and ADR-03's loss rule, on hand-built curves."""

from __future__ import annotations

import numpy as np
import pytest

from t2o.analysis.epoch_budget import CurveTooShortError, epoch_budget, loss_falls, trailing_means


def test_trailing_means_cover_full_windows_only() -> None:
    means = trailing_means([0.0, 1.0, 2.0, 3.0], window=2)

    np.testing.assert_allclose(means, [0.5, 1.5, 2.5])


def test_a_curve_that_keeps_rising_keeps_the_default_budget() -> None:
    verdict = epoch_budget(list(np.linspace(10.0, 60.0, 40)), default=400)

    assert not verdict.fired
    assert verdict.budget == 400
    assert verdict.best_last_epoch == 39


def test_a_peak_then_a_fall_cuts_the_budget_to_the_best_window() -> None:
    # 20 epochs at 50, then 20 at 40: the best window is the first, ending at epoch 9 --
    # ties go to the earliest -- and a budget counts epochs, so 10.
    verdict = epoch_budget([50.0] * 20 + [40.0] * 20, default=400)

    assert verdict.fired
    assert verdict.budget == 10
    assert verdict.best_mean == pytest.approx(50.0)
    assert verdict.final_mean == pytest.approx(40.0)


def test_a_drop_of_exactly_the_threshold_does_not_fire() -> None:
    assert not epoch_budget([52.0] * 10 + [50.0] * 10, default=400).fired
    assert epoch_budget([52.01] * 10 + [50.0] * 10, default=400).fired


def test_one_noisy_epoch_cannot_fire_the_trigger_alone() -> None:
    # ADR-04 rejected a raw-peak trigger because a single lucky reading would fire it.
    curve = [50.0] * 30
    curve[12] = 70.0

    assert not epoch_budget(curve, default=400).fired


def test_the_budget_rule_refuses_a_curve_shorter_than_one_window() -> None:
    with pytest.raises(CurveTooShortError):
        epoch_budget([50.0] * 9, default=400)


def test_the_loss_rule_compares_the_last_five_epochs_with_the_first_five() -> None:
    assert loss_falls([5.0] * 5 + [4.0] * 5)
    assert not loss_falls([4.0] * 5 + [5.0] * 5)
    assert not loss_falls([4.0] * 10)  # flat is not falling


def test_the_loss_rule_refuses_windows_that_would_overlap() -> None:
    with pytest.raises(CurveTooShortError):
        loss_falls([5.0] * 9)
