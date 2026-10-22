"""The paired image bootstrap behind the margin gate (margin-gate plan, Q2 and Q3).

Every arm here is hand-built with real true positives. The untrained fixture detector scores
mAP@50 0.0 on any labels, so an interval bootstrapped from it is always [0, 0] and a seed or
units test against it would pass without testing anything (margin-gate pitfalls).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from t2o.analysis.resolvability import (
    UnpairedImagesError,
    UnpairedSeedsError,
    non_inferior,
    paired_image_bootstrap,
    resolves,
    seed_averaged_bootstrap,
)
from t2o.metrics.task import ImageStats, map50_from_stats

POOL_SIZE = 40
RESAMPLES = 200
SEEDS = (0, 1, 3)  # ModTr's campaign seeds; not contiguous, so nothing may assume 0..n-1


def _arm(hit_rate: float, seed: int) -> dict[str, ImageStats]:
    """One target per image, found with probability `hit_rate`, plus one false positive.

    Image `i`'s target class is `i % 2` whatever the seed, so two arms built here always share
    their labels and differ only in what the detector found.
    """
    rng = np.random.default_rng(seed)
    arm: dict[str, ImageStats] = {}
    for i in range(POOL_SIZE):
        stem = f"img{i:02d}"
        cls = float(i % 2)
        hit = bool(rng.random() < hit_rate)
        arm[stem] = ImageStats(
            stem=stem,
            tp=np.repeat(np.array([[hit], [False]]), 10, axis=1),
            conf=rng.uniform(0.0, 1.0, size=2),
            pred_cls=np.array([cls, cls]),
            target_cls=np.array([cls]),
        )
    return arm


def test_identical_arms_give_a_zero_width_interval() -> None:
    arm = _arm(hit_rate=0.7, seed=0)
    assert map50_from_stats(list(arm.values())) > 0.0  # not the vacuous all-zero case

    interval = paired_image_bootstrap(arm, arm, resamples=RESAMPLES)

    assert (interval.low, interval.high, interval.half_width) == (0.0, 0.0, 0.0)


def test_a_fixed_seed_reproduces_the_interval_and_another_seed_moves_it() -> None:
    candidate, baseline = _arm(hit_rate=0.8, seed=1), _arm(hit_rate=0.5, seed=2)

    first = paired_image_bootstrap(candidate, baseline, resamples=RESAMPLES, seed=0)

    assert first == paired_image_bootstrap(candidate, baseline, resamples=RESAMPLES, seed=0)
    assert first != paired_image_bootstrap(candidate, baseline, resamples=RESAMPLES, seed=1)


def test_the_draw_does_not_depend_on_record_order() -> None:
    candidate, baseline = _arm(hit_rate=0.8, seed=1), _arm(hit_rate=0.5, seed=2)
    reversed_candidate = dict(reversed(candidate.items()))

    assert paired_image_bootstrap(
        reversed_candidate, baseline, resamples=RESAMPLES
    ) == paired_image_bootstrap(candidate, baseline, resamples=RESAMPLES)


def test_the_interval_is_in_map_points_and_brackets_the_pool_delta() -> None:
    candidate, baseline = _arm(hit_rate=0.9, seed=3), _arm(hit_rate=0.3, seed=4)
    pool_delta = map50_from_stats(list(candidate.values())) - map50_from_stats(
        list(baseline.values())
    )

    interval = paired_image_bootstrap(candidate, baseline, resamples=RESAMPLES)

    # A fraction-valued interval could never bracket 100x the pool delta, nor reach past 1.
    assert interval.low < 100 * pool_delta < interval.high
    assert interval.high > 1.0
    assert interval.half_width == pytest.approx((interval.high - interval.low) / 2)


def test_arms_over_different_images_are_refused() -> None:
    candidate, baseline = _arm(hit_rate=0.8, seed=1), _arm(hit_rate=0.5, seed=2)
    del baseline["img07"]

    with pytest.raises(UnpairedImagesError, match="1 only in the candidate"):
        paired_image_bootstrap(candidate, baseline, resamples=RESAMPLES)


def test_arms_scored_against_different_labels_are_refused() -> None:
    candidate, baseline = _arm(hit_rate=0.8, seed=1), _arm(hit_rate=0.5, seed=2)
    baseline["img07"] = dataclasses.replace(baseline["img07"], target_cls=np.array([0.0, 1.0]))

    with pytest.raises(UnpairedImagesError, match="different target classes"):
        paired_image_bootstrap(candidate, baseline, resamples=RESAMPLES)


def test_resolves_at_and_below_the_margin_only() -> None:
    assert resolves(2.0)
    assert not resolves(2.01)
    assert resolves(2.5, margin=3.0)


def _seeded(hit_rate: float, offset: int) -> dict[int, dict[str, ImageStats]]:
    """One arm per training seed, each a different draw of detections over the same labels."""
    return {seed: _arm(hit_rate=hit_rate, seed=offset + seed) for seed in SEEDS}


def test_identical_multi_seed_arms_give_a_zero_interval_and_pass() -> None:
    arms = _seeded(hit_rate=0.7, offset=0)

    interval = seed_averaged_bootstrap(arms, arms, resamples=RESAMPLES)

    assert (interval.low, interval.high, interval.half_width) == (0.0, 0.0, 0.0)
    assert interval.seed_deltas == dict.fromkeys(SEEDS, 0.0)
    assert interval.seed_sd == 0.0
    assert non_inferior(interval.low)


def test_one_seed_draws_the_whole_pool_as_the_gate_would_at_n_equal_pool() -> None:
    """Pins n-out-of-n: the gate's 100 was the test split's size, not the val pool's."""
    candidate, baseline = _arm(hit_rate=0.8, seed=1), _arm(hit_rate=0.5, seed=2)

    interval = seed_averaged_bootstrap({0: candidate}, {0: baseline}, resamples=RESAMPLES)
    gate = paired_image_bootstrap(candidate, baseline, n=POOL_SIZE, resamples=RESAMPLES)

    assert (interval.low, interval.high) == (gate.low, gate.high)
    assert np.isnan(interval.seed_sd)  # one seed has no spread to report


def test_a_fixed_seed_reproduces_the_seed_averaged_interval_and_another_moves_it() -> None:
    candidate, baseline = _seeded(hit_rate=0.8, offset=10), _seeded(hit_rate=0.5, offset=20)

    first = seed_averaged_bootstrap(candidate, baseline, resamples=RESAMPLES, seed=0)

    assert first == seed_averaged_bootstrap(candidate, baseline, resamples=RESAMPLES, seed=0)
    assert first != seed_averaged_bootstrap(candidate, baseline, resamples=RESAMPLES, seed=1)


def test_per_seed_deltas_are_pool_deltas_in_points_and_the_ci_brackets_their_mean() -> None:
    candidate, baseline = _seeded(hit_rate=0.8, offset=10), _seeded(hit_rate=0.5, offset=20)

    interval = seed_averaged_bootstrap(candidate, baseline, resamples=RESAMPLES)

    expected = {
        seed: 100
        * (
            map50_from_stats(list(candidate[seed].values()))
            - map50_from_stats(list(baseline[seed].values()))
        )
        for seed in SEEDS
    }
    assert interval.seed_deltas == pytest.approx(expected)
    assert interval.delta == pytest.approx(np.mean(list(expected.values())))
    assert interval.seed_sd == pytest.approx(np.std(list(expected.values()), ddof=1))
    assert interval.low < interval.delta < interval.high


def test_a_modtr_far_ahead_fails_and_a_loop_far_ahead_passes() -> None:
    weak, strong = _seeded(hit_rate=0.3, offset=10), _seeded(hit_rate=0.9, offset=20)

    behind = seed_averaged_bootstrap(weak, strong, resamples=RESAMPLES)
    ahead = seed_averaged_bootstrap(strong, weak, resamples=RESAMPLES)

    assert behind.high < -5.0  # clear of δ, so the verdict cannot hinge on bootstrap noise
    assert not non_inferior(behind.low)
    assert ahead.low > 0.0
    assert non_inferior(ahead.low)


def test_arms_over_different_seeds_are_refused() -> None:
    candidate, baseline = _seeded(hit_rate=0.8, offset=10), _seeded(hit_rate=0.5, offset=20)
    del baseline[3]

    with pytest.raises(UnpairedSeedsError, match="different seeds"):
        seed_averaged_bootstrap(candidate, baseline, resamples=RESAMPLES)


def test_seeds_over_different_images_are_refused_even_when_each_seed_is_paired() -> None:
    """One draw indexes every seed, so seed 1 losing an image on both arms still breaks it."""
    candidate, baseline = _seeded(hit_rate=0.8, offset=10), _seeded(hit_rate=0.5, offset=20)
    del candidate[1]["img07"], baseline[1]["img07"]

    with pytest.raises(UnpairedImagesError, match="seed 1 against seed 0"):
        seed_averaged_bootstrap(candidate, baseline, resamples=RESAMPLES)


def test_non_inferior_strictly_above_minus_delta_only() -> None:
    assert non_inferior(-4.99)
    assert not non_inferior(-5.0)  # ADR-02: "greater than -δ", so the boundary itself fails
    assert not non_inferior(-5.01)
    assert non_inferior(-2.4, delta=2.5)
    assert not non_inferior(-2.6, delta=2.5)
