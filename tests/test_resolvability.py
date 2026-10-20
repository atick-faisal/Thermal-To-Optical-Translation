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
    paired_image_bootstrap,
    resolves,
)
from t2o.metrics.task import ImageStats, map50_from_stats

POOL_SIZE = 40
RESAMPLES = 200


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
