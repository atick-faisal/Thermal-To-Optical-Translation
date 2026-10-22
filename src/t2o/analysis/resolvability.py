"""Can a CI at the test split's size resolve +2 mAP@50? (margin-gate plan, Q2 and Q3).

`goal.md`'s Margin criterion wants a 95% bootstrap CI over ~100 held-out test images. Before
the test split is read, the 153 val images say how wide that interval will be: draw 100 of
them with replacement (an m-out-of-n bootstrap), score both arms on the same draw, and read
the spread of the paired delta. If the half-width is above 2 mAP points, +2 cannot be told
apart from zero at that size, and Margin becomes non-inferiority to ModTr.

**This is not `aggregate.bootstrap_ci`.** That one resamples one number per training seed.
mAP is not a sum over images, so there is no per-image number to resample here: each draw
recomputes AP from the matched predictions of the images it drew, through
`metrics.task.map50_from_stats` -- ultralytics' own `ap_per_class`, so the AP math is the one
evaluation path's (PLAN.md invariant 1).

**Non-inferiority to ModTr** (modtr-baseline ADR-02) is what Margin became when the gate did not
resolve. Its interval is `seed_averaged_bootstrap`: one draw of the whole val pool per resample,
each seed's mAP@50 recomputed for both arms on that draw, the delta averaged across seeds. It
draws n-out-of-n, not `paired_image_bootstrap`'s 100: that size was the test split's, and with
no test split the val pool is the reporting unit. `non_inferior` is the pre-registered pass rule.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from t2o.metrics.task import ImageStats, map50_from_stats

logger = logging.getLogger(__name__)

# `map50_from_stats` returns a fraction; goal.md's "+2" is in mAP points.
_POINTS_PER_UNIT = 100.0

# Margin's non-inferiority margin δ, in mAP points, pre-registered by modtr-baseline ADR-02
# before any ModTr number existed. Changing it needs a superseding ADR, not an argument.
NON_INFERIORITY_MARGIN = 5.0


class UnpairedImagesError(ValueError):
    """Raised when two arms do not cover the same images with the same labels."""


class UnpairedSeedsError(ValueError):
    """Raised when two arms were not trained on the same seeds, so no per-seed delta exists."""


@dataclass(frozen=True, slots=True)
class PairedInterval:
    """Percentile CI of the paired mAP@50 delta (candidate minus baseline), in mAP points."""

    low: float
    high: float
    half_width: float


@dataclass(frozen=True, slots=True)
class SeedAveragedInterval:
    """Percentile CI of the seed-averaged mAP@50 delta, with the seed spread it averages over.

    All in mAP points, candidate minus baseline. The CI decides the verdict; the per-seed deltas
    and their sd are reported beside it because the average hides them (ADR-02).
    """

    low: float
    high: float
    half_width: float
    delta: float  # the per-seed deltas' mean: the point estimate the CI is around
    seed_deltas: dict[int, float]  # each seed's delta over the whole pool
    seed_sd: float  # sample sd of `seed_deltas`; nan for one seed, where it does not exist


def _pair(
    candidate: Mapping[str, ImageStats], baseline: Mapping[str, ImageStats]
) -> tuple[list[ImageStats], list[ImageStats]]:
    """Both arms' records in one sorted stem order, refusing anything that is not truly paired.

    Sorted so a fixed seed draws the same images whatever order the val loader produced. The
    target classes are compared because a delta is only paired if both arms were scored
    against the same labels -- a stale `labels.cache` would otherwise change one arm's ground
    truth without a word.
    """
    only_candidate = sorted(candidate.keys() - baseline.keys())
    only_baseline = sorted(baseline.keys() - candidate.keys())
    if only_candidate or only_baseline:
        raise UnpairedImagesError(
            f"the arms score different images: {len(only_candidate)} only in the candidate "
            f"(first: {only_candidate[:3]}), {len(only_baseline)} only in the baseline "
            f"(first: {only_baseline[:3]})"
        )

    stems = sorted(candidate)
    relabelled = [
        stem
        for stem in stems
        if not np.array_equal(
            np.sort(candidate[stem].target_cls), np.sort(baseline[stem].target_cls)
        )
    ]
    if relabelled:
        raise UnpairedImagesError(
            f"{len(relabelled)} images carry different target classes in the two arms "
            f"(first: {relabelled[:3]}); delete each split's labels.cache and re-run the val"
        )
    return [candidate[stem] for stem in stems], [baseline[stem] for stem in stems]


def paired_image_bootstrap(
    candidate: Mapping[str, ImageStats],
    baseline: Mapping[str, ImageStats],
    n: int = 100,
    resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedInterval:
    """CI of the candidate's mAP@50 gain over the baseline, as if `n` images had been scored.

    `n` defaults to the held-out test split's size, which is the point: the interval answers
    how wide Margin's CI will be on ~100 test images, not how wide it is on all of val. Both
    arms score the same draw of images on every resample. Seeded, as `bootstrap_ci` is, so a
    reported interval is reproducible.
    """
    candidate_records, baseline_records = _pair(candidate, baseline)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(candidate_records), size=(resamples, n))
    deltas = np.array(
        [
            map50_from_stats([candidate_records[i] for i in draw])
            - map50_from_stats([baseline_records[i] for i in draw])
            for draw in draws
        ]
    )
    low, high = _POINTS_PER_UNIT * np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    logger.info(
        "paired image bootstrap: %d of %d images x %d resamples, delta CI [%.2f, %.2f] mAP points",
        n,
        len(candidate_records),
        resamples,
        low,
        high,
    )
    return PairedInterval(low=float(low), high=float(high), half_width=float(high - low) / 2)


def _pair_seeds(
    candidate: Mapping[int, Mapping[str, ImageStats]],
    baseline: Mapping[int, Mapping[str, ImageStats]],
) -> dict[int, tuple[list[ImageStats], list[ImageStats]]]:
    """Every seed's two arms in one stem order, all over the same images and labels.

    One draw indexes every seed, so it is not enough for each seed's arms to match each other:
    every arm of every seed is checked against the lowest seed's candidate.
    """
    if candidate.keys() != baseline.keys():
        raise UnpairedSeedsError(
            f"the arms ran different seeds: candidate {sorted(candidate)}, "
            f"baseline {sorted(baseline)}"
        )
    seeds = sorted(candidate)
    reference = candidate[seeds[0]]
    paired: dict[int, tuple[list[ImageStats], list[ImageStats]]] = {}
    for seed in seeds:
        try:
            candidate_records, _ = _pair(candidate[seed], reference)
            baseline_records, _ = _pair(baseline[seed], reference)
        except UnpairedImagesError as error:
            raise UnpairedImagesError(
                f"seed {seed} against seed {seeds[0]}'s candidate: {error}"
            ) from error
        paired[seed] = (candidate_records, baseline_records)
    return paired


def seed_averaged_bootstrap(
    candidate: Mapping[int, Mapping[str, ImageStats]],
    baseline: Mapping[int, Mapping[str, ImageStats]],
    resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> SeedAveragedInterval:
    """CI of the candidate's seed-averaged mAP@50 gain over the baseline (modtr-baseline ADR-02).

    Each arm maps a training seed to its per-image records. Seeds are paired by number, and one
    draw of image indices serves every seed of both arms, so a resample moves all of them
    together. Seeded, so a reported interval is reproducible.
    """
    paired = _pair_seeds(candidate, baseline)
    pool = len(next(iter(paired.values()))[0])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, pool, size=(resamples, pool))
    deltas = np.array(
        [
            np.mean(
                [
                    map50_from_stats([candidate_records[i] for i in draw])
                    - map50_from_stats([baseline_records[i] for i in draw])
                    for candidate_records, baseline_records in paired.values()
                ]
            )
            for draw in draws
        ]
    )
    low, high = _POINTS_PER_UNIT * np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    seed_deltas = {
        s: _POINTS_PER_UNIT
        * (map50_from_stats(candidate_records) - map50_from_stats(baseline_records))
        for s, (candidate_records, baseline_records) in paired.items()
    }
    values = list(seed_deltas.values())
    seed_sd = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
    logger.info(
        "seed-averaged bootstrap: %d seed(s) x %d images x %d resamples, "
        "delta CI [%.2f, %.2f] mAP points",
        len(paired),
        pool,
        resamples,
        low,
        high,
    )
    return SeedAveragedInterval(
        low=float(low),
        high=float(high),
        half_width=float(high - low) / 2,
        delta=float(np.mean(values)),
        seed_deltas=seed_deltas,
        seed_sd=seed_sd,
    )


def non_inferior(lower_bound: float, delta: float = NON_INFERIORITY_MARGIN) -> bool:
    """ADR-02's pass rule: the CI's lower bound is above -`delta`, strictly. In mAP points."""
    return lower_bound > -delta


def resolves(half_width: float, margin: float = 2.0) -> bool:
    """Whether an interval this narrow could tell a true +`margin` gain from zero (Q3).

    If the true delta were +`margin`, an interval of half-width at most `margin` around it
    would exclude zero. Both arguments are in mAP points.
    """
    return half_width <= margin
