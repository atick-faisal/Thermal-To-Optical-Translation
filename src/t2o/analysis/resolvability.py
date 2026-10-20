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
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from t2o.metrics.task import ImageStats, map50_from_stats

logger = logging.getLogger(__name__)

# `map50_from_stats` returns a fraction; goal.md's "+2" is in mAP points.
_POINTS_PER_UNIT = 100.0


class UnpairedImagesError(ValueError):
    """Raised when two arms do not cover the same images with the same labels."""


@dataclass(frozen=True, slots=True)
class PairedInterval:
    """Percentile CI of the paired mAP@50 delta (candidate minus baseline), in mAP points."""

    low: float
    high: float
    half_width: float


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


def resolves(half_width: float, margin: float = 2.0) -> bool:
    """Whether an interval this narrow could tell a true +`margin` gain from zero (Q3).

    If the true delta were +`margin`, an interval of half-width at most `margin` around it
    would exclude zero. Both arguments are in mAP points.
    """
    return half_width <= margin
