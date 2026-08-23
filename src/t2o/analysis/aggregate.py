"""Read a whole campaign of runs as one experiment (TASKS.md M1.2 step 4).

Nothing else in the repo reads more than one run: ``engine/loop.py`` writes one
``metrics.json`` per run and that is where the data stops. E3 needs the opposite -- twelve
runs (six seeds x {control, loop}) read as a single paired comparison -- because M1.2's gate
re-score put the run-to-run noise floor at 0.059 mAP50, the same order as the lambda_det
effect being claimed. At n = 1 the two lambda = 0 comparators disagreed by more than the
effect itself.

**Runs are joined to their sibling ``config.yaml`` snapshot, and that snapshot is parsed as
plain YAML.** ``metrics.json`` records neither the seed nor the config hash, so the seed and
the arm have to come from somewhere; changing ``metrics.json``'s format would be the obvious
alternative and would strand the two M1 server runs that already exist. Parsing the snapshot
through :meth:`Config.load` is *also* ruled out: M1.2 step 2b moved ``workers`` from
``TrainConfig`` to ``RuntimeConfig``, so those older snapshots carry a ``train.workers`` key
that ``extra="forbid"`` now rejects. Reading the two keys this module needs out of raw YAML
is what keeps every run ever produced readable.

The statistics are deliberately assumption-free. The comparison is paired (seed *i*'s control
against seed *i*'s loop), so the exact test on it is a sign-flip permutation over all 2**n
assignments -- no scipy, no normality assumption, and no minimum-n fudge. Its smallest
attainable two-sided p is 2/2**n, which is why E3 runs six seeds: n = 6 is the first size
where a distribution-free two-sided test can clear 0.05 at all (0.031), whatever the effect
size.
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from t2o.config.schema import CONFIG_FILENAME
from t2o.engine.loop import METRICS_FILENAME

logger = logging.getLogger(__name__)

# 2**20 = ~1M sign assignments, about a second. Past that a mis-globbed --runs would hang
# rather than fail, which is the failure mode this cap exists to convert into a message.
MAX_EXACT_PERMUTATION_N = 20


class AggregationError(ValueError):
    """Raised for a malformed run directory, a missing metric, or an unpaired campaign."""


class Arm(StrEnum):
    """Which side of the ablation a run is on.

    Derived from ``coupling.task_weights``, never read from a field, using the *same*
    predicate as ``tracking.py::run_tags``' ``lambda_det:on|off`` tag -- so a run's arm here
    and its W&B tag cannot disagree. Derived from the config snapshot rather than from
    ``metrics.json`` for a concrete reason: the loop arm's stage 0 is itself lambda = 0, so a
    run that has only finished stage 0 has an all-zero ``task_weight`` column recorded and
    would be misfiled as a control.
    """

    CONTROL = "control"
    LOOP = "loop"


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One finished (or in-progress) run: who it is, and its raw per-stage metric records."""

    path: Path
    name: str
    seed: int
    arm: Arm
    task_weights: tuple[float, ...]
    # Left as the raw JSON `metrics.json` entries rather than rebuilt into `StageResult`s.
    # This module only ever pulls leaf floats out of them, so reconstructing
    # DetectorResult/TaskMetrics/FidelityMetrics would couple the aggregator to
    # engine/loop.py's dataclass shape and buy nothing.
    stages: tuple[dict[str, Any], ...]

    @property
    def stage_indices(self) -> tuple[int, ...]:
        return tuple(int(stage["stage"]) for stage in self.stages)


@dataclass(frozen=True, slots=True)
class ArmSummary:
    """One arm's spread at one stage, for one metric."""

    arm: Arm
    stage: int
    metric: str
    n: int
    mean: float
    std: float  # sample std (ddof=1); nan at n = 1, where spread is undefined rather than 0
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PairedResult:
    """The paired loop-minus-control comparison at one stage, for one metric."""

    stage: int
    metric: str
    seeds: tuple[int, ...]
    differences: tuple[float, ...]  # loop - control, in `seeds` order
    mean_difference: float
    p_value: float
    ci_low: float
    ci_high: float

    @property
    def n(self) -> int:
        return len(self.differences)


@dataclass(frozen=True, slots=True)
class TrajectoryResult:
    """Paired difference-of-differences: how much more one arm *gained* than the other.

    The stage-N paired difference asks "how far apart are the arms at the finish line".
    This asks "how much further did each arm travel", by subtracting each arm's own
    baseline-stage value from its own stage-N value before pairing. The two answer the same
    question only when the arms start level -- and stage 0 is a *measurement* of run-to-run
    noise, not a guarantee of levelness, so when that draw comes out wide the finish-line
    contrast inherits an offset this one is immune to by construction.

    Not a new hypothesis: ``aggregate``'s own decision rule already directs stage N to be
    read against stage 0 ("if the stage-3 effect is not clearly larger than the stage-0
    difference, E3 is negative"). This is that rule's arithmetic, made exact instead of
    eyeballed. It is a sensitivity analysis and must be reported as one -- the pre-registered
    endpoint stays the paired stage-N difference (TASKS.md M1.2 step 4, step 8).

    Algebraically ``differences[i]`` is the stage-N paired difference minus the baseline
    paired difference for that seed, which is how it is computed -- so it is exactly the
    contrast the rule names, not an approximation of it.
    """

    stage: int
    baseline_stage: int
    metric: str
    seeds: tuple[int, ...]
    differences: tuple[float, ...]  # (loop gain) - (control gain), in `seeds` order
    mean_difference: float
    p_value: float
    ci_low: float
    ci_high: float
    # Each arm's own mean baseline -> stage-N movement. Printed beside the contrast because
    # a difference of gains is unreadable without knowing whether both arms rose.
    control_gain: float
    loop_gain: float

    @property
    def n(self) -> int:
        return len(self.differences)


@dataclass(frozen=True, slots=True)
class AggregateReport:
    """Everything one `t2o aggregate` invocation computed."""

    runs: tuple[RunRecord, ...]
    stages: tuple[int, ...]
    metrics: tuple[str, ...]
    summaries: tuple[ArmSummary, ...]
    paired: tuple[PairedResult, ...]
    # Empty when only one stage is common to every run: a gain needs two points.
    trajectory: tuple[TrajectoryResult, ...] = ()


def load_run(run_dir: Path | str) -> RunRecord:
    """Read one run directory's ``metrics.json`` and its sibling ``config.yaml`` snapshot."""
    path = Path(run_dir)
    metrics_path = path / METRICS_FILENAME
    config_path = path / CONFIG_FILENAME
    for required in (metrics_path, config_path):
        if not required.is_file():
            raise AggregationError(
                f"{path} is not a run directory: expected {required.name} in it. "
                f"A run written by `t2o loop` always has both {METRICS_FILENAME} and "
                f"{CONFIG_FILENAME}."
            )

    stages = json.loads(metrics_path.read_text())
    # Plain YAML, never Config.load -- see the module docstring. Only two keys are read, and
    # both have been in the schema since M0.2.
    snapshot = yaml.safe_load(config_path.read_text())
    try:
        seed = int(snapshot["train"]["seed"])
        task_weights = tuple(float(weight) for weight in snapshot["coupling"]["task_weights"])
    except (KeyError, TypeError) as error:
        raise AggregationError(
            f"{config_path} is missing train.seed or coupling.task_weights; it does not look "
            "like a config snapshot written by Config.snapshot()"
        ) from error

    arm = Arm.LOOP if any(weight > 0.0 for weight in task_weights) else Arm.CONTROL
    return RunRecord(
        path=path,
        name=snapshot.get("runtime", {}).get("name", path.name),
        seed=seed,
        arm=arm,
        task_weights=task_weights,
        stages=tuple(stages),
    )


def metric_value(stage: dict[str, Any], metric: str) -> float:
    """Pull a dotted metric out of one stage record, e.g. ``zero_shot.map50``.

    A ``None`` anywhere along the path raises rather than returning a sentinel: it means the
    stage never computed that arm (``--no-detector`` skips export, the zero-shot pass and
    fidelity together), and a hole in one cell of a paired comparison must not be papered
    over into a number.
    """
    node: Any = stage
    for key in metric.split("."):
        if not isinstance(node, dict) or key not in node:
            raise AggregationError(
                f"stage {stage.get('stage')} has no metric '{metric}' (failed at '{key}'). "
                "Available top-level keys: " + ", ".join(sorted(stage))
            )
        node = node[key]
        if node is None:
            raise AggregationError(
                f"stage {stage.get('stage')} recorded no '{metric}' -- '{key}' is null. "
                "A run made with --no-detector computes neither the zero-shot nor the "
                "fidelity arm."
            )
    return float(node)


def pair_runs(runs: Sequence[RunRecord]) -> dict[int, tuple[RunRecord, RunRecord]]:
    """Match each seed's control run to its loop run, refusing anything unpaired.

    Both failure modes here are real launch mistakes rather than defensive padding: a seed
    whose second arm crashed silently halves the campaign, and two runs sharing an ``(arm,
    seed)`` cell is what a forgotten ``--name`` produces (``runs/<name>`` has no seed
    component, so the second launch overwrites the first). Dropping either quietly would
    produce a p-value from a comparison that was never actually paired.
    """
    by_cell: dict[tuple[Arm, int], RunRecord] = {}
    for run in runs:
        cell = (run.arm, run.seed)
        if cell in by_cell:
            raise AggregationError(
                f"two runs claim arm={run.arm} seed={run.seed}: {by_cell[cell].path} and "
                f"{run.path}. runs/<name> has no seed component, so a launch that forgot "
                "--name overwrites the previous seed's outputs."
            )
        by_cell[cell] = run

    control_seeds = {seed for arm, seed in by_cell if arm is Arm.CONTROL}
    loop_seeds = {seed for arm, seed in by_cell if arm is Arm.LOOP}
    unpaired = control_seeds ^ loop_seeds
    if unpaired:
        raise AggregationError(
            f"seeds {sorted(unpaired)} appear in only one arm; the paired test needs every "
            f"seed in both. control={sorted(control_seeds)} loop={sorted(loop_seeds)}"
        )
    if not control_seeds:
        raise AggregationError("no control/loop pairs found in the given runs")

    return {
        seed: (by_cell[(Arm.CONTROL, seed)], by_cell[(Arm.LOOP, seed)])
        for seed in sorted(control_seeds)
    }


def sign_flip_p_value(differences: Sequence[float]) -> float:
    """Exact two-sided sign-flip permutation p-value on paired differences.

    Under the null the sign of each paired difference is arbitrary, so enumerating all 2**n
    sign assignments gives the exact null distribution of the mean -- no distributional
    assumption, and no approximation to check. p is the fraction of assignments whose |mean|
    reaches the observed |mean|; the identity assignment is always among them, so p is never
    below 2/2**n (both it and its full negation always count).
    """
    n = len(differences)
    if n == 0:
        raise AggregationError("cannot test an empty set of paired differences")
    if n > MAX_EXACT_PERMUTATION_N:
        raise AggregationError(
            f"exact enumeration needs 2**{n} sign assignments; refusing above "
            f"n={MAX_EXACT_PERMUTATION_N}. Check that --runs matched only the intended runs."
        )

    values = np.asarray(differences, dtype=float)
    observed = abs(float(values.mean()))
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    means = np.abs(signs @ values / n)
    # >= with a tolerance, not >: floating-point sums make the identity assignment's own mean
    # differ from `observed` in the last bits, which would otherwise exclude it and let p
    # dip below its true floor of 2/2**n.
    return float(np.count_nonzero(means >= observed - 1e-12) / means.size)


def bootstrap_ci(
    differences: Sequence[float],
    resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean paired difference.

    Seeded so a reported interval is reproducible -- an interval that moves between two
    readings of the same finished runs is not something to paste into a paper.
    """
    if not differences:
        raise AggregationError("cannot bootstrap an empty set of paired differences")
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def common_stages(runs: Sequence[RunRecord]) -> tuple[int, ...]:
    """Stage indices present in *every* run -- the only ones a paired test is defined on."""
    if not runs:
        return ()
    shared = set(runs[0].stage_indices)
    for run in runs[1:]:
        shared &= set(run.stage_indices)
    return tuple(sorted(shared))


def aggregate(
    run_dirs: Iterable[Path | str],
    metrics: Sequence[str] = ("zero_shot.map50",),
    stages: Sequence[int] | None = None,
    resamples: int = 10000,
    seed: int = 0,
) -> AggregateReport:
    """Load every run, summarise each arm, and run the paired test at every stage.

    ``stages=None`` uses every stage present in all runs, which is what makes stage 0's null
    control (both arms are lambda = 0 there) come out beside stage 3 for free rather than
    needing a flag of its own. If the stage-3 effect is not clearly larger than the stage-0
    difference, E3 is negative -- so the two are only useful side by side.

    Every stage above the baseline also gets a :class:`TrajectoryResult`, which applies the
    same test to each arm's own movement away from that baseline. When the stage-0 draw comes
    out level the two contrasts agree; when it does not, the trajectory is the one that still
    means what the rule intended. See that class for why it is a sensitivity analysis rather
    than a second endpoint.
    """
    runs = tuple(load_run(run_dir) for run_dir in run_dirs)
    if not runs:
        raise AggregationError("no run directories given")
    pairs = pair_runs(runs)

    stage_indices = tuple(stages) if stages is not None else common_stages(runs)
    if not stage_indices:
        raise AggregationError(
            "the given runs share no stage index; they are at different points in the "
            "campaign, or one of them recorded no stages at all"
        )

    summaries: list[ArmSummary] = []
    paired: list[PairedResult] = []
    trajectory: list[TrajectoryResult] = []
    # The earliest stage every run shares, not a hardcoded 0: `stages=[1, 2, 3]` is a legal
    # request, and a gain is only defined against a baseline that is actually present.
    baseline_stage = stage_indices[0]
    for metric in metrics:
        arm_means: dict[tuple[Arm, int], float] = {}
        differences_by_stage: dict[int, tuple[float, ...]] = {}
        for stage in stage_indices:
            for arm in (Arm.CONTROL, Arm.LOOP):
                values = tuple(_stage_metric(run, stage, metric) for run in runs if run.arm is arm)
                arm_means[(arm, stage)] = float(np.mean(values))
                summaries.append(
                    ArmSummary(
                        arm=arm,
                        stage=stage,
                        metric=metric,
                        n=len(values),
                        mean=float(np.mean(values)),
                        std=float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
                        values=values,
                    )
                )

            seeds = tuple(pairs)
            differences = tuple(
                _stage_metric(loop, stage, metric) - _stage_metric(control, stage, metric)
                for control, loop in pairs.values()
            )
            differences_by_stage[stage] = differences
            low, high = bootstrap_ci(differences, resamples=resamples, seed=seed)
            paired.append(
                PairedResult(
                    stage=stage,
                    metric=metric,
                    seeds=seeds,
                    differences=differences,
                    mean_difference=float(np.mean(differences)),
                    p_value=sign_flip_p_value(differences),
                    ci_low=low,
                    ci_high=high,
                )
            )

        # Each seed's stage-N paired difference minus its own baseline paired difference --
        # identically (loop gain) - (control gain), since the two control terms cancel.
        # Computed this way rather than from four raw values so it cannot drift from the
        # paired block above.
        baseline_differences = differences_by_stage[baseline_stage]
        for stage in stage_indices[1:]:
            gains = tuple(
                later - baseline
                for later, baseline in zip(
                    differences_by_stage[stage], baseline_differences, strict=True
                )
            )
            low, high = bootstrap_ci(gains, resamples=resamples, seed=seed)
            trajectory.append(
                TrajectoryResult(
                    stage=stage,
                    baseline_stage=baseline_stage,
                    metric=metric,
                    seeds=tuple(pairs),
                    differences=gains,
                    mean_difference=float(np.mean(gains)),
                    p_value=sign_flip_p_value(gains),
                    ci_low=low,
                    ci_high=high,
                    control_gain=arm_means[(Arm.CONTROL, stage)]
                    - arm_means[(Arm.CONTROL, baseline_stage)],
                    loop_gain=arm_means[(Arm.LOOP, stage)] - arm_means[(Arm.LOOP, baseline_stage)],
                )
            )

    return AggregateReport(
        runs=runs,
        stages=stage_indices,
        metrics=tuple(metrics),
        summaries=tuple(summaries),
        paired=tuple(paired),
        trajectory=tuple(trajectory),
    )


def _stage_metric(run: RunRecord, stage: int, metric: str) -> float:
    for record in run.stages:
        if int(record["stage"]) == stage:
            return metric_value(record, metric)
    raise AggregationError(f"{run.path} has no stage {stage}")


def tidy_rows(report: AggregateReport) -> list[dict[str, Any]]:
    """One row per (run, seed, arm, stage, metric, value) -- the long format for re-analysis."""
    return [
        {
            "run": run.name,
            "path": str(run.path),
            "seed": run.seed,
            "arm": run.arm.value,
            "stage": stage,
            "task_weight": run.task_weights[stage] if stage < len(run.task_weights) else math.nan,
            "metric": metric,
            "value": _stage_metric(run, stage, metric),
        }
        for run in report.runs
        for stage in report.stages
        for metric in report.metrics
    ]


def write_csv(rows: Sequence[dict[str, Any]], path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination
