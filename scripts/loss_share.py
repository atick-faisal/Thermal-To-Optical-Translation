"""How much of the generator's objective was the detection term actually worth?

E3 came back null on `zero_shot.map50` (TASKS.md M1.2 step 6), and the null is only
interpretable once the detection term's size is known. The effective lambda_det is **not**
`coupling.task_weights`: `translators/pix2pix.py::fit` adds `task_weight * detection` to the
total while `coupling/detection_loss.py::DetectionTaskLoss.forward` has already multiplied by
`grad_scale`. At E3's `grad_scale: 1.0e-2` the real ramp was 0.01/0.02/0.03, against
`l2: 1.0` + `lpips: 5.0` + `gan: 1.0`.

So this reports, per stage, the detection term's share of the objective the optimiser actually
saw -- `task_weight * loss_det / loss_total`, pooled over epochs and over a campaign's loop
runs. `loss_det` as recorded is already post-`grad_scale` (`fit` stores the value it was
handed), so the share needs no reconstruction of the weight chain; `grad_scale` is read from the
config snapshot for the report line only.

Nothing here is available any other way. `analysis/aggregate.py::metric_value` walks dicts and
`epochs` is a list, so `t2o aggregate --metric` cannot reach these; and W&B is no help either --
the step bug fixed in `accfe56` dropped exactly the stages that have a detection term at all.

Every figure is a mean over epochs, so two runs of different length are not comparable even
at identical settings: the opening epochs are the loud ones, and a mean over the first 25 sits
above a mean over 100 in every term at once. `--first-epochs` truncates the pool so a probe can
be read against a campaign; the `epochs` column exists so that a run of the wrong length cannot
be mistaken for a run in a different condition.

Control runs are skipped rather than pooled in: their `loss_det` does not exist (`fit` only
records the key when `task_weight > 0`), and averaging them into a loop campaign's share would
dilute the number the decision rests on.

`--terms-only` reports a control arm's own composition, with the detection and share columns
empty. That is not a share question but a fidelity one: step 8's finding 9 measured a +0.0097
stage-3 LPIPS cost at evaluation, and deciding whether the same gap exists in *training* loss
needs the control arm's `loss_lpips`/`loss_gan` trajectory, which lives nowhere else. Behind a
flag rather than allowed by default, because the failure it guards is a mis-glob: asking for
`runs/*-control-*` when you meant `runs/*-*` would otherwise report a meaningless 0.0% share.

Standalone script, not part of the `t2o` package -- it owns its own `logging.basicConfig` the
way `t2o/cli.py` does for the package proper.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

import yaml

# Private, and imported rather than reimplemented on purpose: a literal `--runs` path that does
# not exist must raise instead of silently shrinking the campaign, and that rule belongs in one
# place. Same reasoning for `load_run` -- the arm predicate below has to agree with the
# aggregator's, or a run could be a control here and a loop run there.
from t2o.analysis.aggregate import Arm, RunRecord, load_run
from t2o.cli import _expand_run_globs

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

CONFIG_FILENAME = "config.yaml"
FIDELITY_KEYS = ("loss_l2", "loss_lpips", "loss_gan")
DETECTION_KEY = "loss_det"
TOTAL_KEY = "loss_total"


@dataclass(frozen=True, slots=True)
class StageShare:
    """One stage's loss composition, pooled over epochs and over a campaign's loop runs."""

    stage: int
    n_runs: int
    # Epoch counts pooled, one entry per distinct value across the runs. Reported because a
    # short probe and a full campaign are not comparable and nothing else on the row says so:
    # a mean over the first 25 epochs sits above a mean over 100 in every term at once, which
    # reads as a changed condition rather than a shorter one (M1.2 step 8).
    n_epochs: tuple[int, ...]
    task_weight: float
    effective_lambda: float
    means: dict[str, float]
    detection_term: float  # task_weight * mean(loss_det) -- what the total actually carried
    share: float  # detection_term / mean(loss_total); nan if the total is zero


def epoch_means(stage: dict[str, Any], first_epochs: int | None = None) -> dict[str, float]:
    """Mean over a stage's epochs of every loss key its `train_losses` recorded.

    Keys are whatever `translator.fit()` returned, so a term the config disabled is simply
    absent rather than zero -- and `loss_det` is absent from every stage at weight 0.

    `first_epochs` truncates to the stage's opening epochs, which is what makes a short probe
    comparable to a full-length campaign run rather than merely adjacent to one.
    """
    epochs = (stage.get("epochs") or [])[:first_epochs]
    keys = {key for epoch in epochs for key in epoch.get("train_losses", {})}
    return {
        key: fmean([epoch["train_losses"][key] for epoch in epochs if key in epoch["train_losses"]])
        for key in sorted(keys)
    }


def grad_scale_of(run: RunRecord) -> float:
    """`coupling.grad_scale` from the run's own config snapshot, read as plain YAML.

    Never through `Config.load`: M1's snapshots carry a `train.workers` key that the current
    `extra="forbid"` schema rejects, which is the same reason `load_run` reads YAML directly.
    """
    snapshot = yaml.safe_load((run.path / CONFIG_FILENAME).read_text())
    return float(snapshot["coupling"]["grad_scale"])


def stage_shares(runs: Sequence[RunRecord], first_epochs: int | None = None) -> list[StageShare]:
    per_stage: dict[int, list[tuple[RunRecord, dict[str, Any]]]] = {}
    for run in runs:
        for stage in run.stages:
            per_stage.setdefault(int(stage["stage"]), []).append((run, stage))

    shares: list[StageShare] = []
    for stage_index in sorted(per_stage):
        entries = per_stage[stage_index]
        weights = {float(stage["task_weight"]) for _, stage in entries}
        if len(weights) > 1:
            # Runs of one campaign share a ramp; more than one weight here means --runs pulled
            # in two different experiments, and pooling them would average unlike conditions.
            raise SystemExit(
                f"stage {stage_index}: runs disagree on task_weight ({sorted(weights)}) -- "
                "--runs matched more than one experiment"
            )
        task_weight = weights.pop()
        scales = {grad_scale_of(run) for run, _ in entries}
        if len(scales) > 1:
            raise SystemExit(f"stage {stage_index}: runs disagree on grad_scale ({sorted(scales)})")

        per_run = [epoch_means(stage, first_epochs) for _, stage in entries]
        n_epochs = tuple(
            sorted({len((stage.get("epochs") or [])[:first_epochs]) for _, stage in entries})
        )
        keys = {key for means in per_run for key in means}
        means = {
            key: fmean([means[key] for means in per_run if key in means]) for key in sorted(keys)
        }
        detection_term = task_weight * means.get(DETECTION_KEY, 0.0)
        total = means.get(TOTAL_KEY, 0.0)
        if not total:
            logger.warning("stage %d: mean %s is zero; share is undefined", stage_index, TOTAL_KEY)
        shares.append(
            StageShare(
                stage=stage_index,
                n_runs=len(entries),
                n_epochs=n_epochs,
                task_weight=task_weight,
                effective_lambda=task_weight * scales.pop(),
                means=means,
                detection_term=detection_term,
                share=detection_term / total if total else float("nan"),
            )
        )
    return shares


def _format_epochs(counts: tuple[int, ...]) -> str:
    """`25`, or `25-100` when the pooled runs were not all the same length."""
    if not counts:
        return "0"
    return str(counts[0]) if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"


def report(shares: Sequence[StageShare]) -> None:
    header = (
        f"{'stage':>5} {'runs':>4} {'epochs':>7} {'w':>5} {'lambda_eff':>10} "
        + " ".join(f"{key:>11}" for key in (*FIDELITY_KEYS, DETECTION_KEY, TOTAL_KEY))
        + f" {'w*det':>9} {'share':>7}"
    )
    logger.info(header)
    for share in shares:
        cells = " ".join(
            f"{share.means[key]:>11.4f}" if key in share.means else f"{'--':>11}"
            for key in (*FIDELITY_KEYS, DETECTION_KEY, TOTAL_KEY)
        )
        logger.info(
            "%5d %4d %7s %5.1f %10.4f %s %9.4f %6.1f%%",
            share.stage,
            share.n_runs,
            _format_epochs(share.n_epochs),
            share.task_weight,
            share.effective_lambda,
            cells,
            share.detection_term,
            100.0 * share.share,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="run directories, or globs over them -- quote the glob ('runs/e3-*') so it "
        "reaches this process unexpanded",
    )
    parser.add_argument(
        "--terms-only",
        action="store_true",
        help="report per-term means for runs with no detection term at all (a control arm); "
        "the detection and share columns come back empty",
    )
    parser.add_argument(
        "--first-epochs",
        type=int,
        default=None,
        help="pool only each stage's first N epochs, so a short probe can be compared against "
        "a full-length campaign run on equal terms",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    runs = [load_run(path) for path in _expand_run_globs(args.runs)]
    if args.terms_only:
        # Every matched run, whichever arm: the question is what the objective was made of,
        # and a control arm answers it for the terms it does have.
        logger.info("--terms-only: reporting %d run(s) with no detection share", len(runs))
        report(stage_shares(runs, args.first_epochs))
        return 0

    loop_runs = [run for run in runs if run.arm is Arm.LOOP]
    if not loop_runs:
        raise SystemExit(
            f"none of the {len(runs)} matched run(s) is a loop arm; a control run records no "
            f"{DETECTION_KEY} at any stage, so there is no share to report. Pass --terms-only "
            "to report its fidelity terms instead."
        )
    if len(loop_runs) < len(runs):
        logger.info("skipped %d control run(s)", len(runs) - len(loop_runs))

    report(stage_shares(loop_runs, args.first_epochs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
