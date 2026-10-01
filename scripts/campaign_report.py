"""One campaign of runs -> one block of text you can copy out of a terminal.

`runs/` never comes back from the server (PLAN.md §5: configs go out through git, results do
not come back through it), and on the box that holds the E9 cell *nothing* comes back except
text that can be pasted. Every number this project has recorded from a campaign so far was
read off a terminal by hand across several commands; this script is the single command that
prints all of them at once, so a finished cell costs one paste rather than one round-trip per
question.

**What it adds over running the commands by hand**, each of which is a thing nothing else
reports:

* the **config-variance block** -- the twelve snapshots flattened, printing only the keys that
  *differ*. `train.seed` and `coupling.task_weights` are the two that must; anything else is a
  confound, and `test_control_and_loop_configs_differ_only_by_design` cannot see one that
  arrives as a CLI flag. The resolved flag set of one run is printed beside it, which is how a
  campaign launched without its command written down (M3 E9 step 5) gets a reproducible record
  after the fact.
* the **wall clock**, from `EpochStats.seconds` for the training term and run-directory mtimes
  for the boundary term -- the method M3 E9 step 4 (b) established, which prices a stage
  without the mtime archaeology step 0 needed.
* the **per-run wide table**, the row-level record that makes any later re-analysis possible
  once the run directories are gone (the reasoning behind tracking `docs/results/e8-tidy.csv`).

**The tables it does not render itself.** The arm summaries, the paired sign-flip block, the
trajectory block and the loss shares are printed by `t2o aggregate` and
`scripts/loss_share.py`, called in process. Re-implementing their formatting would create a
second renderer that can disagree with the one a reader reproduces by hand. They log through
`logging`, and `logging.basicConfig` is a no-op once the root logger has a handler -- so
configuring one here, bare-format onto stdout, is what turns their output into report body
instead of timestamped terminal chatter.

Campaign-agnostic on purpose: `--runs` plus `--primary-classes` serves the e3b and e3t cells
and M4's re-reads as well as the FLIR one it was written for.
"""

from __future__ import annotations

import argparse
import logging
import platform
import statistics
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from scripts.loss_share import report as loss_report
from scripts.loss_share import stage_shares
from t2o.analysis.aggregate import (
    AggregationError,
    Arm,
    RunRecord,
    add_primary_mean,
    common_stages,
    load_run,
    metric_is_recorded,
    metric_value,
)
from t2o.cli import _expand_run_globs
from t2o.cli import main as cli_main

logger = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600.0

# Solo, one card, `--val-loss-images 153`, measured by M3 E9 step 4 (b)'s three-run probe. The
# cell itself runs two-up, and the difference between these and the campaign's own medians is
# the contention cost that projection explicitly left unmeasured ("read 77-103 GPU-h as a
# floor, not a forecast").
PROBE_EPOCH_SECONDS = {Arm.CONTROL: 43.524, Arm.LOOP: 51.156}

# Keys that MUST differ run to run, or are machine-specific by design. Everything else varying
# is a confound: `test_control_and_loop_configs_differ_only_by_design` guards the two config
# files, and nothing guards a flag that arrived differently on one of the two launch shells.
# `runtime.device` is here because the cell runs two-up by seed -- the pairing invariant it
# has to satisfy is checked separately below, not by being constant.
EXPECTED_VARYING = frozenset(
    {
        "train.seed",
        "coupling.task_weights",
        "runtime.name",
        "runtime.run_dir",
        "runtime.device",
    }
)

# Printed verbatim at the end so the numbers are read against what was written before them.
# Every figure here predates the campaign; none is derived from it.
PRE_REGISTERED = """\
primary endpoint   paired stage-3 difference in the 3-class mAP50 (bicycle/car/person),
                   exact two-sided sign-flip p over 2^6 assignments. Never a bootstrap CI
                   excluding zero (PLAN.md §12); CIs are descriptive spread only.
null control       the stage-0 paired difference -- lambda = 0 in BOTH arms there.
dog                13 val instances. Noise in both arms (gate: 0.2396 ceiling / 0.0260 floor).
                   Stated separately, never in the headline.
bicycle            predicted the WEAK class: gate headroom +0.1295, below the 0.15 kill line
                   on its own. A gain concentrated in car/person was predicted, not observed.
person             predicted the STRONGEST: gate headroom +0.2932.
noise floor        0.059 mAP50 (custom set, M1.2) and 0.048 (FLIR's own A/B probe pair).
                   An effect below these is not an effect.
headroom           the cell is read against +0.2066 3-class (ceiling 0.6566 / floor 0.4499),
                   a WEAK PASS with +0.0566 of margin. Custom set was +0.733.
cost               77-103 GPU-h was a FLOOR: measured solo, stage 0 only, +20%/stage carried.
confound           the 5.90 px roll still sits between input and the l2+lpips target. De-roll
                   fixed the LABELS, buying an honest floor, not an aligned training set.
                   This confound exists on the public cell and nowhere else.
"""


def _run(command: Sequence[str]) -> str:
    """Best-effort provenance. A box without git or nvidia-smi still gets a report."""
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return done.stdout.strip() or done.stderr.strip() or "unavailable"


def _flatten(node: Any, prefix: str = "") -> dict[str, str]:
    """A config snapshot as `dotted.key -> repr`, so two runs can be diffed key by key."""
    if not isinstance(node, dict):
        return {prefix: repr(node)}
    flat: dict[str, str] = {}
    for key, value in node.items():
        flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return flat


def _record(run: RunRecord, stage: int) -> dict[str, Any]:
    return next(record for record in run.stages if int(record["stage"]) == stage)


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n== {title}\n{'=' * 78}")


def _provenance(runs: Sequence[RunRecord], data: Path | None) -> None:
    _section("1. PROVENANCE")
    print(f"generated      {datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}")
    print(f"host           {platform.node()}  ({platform.system()} {platform.release()})")
    print(f"git HEAD       {_run(['git', 'rev-parse', 'HEAD'])}")
    print(f"git status     {_run(['git', 'status', '--porcelain']) or 'clean'}")
    print(f"gpus           {_run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])}")
    print(f"runs matched   {len(runs)}")

    if data is None:
        return
    # ultralytics honours `path:` literally while t2o's loader falls through a stale one, so a
    # tree that moved since it was adapted fails only on the ultralytics-facing commands --
    # which is every detector stage of every run here (M3 E9 step 1).
    manifest = yaml.safe_load(data.read_text()) if data.is_file() else {}
    print(f"data           {data}")
    print(f"  path:        {manifest.get('path')}")
    print(f"  names:       {manifest.get('names')}   <- the per-class key spelling")
    print(f"  train/val:   {manifest.get('train')} | {manifest.get('val')}")
    for split in ("train", "val"):
        marker = data.parent / split / "infrared" / "LABELS_PROVENANCE.json"
        state = marker.read_text().strip().replace("\n", " ") if marker.is_file() else "absent"
        print(f"  {split} labels: {state}")


def _completeness(runs: Sequence[RunRecord], stage: int) -> bool:
    """One line per run. Returns whether every run reached `stage`.

    Read this before anything below it. `aggregate` computes on the stages shared by *every*
    run, so one stump swept in by a glob collapses them to [0] -- which produced a plausible,
    wrong table once already (TASKS.md M2a step 5).
    """
    _section("2. COMPLETENESS -- read this first")
    print(f"{'run':28} {'seed':>4} {'arm':8} {'stages':14} {'epochs/stage':16} task_weights")
    complete = True
    for run in sorted(runs, key=lambda r: (r.arm.value, r.seed)):
        epochs = [len(record.get("epochs") or []) for record in run.stages]
        reached = stage in run.stage_indices
        complete &= reached
        print(
            f"{run.name:28} {run.seed:>4} {run.arm.value:8} "
            f"{list(run.stage_indices)!s:14} {epochs!s:16} {list(run.task_weights)}"
            f"{'' if reached else '   <-- INCOMPLETE'}"
        )
    print(f"\nverdict: {len(runs)} runs, every run reached stage {stage}: {complete}")
    return complete


def _config_variance(runs: Sequence[RunRecord]) -> None:
    _section("3. CONFIG VARIANCE -- the confound check")
    flats = {
        run.name: _flatten(yaml.safe_load((run.path / "config.yaml").read_text())) for run in runs
    }
    keys = sorted({key for flat in flats.values() for key in flat})
    varying = [
        key for key in keys if len({flat.get(key, "<absent>") for flat in flats.values()}) > 1
    ]

    print("keys whose value differs across the campaign, grouped by value:")
    for key in varying:
        print(f"  {key}{'' if key in EXPECTED_VARYING else '   <-- UNEXPECTED'}")
        by_value: dict[str, list[str]] = {}
        for name, flat in sorted(flats.items()):
            by_value.setdefault(flat.get(key, "<absent>"), []).append(name)
        # A key that is unique per run carries no information in a listing of it -- the run
        # name is already the row label everywhere else in the report.
        if len(by_value) == len(flats) and key in EXPECTED_VARYING:
            print(f"      {len(by_value)} distinct values, one per run")
            continue
        for value, names in by_value.items():
            print(f"      {value:28} {' '.join(names)}")
    unexpected = [key for key in varying if key not in EXPECTED_VARYING]
    print(f"\nverdict: {len(unexpected)} unexpected varying key(s): {unexpected or 'none'}")

    # The launch rule pins each *pair* to one card, never each arm, because a pair split across
    # two GPUs puts whatever differs between the cards inside the paired difference being
    # measured (TASKS.md M2a step 5). `runtime.device` is therefore expected to vary across
    # seeds and required to agree within one.
    split: list[int] = []
    for run in runs:
        partner = next(
            (other for other in runs if other.seed == run.seed and other.arm is not run.arm), None
        )
        if partner is not None and flats[run.name].get("runtime.device") != flats[partner.name].get(
            "runtime.device"
        ):
            split.append(run.seed)
    print(
        f"pairs split across cards: {sorted(set(split)) or 'none'} "
        "(a pair must share one card, or the GPU difference lands in the paired contrast)"
    )

    # The campaign's own launch command, recovered from a snapshot. M3 E9 step 5 ran without
    # one written down, so this is the only record of what was actually asked for.
    sample = sorted(flats)[0]
    print(f"\nfully resolved config of {sample} (the reconstructed launch):")
    for key in keys:
        print(f"  {key:44} {flats[sample].get(key, '<absent>')}")


def _wide_table(runs: Sequence[RunRecord], metrics: Sequence[str]) -> None:
    _section("4. PER-RUN x PER-STAGE ROWS (csv)")
    header = ["run", "seed", "arm", "stage", "task_weight", *metrics]
    print(",".join(header))
    for run in sorted(runs, key=lambda r: (r.arm.value, r.seed)):
        for record in run.stages:
            stage = int(record["stage"])
            weight = run.task_weights[stage] if stage < len(run.task_weights) else float("nan")
            cells = [run.name, str(run.seed), run.arm.value, str(stage), f"{weight:g}"]
            for metric in metrics:
                # Absent and null are different facts: faithfulness.* is absent until the
                # post-hoc pass scores that stage, while a null is what --no-detector wrote.
                if not metric_is_recorded(record, metric):
                    cells.append("")
                    continue
                try:
                    cells.append(f"{metric_value(record, metric):.6f}")
                except AggregationError:
                    # An explicit null -- what `--no-detector` writes. Reported as a hole
                    # rather than aborting the report over one cell.
                    cells.append("null")
            print(",".join(cells))


def _wall_clock(runs: Sequence[RunRecord]) -> None:
    """Training time from the measured per-epoch clock; boundaries from mtimes.

    `EpochStats.seconds` accounts for ~99% of in-stage translator time (step 4 (b)), so the
    training term is a measurement. The boundary -- export, zero-shot, FID, the 50-epoch
    adapted fine-tune -- is wall clock between consecutive `translator_last.pt` writes minus
    the next stage's measured training, the same subtraction step 4 (b) used.

    Medians, not totals. One run was stopped and resumed, and an interval that is wildly large
    is idle wall clock, which mtimes cannot distinguish from compute.
    """
    _section("5. WALL CLOCK")
    print(f"{'run':28} {'stage':>5} {'epochs':>6} {'median_s':>9} {'train_h':>8} {'bound_h':>8}")
    per_arm_epoch_seconds: dict[Arm, list[float]] = {Arm.CONTROL: [], Arm.LOOP: []}
    growth: list[float] = []

    for run in sorted(runs, key=lambda r: (r.arm.value, r.seed)):
        marks = {
            int(record["stage"]): (run.path / f"stage{int(record['stage'])}" / "translator_last.pt")
            for record in run.stages
        }
        times = {
            stage: path.stat().st_mtime for stage, path in sorted(marks.items()) if path.is_file()
        }
        end = (run.path / "metrics.json").stat().st_mtime
        stage_train: dict[int, float] = {}
        for record in run.stages:
            stage = int(record["stage"])
            seconds = [float(epoch.get("seconds", 0.0)) for epoch in record.get("epochs") or []]
            stage_train[stage] = sum(seconds)
            per_arm_epoch_seconds[run.arm].extend(second for second in seconds if second > 0)
            # The boundary that FOLLOWS this stage: next checkpoint, less that stage's own
            # measured training. The last one runs to metrics.json, written after everything.
            nxt = stage + 1
            if stage in times:
                after = times.get(nxt, end if nxt not in stage_train else None)
                boundary = (
                    (after - times[stage] - stage_train.get(nxt, 0.0)) / SECONDS_PER_HOUR
                    if after is not None
                    else float("nan")
                )
            else:
                boundary = float("nan")
            print(
                f"{run.name:28} {stage:>5} {len(seconds):>6} "
                f"{statistics.median(seconds) if seconds else float('nan'):>9.3f} "
                f"{stage_train[stage] / SECONDS_PER_HOUR:>8.2f} {boundary:>8.2f}"
            )
        if stage_train.get(0) and stage_train.get(max(stage_train)):
            growth.append(stage_train[max(stage_train)] / stage_train[0])
        span = (end - (run.path / "config.yaml").stat().st_mtime) / SECONDS_PER_HOUR
        print(f"{run.name:28} {'TOTAL':>5} span {span:.2f} h (config.yaml -> metrics.json)")

    print("\ntwo-up contention -- campaign median epoch vs step 4 (b)'s SOLO probe:")
    for arm, seconds in per_arm_epoch_seconds.items():
        if not seconds:
            continue
        median = statistics.median(seconds)
        solo = PROBE_EPOCH_SECONDS[arm]
        print(f"  {arm.value:8} {median:7.3f} s   solo {solo:7.3f} s   x{median / solo:.2f}")
    if growth:
        print(
            f"\nper-stage growth, last stage's training / stage 0's: median "
            f"x{statistics.median(growth):.2f} over {len(growth)} runs "
            "(step 0 measured ~+20%/stage on the custom set and never explained it)"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="run directories, or globs over them -- quote the glob ('runs/e3f-*')",
    )
    parser.add_argument(
        "--stage", type=int, default=3, help="the headline stage every run must have reached"
    )
    parser.add_argument(
        "--primary-classes",
        nargs="+",
        help="derive the class-subset mAP headline over these classes, e.g. bicycle car person",
    )
    parser.add_argument("--data", type=Path, help="the dataset data.yaml, for the provenance block")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Bare format onto stdout, before anything logs: `t2o aggregate`'s and `loss_share`'s own
    # `basicConfig` calls then no-op, and their tables arrive as report body rather than as
    # timestamped lines nobody wants in a paste.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    run_dirs = _expand_run_globs(args.runs)
    runs = [load_run(run_dir) for run_dir in run_dirs]
    primary = list(args.primary_classes or [])
    if primary:
        add_primary_mean(runs, primary)

    classes = sorted(
        {
            name
            for run in runs
            for record in run.stages
            for name in (record.get("zero_shot") or {}).get("per_class_ap50", {})
        }
    )
    headline = ["zero_shot.primary_map50", "zero_shot.primary_n_classes"] if primary else []
    metrics = [
        *headline,
        "zero_shot.map50",
        "zero_shot.map50_95",
        "zero_shot.precision",
        "zero_shot.recall",
        *(f"zero_shot.per_class_ap50.{name}" for name in classes),
        "fidelity.lpips",
        "fidelity.fid",
        "fidelity.psnr",
        "fidelity.ssim",
        "detector.map50",
        "faithfulness.false_object_rate",
        "faithfulness.missed_object_rate",
        "faithfulness.detection_consistency",
    ]

    print(f"CAMPAIGN REPORT -- {', '.join(args.runs)}")
    _provenance(runs, args.data)
    complete = _completeness(runs, args.stage)
    _config_variance(runs)
    _wide_table(runs, metrics)
    _wall_clock(runs)

    _section("6. PAIRED STATISTICS (t2o aggregate, verbatim)")
    # `aggregate` drops a stage where only some runs record a metric, but raises when NO shared
    # stage records it in every run -- which would abort the whole invocation over one metric
    # the post-hoc pass has not reached yet. So the filter mirrors exactly that condition, and
    # the wide table above stays the place a partly-scored metric is read from.
    shared = common_stages(runs)
    aggregable = [
        metric
        for metric in metrics
        if any(
            all(metric_is_recorded(_record(run, stage), metric) for run in runs) for stage in shared
        )
    ]
    argv_aggregate = ["aggregate", "--runs", *args.runs, "--metric", *aggregable]
    if primary:
        argv_aggregate += ["--primary-classes", *primary]
    if complete:
        argv_aggregate += ["--stage", str(args.stage)]
    print(f"$ t2o {' '.join(argv_aggregate)}\n")
    cli_main(argv_aggregate)

    _section("7. LOSS SPACE -- the realised dose")
    for label, arm in (("loop arm", Arm.LOOP), ("control arm (terms only)", Arm.CONTROL)):
        chosen = [run for run in runs if run.arm is arm]
        if not chosen:
            continue
        print(f"\n-- {label}, {len(chosen)} run(s)")
        loss_report(stage_shares(chosen))

    _section("8. PRE-REGISTERED READINGS -- written before these numbers")
    print(PRE_REGISTERED)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
