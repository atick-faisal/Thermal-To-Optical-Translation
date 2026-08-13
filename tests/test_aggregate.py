"""`analysis/aggregate.py` -- reading a campaign of runs as one paired experiment.

Run directories are hand-written here rather than produced by `run_loop`, for the same
reason `tests/conftest.py` builds a synthetic dataset instead of committing images (M0.3):
the pathologies that matter -- an unpaired seed, a duplicate (arm, seed) cell, a run one
stage short, a null metric -- can be constructed exactly, instead of hoping a real run
happens to contain one. The one `slow` test at the bottom covers what hand-written fixtures
structurally cannot: that this module parses what `engine/loop.py` *actually* writes.

The statistics tests check hand-computed answers, not just self-consistency. n=6 all-positive
differences must give exactly 2/64 = 0.03125 -- that number is the whole justification for
E3's six-seed budget (TASKS.md M1.2), so pinning it here pins the claim, not only the code.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pytest

from t2o.analysis.aggregate import (
    AggregationError,
    Arm,
    aggregate,
    bootstrap_ci,
    common_stages,
    load_run,
    metric_value,
    pair_runs,
    sign_flip_p_value,
    tidy_rows,
    write_csv,
)


def _write_run(
    root: Path,
    name: str,
    seed: int,
    task_weights: list[float],
    map50_per_stage: list[float],
    lpips_per_stage: list[float] | None = None,
) -> Path:
    """Write a minimal but real-shaped run directory: metrics.json + config.yaml."""
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    stages: list[dict[str, Any]] = []
    for index, map50 in enumerate(map50_per_stage):
        lpips = lpips_per_stage[index] if lpips_per_stage is not None else 0.3 - 0.01 * index
        stages.append(
            {
                "stage": index,
                "task_weight": task_weights[index],
                "epochs": [{"epoch": 0, "train_losses": {"total": 1.0}, "val_loss": 1.0}],
                "detector": {
                    "weights": "w.pt",
                    "precision": 0.8,
                    "recall": 0.8,
                    "map50": 0.9,
                    "map50_95": 0.6,
                },
                "zero_shot": {
                    "precision": 0.8,
                    "recall": 0.8,
                    "map50": map50,
                    "map50_95": map50 * 0.7,
                    "per_class_ap50": {"Switch": map50 * 0.6, "Pole": 0.95},
                    "per_class_ap50_95": {"Switch": map50 * 0.4, "Pole": 0.7},
                },
                "fidelity": {
                    "psnr": 15.5,
                    "ssim": 0.5,
                    "lpips": lpips,
                    "fid": 90.0,
                    "kid_mean": 0.02,
                    "kid_std": 0.001,
                },
            }
        )
    (run_dir / "metrics.json").write_text(json.dumps(stages))

    # Deliberately not written via Config.snapshot: this mirrors what a snapshot looks like
    # while letting a test include a key the current schema would reject (see the
    # stale-snapshot test below).
    (run_dir / "config.yaml").write_text(
        "train:\n"
        f"  seed: {seed}\n"
        "coupling:\n"
        f"  task_weights: {task_weights}\n"
        "runtime:\n"
        f"  name: {name}\n"
    )
    return run_dir


def _campaign(root: Path, seeds: list[int], loop_map50: list[float]) -> list[Path]:
    """One control + one loop run per seed, two stages each."""
    paths: list[Path] = []
    for index, seed in enumerate(seeds):
        paths.append(
            _write_run(root, f"e3-control-s{seed}", seed, [0.0, 0.0], [0.70, 0.72 + 0.001 * index])
        )
        paths.append(
            _write_run(root, f"e3-loop-s{seed}", seed, [0.0, 1.0], [0.70, loop_map50[index]])
        )
    return paths


def test_arm_is_derived_from_task_weights(tmp_path: Path) -> None:
    control = load_run(_write_run(tmp_path, "c", 0, [0.0, 0.0], [0.7, 0.7]))
    loop = load_run(_write_run(tmp_path, "l", 0, [0.0, 1.0], [0.7, 0.8]))

    assert control.arm is Arm.CONTROL
    assert loop.arm is Arm.LOOP
    assert control.seed == 0
    assert loop.task_weights == (0.0, 1.0)


def test_a_loop_run_that_only_finished_stage_zero_is_still_the_loop_arm(tmp_path: Path) -> None:
    """The reason the arm comes from config.yaml and not metrics.json.

    The loop arm's stage 0 is itself lambda = 0, so a run interrupted after it records an
    all-zero task_weight column -- indistinguishable from a control on metrics.json alone.
    """
    run = load_run(_write_run(tmp_path, "l", 0, [0.0, 1.0, 2.0, 3.0], [0.7]))

    assert run.arm is Arm.LOOP
    assert run.stage_indices == (0,)


def test_a_snapshot_with_a_key_the_current_schema_rejects_still_loads(tmp_path: Path) -> None:
    """M1's completed runs carry `train.workers`, moved to `runtime` by M1.2 step 2b.

    `Config.load` rejects it (`extra="forbid"`), which is why this module parses plain YAML.
    """
    run_dir = _write_run(tmp_path, "old", 3, [0.0], [0.78])
    (run_dir / "config.yaml").write_text(
        "train:\n  seed: 3\n  workers: 4\ncoupling:\n  task_weights: [0.0]\n"
    )

    assert load_run(run_dir).seed == 3


def test_load_run_names_the_missing_file(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    with pytest.raises(AggregationError, match=r"metrics\.json"):
        load_run(tmp_path / "empty")


def test_metric_value_walks_a_dotted_path_including_per_class(tmp_path: Path) -> None:
    run = load_run(_write_run(tmp_path, "r", 0, [0.0], [0.80]))
    stage = run.stages[0]

    assert metric_value(stage, "zero_shot.map50") == pytest.approx(0.80)
    assert metric_value(stage, "fidelity.lpips") == pytest.approx(0.30)
    assert metric_value(stage, "zero_shot.per_class_ap50.Switch") == pytest.approx(0.48)


def test_metric_value_raises_on_a_null_arm_rather_than_returning_a_sentinel(
    tmp_path: Path,
) -> None:
    """A --no-detector run records `zero_shot: null`; a hole in a paired cell is not a 0."""
    run_dir = _write_run(tmp_path, "r", 0, [0.0], [0.80])
    stages = json.loads((run_dir / "metrics.json").read_text())
    stages[0]["zero_shot"] = None
    (run_dir / "metrics.json").write_text(json.dumps(stages))

    with pytest.raises(AggregationError, match="null"):
        metric_value(load_run(run_dir).stages[0], "zero_shot.map50")


def test_metric_value_raises_on_an_unknown_path(tmp_path: Path) -> None:
    run = load_run(_write_run(tmp_path, "r", 0, [0.0], [0.80]))

    with pytest.raises(AggregationError, match="no metric"):
        metric_value(run.stages[0], "zero_shot.nonexistent")


def test_pair_runs_matches_each_seed(tmp_path: Path) -> None:
    runs = [load_run(path) for path in _campaign(tmp_path, [0, 1], [0.80, 0.81])]

    pairs = pair_runs(runs)

    assert sorted(pairs) == [0, 1]
    assert pairs[0][0].arm is Arm.CONTROL
    assert pairs[0][1].arm is Arm.LOOP


def test_pair_runs_refuses_a_seed_present_in_only_one_arm(tmp_path: Path) -> None:
    paths = _campaign(tmp_path, [0, 1], [0.80, 0.81])
    runs = [load_run(path) for path in paths if path.name != "e3-loop-s1"]

    with pytest.raises(AggregationError, match=r"seeds \[1\]"):
        pair_runs(runs)


def test_pair_runs_refuses_two_runs_in_the_same_cell(tmp_path: Path) -> None:
    """What a forgotten --name produces: runs/<name> has no seed component."""
    runs = [
        load_run(_write_run(tmp_path, "a", 0, [0.0], [0.7])),
        load_run(_write_run(tmp_path, "b", 0, [0.0], [0.7])),
        load_run(_write_run(tmp_path, "c", 0, [1.0], [0.8])),
    ]

    with pytest.raises(AggregationError, match="two runs claim"):
        pair_runs(runs)


def test_sign_flip_p_value_is_exact_and_bottoms_out_at_two_over_two_to_the_n() -> None:
    """0.03125 at n=6 is the number E3's six-seed budget rests on (TASKS.md M1.2)."""
    assert sign_flip_p_value([0.1]) == pytest.approx(1.0)
    assert sign_flip_p_value([0.1] * 5) == pytest.approx(2 / 32)
    assert sign_flip_p_value([0.1] * 6) == pytest.approx(2 / 64)
    assert sign_flip_p_value([0.08, 0.11, 0.09, 0.13, 0.10, 0.12]) == pytest.approx(2 / 64)


def test_sign_flip_p_value_is_uninformative_when_the_differences_cancel() -> None:
    # Mean 0: every sign assignment reaches |0|, so p is 1.0 -- the honest answer.
    assert sign_flip_p_value([0.1, -0.1]) == pytest.approx(1.0)
    # One large negative among positives: not separable at n = 4 (min p there is 0.125).
    assert sign_flip_p_value([0.1, 0.1, 0.1, -0.4]) > 0.5


def test_sign_flip_p_value_refuses_an_intractable_n() -> None:
    with pytest.raises(AggregationError, match="refusing above"):
        sign_flip_p_value([0.1] * 21)


def test_bootstrap_ci_is_reproducible_and_brackets_the_mean() -> None:
    differences = [0.08, 0.11, 0.09, 0.13, 0.10, 0.12]

    low, high = bootstrap_ci(differences, resamples=2000, seed=0)

    assert (low, high) == bootstrap_ci(differences, resamples=2000, seed=0)
    assert low < sum(differences) / len(differences) < high
    assert low > 0.0  # a consistently positive effect must not straddle zero


def test_common_stages_is_the_intersection(tmp_path: Path) -> None:
    runs = [
        load_run(_write_run(tmp_path, "a", 0, [0.0, 0.0, 0.0], [0.7, 0.7, 0.7])),
        load_run(_write_run(tmp_path, "b", 0, [0.0, 1.0], [0.7, 0.8])),
    ]

    assert common_stages(runs) == (0, 1)


def test_aggregate_reports_stage_zero_beside_the_last_stage(tmp_path: Path) -> None:
    """Six seeds, a consistent positive effect at stage 1 and none at stage 0."""
    paths = _campaign(tmp_path, [0, 1, 2, 3, 4, 5], [0.80, 0.81, 0.80, 0.82, 0.81, 0.83])

    report = aggregate(paths, metrics=["zero_shot.map50"], resamples=500)

    assert report.stages == (0, 1)
    null, effect = report.paired
    assert null.stage == 0
    assert null.mean_difference == pytest.approx(0.0)
    assert null.p_value == pytest.approx(1.0)
    assert effect.stage == 1
    assert effect.n == 6
    assert effect.mean_difference > 0.07
    assert effect.p_value == pytest.approx(2 / 64)
    assert effect.ci_low > 0.0


def test_aggregate_summarises_every_arm_and_stage(tmp_path: Path) -> None:
    paths = _campaign(tmp_path, [0, 1], [0.80, 0.81])

    report = aggregate(paths, metrics=["zero_shot.map50"], resamples=200)

    assert len(report.summaries) == 4  # 2 stages x 2 arms x 1 metric
    stage1_loop = next(s for s in report.summaries if s.stage == 1 and s.arm is Arm.LOOP)
    assert stage1_loop.n == 2
    assert stage1_loop.mean == pytest.approx(0.805)
    assert stage1_loop.std > 0.0


def test_a_single_seed_reports_nan_std_rather_than_zero(tmp_path: Path) -> None:
    report = aggregate(_campaign(tmp_path, [0], [0.80]), resamples=100)

    assert all(math.isnan(summary.std) for summary in report.summaries)


def test_aggregate_handles_several_metrics_in_one_pass(tmp_path: Path) -> None:
    """mAP and LPIPS together is the reward-hacking read (M1.1)."""
    paths = _campaign(tmp_path, [0, 1], [0.80, 0.81])

    report = aggregate(paths, metrics=["zero_shot.map50", "fidelity.lpips"], resamples=200)

    assert report.metrics == ("zero_shot.map50", "fidelity.lpips")
    assert {result.metric for result in report.paired} == set(report.metrics)
    # Both arms carry the same synthetic LPIPS, so its paired difference is exactly zero --
    # what "fidelity held while mAP rose" looks like.
    lpips = [r for r in report.paired if r.metric == "fidelity.lpips"]
    assert all(result.mean_difference == pytest.approx(0.0) for result in lpips)


def test_aggregate_refuses_runs_that_share_no_stage(tmp_path: Path) -> None:
    paths = [
        _write_run(tmp_path, "c", 0, [0.0, 0.0], [0.7, 0.7]),
        _write_run(tmp_path, "l", 0, [0.0, 1.0], []),
    ]

    with pytest.raises(AggregationError, match="share no stage"):
        aggregate(paths)


def test_tidy_rows_and_write_csv(tmp_path: Path) -> None:
    paths = _campaign(tmp_path, [0, 1], [0.80, 0.81])
    report = aggregate(paths, metrics=["zero_shot.map50", "fidelity.lpips"], resamples=100)

    rows = tidy_rows(report)
    assert len(rows) == 4 * 2 * 2  # runs x stages x metrics
    assert {row["arm"] for row in rows} == {"control", "loop"}
    assert {row["seed"] for row in rows} == {0, 1}

    destination = write_csv(rows, tmp_path / "out" / "e3.csv")
    with destination.open() as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == len(rows)
    assert written[0]["metric"] == "zero_shot.map50"


@pytest.mark.slow
def test_aggregate_reads_what_run_loop_actually_writes(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    """The seam no hand-written fixture can cover.

    Every fast test above asserts against a `metrics.json`/`config.yaml` pair this file
    wrote, so all of them would keep passing if `engine/loop.py`'s format drifted. This one
    runs the real thing -- a control arm and a loop arm, one stage each, through
    `run_loop` and `Config.snapshot` -- and aggregates the result.
    """
    from t2o.config import Config
    from t2o.engine.loop import run_loop
    from t2o.translators import build_translator

    def _run(name: str, task_weights: list[float]) -> Path:
        config = Config.load(
            overrides={
                "data": {"manifest": str(data_yaml)},
                "train": {"batch_size": 2, "epochs_per_stage": 1, "seed": 0},
                "coupling": {"task_weights": task_weights},
                "detector": {
                    "in_loop": {"weights": str(detector_weights)},
                    "evaluation": {
                        "init_weights": str(detector_weights),
                        "epochs": 1,
                        "batch": 2,
                    },
                },
                "runtime": {"device": "cpu", "workers": 0, "run_dir": str(tmp_path), "name": name},
            }
        )
        run_loop(config, build_translator(config), run_dir=config.runtime.path)
        return config.runtime.path

    control = _run("e3-control-s0", [0.0])
    loop = _run("e3-loop-s0", [1.0])

    report = aggregate(
        [control, loop], metrics=["zero_shot.map50", "fidelity.lpips"], resamples=100
    )

    assert report.stages == (0,)
    assert {run.arm for run in report.runs} == {Arm.CONTROL, Arm.LOOP}
    assert all(run.seed == 0 for run in report.runs)
    assert len(report.paired) == 2
    assert all(math.isfinite(result.mean_difference) for result in report.paired)
    assert len(tidy_rows(report)) == 4  # 2 runs x 1 stage x 2 metrics
