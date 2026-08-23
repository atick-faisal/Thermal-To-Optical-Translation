"""`scripts/loss_share.py` -- the detection term's share of the objective.

Run directories are hand-written, as in `tests/test_aggregate.py` and for the same reason:
the composition being measured has to be known exactly for the share to be checkable. The
loss values here are internally consistent (`loss_total == l2 + lpips + gan + w*det`), which
is what `translators/pix2pix.py::fit` actually writes, so the fixture cannot drift into a
shape the script would never meet.

The numbers are hand-computed rather than recomputed from the fixture: this script's whole
purpose is to decide whether E3's null was dose-limited (TASKS.md M1.2 step 7), so an
arithmetic slip in the pooling would misdirect a ~72 GPU-hour decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scripts.loss_share import _format_epochs, epoch_means, main, stage_shares

from t2o.analysis.aggregate import load_run


def _write_run(
    root: Path,
    name: str,
    seed: int,
    task_weights: list[float],
    det_per_epoch: list[list[float]],
    grad_scale: float = 1.0e-2,
) -> Path:
    """One run directory whose stage *i* has `len(det_per_epoch[i])` epochs.

    A stage at weight 0 records no `loss_det` key at all -- `fit` only sets it when
    `task_weight > 0` -- so an empty list for that stage is the faithful shape, not a hole.
    """
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    stages: list[dict[str, Any]] = []
    for index, weight in enumerate(task_weights):
        dets = det_per_epoch[index]
        epochs: list[dict[str, Any]] = []
        for epoch, det in enumerate(dets or [None]):
            losses = {"loss_l2": 0.02, "loss_lpips": 1.0, "loss_gan": 0.5}
            if det is not None:
                losses["loss_det"] = det
            losses["loss_total"] = 1.52 + weight * (det or 0.0)
            epochs.append({"epoch": epoch, "train_losses": losses, "val_loss": 1.0})
        stages.append({"stage": index, "task_weight": weight, "epochs": epochs})
    (run_dir / "metrics.json").write_text(json.dumps(stages))

    (run_dir / "config.yaml").write_text(
        "train:\n"
        f"  seed: {seed}\n"
        "coupling:\n"
        f"  task_weights: {task_weights}\n"
        f"  grad_scale: {grad_scale}\n"
        "runtime:\n"
        f"  name: {name}\n"
    )
    return run_dir


def test_epoch_means_averages_over_epochs_and_omits_an_absent_term(tmp_path: Path) -> None:
    run = load_run(_write_run(tmp_path, "r", 0, [0.0, 2.0], [[], [0.05, 0.15]]))

    assert "loss_det" not in epoch_means(run.stages[0])
    assert epoch_means(run.stages[1])["loss_det"] == pytest.approx(0.10)


def test_share_is_the_weighted_detection_term_over_the_total(tmp_path: Path) -> None:
    """Pooled over epochs *and* runs, and weighted -- the three places a slip would hide.

    Seed 0's stage 1 averages `loss_det` 0.05/0.15 -> 0.10 with totals 1.62/1.82 -> 1.72;
    seed 1's is 0.20 flat with total 1.92. Pooling the two runs gives det 0.15, total 1.82,
    so at `task_weight` 2.0 the objective carried 0.30 of 1.82 == 16.48%.
    """
    runs = [
        load_run(_write_run(tmp_path, "loop-s0", 0, [0.0, 2.0], [[], [0.05, 0.15]])),
        load_run(_write_run(tmp_path, "loop-s1", 1, [0.0, 2.0], [[], [0.20, 0.20]])),
    ]

    stage0, stage1 = stage_shares(runs)

    assert stage1.n_runs == 2
    assert stage1.means["loss_det"] == pytest.approx(0.15)
    assert stage1.means["loss_total"] == pytest.approx(1.82)
    assert stage1.detection_term == pytest.approx(0.30)
    assert stage1.share == pytest.approx(0.30 / 1.82)

    # Stage 0 is lambda = 0 in both arms -- the null control. No detection term exists there.
    assert stage0.detection_term == 0.0
    assert stage0.share == 0.0


def test_effective_lambda_is_the_weight_times_grad_scale(tmp_path: Path) -> None:
    """The arithmetic the whole script exists to surface: E3 ran at 0.01/0.02/0.03, not 1/2/3."""
    runs = [load_run(_write_run(tmp_path, "loop", 0, [0.0, 3.0], [[], [0.1]], grad_scale=1.0e-2))]

    assert stage_shares(runs)[1].effective_lambda == pytest.approx(0.03)


def test_runs_from_two_different_experiments_are_refused(tmp_path: Path) -> None:
    runs = [
        load_run(_write_run(tmp_path, "a", 0, [0.0, 2.0], [[], [0.1]])),
        load_run(_write_run(tmp_path, "b", 1, [0.0, 3.0], [[], [0.1]])),
    ]

    with pytest.raises(SystemExit, match="disagree on task_weight"):
        stage_shares(runs)


def test_first_epochs_truncates_the_pool_and_changes_the_share(tmp_path: Path) -> None:
    """The confound that cost M1.2 step 8 a round trip: a 25-epoch probe is not a 100-epoch run.

    Stage 1's `loss_det` runs 0.05/0.15/0.55/0.85 against totals 1.62/1.82/2.62/3.22. Pooled
    whole that is det 0.40 and total 2.32, so at weight 2.0 the share is 0.80/2.32 == 34.48%.
    Pooled over the opening two epochs it is det 0.10 and total 1.72 -- share 0.20/1.72 ==
    11.63%. Same run, same lambda, three-fold difference in the number a campaign is launched
    on, which is why the epoch count is on the report row rather than left implicit.
    """
    run = load_run(_write_run(tmp_path, "r", 0, [0.0, 2.0], [[], [0.05, 0.15, 0.55, 0.85]]))

    whole = stage_shares([run])[1]
    assert whole.n_epochs == (4,)
    assert whole.share == pytest.approx(0.80 / 2.32)

    opening = stage_shares([run], first_epochs=2)[1]
    assert opening.n_epochs == (2,)
    assert opening.share == pytest.approx(0.20 / 1.72)


def test_pooling_runs_of_unequal_length_reports_the_span(tmp_path: Path) -> None:
    """Unequal lengths are pooled, not refused -- but the row has to admit it happened."""
    short = load_run(_write_run(tmp_path, "short", 0, [0.0, 2.0], [[], [0.05, 0.15]]))
    long = load_run(_write_run(tmp_path, "long", 1, [0.0, 2.0], [[], [0.05, 0.15, 0.55, 0.85]]))

    assert stage_shares([short, long])[1].n_epochs == (2, 4)
    assert _format_epochs((2, 4)) == "2-4"
    assert _format_epochs((25,)) == "25"


def test_a_control_only_glob_is_refused_and_names_the_way_out(tmp_path: Path) -> None:
    """The default refusal exists for the mis-glob, so it must survive --terms-only existing."""
    _write_run(tmp_path, "control-s0", 0, [0.0, 0.0], [[], []])

    with pytest.raises(SystemExit, match="--terms-only"):
        main(["--runs", str(tmp_path / "control-s0")])


def test_terms_only_reports_a_control_arms_composition_with_no_detection_term(
    tmp_path: Path,
) -> None:
    """The control arm's fidelity trajectory, which nothing else in the repo can read.

    Its `loss_det` key genuinely does not exist, so the detection column must come back
    absent rather than as a zero that reads like a measured value.
    """
    runs = [load_run(_write_run(tmp_path, "control-s0", 0, [0.0, 0.0], [[], []]))]

    shares = stage_shares(runs)

    assert [s.stage for s in shares] == [0, 1]
    assert all("loss_det" not in s.means for s in shares)
    assert all(s.means["loss_lpips"] == pytest.approx(1.0) for s in shares)
    # w * det is 0 * absent, and the share with it -- an honest 0% detection, not a nan.
    assert all(s.detection_term == 0.0 and s.share == 0.0 for s in shares)


def test_terms_only_accepts_a_mixed_glob_without_dropping_the_control_runs(
    tmp_path: Path,
) -> None:
    """Unlike the default path, which skips control runs to keep the share undiluted."""
    _write_run(tmp_path, "control-s0", 0, [0.0, 0.0], [[], []])
    _write_run(tmp_path, "control-s1", 1, [0.0, 0.0], [[], []])

    assert main(["--runs", str(tmp_path / "control-s*"), "--terms-only"]) == 0
