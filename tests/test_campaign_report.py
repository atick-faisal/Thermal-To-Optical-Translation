"""`scripts/campaign_report.py` -- a whole campaign as one pasteable block.

Run directories are hand-written, the same reasoning `tests/test_aggregate.py` records: the
states that matter -- a run one stage short, a config key that drifted between the two launch
shells, a metric only some stages carry -- can be constructed exactly instead of hoping a real
campaign contains one. The real campaign this was written for cannot be reproduced here at
all; it cost ~80 GPU-hours.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.campaign_report import main

FLIR_CLASSES = {"bicycle": 0.30, "car": 0.60, "person": 0.90, "dog": 0.02}


def _write_run(
    root: Path,
    name: str,
    seed: int,
    weights: list[float],
    map50: list[float],
    *,
    extra_config: str = "",
    scored_stages: tuple[int, ...] = (3,),
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    stages = []
    for index, value in enumerate(map50):
        (run_dir / f"stage{index}").mkdir()
        (run_dir / f"stage{index}" / "translator_last.pt").write_bytes(b"x")
        record = {
            "stage": index,
            "task_weight": weights[index],
            "epochs": [
                {"epoch": e, "train_losses": {"loss_total": 1.0}, "val_loss": 1.0, "seconds": 40.0}
                for e in range(2)
            ],
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
                "map50": value,
                "map50_95": value * 0.7,
                "per_class_ap50": dict(FLIR_CLASSES),
                "per_class_ap50_95": dict(FLIR_CLASSES),
            },
            "fidelity": {
                "psnr": 15.5,
                "ssim": 0.5,
                "lpips": 0.30,
                "fid": 90.0,
                "kid_mean": 0.02,
                "kid_std": 0.001,
            },
        }
        if index in scored_stages:
            record["faithfulness"] = {
                "false_object_rate": 0.19,
                "missed_object_rate": 0.19,
                "detection_consistency": 0.77,
            }
        stages.append(record)
    (run_dir / "metrics.json").write_text(json.dumps(stages))
    (run_dir / "config.yaml").write_text(
        f"train:\n  seed: {seed}\n  batch_size: 8\n"
        f"coupling:\n  task_weights: {weights}\n  grad_scale: 0.15\n"
        f"data:\n  max_train_images: 600\n  subset_seed: 0\n"
        f"runtime:\n  name: {name}\n  workers: 8\n{extra_config}"
    )
    return run_dir


def _campaign(root: Path, **kwargs: object) -> None:
    """Twelve runs, six paired seeds -- the real shape of an E3 cell."""
    for seed in range(6):
        _write_run(root, f"e3f-control-s{seed}", seed, [0.0] * 4, [0.50, 0.51, 0.52, 0.53])
        _write_run(
            root,
            f"e3f-loop-s{seed}",
            seed,
            [0.0, 1.0, 2.0, 3.0],
            [0.50, 0.54, 0.56, 0.58],
            **kwargs,  # type: ignore[arg-type]
        )


def test_the_whole_report_renders_for_a_complete_campaign(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _campaign(tmp_path)

    assert (
        main(["--runs", f"{tmp_path}/e3f-*", "--primary-classes", "bicycle", "car", "person"]) == 0
    )

    out = capsys.readouterr().out
    for title in (
        "1. PROVENANCE",
        "2. COMPLETENESS",
        "3. CONFIG VARIANCE",
        "4. PER-RUN x PER-STAGE ROWS",
        "5. WALL CLOCK",
        "6. PAIRED STATISTICS",
        "7. LOSS SPACE",
        "8. PRE-REGISTERED READINGS",
    ):
        assert title in out
    assert "12 runs, every run reached stage 3: True" in out
    # The headline is the 3-class mean, and dog's wild 0.02 is not in it.
    assert "0.600000" in out
    assert "zero_shot.primary_map50" in out
    assert "0 unexpected varying key(s)" in out
    assert "pairs split across cards: none" in out
    # primary_n_classes at std 0.0000 is the check that the denominator never moved.
    assert "zero_shot.primary_n_classes" in out


def test_a_run_one_stage_short_is_reported_not_averaged_over(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure that produced a plausible, wrong table once (TASKS.md M2a step 5).

    The report must still render -- a half-finished campaign is exactly when someone needs to
    see where it got to -- but it must say so, and it must not ask `aggregate` for a stage the
    runs do not share.
    """
    _campaign(tmp_path)
    _write_run(tmp_path, "e3f-loop-s9", 9, [0.0, 1.0], [0.50, 0.54], scored_stages=())
    _write_run(tmp_path, "e3f-control-s9", 9, [0.0, 0.0], [0.50, 0.51], scored_stages=())

    assert main(["--runs", f"{tmp_path}/e3f-*"]) == 0

    out = capsys.readouterr().out
    assert "<-- INCOMPLETE" in out
    assert "every run reached stage 3: False" in out


def test_a_config_key_that_drifted_between_shells_is_surfaced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A flag that differs between the two launch shells is a confound no config test can see.

    `test_control_and_loop_configs_differ_only_by_design` guards the files; `--batch` arriving
    differently on one shell never touches them.
    """
    _campaign(tmp_path, extra_config="export:\n  normalize: minmax\n")

    assert main(["--runs", f"{tmp_path}/e3f-*"]) == 0

    out = capsys.readouterr().out
    assert "export.normalize   <-- UNEXPECTED" in out
    assert "1 unexpected varying key(s)" in out


def test_a_metric_only_some_stages_carry_is_a_blank_cell_not_a_crash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`t2o faithfulness --write-back` scores one stage at a time, so a sparse column is the
    normal state of this file. It belongs in the wide table and out of the paired block."""
    _campaign(tmp_path)

    assert main(["--runs", f"{tmp_path}/e3f-*"]) == 0

    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if line.startswith("e3f-control-s0,")]
    assert rows[0].endswith(",,,")  # stage 0 has no faithfulness block
    assert not rows[3].endswith(",,,")  # stage 3 does
