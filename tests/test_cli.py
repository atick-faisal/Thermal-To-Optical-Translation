"""`cli.py`'s override table, config merging, and each subcommand end-to-end.

Built on the synthetic `data_yaml`/`detector_weights` fixtures (`tests/conftest.py`), same
as `test_loop.py`/`test_trainer.py`/`test_export.py`. `evaluate` is the one subcommand
marked `slow` -- it is the first real exercise of `metrics.task.evaluate_detector` through
the CLI and, like `test_task.py`'s own end-to-end case, calls ultralytics' real `model.val()`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from t2o.cli import (
    _export_run_and_stage,
    build_parser,
    config_from_args,
    main,
    overrides_from_args,
)
from t2o.config import Config
from t2o.metrics.faithfulness import FaithfulnessMetrics
from t2o.translators import StubTranslator, build_translator


def test_overrides_from_args_maps_flags_including_a_three_level_path() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--data",
            "foo/data.yaml",
            "--in-loop-weights",
            "w.pt",
            "--eval-init-weights",
            "e.pt",
            "--reference-weights",
            "r.pt",
            "--detector-epochs",
            "3",
        ]
    )

    overrides = overrides_from_args(args)

    assert overrides["data"] == {"manifest": "foo/data.yaml"}
    # Three flags land under the same top-level "detector" key -- this is the merge that
    # a 1-tuple/2-tuple-only override table (Clean-SeAFusion's own) cannot express, and all
    # three sibling sub-dicts (in_loop/evaluation/reference) must survive rather than one
    # clobbering the others.
    assert overrides["detector"] == {
        "in_loop": {"weights": "w.pt"},
        "evaluation": {"init_weights": "e.pt", "epochs": 3},
        "reference": {"weights": "r.pt"},
    }


def test_group_flag_is_wired_through_to_runtime() -> None:
    """Otherwise E3's campaign can only set the group by editing the tracked YAML, which is
    the one thing a 12-run sweep of two files should not have to do per launch.
    """
    args = build_parser().parse_args(["loop", "--group", "e3-pix2pix"])

    assert overrides_from_args(args)["runtime"] == {"group": "e3-pix2pix"}


def test_overrides_from_args_omits_unset_flags() -> None:
    args = build_parser().parse_args(["train"])

    overrides = overrides_from_args(args)

    assert overrides == {}


def test_config_from_args_merges_file_with_cli_overrides_winning(tmp_path: Path) -> None:
    config_path = tmp_path / "exp.yaml"
    config_path.write_text("train:\n  batch_size: 4\n  lr: 0.0005\n")

    args = build_parser().parse_args(["train", "--config", str(config_path), "--batch-size", "8"])
    config = config_from_args(args)

    assert config.train.batch_size == 8  # CLI overrides the file
    assert config.train.lr == 0.0005  # file-only field survives untouched


def test_build_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_version_short_circuits_without_a_subcommand() -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--version"])

    assert exc_info.value.code == 0


def test_build_translator_returns_a_stub_sized_by_config() -> None:
    config = Config.load(overrides={"translator": {"backbone": "stub", "hidden_channels": 8}})

    translator = build_translator(config)

    assert isinstance(translator, StubTranslator)
    assert translator.net[0].out_channels == 8


def test_main_train_writes_a_checkpoint(data_yaml: Path, tmp_path: Path) -> None:
    exit_code = main(
        [
            "train",
            "--data",
            str(data_yaml),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--workers",
            "0",
            "--epochs",
            "1",
            "--run-dir",
            str(tmp_path),
            "--name",
            "run",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "run" / "translator_last.pt").is_file()


def test_main_loop_no_detector_writes_metrics(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    exit_code = main(
        [
            "loop",
            "--data",
            str(data_yaml),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--workers",
            "0",
            "--epochs",
            "1",
            "--task-weights",
            "0.0",
            "1.0",
            "--in-loop-weights",
            str(detector_weights),
            "--run-dir",
            str(tmp_path),
            "--name",
            "run",
            "--no-detector",
        ]
    )

    assert exit_code == 0
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert len(metrics) == 2
    assert all(stage["detector"] is None for stage in metrics)


def test_main_loop_resume_skips_completed_stages(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    common = [
        "--data",
        str(data_yaml),
        "--device",
        "cpu",
        "--batch-size",
        "2",
        "--workers",
        "0",
        "--epochs",
        "1",
        "--in-loop-weights",
        str(detector_weights),
        "--run-dir",
        str(tmp_path),
        "--name",
        "run",
        "--no-detector",
    ]

    first_exit = main(["loop", *common, "--task-weights", "0.0"])
    assert first_exit == 0
    assert len(json.loads((tmp_path / "run" / "metrics.json").read_text())) == 1

    resumed_exit = main(["loop", *common, "--task-weights", "0.0", "1.0", "--resume"])
    assert resumed_exit == 0
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert len(metrics) == 2


def test_main_export_writes_a_translated_dataset(data_yaml: Path, tmp_path: Path) -> None:
    train_exit = main(
        [
            "train",
            "--data",
            str(data_yaml),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--workers",
            "0",
            "--epochs",
            "1",
            "--run-dir",
            str(tmp_path),
            "--name",
            "train_run",
        ]
    )
    assert train_exit == 0
    checkpoint = tmp_path / "train_run" / "translator_last.pt"

    export_exit = main(
        [
            "export",
            "--data",
            str(data_yaml),
            "--device",
            "cpu",
            "--workers",
            "0",
            "--detector-batch",
            "2",
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(tmp_path / "exported"),
        ]
    )

    assert export_exit == 0
    assert (tmp_path / "exported" / "data.yaml").is_file()


@pytest.mark.slow
def test_main_evaluate_scores_a_detector_checkpoint(
    detector_weights: Path, data_yaml: Path
) -> None:
    exit_code = main(
        [
            "evaluate",
            "--weights",
            str(detector_weights),
            "--data",
            str(data_yaml),
            "--device",
            "cpu",
            "--batch",
            "2",
        ]
    )

    assert exit_code == 0


def test_fidelity_accepts_an_export_root_or_an_images_dir(tmp_path: Path) -> None:
    """`run_loop` hands out `.../translated`; the data.yaml inside it names
    `.../translated/val/images`. Both are things a user will paste, so both must work."""
    parser = build_parser()
    root = tmp_path / "translated"
    (root / "val" / "images").mkdir(parents=True)

    args = parser.parse_args(
        ["fidelity", "--translated", str(root), "--data", "d.yaml", "--split", "val"]
    )
    assert args.translated == root
    assert args.split == "val"
    assert (args.translated / args.split / "images").is_dir()


@pytest.mark.slow
def test_main_fidelity_scores_an_export_against_the_real_visible_split(
    data_yaml: Path, tmp_path: Path
) -> None:
    from t2o.engine.export import export_translated

    config = Config.load(
        overrides={
            "data": {"manifest": str(data_yaml)},
            "translator": {"backbone": "stub", "hidden_channels": 4},
            "runtime": {"device": "cpu"},
        }
    )
    translator = build_translator(config)
    export_translated(translator, config, tmp_path / "translated")

    exit_code = main(
        [
            "fidelity",
            "--translated",
            str(tmp_path / "translated"),
            "--data",
            str(data_yaml),
            "--kid-subset-size",
            "2",
            "--batch",
            "2",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0


def test_faithfulness_accepts_an_export_root_or_an_images_dir(tmp_path: Path) -> None:
    """Same two-shapes convention as `fidelity` above, and the same reason."""
    parser = build_parser()
    root = tmp_path / "translated"
    (root / "val" / "images").mkdir(parents=True)

    args = parser.parse_args(
        ["faithfulness", "--translated", str(root), "--data", "d.yaml", "--weights", "w.pt"]
    )
    assert args.translated == root
    assert args.split == "val"
    assert (args.translated / args.split / "images").is_dir()


@pytest.mark.slow
def test_main_faithfulness_scores_an_export_against_the_real_visible_split(
    data_yaml: Path, detector_weights: Path, tmp_path: Path
) -> None:
    """The seam the fast tests cannot reach: a real ultralytics `predict` over a real export.

    The detector is conftest's randomly-initialised yolo11n, so the *rates* are arbitrary --
    what this pins is that `evaluate_faithfulness` reads what `export_translated` actually
    writes, through the genuine `Results` API rather than the stand-in the unit tests use.
    """
    from t2o.engine.export import export_translated

    config = Config.load(
        overrides={
            "data": {"manifest": str(data_yaml)},
            "translator": {"backbone": "stub", "hidden_channels": 4},
            "runtime": {"device": "cpu"},
        }
    )
    export_translated(build_translator(config), config, tmp_path / "translated")

    exit_code = main(
        [
            "faithfulness",
            "--translated",
            str(tmp_path / "translated"),
            "--data",
            str(data_yaml),
            "--weights",
            str(detector_weights),
            "--imgsz",
            "64",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0


# --- --write-back: locating the run and stage an export belongs to -------------------------


def _finished_run(tmp_path: Path, stages: list[int]) -> Path:
    """A run directory with a metrics.json and one export tree per stage."""
    run_dir = tmp_path / "runs" / "e3b-loop-s0"
    for stage in stages:
        (run_dir / f"stage{stage}" / "translated" / "val" / "images").mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps([{"stage": stage, "task_weight": float(stage)} for stage in stages])
    )
    return run_dir


@pytest.mark.parametrize("suffix", ["", "val/images"])
def test_write_back_finds_the_run_and_stage_from_the_export_path(
    tmp_path: Path, suffix: str
) -> None:
    """Both shapes `--translated` accepts resolve to the same run and stage."""
    run_dir = _finished_run(tmp_path, [0, 3])
    translated = run_dir / "stage3" / "translated"

    found_dir, found_stage = _export_run_and_stage(translated / suffix if suffix else translated)

    assert found_dir == run_dir.resolve()
    assert found_stage == 3


def test_write_back_refuses_an_export_outside_a_run(tmp_path: Path) -> None:
    """A `stage3/` that is not under a run directory must not be guessed at: a rate written
    into the wrong stage record is silent here and wrong in the paper.
    """
    stray = tmp_path / "stage3" / "translated"
    stray.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="cannot tell which run and stage"):
        _export_run_and_stage(stray)


def test_write_back_records_the_rates_into_metrics_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through `main`, with the scoring pass itself stubbed out.

    What this pins is the wiring -- that the rates reach the right stage entry of the right
    run -- not the rates, which `tests/test_faithfulness.py` already covers.
    """
    run_dir = _finished_run(tmp_path, [0, 3])
    monkeypatch.setattr(
        "t2o.data.manifest.DatasetManifest.load",
        classmethod(lambda cls, path: SimpleNamespace(val_images=tmp_path, pairing=None)),
    )
    monkeypatch.setattr(
        "t2o.metrics.faithfulness.evaluate_faithfulness",
        lambda *args, **kwargs: FaithfulnessMetrics(0.125, 0.25, 0.75),
    )

    exit_code = main(
        [
            "faithfulness",
            "--translated",
            str(run_dir / "stage3" / "translated"),
            "--data",
            "d.yaml",
            "--weights",
            "w.pt",
            "--write-back",
        ]
    )

    assert exit_code == 0
    entries = json.loads((run_dir / "metrics.json").read_text())
    assert "faithfulness" not in entries[0]
    assert entries[1]["faithfulness"]["false_object_rate"] == pytest.approx(0.125)


def test_faithfulness_write_back_is_off_by_default(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["faithfulness", "--translated", str(tmp_path), "--data", "d.yaml", "--weights", "w.pt"]
    )
    assert args.write_back is False


def test_faithfulness_takes_no_config(tmp_path: Path) -> None:
    """Standalone like `fidelity`/`aggregate`: paths only, never a Config (M1.2 step 4)."""
    parser = build_parser()
    args = parser.parse_args(
        ["faithfulness", "--translated", str(tmp_path), "--data", "d.yaml", "--weights", "w.pt"]
    )
    assert not hasattr(args, "config")


def test_train_detector_splits_out_into_ultralytics_project_and_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--out` names the directory the weights land in, not its parent.

    ultralytics writes into `project/name`, so passing `--out` straight through as `project`
    would bury the checkpoint one level deeper than the flag promises -- and the printed path
    is what gets pasted into `--reference-weights` (M1.2).
    """
    from t2o.engine.detector_stage import DetectorResult

    captured: dict[str, object] = {}

    def fake_train_detector(**kwargs: object) -> DetectorResult:
        captured.update(kwargs)
        return DetectorResult(
            weights=tmp_path / "judge" / "weights" / "best.pt",
            precision=0.0,
            recall=0.0,
            map50=0.0,
            map50_95=0.0,
        )

    monkeypatch.setattr("t2o.engine.detector_stage.train_detector", fake_train_detector)

    exit_code = main(
        ["train-detector", "--data", "d.yaml", "--out", str(tmp_path / "judge"), "--epochs", "3"]
    )

    assert exit_code == 0
    assert captured["project"] == tmp_path
    assert captured["name"] == "judge"
    assert captured["epochs"] == 3
    # Defaults that exist to keep E3's judge independent of the in-loop detector: a
    # different architecture, and a seed that is not train.seed's 0.
    assert captured["seed"] == 1
    assert Path(str(captured["init_weights"])).name == "yolo11s.pt"


def test_train_detector_takes_no_config(tmp_path: Path) -> None:
    """Standalone like `evaluate`/`fidelity`: an experiment config must not be able to reach
    the judge, because the judge has to be independent of the experiment it will score."""
    args = build_parser().parse_args(["train-detector", "--data", "d.yaml", "--out", str(tmp_path)])

    assert not hasattr(args, "config")


@pytest.mark.slow
def test_main_train_detector_writes_a_checkpoint(
    detector_weights: Path, data_yaml: Path, tmp_path: Path
) -> None:
    exit_code = main(
        [
            "train-detector",
            "--data",
            str(data_yaml),
            "--init-weights",
            str(detector_weights),
            "--out",
            str(tmp_path / "judge"),
            "--epochs",
            "1",
            "--imgsz",
            "64",
            "--batch",
            "2",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    weights = tmp_path / "judge" / "weights"
    assert (weights / "best.pt").is_file() or (weights / "last.pt").is_file()


def _write_e3_run(root: Path, name: str, seed: int, weights: list[float], map50: float) -> None:
    """A minimal run directory, mirroring what `run_loop` + `Config.snapshot` leave behind."""
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            [
                {
                    "stage": 0,
                    "task_weight": weights[0],
                    "epochs": [],
                    "detector": None,
                    "zero_shot": {
                        "precision": 0.8,
                        "recall": 0.8,
                        "map50": map50,
                        "map50_95": map50 * 0.7,
                        "per_class_ap50": {},
                        "per_class_ap50_95": {},
                    },
                    "fidelity": None,
                }
            ]
        )
    )
    (run_dir / "config.yaml").write_text(
        f"train:\n  seed: {seed}\ncoupling:\n  task_weights: {weights}\nruntime:\n  name: {name}\n"
    )


def test_main_aggregate_expands_a_glob_and_writes_the_csv(tmp_path: Path) -> None:
    """`--runs 'runs/e3-*'` is quoted in TASKS.md M1.2 step 5, so the glob reaches argparse
    unexpanded and this is the code that has to handle it."""
    for seed in (0, 1):
        _write_e3_run(tmp_path, f"e3-control-s{seed}", seed, [0.0], 0.70)
        _write_e3_run(tmp_path, f"e3-loop-s{seed}", seed, [1.0], 0.80)
    csv_path = tmp_path / "e3.csv"

    exit_code = main(
        [
            "aggregate",
            "--runs",
            f"{tmp_path}/e3-*",
            "--metric",
            "zero_shot.map50",
            "zero_shot.map50_95",
            "--csv",
            str(csv_path),
            "--resamples",
            "100",
        ]
    )

    assert exit_code == 0
    rows = csv_path.read_text().splitlines()
    assert len(rows) == 1 + 4 * 1 * 2  # header + runs x stages x metrics


def test_main_aggregate_reports_a_missing_literal_run(tmp_path: Path) -> None:
    """`glob` returns nothing for a non-existent literal path, which would silently shrink
    a campaign rather than fail it."""
    with pytest.raises(FileNotFoundError, match="matched nothing"):
        main(["aggregate", "--runs", str(tmp_path / "absent")])


def test_aggregate_takes_no_config(tmp_path: Path) -> None:
    args = build_parser().parse_args(["aggregate", "--runs", str(tmp_path)])

    assert not hasattr(args, "config")
    assert args.metric == ["zero_shot.map50"]
    assert args.stage is None
