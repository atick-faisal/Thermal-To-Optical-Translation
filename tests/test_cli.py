"""`cli.py`'s override table, config merging, and each subcommand end-to-end.

Built on the synthetic `data_yaml`/`detector_weights` fixtures (`tests/conftest.py`), same
as `test_loop.py`/`test_trainer.py`/`test_export.py`. `evaluate` is the one subcommand
marked `slow` -- it is the first real exercise of `metrics.task.evaluate_detector` through
the CLI and, like `test_task.py`'s own end-to-end case, calls ultralytics' real `model.val()`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from t2o.cli import build_parser, config_from_args, main, overrides_from_args
from t2o.config import Config
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
