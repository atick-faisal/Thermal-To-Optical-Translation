"""Ownership of the wandb run.

Ported from ``../Clean-SeAFusion/tests/test_tracking.py``'s "the tracker" section only —
its "the ultralytics conflict" section tests ``wandb_integration_disabled``, already
covered since M0.4 in ``tests/test_detector_stage.py``. The two rank/world_size cases
(``test_disabled_on_non_main_ranks``) are dropped: ``t2o`` has no DDP and
``RuntimeConfig`` has no rank concept to gate on (PLAN.md §3, tracking.py's module
docstring).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from t2o.config import Config
from t2o.tracking import RunTracker, run_tags


def _config(tmp_path: Path, **runtime: Any) -> Config:
    return Config.load(overrides={"runtime": {"run_dir": str(tmp_path), "name": "t", **runtime}})


def test_disabled_by_default(tmp_path: Path) -> None:
    tracker = RunTracker(_config(tmp_path))
    assert not tracker.enabled
    tracker.log({"loss": 1.0})  # must not raise
    tracker.finish()


def test_project_and_name_are_not_swapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run belongs under a stable project, named by `name`.

    Passing the run name as the project spawns a new project per run and makes runs
    impossible to compare -- the reason to use wandb at all.
    """
    import wandb

    captured: dict[str, Any] = {}

    def fake_init(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(log=lambda *a, **k: None, finish=lambda: None)

    monkeypatch.setattr(wandb, "init", fake_init)
    config = _config(tmp_path, wandb=True, name="loop_v1", wandb_project="t2o-experiments")
    RunTracker(config)

    assert captured["project"] == "t2o-experiments"
    assert captured["name"] == "loop_v1"


def test_group_and_derived_tags_reach_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E3's campaign is 12 runs off two YAML files, with `--seed`/`--name` overridden on
    every launch -- so tags are derived from the resolved config (tracking.py::run_tags)
    rather than authored, and cannot disagree with the run they label.
    """
    import wandb

    captured: dict[str, Any] = {}

    def fake_init(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(log=lambda *a, **k: None, finish=lambda: None)

    monkeypatch.setattr(wandb, "init", fake_init)
    config = Config.load(
        overrides={
            "runtime": {"run_dir": str(tmp_path), "name": "t", "wandb": True, "group": "e3"},
            "train": {"seed": 4},
            "coupling": {"task_weights": [0.0, 1.0]},
        }
    )
    RunTracker(config)

    assert captured["group"] == "e3"
    assert "seed:4" in captured["tags"]
    assert "lambda_det:on" in captured["tags"]


def test_lambda_det_tag_is_off_for_an_all_zero_ramp(tmp_path: Path) -> None:
    """The control arm must be filterable as such -- a `[0,0,0,0]` ramp has four stages but
    no coupling term, and tagging it `on` would mislabel exactly half of E3's runs.
    """
    config = Config.load(overrides={"coupling": {"task_weights": [0.0, 0.0, 0.0, 0.0]}})

    assert "lambda_det:off" in run_tags(config)


def test_run_directory_is_created_before_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # wandb writes into `dir`; the trainer has not necessarily created it yet.
    import wandb

    seen: dict[str, Any] = {}

    def fake_init(**kwargs: Any) -> Any:
        seen["dir_exists"] = Path(kwargs["dir"]).is_dir()
        return SimpleNamespace(log=lambda *a, **k: None, finish=lambda: None)

    monkeypatch.setattr(wandb, "init", fake_init)
    RunTracker(_config(tmp_path / "fresh", wandb=True))

    assert seen["dir_exists"]


def test_logging_failure_does_not_kill_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Telemetry must never take down a long training run."""
    import wandb

    def exploding_log(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Run (tyy5v8d) is finished. The call to log will be ignored")

    monkeypatch.setattr(
        wandb, "init", lambda **kw: SimpleNamespace(log=exploding_log, finish=lambda: None)
    )
    tracker = RunTracker(_config(tmp_path, wandb=True))

    with caplog.at_level("WARNING"):
        tracker.log({"loss": 1.0})
    assert "wandb logging failed" in caplog.text

    # Disabled after the first failure rather than repeating the traceback every epoch.
    assert not tracker.enabled
    tracker.log({"loss": 2.0})  # must still not raise
    tracker.finish()


def test_finish_failure_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import wandb

    def exploding_finish() -> None:
        raise RuntimeError("network error")

    monkeypatch.setattr(
        wandb,
        "init",
        lambda **kw: SimpleNamespace(log=lambda *a, **k: None, finish=exploding_finish),
    )
    tracker = RunTracker(_config(tmp_path, wandb=True))

    with caplog.at_level("WARNING"):
        tracker.finish()  # must not raise
    assert "wandb finish failed" in caplog.text


def test_context_manager_calls_finish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import wandb

    finished = False

    def fake_finish() -> None:
        nonlocal finished
        finished = True

    monkeypatch.setattr(
        wandb, "init", lambda **kw: SimpleNamespace(log=lambda *a, **k: None, finish=fake_finish)
    )

    with RunTracker(_config(tmp_path, wandb=True)) as tracker:
        assert tracker.enabled

    assert finished
