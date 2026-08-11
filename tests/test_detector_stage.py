"""Locating the checkpoint ultralytics actually wrote.

Ported from ``../Clean-SeAFusion/tests/test_detector_stage.py``. `_resolve_weights` and
`_extract_metrics` are unchanged by the M0.4 port (see t2o/engine/detector_stage.py's module
docstring for what *did* change), so these tests carry over verbatim.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from t2o.engine.detector_stage import _extract_metrics, _resolve_weights


def _trainer(save_dir: Path) -> SimpleNamespace:
    weights = save_dir / "weights"
    return SimpleNamespace(save_dir=save_dir, best=weights / "best.pt", last=weights / "last.pt")


def _model(save_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(trainer=_trainer(save_dir))


def _write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")
    return path


def test_prefers_best(tmp_path: Path) -> None:
    _write(tmp_path / "weights" / "best.pt")
    _write(tmp_path / "weights" / "last.pt")
    assert _resolve_weights(_model(tmp_path), tmp_path, "s0").name == "best.pt"


def test_falls_back_to_last_when_best_was_never_written(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The regression test for a 50-epoch run ending with no best.pt.

    ultralytics writes last.pt every epoch but best.pt only when
    `best_fitness == fitness` (trainer.py:767). best_fitness is assigned from that same
    value (trainer.py:873), so a NaN validation fitness makes the comparison False on
    every epoch and best.pt is never written for the entire run. Discarding a fully
    trained last.pt over that would throw away the whole stage.
    """
    _write(tmp_path / "weights" / "last.pt")

    with caplog.at_level("WARNING"):
        resolved = _resolve_weights(_model(tmp_path), tmp_path, "s0")

    assert resolved.name == "last.pt"
    assert "no best.pt" in caplog.text
    assert "NaN" in caplog.text


def test_reads_the_path_from_the_trainer_not_the_convention(tmp_path: Path) -> None:
    # ultralytics may increment the run directory; the trainer knows where it landed and
    # the project/name convention does not.
    actual = tmp_path / "detector" / "stage02"
    _write(actual / "weights" / "best.pt")

    resolved = _resolve_weights(_model(actual), tmp_path / "detector", "stage0")
    assert resolved == actual / "weights" / "best.pt"


def test_falls_back_to_the_convention_without_a_trainer(tmp_path: Path) -> None:
    _write(tmp_path / "s0" / "weights" / "best.pt")
    resolved = _resolve_weights(SimpleNamespace(), tmp_path, "s0")
    assert resolved == tmp_path / "s0" / "weights" / "best.pt"


def test_no_checkpoint_at_all_reports_what_is_there(tmp_path: Path) -> None:
    (tmp_path / "weights").mkdir(parents=True)
    (tmp_path / "weights" / "epoch3.pt").write_bytes(b"x")

    with pytest.raises(FileNotFoundError, match="no usable checkpoint") as exc:
        _resolve_weights(_model(tmp_path), tmp_path, "s0")
    assert "epoch3.pt" in str(exc.value)


def test_missing_weights_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _resolve_weights(_model(tmp_path), tmp_path, "s0")


def test_metrics_from_results_dict() -> None:
    results = SimpleNamespace(
        results_dict={
            "metrics/precision(B)": 0.7,
            "metrics/recall(B)": 0.6,
            "metrics/mAP50(B)": 0.5,
            "metrics/mAP50-95(B)": 0.4,
        }
    )
    assert _extract_metrics(results) == {
        "precision": 0.7,
        "recall": 0.6,
        "map50": 0.5,
        "map50_95": 0.4,
    }


def test_metrics_fall_back_to_the_box_object() -> None:
    results = SimpleNamespace(
        results_dict={}, box=SimpleNamespace(mp=0.3, mr=0.2, map50=0.1, map=0.05)
    )
    assert _extract_metrics(results)["precision"] == 0.3
    assert _extract_metrics(results)["map50_95"] == 0.05


def test_metrics_degrade_to_zero_rather_than_raising() -> None:
    # A shape change in either surface must not take down a run mid-loop.
    assert _extract_metrics(SimpleNamespace()) == {
        "precision": 0.0,
        "recall": 0.0,
        "map50": 0.0,
        "map50_95": 0.0,
    }
