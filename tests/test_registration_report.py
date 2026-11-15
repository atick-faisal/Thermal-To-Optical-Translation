"""`scripts/registration_report.py` -- the registration axis, read off both halves.

The per-image collector is stubbed with hand-built records, as `test_margin_resolvability.py`
stubs it: no detector is trained or loaded. Run directories are written by hand in the layout
`t2o loop` leaves (`config.yaml`, `metrics.json`, `stage3/translated/data.yaml`), with the
recorded zero-shot mAP set to what the stub's records score, so the re-score check passes unless
a test breaks it on purpose.
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import pytest
import yaml
from scripts.registration_report import main

from t2o.metrics.task import ImageStats, map50_from_stats, restrict_classes
from test_resolvability import _arm

NAMES = ["car", "dog"]  # `_arm` labels image i with class i % 2
DATASET = "custom"

# Per (run, arm): what the stubbed val pass returns. 12 px is worse than 0 px on both arms.
RECORDS: dict[tuple[str, str], dict[str, ImageStats]] = {
    ("reg-00px", "zero_shot"): _arm(hit_rate=0.9, seed=1),
    ("reg-00px", "adapted"): _arm(hit_rate=0.95, seed=2),
    ("reg-12px", "zero_shot"): _arm(hit_rate=0.4, seed=3),
    ("reg-12px", "adapted"): _arm(hit_rate=0.7, seed=4),
}


def _map50(run: str, arm: str) -> float:
    return map50_from_stats(list(RECORDS[(run, arm)].values()))


def _make_run(
    root: Path,
    name: str,
    level: float | None,
    *,
    zero_shot: float | None = None,
    adapted: float = 0.9,
    fid: float = 100.0,
    seed: int = 0,
) -> Path:
    """One run directory as `t2o loop` writes it, stage 3 only. `level=None` omits the field."""
    run = root / name
    export = run / "stage3" / "translated"
    (export / "val" / "images").mkdir(parents=True)
    (export / "data.yaml").write_text(
        yaml.safe_dump(
            {"path": str(export), "train": "val/images", "val": "val/images", "names": NAMES}
        )
    )
    judge = root / "judge.pt"
    judge.write_bytes(b"")
    adapted_weights = run / "stage3" / "detector" / "stage3" / "weights" / "best.pt"
    adapted_weights.parent.mkdir(parents=True)
    adapted_weights.write_bytes(b"")

    data: dict[str, Any] = {"manifest": "data.yaml"}
    if level is not None:
        data["misalignment_px"] = level
    snapshot = {
        "data": data,
        "train": {"seed": seed},
        "coupling": {"task_weights": [0.0, 1.0, 2.0, 3.0]},
        "detector": {
            "reference": {"weights": str(judge), "imgsz": 640, "batch": 16},
            "evaluation": {"init_weights": "yolo11n.pt", "imgsz": 640, "batch": 16},
        },
        "runtime": {"name": name},
    }
    (run / "config.yaml").write_text(yaml.safe_dump(snapshot))
    recorded = zero_shot if zero_shot is not None else _map50(name, "zero_shot")
    stage = {
        "stage": 3,
        "zero_shot": {"map50": recorded, "per_class_ap50": {"car": recorded, "dog": recorded}},
        "detector": {"weights": str(adapted_weights), "map50": adapted},
        "fidelity": {"fid": fid},
    }
    (run / "metrics.json").write_text(json.dumps([stage]))
    return run


def _sweep(path: Path, values: dict[float, list[float]], primary: bool = True) -> Path:
    """A sweep CSV with the columns the report reads, one row per (level, seed)."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "misalignment_px", "seed", "map50", "primary_map50"]
        )
        writer.writeheader()
        for level, by_seed in values.items():
            for seed, value in enumerate(by_seed):
                writer.writerow(
                    {
                        "dataset": DATASET,
                        # Measured off the field, as the sweep stamps it.
                        "misalignment_px": level + 1e-9,
                        "seed": seed,
                        "map50": value,
                        "primary_map50": value - 0.01 if primary else "",
                    }
                )
    return path


@pytest.fixture
def collected(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every val pass asked for, and answer it from `RECORDS`."""
    calls: list[dict[str, Any]] = []

    def fake_collect(**kwargs: Any) -> dict[str, ImageStats]:
        calls.append(kwargs)
        run = Path(kwargs["data_yaml"]).parents[2].name
        arm = "zero_shot" if Path(kwargs["weights"]).name == "judge.pt" else "adapted"
        return RECORDS[(run, arm)]

    monkeypatch.setattr("t2o.metrics.task.collect_image_stats", fake_collect)
    return calls


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    # The 0 px run has no misalignment field at all, as e3b's August snapshot does not.
    _make_run(root, "reg-00px", None)
    _make_run(root, "reg-12px", 12.0)
    return root


def _argv(sweep: Path, runs: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--dataset",
        DATASET,
        "--sweep-csv",
        str(sweep),
        "--runs",
        str(runs / "reg-*"),
        "--out",
        str(out),
        "--resamples",
        "50",
        *extra,
    ]


def _rows(out: Path) -> list[dict[str, str]]:
    with (out / f"{DATASET}-registration-curve.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def _row(rows: list[dict[str, str]], route: str, level: float) -> dict[str, str]:
    return next(r for r in rows if r["route"] == route and float(r["level_px"]) == level)


def test_reports_both_halves_at_every_level(
    runs: Path, tmp_path: Path, collected: list[dict[str, Any]]
) -> None:
    sweep = _sweep(tmp_path / "sweep.csv", {0.0: [0.93, 0.92, 0.94], 12.0: [0.91, 0.90, 0.92]})
    out = tmp_path / "out"

    assert main(_argv(sweep, runs, out)) == 0

    rows = _rows(out)
    assert sorted((r["route"], float(r["level_px"])) for r in rows) == [
        ("adapted", 0.0),
        ("adapted", 12.0),
        ("label_transfer", 0.0),
        ("label_transfer", 12.0),
        ("zero_shot", 0.0),
        ("zero_shot", 12.0),
    ]
    # Two val passes per run, each by the scorer its own snapshot and metrics.json name.
    assert len(collected) == 4

    lt = _row(rows, "label_transfer", 12.0)
    assert float(lt["map50"]) == pytest.approx(0.91)
    assert float(lt["seed_sd"]) == pytest.approx(statistics.stdev([0.91, 0.90, 0.92]))
    assert float(lt["delta"]) == pytest.approx(0.91 - 0.93)
    assert lt["ci_low_pts"] == ""  # seed spread, never an image CI, on this half

    for arm in ("zero_shot", "adapted"):
        zero, misaligned = _row(rows, arm, 0.0), _row(rows, arm, 12.0)
        assert float(misaligned["map50"]) == pytest.approx(_map50("reg-12px", arm))
        assert float(misaligned["delta"]) == pytest.approx(
            _map50("reg-12px", arm) - _map50("reg-00px", arm)
        )
        assert float(misaligned["delta_low_pts"]) <= float(misaligned["delta_high_pts"]) < 0.0
        assert float(zero["delta"]) == 0.0
        assert zero["delta_low_pts"] == ""  # 0 px has no partner to be differenced against
        assert float(zero["ci_low_pts"]) <= float(zero["ci_high_pts"])
        assert zero["collapsed"] == "False"


def test_a_rescore_that_misses_metrics_json_stops(
    tmp_path: Path, collected: list[dict[str, Any]]
) -> None:
    root = tmp_path / "runs"
    _make_run(root, "reg-00px", 0.0, zero_shot=_map50("reg-00px", "zero_shot") + 0.01)
    _make_run(root, "reg-12px", 12.0)
    sweep = _sweep(tmp_path / "sweep.csv", {0.0: [0.93]})
    out = tmp_path / "out"

    with pytest.raises(SystemExit, match="re-score"):
        main(_argv(sweep, root, out))
    assert not out.exists()  # nothing written from a failed check


def test_identical_seeds_are_n_1_with_no_band(
    runs: Path, tmp_path: Path, collected: list[dict[str, Any]], caplog: pytest.LogCaptureFixture
) -> None:
    sweep = _sweep(tmp_path / "sweep.csv", {0.0: [0.61, 0.61, 0.61], 12.0: [0.56, 0.56, 0.56]})
    out = tmp_path / "out"

    with caplog.at_level(logging.INFO):
        main(_argv(sweep, runs, out))

    lt = _row(_rows(out), "label_transfer", 0.0)
    assert lt["seeds_identical"] == "True"
    assert lt["seed_sd"] == ""
    assert any("F227" in r.getMessage() for r in caplog.records)


def test_a_high_fid_marks_the_point_collapsed(
    tmp_path: Path, collected: list[dict[str, Any]], caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "runs"
    _make_run(root, "reg-00px", 0.0, fid=227.62)
    _make_run(root, "reg-12px", 12.0, fid=110.0)
    sweep = _sweep(tmp_path / "sweep.csv", {0.0: [0.93]})
    out = tmp_path / "out"

    with caplog.at_level(logging.INFO):
        main(_argv(sweep, root, out))

    rows = _rows(out)
    assert _row(rows, "zero_shot", 0.0)["collapsed"] == "True"
    assert _row(rows, "adapted", 0.0)["collapsed"] == "True"  # one export, one FID
    assert _row(rows, "zero_shot", 12.0)["collapsed"] == "False"
    assert any("F231" in r.getMessage() for r in caplog.records)


def test_primary_classes_restrict_both_halves(
    runs: Path, tmp_path: Path, collected: list[dict[str, Any]]
) -> None:
    sweep = _sweep(tmp_path / "sweep.csv", {0.0: [0.93], 12.0: [0.91]})
    out = tmp_path / "out"

    main(_argv(sweep, runs, out, "--primary-classes", "car"))

    rows = _rows(out)
    assert float(_row(rows, "label_transfer", 12.0)["map50"]) == pytest.approx(0.90)
    car_only = restrict_classes(RECORDS[("reg-12px", "zero_shot")], [NAMES.index("car")])
    assert float(_row(rows, "zero_shot", 12.0)["map50"]) == pytest.approx(
        map50_from_stats(list(car_only.values()))
    )


def test_primary_classes_refuse_a_sweep_run_without_them(
    runs: Path, tmp_path: Path, collected: list[dict[str, Any]]
) -> None:
    sweep = _sweep(tmp_path / "sweep.csv", {0.0: [0.93]}, primary=False)

    with pytest.raises(SystemExit, match="primary_map50"):
        main(_argv(sweep, runs, tmp_path / "out", "--primary-classes", "car"))
    assert collected == []  # refused before any val pass


def test_a_curve_needs_one_run_per_level_and_a_zero(
    tmp_path: Path, collected: list[dict[str, Any]]
) -> None:
    sweep = _sweep(tmp_path / "sweep.csv", {0.0: [0.93]})
    missing_zero = tmp_path / "missing-zero"
    _make_run(missing_zero, "reg-12px", 12.0)
    with pytest.raises(SystemExit, match="no 0 px run"):
        main(_argv(sweep, missing_zero, tmp_path / "out"))

    doubled = tmp_path / "doubled"
    _make_run(doubled, "reg-00px", 0.0)
    _make_run(doubled, "reg-00px-s1", 0.0, zero_shot=0.8, seed=1)
    with pytest.raises(SystemExit, match="two runs at 0 px"):
        main(_argv(sweep, doubled, tmp_path / "out"))
    assert collected == []


def test_seed_runs_give_the_zero_point_its_spread(
    runs: Path, tmp_path: Path, collected: list[dict[str, Any]]
) -> None:
    seeds = tmp_path / "seeds"
    recorded = [0.85, 0.83, 0.86]
    for seed, value in enumerate(recorded):
        _make_run(
            seeds, f"e3b-loop-s{seed}", None, zero_shot=value, adapted=value + 0.05, seed=seed
        )
    sweep = _sweep(tmp_path / "sweep.csv", {0.0: [0.93]})
    out = tmp_path / "out"

    main(_argv(sweep, runs, out, "--seed-runs", str(seeds / "e3b-loop-s*")))

    rows = _rows(out)
    zero = _row(rows, "zero_shot", 0.0)
    assert float(zero["seed_spread_sd"]) == pytest.approx(statistics.stdev(recorded))
    assert int(zero["seed_spread_n"]) == 3
    assert _row(rows, "zero_shot", 12.0)["seed_spread_sd"] == ""
    assert len(collected) == 4  # the seed set is read off metrics.json, never re-scored


def test_an_existing_report_is_refused(
    runs: Path, tmp_path: Path, collected: list[dict[str, Any]]
) -> None:
    sweep = _sweep(tmp_path / "sweep.csv", {0.0: [0.93]})
    out = tmp_path / "out"
    main(_argv(sweep, runs, out))

    with pytest.raises(SystemExit, match="already exists"):
        main(_argv(sweep, runs, out))
