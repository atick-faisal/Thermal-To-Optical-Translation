"""`scripts/gate_table.py` -- E9's kill-test driver.

The GPU entry point is stubbed. What matters here is not ultralytics' mAP but the wiring around
it, because every way this script comes back *wrong* rather than *late* is silent:

- the floor arm reading visible pixels while the table calls them thermal,
- a thin class with a dozen instances dragging the number the verdict is read off,
- and an unmirrored `infrared/labels`, which ultralytics scores as an unlabelled split without
  complaining at all.

The fixture applies `t2o.data.mirror.mirror_labels_onto_infrared` rather than copying labels by
hand, so these also cover the E9 step 1 -> step 3 chain in one go.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pytest
from scripts.gate_table import (
    CEILING,
    FLOOR,
    KILL_THRESHOLD,
    PASSIVE_THRESHOLD,
    STRONG_THRESHOLD,
    THERMAL_TRAINED,
    Row,
    _per_class_gap,
    _sensor_gap_verdict,
    _summarise,
    _verdict,
    main,
)

from t2o.data.calibration import load_calibration
from t2o.data.manifest import DatasetManifest
from t2o.data.mirror import mirror_labels_onto_infrared
from t2o.metrics.task import TaskMetrics
from test_calibration import make_calibration

CLASSES = ("Fuse", "Pole", "Switch", "Transformer")
JUDGE = "best.pt"


@pytest.fixture
def mirrored(dataset_root: Path, tmp_path: Path) -> Path:
    """A throwaway copy of the shared fixture with its labels mirrored onto the thermal side.

    Copied because `dataset_root` is session-scoped (`tests/conftest.py`) and mirroring writes
    into the tree -- mutating it here would leak into every other test in the run.
    """
    root = tmp_path / "mirrored"
    shutil.copytree(dataset_root, root)
    copied = root / "data.yaml"
    copied.write_text(copied.read_text().replace(f"path: {dataset_root}", f"path: {root}"))
    mirror_labels_onto_infrared(DatasetManifest.load(copied))
    return copied


@pytest.fixture
def unmirrored(dataset_root: Path, tmp_path: Path) -> Path:
    """The same copy, left as every adapted public dataset arrives: visible labels only."""
    root = tmp_path / "unmirrored"
    shutil.copytree(dataset_root, root)
    copied = root / "data.yaml"
    copied.write_text(copied.read_text().replace(f"path: {dataset_root}", f"path: {root}"))
    return copied


ROLL = ((0.0, 6.0), (-12.0, 1.0), (0.0, -6.0), (13.0, 0.0))


def _copy(dataset_root: Path, tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    shutil.copytree(dataset_root, root)
    copied = root / "data.yaml"
    copied.write_text(copied.read_text().replace(f"path: {dataset_root}", f"path: {root}"))
    return copied


@pytest.fixture
def derolled(dataset_root: Path, tmp_path: Path) -> tuple[Path, Path]:
    """A tree whose thermal labels are de-rolled, and the constant file that did it.

    `validate_for` compares the constant's dataset against the tree's own directory name, so the
    two have to be built together.
    """
    copied = _copy(dataset_root, tmp_path, "derolled")
    calibration = make_calibration(dataset=copied.parent.name, corner_shift=ROLL)
    mirror_labels_onto_infrared(DatasetManifest.load(copied), calibration=calibration)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(asdict(calibration)))
    return copied, path


@pytest.fixture
def judge(tmp_path: Path) -> Path:
    """A stand-in for the weights file. Never loaded -- `evaluate_detector` is stubbed."""
    path = tmp_path / JUDGE
    path.write_bytes(b"")
    return path


def _metrics(per_class: dict[str, float], map50: float = 0.5) -> TaskMetrics:
    return TaskMetrics(
        precision=0.1,
        recall=0.2,
        map50=map50,
        map50_95=0.3,
        per_class_ap50=per_class,
        per_class_ap50_95=dict.fromkeys(per_class, 0.1),
    )


@dataclass
class StubbedGpu:
    """What the script asked for, and what it got back. `scores` is read in call order."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    scores: list[TaskMetrics] = field(default_factory=list)


@pytest.fixture
def stubbed_gpu(monkeypatch: pytest.MonkeyPatch) -> StubbedGpu:
    """Record each validation pass and return the score the test lined up for it."""
    stub = StubbedGpu()

    def fake_evaluate(**kwargs: Any) -> TaskMetrics:
        stub.calls.append(kwargs)
        index = len(stub.calls) - 1
        if index < len(stub.scores):
            return stub.scores[index]
        # Every class scored, so a test that only cares about wiring is not tripped by
        # the script's (correct) refusal of a primary set with nothing in it.
        return _metrics(dict.fromkeys(CLASSES, 0.5))

    monkeypatch.setattr("t2o.metrics.task.evaluate_detector", fake_evaluate)
    return stub


def _argv(
    data: Path, judge: Path, out: Path, *primary: str, calibration: Path | None = None
) -> list[str]:
    argv = ["--data", str(data), "--weights", str(judge), "--out", str(out), "--device", "cpu"]
    if primary:
        argv += ["--primary-classes", *primary]
    if calibration is not None:
        argv += ["--calibration", str(calibration)]
    return argv


def _rows(out: Path) -> list[dict[str, str]]:
    with (out / "gate.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_scores_both_arms_and_writes_them(
    mirrored: Path, judge: Path, tmp_path: Path, stubbed_gpu: StubbedGpu
) -> None:
    stubbed_gpu.scores.extend(
        [
            _metrics(dict.fromkeys(CLASSES, 0.90), map50=0.90),
            _metrics(dict.fromkeys(CLASSES, 0.20), map50=0.20),
        ]
    )
    out = tmp_path / "gate"
    assert main(_argv(mirrored, judge, out)) == 0

    rows = _rows(out)
    assert [row["arm"] for row in rows] == ["ceiling", "floor"]
    assert [row["val_pixels"] for row in rows] == ["visible", "thermal"]
    assert float(rows[0]["primary_map50"]) == pytest.approx(0.90)
    assert float(rows[1]["primary_map50"]) == pytest.approx(0.20)
    assert float(rows[0]["ap50_Switch"]) == pytest.approx(0.90)


def test_each_arm_reads_the_pixels_its_row_claims(
    mirrored: Path, judge: Path, tmp_path: Path, stubbed_gpu: StubbedGpu
) -> None:
    """The failure this guards scores a real, meaningless number instead of raising.

    Both arms are also checked to carry no `path:` -- ultralytics honours that field literally,
    so a manifest built on a tree that has since moved would raise `FileNotFoundError` on the
    ceiling arm even though `t2o`'s own loader tolerates it.
    """
    assert main(_argv(mirrored, judge, tmp_path / "gate")) == 0

    ceiling, floor = (Path(call["data_yaml"]) for call in stubbed_gpu.calls)
    for built, modality in ((ceiling, "visible"), (floor, "infrared")):
        lines = built.read_text().splitlines()
        val = next(line for line in lines if line.startswith("val:"))
        assert val.endswith(str(Path("val") / modality / "images"))
        assert not any(line.startswith("path:") for line in lines)


def test_the_primary_mean_ignores_the_classes_left_out(
    mirrored: Path, judge: Path, tmp_path: Path, stubbed_gpu: StubbedGpu
) -> None:
    """FLIR val carries 13 dog instances: a thin class must not reach the verdict's number.

    `Switch` here stands in for dog -- wild in both arms, and excluded from `--primary-classes`.
    """
    stubbed_gpu.scores.extend(
        [
            _metrics({"Fuse": 0.80, "Pole": 0.90, "Transformer": 0.70, "Switch": 0.00}),
            _metrics({"Fuse": 0.10, "Pole": 0.20, "Transformer": 0.30, "Switch": 1.00}),
        ]
    )
    out = tmp_path / "gate"
    assert main(_argv(mirrored, judge, out, "Fuse", "Pole", "Transformer")) == 0

    rows = _rows(out)
    assert float(rows[0]["primary_map50"]) == pytest.approx(0.80)
    assert float(rows[1]["primary_map50"]) == pytest.approx(0.20)
    # Left in the all-class means, so nothing is hidden -- just not what the verdict reads.
    assert float(rows[1]["ap50_Switch"]) == pytest.approx(1.00)


def test_a_class_with_no_instances_is_skipped_not_scored_as_zero(
    mirrored: Path, judge: Path, tmp_path: Path, stubbed_gpu: StubbedGpu
) -> None:
    """`_extract_per_class_ap` omits a zero-instance class; averaging in a 0.0 would invent one."""
    stubbed_gpu.scores.extend(
        [_metrics({"Fuse": 0.80, "Pole": 0.60}), _metrics({"Fuse": 0.20, "Pole": 0.40})]
    )
    out = tmp_path / "gate"
    assert main(_argv(mirrored, judge, out, "Fuse", "Pole", "Switch")) == 0

    rows = _rows(out)
    assert float(rows[0]["primary_map50"]) == pytest.approx(0.70)
    assert rows[0]["ap50_Switch"] == ""


def test_rejects_a_primary_class_the_manifest_does_not_have(
    mirrored: Path, judge: Path, tmp_path: Path, stubbed_gpu: StubbedGpu
) -> None:
    """A misspelt class name would otherwise silently narrow the verdict's basis."""
    with pytest.raises(SystemExit, match="Switchh"):
        main(_argv(mirrored, judge, tmp_path / "gate", "Fuse", "Switchh"))
    assert stubbed_gpu.calls == []


def test_refuses_a_tree_whose_thermal_side_has_no_labels(
    unmirrored: Path, judge: Path, tmp_path: Path, stubbed_gpu: StubbedGpu
) -> None:
    """E9 step 1 is a hard precondition, and it fails before the first validation pass."""
    from t2o.data.budget import BudgetError

    with pytest.raises(BudgetError, match="has no labels beside it"):
        main(_argv(unmirrored, judge, tmp_path / "gate"))
    assert stubbed_gpu.calls == []


def test_refuses_a_calibration_the_tree_was_not_derolled_by(
    mirrored: Path, judge: Path, tmp_path: Path, stubbed_gpu: StubbedGpu
) -> None:
    """The digest column must be a checked fact, not a record that a flag was passed.

    E9's first de-rolled gate stamped a digest on a floor scored from a stale ultralytics label
    cache, i.e. on uncorrected boxes -- and a floor that moved for the wrong reason is exactly
    the failure the column exists to make visible.
    """
    calibration = tmp_path / "other.json"
    calibration.write_text(json.dumps(asdict(make_calibration(dataset=mirrored.parent.name))))

    with pytest.raises(SystemExit, match="was not de-rolled by"):
        main(_argv(mirrored, judge, tmp_path / "gate", calibration=calibration))
    assert stubbed_gpu.calls == []


def test_refuses_a_derolled_tree_with_no_calibration(
    derolled: tuple[Path, Path], judge: Path, tmp_path: Path, stubbed_gpu: StubbedGpu
) -> None:
    """The check is symmetric: otherwise the column is only right when someone remembers the flag,
    and a de-rolled floor would be written as a plain-mirror one."""
    data, _ = derolled

    with pytest.raises(SystemExit, match="no --calibration was given"):
        main(_argv(data, judge, tmp_path / "gate"))
    assert stubbed_gpu.calls == []


def test_stamps_the_verified_digest_on_both_rows(
    derolled: tuple[Path, Path], judge: Path, tmp_path: Path, stubbed_gpu: StubbedGpu
) -> None:
    data, calibration = derolled
    out = tmp_path / "gate"

    assert main(_argv(data, judge, out, calibration=calibration)) == 0

    expected = load_calibration(calibration).digest()
    assert [row["calibration_digest"] for row in _rows(out)] == [expected, expected]


@pytest.mark.parametrize(
    ("headroom", "expected"),
    [
        (STRONG_THRESHOLD + 0.01, "PASS --"),
        (STRONG_THRESHOLD, "PASS --"),
        (KILL_THRESHOLD, "WEAK PASS"),
        (STRONG_THRESHOLD - 0.01, "WEAK PASS"),
        (KILL_THRESHOLD - 0.01, "KILL"),
        (0.0, "KILL"),
    ],
)
def test_the_verdict_bands_are_the_pre_registered_ones(headroom: float, expected: str) -> None:
    assert _verdict(headroom).startswith(expected)


# --- the thermal-trained arm (T) and seeds: plan.md Q5-Q7 -------------------------------------


def _seeded_argv(
    data: Path, out: Path, weights: list[Path], thermal: list[Path] | None = None
) -> list[str]:
    argv = ["--data", str(data), "--weights", *map(str, weights), "--out", str(out)]
    if thermal:
        argv += ["--thermal-weights", *map(str, thermal)]
    return [*argv, "--device", "cpu"]


def _checkpoints(tmp_path: Path, *names: str) -> list[Path]:
    """Stand-ins like `judge`: never loaded, only checked to exist."""
    paths = [tmp_path / name for name in names]
    for path in paths:
        path.write_bytes(b"")
    return paths


def _row(arm: str, per_class: dict[str, float]) -> Row:
    mean = sum(per_class.values()) / len(per_class)
    return Row(
        arm=arm,
        val_pixels="thermal",
        map50=mean,
        primary_map50=mean,
        map50_95=0.3,
        precision=0.1,
        recall=0.2,
        per_class_ap50=per_class,
        weights="w.pt",
        data="data.yaml",
        calibration_digest="",
    )


def test_the_thermal_arm_is_a_third_row_scored_on_the_floors_own_manifest(
    mirrored: Path, judge: Path, tmp_path: Path, stubbed_gpu: StubbedGpu
) -> None:
    """V - T only measures the sensor if T and J read the very same thermal val split."""
    (thermal,) = _checkpoints(tmp_path, "thermal.pt")
    out = tmp_path / "gate"
    assert main(_seeded_argv(mirrored, out, [judge], [thermal])) == 0

    rows = _rows(out)
    assert [row["arm"] for row in rows] == [CEILING, FLOOR, THERMAL_TRAINED]
    assert rows[2]["val_pixels"] == "thermal"
    assert rows[2]["weights"] == str(thermal)
    assert stubbed_gpu.calls[2]["data_yaml"] == stubbed_gpu.calls[1]["data_yaml"]
    assert stubbed_gpu.calls[2]["weights"] == thermal


def test_each_seed_is_its_own_csv_row_and_v_and_j_share_checkpoints(
    mirrored: Path, tmp_path: Path, stubbed_gpu: StubbedGpu
) -> None:
    """The CSV keeps raw passes so it can be re-averaged; J pairs with V seed by seed."""
    judges = _checkpoints(tmp_path, "v0.pt", "v1.pt")
    thermals = _checkpoints(tmp_path, "t0.pt", "t1.pt", "t2.pt")
    out = tmp_path / "gate"
    assert main(_seeded_argv(mirrored, out, judges, thermals)) == 0

    rows = _rows(out)
    assert [row["arm"] for row in rows] == [CEILING] * 2 + [FLOOR] * 2 + [THERMAL_TRAINED] * 3
    assert [call["weights"] for call in stubbed_gpu.calls] == [*judges, *judges, *thermals]


def test_an_arm_summary_is_the_seed_mean_with_sample_std() -> None:
    v, t = _summarise(
        [
            _row(CEILING, {"Fuse": 0.90, "Pole": 0.60}),
            _row(CEILING, {"Fuse": 0.70, "Pole": 0.60}),
            _row(THERMAL_TRAINED, {"Fuse": 0.40}),
        ]
    )
    assert (v.arm, v.seeds) == (CEILING, 2)
    assert v.per_class_ap50["Fuse"].mean == pytest.approx(0.80)
    assert v.per_class_ap50["Fuse"].std == pytest.approx(statistics.stdev([0.90, 0.70]))
    assert v.per_class_ap50["Pole"].std == pytest.approx(0.0)
    # One seed: spread is undefined, not zero -- and a class nobody scored is absent, not 0.0.
    assert math.isnan(t.per_class_ap50["Fuse"].std)
    assert "Pole" not in t.per_class_ap50


def test_the_gaps_are_differences_of_seed_means_over_classes_both_arms_scored() -> None:
    v, t = _summarise(
        [
            _row(CEILING, {"Fuse": 0.90, "Pole": 0.80}),
            _row(CEILING, {"Fuse": 0.70, "Pole": 0.80}),
            _row(THERMAL_TRAINED, {"Fuse": 0.30}),
        ]
    )
    assert _per_class_gap(v, t) == {"Fuse": pytest.approx(0.50)}


def test_the_report_carries_spread_and_both_gap_lines(
    mirrored: Path, tmp_path: Path, stubbed_gpu: StubbedGpu, caplog: pytest.LogCaptureFixture
) -> None:
    judges = _checkpoints(tmp_path, "v0.pt", "v1.pt")
    (thermal,) = _checkpoints(tmp_path, "t0.pt")
    stubbed_gpu.scores.extend(
        [
            _metrics(dict.fromkeys(CLASSES, 0.90), map50=0.90),
            _metrics(dict.fromkeys(CLASSES, 0.70), map50=0.70),
            _metrics(dict.fromkeys(CLASSES, 0.20), map50=0.20),
            _metrics(dict.fromkeys(CLASSES, 0.20), map50=0.20),
            _metrics(dict.fromkeys(CLASSES, 0.50), map50=0.50),
        ]
    )
    with caplog.at_level("INFO"):
        assert main(_seeded_argv(mirrored, tmp_path / "gate", judges, [thermal])) == 0

    spread = statistics.stdev([0.90, 0.70])
    assert f"| ceiling (visible) | 0.8000 ± {spread:.4f} |" in caplog.text
    assert "| thermal-trained (thermal) | 0.5000 |" in caplog.text
    assert "| V - T (sensor gap) | +0.3000 | **+0.3000** |" in caplog.text
    assert "| T - J (domain gap) | +0.3000 | **+0.3000** |" in caplog.text


def test_without_thermal_weights_the_report_is_todays(
    mirrored: Path,
    judge: Path,
    tmp_path: Path,
    stubbed_gpu: StubbedGpu,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO"):
        assert main(_argv(mirrored, judge, tmp_path / "gate")) == 0

    assert "| ceiling (visible) | 0.5000 | **0.5000** |" in caplog.text
    assert "±" not in caplog.text
    assert "sensor gap" not in caplog.text
    assert "domain gap" not in caplog.text


def test_a_passing_gate_marks_each_primary_class_by_its_sensor_gap(
    mirrored: Path,
    judge: Path,
    tmp_path: Path,
    stubbed_gpu: StubbedGpu,
    caplog: pytest.LogCaptureFixture,
) -> None:
    (thermal,) = _checkpoints(tmp_path, "thermal.pt")
    stubbed_gpu.scores.extend(
        [
            _metrics({"Fuse": 0.90, "Pole": 0.90}),
            _metrics({"Fuse": 0.20, "Pole": 0.20}),
            _metrics({"Fuse": 0.50, "Pole": 0.85}),
        ]
    )
    with caplog.at_level("INFO"):
        assert main(_seeded_argv(mirrored, tmp_path / "gate", [judge], [thermal])) == 0

    assert "Fuse: V - T +0.4000 -> passive GO" in caplog.text
    assert "Pole: V - T +0.0500 -> domain-gap GO" in caplog.text


def test_a_killed_gate_withholds_the_per_class_marks(
    mirrored: Path,
    judge: Path,
    tmp_path: Path,
    stubbed_gpu: StubbedGpu,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Q6 reads V - T only once V - J clears the kill line: no room at zero labels ends it."""
    (thermal,) = _checkpoints(tmp_path, "thermal.pt")
    stubbed_gpu.scores.extend(
        [
            _metrics(dict.fromkeys(CLASSES, 0.30)),
            _metrics(dict.fromkeys(CLASSES, 0.25)),
            _metrics(dict.fromkeys(CLASSES, 0.05)),
        ]
    )
    with caplog.at_level("INFO"):
        assert main(_seeded_argv(mirrored, tmp_path / "gate", [judge], [thermal])) == 0

    assert "KILL --" in caplog.text
    assert "no per-class GO marks" in caplog.text
    assert "passive GO" not in caplog.text


def test_refuses_a_missing_thermal_checkpoint_before_any_pass(
    mirrored: Path, judge: Path, tmp_path: Path, stubbed_gpu: StubbedGpu
) -> None:
    with pytest.raises(SystemExit, match="--thermal-weights not found"):
        main(_seeded_argv(mirrored, tmp_path / "gate", [judge], [tmp_path / "absent.pt"]))
    assert stubbed_gpu.calls == []


@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        (PASSIVE_THRESHOLD + 0.01, "passive GO"),
        (PASSIVE_THRESHOLD, "passive GO"),
        (PASSIVE_THRESHOLD - 0.01, "domain-gap GO"),
        (0.0, "domain-gap GO"),
        (-0.05, "domain-gap GO"),
    ],
)
def test_the_sensor_gap_band_is_the_pre_registered_one(gap: float, expected: str) -> None:
    """Driven off the constant, not 0.90 - 0.80: that is 0.0999... in floating point."""
    assert _sensor_gap_verdict(gap).startswith(expected)
