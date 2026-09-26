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
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from scripts.gate_table import KILL_THRESHOLD, STRONG_THRESHOLD, _verdict, main

from t2o.data.manifest import DatasetManifest
from t2o.data.mirror import mirror_labels_onto_infrared
from t2o.metrics.task import TaskMetrics

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


def _argv(data: Path, judge: Path, out: Path, *primary: str) -> list[str]:
    argv = ["--data", str(data), "--weights", str(judge), "--out", str(out), "--device", "cpu"]
    if primary:
        argv += ["--primary-classes", *primary]
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
