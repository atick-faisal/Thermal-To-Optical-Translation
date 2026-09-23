"""`scripts/annotation_sweep.py` -- E8's low-annotation sweep driver.

Both GPU entry points are stubbed. What is worth testing here is not ultralytics -- it has
its own tests -- but the wiring around it: that each arm reads the pixels it claims to read,
that a resumed sweep does not re-pay for a cell it already has, and that the cheap anchors
run before the expensive fine-tunes. Those are the three ways an eight-hour run comes back
wrong rather than late.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image
from scripts.annotation_sweep import Row, _append, _budget_manifest, _completed, _plan, main

from t2o.data.manifest import DatasetManifest
from t2o.engine.detector_stage import DetectorResult
from t2o.metrics.task import TaskMetrics

N_EXPORT_TRAIN = 5
N_EXPORT_VAL = 3


@pytest.fixture
def thermal_labelled(dataset_root: Path, tmp_path: Path) -> Path:
    """The paired fixture with labels mirrored onto the infrared side.

    Same reasoning as `tests/test_budget.py`: the shared fixture leaves `infrared/labels`
    empty like every adapted public dataset, and arm A is only buildable on a dataset whose
    thermal side carries boxes.
    """
    root = tmp_path / "paired"
    shutil.copytree(dataset_root, root)
    copied = root / "data.yaml"
    copied.write_text(copied.read_text().replace(f"path: {dataset_root}", f"path: {root}"))
    for split in ("train", "val"):
        for label in (root / split / "visible" / "labels").glob("*.txt"):
            shutil.copy(label, root / split / "infrared" / "labels" / label.name)
    return copied


@pytest.fixture
def export_template(tmp_path: Path) -> str:
    """Two stage-3 translated exports, one per seed, in ultralytics' own export layout.

    `{split}/{images,labels}` rather than the source's `{split}/{modality}/{images,labels}`
    -- that difference is exactly what the sweep has to survive, since arm A reads one layout
    and arm B the other through the same code path.
    """
    for seed in (0, 1):
        root = tmp_path / f"export-s{seed}" / "translated"
        for split, count in (("train", N_EXPORT_TRAIN), ("val", N_EXPORT_VAL)):
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir(parents=True)
            for i in range(count):
                pixels = np.full((8, 8, 3), seed * 40 + i, dtype=np.uint8)
                Image.fromarray(pixels, mode="RGB").save(root / split / "images" / f"{i:03d}.png")
                (root / split / "labels" / f"{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (root / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": str(root),
                    "train": "train/images",
                    "val": "val/images",
                    "nc": 4,
                    "names": ["Fuse", "Pole", "Switch", "Transformer"],
                }
            )
        )
    return str(tmp_path / "export-s{seed}" / "translated")


@pytest.fixture
def stubbed_gpu(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record what would have been trained and evaluated, and run neither."""
    calls: list[dict[str, Any]] = []

    def fake_train(**kwargs: Any) -> DetectorResult:
        calls.append({"kind": "train", **kwargs})
        return DetectorResult(
            weights=Path("best.pt"), precision=0.1, recall=0.2, map50=0.3, map50_95=0.4
        )

    def fake_evaluate(**kwargs: Any) -> TaskMetrics:
        calls.append({"kind": "evaluate", **kwargs})
        return TaskMetrics(
            precision=0.5,
            recall=0.6,
            map50=0.7,
            map50_95=0.8,
            per_class_ap50={},
            per_class_ap50_95={},
        )

    monkeypatch.setattr("t2o.engine.detector_stage.train_detector", fake_train)
    monkeypatch.setattr("t2o.metrics.task.evaluate_detector", fake_evaluate)
    return calls


def _sweep_argv(
    thermal: Path, template: str, out: Path, judge: Path, **overrides: str
) -> list[str]:
    argv = [
        "--data",
        str(thermal),
        "--out",
        str(out),
        "--control-template",
        template,
        "--loop-template",
        template,
        "--reference-weights",
        str(judge),
        "--budgets",
        "2",
        "4",
        "--seeds",
        "0",
        "1",
        "--epochs",
        "1",
    ]
    for flag, value in overrides.items():
        argv += [f"--{flag.replace('_', '-')}", *value.split()]
    return argv


@pytest.fixture
def judge(tmp_path: Path) -> Path:
    weights = tmp_path / "reference.pt"
    weights.write_bytes(b"not a real checkpoint")
    return weights


def _rows(out: Path) -> list[dict[str, str]]:
    import csv

    with (out / "e8-tidy.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_cheap_anchors_run_before_any_fine_tune() -> None:
    """An interrupted sweep should leave a curve with a thin tail, not a tall left edge.

    The zero-shot arms are minutes and they are the flat lines the trained arms are read
    against, so ordering them first is what makes a partial CSV interpretable at all.
    """
    cells = _plan(("A0", "C", "D", "A", "B"), budgets=(50, 10), seeds=(0, 1), total=600)

    arms = [arm for arm, _, _ in cells]
    assert set(arms[: arms.index("A")]) == {"A0", "C", "D"}
    trained = [(arm, n) for arm, n, _ in cells if arm in ("A", "B")]
    assert [n for _, n in trained] == sorted(n for _, n in trained)


def test_the_unseeded_anchor_is_written_once() -> None:
    """Arm A0 is one judge on one fixed split; duplicating it per seed would invent spread."""
    cells = _plan(("A0", "C"), budgets=(10,), seeds=(0, 1, 2), total=600)

    assert cells.count(("A0", 0, -1)) == 1
    assert sum(1 for arm, _, _ in cells if arm == "C") == 3


def test_a_resumed_sweep_skips_exactly_the_cells_already_written(
    tmp_path: Path,
    thermal_labelled: Path,
    export_template: str,
    judge: Path,
    stubbed_gpu: list[dict[str, Any]],
) -> None:
    out = tmp_path / "sweep"
    assert main(_sweep_argv(thermal_labelled, export_template, out, judge)) == 0
    first = len(stubbed_gpu)
    rows = _rows(out)

    stubbed_gpu.clear()
    assert main(_sweep_argv(thermal_labelled, export_template, out, judge)) == 0

    assert first > 0
    assert stubbed_gpu == [], "a completed cell was paid for twice"
    assert _rows(out) == rows


def test_arm_a_trains_on_thermal_and_arm_b_on_the_export(
    tmp_path: Path,
    thermal_labelled: Path,
    export_template: str,
    judge: Path,
    stubbed_gpu: list[dict[str, Any]],
) -> None:
    """The one claim the whole sweep rests on: the arms differ in pixels and nothing else."""
    out = tmp_path / "sweep"
    main(_sweep_argv(thermal_labelled, export_template, out, judge, arms="A B", budgets="4"))

    by_arm = {}
    for call in stubbed_gpu:
        arm = call["name"].split("-")[0]
        listed = Path(yaml.safe_load(Path(call["data_yaml"]).read_text())["train"]).read_text()
        by_arm.setdefault(arm, []).append([Path(line) for line in listed.splitlines()])

    # The two layouts differ as well as the pixels: the source nests a modality segment
    # (`train/infrared/images`) and an export does not (`train/images`).
    for paths in by_arm["A"]:
        assert all(p.parent.parent.name == "infrared" for p in paths)
    for paths in by_arm["B"]:
        assert all(p.parents[2].name == "translated" and "infrared" not in p.parts for p in paths)
    # Same scenes on both sides at a given seed is what makes the contrast paired.
    assert [len(p) for p in by_arm["A"]] == [len(p) for p in by_arm["B"]] == [4, 4]


def test_each_arm_records_which_model_and_which_pixels_produced_it(
    tmp_path: Path,
    thermal_labelled: Path,
    export_template: str,
    judge: Path,
    stubbed_gpu: list[dict[str, Any]],
) -> None:
    """The arms are not one model family, and a CSV that hid that would be wrong."""
    out = tmp_path / "sweep"
    main(_sweep_argv(thermal_labelled, export_template, out, judge))

    by_arm = {row["arm"]: row for row in _rows(out)}
    assert by_arm["A"]["detector"] == by_arm["B"]["detector"] == "finetuned"
    assert by_arm["A0"]["detector"] == by_arm["C"]["detector"] == "reference"
    assert by_arm["A"]["val_pixels"] == by_arm["A0"]["val_pixels"] == "thermal"
    assert by_arm["C"]["val_pixels"] == "translated-l0"
    assert by_arm["D"]["val_pixels"] == "translated-lpos"
    # Arm D's annotations were spent inside the translator, not on a detector -- but they were
    # spent, and the accounting has to say so.
    assert int(by_arm["D"]["n_annotated"]) == 5
    assert int(by_arm["C"]["n_annotated"]) == 0


def test_the_export_seed_is_paired_with_the_detector_seed(
    tmp_path: Path,
    thermal_labelled: Path,
    export_template: str,
    judge: Path,
    stubbed_gpu: list[dict[str, Any]],
) -> None:
    """Otherwise arm B's error bars would look like translator variance and would not be."""
    out = tmp_path / "sweep"
    main(_sweep_argv(thermal_labelled, export_template, out, judge, arms="B", budgets="4"))

    for call in stubbed_gpu:
        seed = call["seed"]
        listed = Path(yaml.safe_load(Path(call["data_yaml"]).read_text())["train"]).read_text()
        assert all(f"export-s{seed}" in line for line in listed.splitlines())


def test_a_budget_bigger_than_the_train_split_is_refused(
    tmp_path: Path,
    thermal_labelled: Path,
    export_template: str,
    judge: Path,
) -> None:
    out = tmp_path / "sweep"
    argv = _sweep_argv(thermal_labelled, export_template, out, judge, budgets="600")
    with pytest.raises(SystemExit, match="not in 1"):
        main(argv)


def test_a_missing_judge_fails_before_the_first_gpu_hour(
    tmp_path: Path,
    thermal_labelled: Path,
    export_template: str,
) -> None:
    """Discovering this at cell 30 would waste every cell before it."""
    out = tmp_path / "sweep"
    argv = _sweep_argv(thermal_labelled, export_template, out, tmp_path / "absent.pt")
    with pytest.raises(SystemExit, match="reference-weights"):
        main(argv)


def test_a_budget_manifest_holds_exactly_the_requested_count(
    thermal_labelled: Path, tmp_path: Path
) -> None:
    """`N=25` labelling a 24-image run would corrupt the x-axis silently."""
    manifest = DatasetManifest.load(thermal_labelled)
    for n in (1, 3, 5):
        written = _budget_manifest(manifest, tmp_path / f"n{n}", n, seed=0, infrared=True)
        listed = Path(yaml.safe_load(written.read_text())["train"]).read_text().splitlines()
        assert len(listed) == n


def test_a_written_row_round_trips_through_the_resume_index(tmp_path: Path) -> None:
    csv_path = tmp_path / "e8-tidy.csv"
    row = Row(
        arm="A",
        n_annotated=25,
        seed=1,
        detector="finetuned",
        val_pixels="thermal",
        epochs=50,
        precision=0.1,
        recall=0.2,
        map50=0.3,
        map50_95=0.4,
        weights="w.pt",
        data="d.yaml",
    )
    _append(csv_path, row)
    _append(csv_path, row)  # a duplicate on disk must not confuse the index

    assert _completed(csv_path) == {("A", 25, 1)}
