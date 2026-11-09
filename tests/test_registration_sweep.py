"""`scripts/registration_sweep.py` -- the label-transfer half of C2's registration-error axis.

The GPU entry points are stubbed in every test but the `slow` one. What can go wrong here is
the wiring, not ultralytics: a level trained on the wrong boxes, scored on a perturbed val, a
row that cannot be traced to its field, or a resumed sweep paying for a cell twice.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from scripts.registration_sweep import CSV_FILENAME, _plan, main

from conftest import IMG_SIZE
from t2o.data.manifest import DatasetManifest
from t2o.data.mirror import PROVENANCE_FILENAME, mirror_labels_onto_infrared
from t2o.data.misalignment import Misalignment
from t2o.engine.detector_stage import DetectorResult
from t2o.metrics.task import TaskMetrics

HEIGHT, WIDTH = IMG_SIZE


@pytest.fixture
def aligned(dataset_root: Path, tmp_path: Path) -> Path:
    """A copy of the shared fixture with thermal labels, standing in for an aligned dataset.

    Copied and re-pathed as in `test_misalignment.py::aligned`: the fixture is session-scoped
    and its `path:` is absolute.
    """
    root = tmp_path / "aligned"
    shutil.copytree(dataset_root, root)
    manifest_path = root / "data.yaml"
    manifest_path.write_text(
        manifest_path.read_text().replace(f"path: {dataset_root}", f"path: {root}")
    )
    mirror_labels_onto_infrared(DatasetManifest.load(manifest_path))
    return manifest_path


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
            per_class_ap50={"Fuse": 0.2, "Pole": 0.4},
            per_class_ap50_95={"Fuse": 0.1, "Pole": 0.2},
        )

    monkeypatch.setattr("t2o.engine.detector_stage.train_detector", fake_train)
    monkeypatch.setattr("t2o.metrics.task.evaluate_detector", fake_evaluate)
    return calls


def _argv(data: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--data",
        str(data),
        "--dataset",
        "custom",
        "--out",
        str(out),
        "--levels",
        "0",
        "24",
        "--seeds",
        "0",
        "1",
        "--epochs",
        "1",
        *extra,
    ]


def _rows(out: Path) -> list[dict[str, str]]:
    with (out / CSV_FILENAME).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _trained(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [call for call in calls if call["kind"] == "train"]


def test_every_level_runs_at_one_seed_before_the_next_seed_starts() -> None:
    """An interrupted sweep should leave a whole curve at fewer seeds, not one end of it."""
    cells = _plan([12.0, 0.0, 24.0, 12.0], seeds=[0, 1])

    assert cells == [(0.0, 0), (12.0, 0), (24.0, 0), (0.0, 1), (12.0, 1), (24.0, 1)]


def test_a_resumed_sweep_skips_exactly_the_cells_already_written(
    aligned: Path, tmp_path: Path, stubbed_gpu: list[dict[str, Any]]
) -> None:
    out = tmp_path / "sweep"
    assert main(_argv(aligned, out)) == 0
    rows = _rows(out)
    assert len(_trained(stubbed_gpu)) == len(rows) == 4

    stubbed_gpu.clear()
    assert main(_argv(aligned, out)) == 0

    assert stubbed_gpu == [], "a completed cell was paid for twice"
    assert _rows(out) == rows


def test_each_level_trains_on_its_own_moved_boxes_and_scores_on_the_untouched_val(
    aligned: Path, tmp_path: Path, stubbed_gpu: list[dict[str, Any]]
) -> None:
    """The claim the curve rests on: only the train boxes differ between levels."""
    out = tmp_path / "sweep"
    main(_argv(aligned, out))
    source = aligned.parent

    by_level: dict[str, set[str]] = {}
    for call in _trained(stubbed_gpu):
        manifest = yaml.safe_load(Path(call["data_yaml"]).read_text())
        train = [Path(line) for line in Path(manifest["train"]).read_text().splitlines()]
        assert all(p.parent.parent.name == "infrared" for p in train)
        by_level.setdefault(call["name"].split("-")[1], set()).add(str(train[0].parents[2]))

        # Val is the thermal side of a byte-for-byte copy of the source's val split.
        val = Path(manifest["val"])
        assert val.parent.name == "infrared"
        for label in (source / "val" / "infrared" / "labels").glob("*.txt"):
            assert (val.parent / "labels" / label.name).read_bytes() == label.read_bytes()

    # One tree per level, shared by both seeds.
    assert {level: len(trees) for level, trees in by_level.items()} == {"l0": 1, "l24": 1}
    moved = Path(next(iter(by_level["l24"]))) / "infrared" / "labels"
    changed = [
        label.name
        for label in (source / "train" / "infrared" / "labels").glob("*.txt")
        if (moved / label.name).read_bytes() != label.read_bytes()
    ]
    assert changed, "the 24 px level trained on the source's boxes"


def test_every_row_is_stamped_with_its_field_and_scored_by_the_val_pass(
    aligned: Path, tmp_path: Path, stubbed_gpu: list[dict[str, Any]]
) -> None:
    out = tmp_path / "sweep"
    main(_argv(aligned, out, "--primary-classes", "Fuse", "Pole"))

    for row in _rows(out):
        level = float(row["misalignment_px"])
        assert level in (0.0, 24.0)
        assert row["misalignment_digest"] == Misalignment(HEIGHT, WIDTH, level).digest()
        assert row["dataset"] == "custom"
        assert (row["init_weights"], row["epochs"]) == ("yolo11n.pt", "1")
        assert row["boxes_dropped"] == "0"
        assert row["calibration_digest"] == ""  # a plain mirror carries no de-roll
        assert float(row["map50"]) == 0.7  # the val pass's, not the training stub's 0.3
        assert float(row["primary_map50"]) == pytest.approx(0.3)
        assert row["ap50_Switch"] == ""  # never scored, so blank rather than 0.0


def test_a_de_rolled_source_stamps_its_calibration_digest(
    aligned: Path, tmp_path: Path, stubbed_gpu: list[dict[str, Any]]
) -> None:
    """FLIR's 0 px point is its label-transfer reference only on the de-rolled labels."""
    marker = aligned.parent / "train" / "infrared" / PROVENANCE_FILENAME
    marker.write_text(json.dumps({"calibration_digest": "abc123"}))
    out = tmp_path / "sweep"
    main(_argv(aligned, out))

    assert {row["calibration_digest"] for row in _rows(out)} == {"abc123"}


def test_a_finished_tree_for_another_field_is_refused(
    aligned: Path, tmp_path: Path, stubbed_gpu: list[dict[str, Any]]
) -> None:
    out = tmp_path / "sweep"
    main(_argv(aligned, out, "--seeds", "0"))
    provenance = out / "trees" / "level-24px" / "train" / "infrared" / PROVENANCE_FILENAME
    provenance.write_text(json.dumps({"misalignment_digest": "someone-else"}))
    (out / CSV_FILENAME).unlink()

    with pytest.raises(SystemExit, match="another level or frame size"):
        main(_argv(aligned, out, "--seeds", "0"))


def test_a_half_written_tree_is_refused_not_overwritten(
    aligned: Path, tmp_path: Path, stubbed_gpu: list[dict[str, Any]]
) -> None:
    """No `data.yaml` means an interrupted copy, and deleting it is the human's call."""
    leftover = tmp_path / "sweep" / "trees" / "level-0px" / "leftover.txt"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("")

    with pytest.raises(SystemExit, match="not empty"):
        main(_argv(aligned, tmp_path / "sweep"))
    assert leftover.is_file()


@pytest.mark.parametrize(
    ("extra", "message"),
    [(("--primary-classes", "car_stop"), "primary-classes"), (("--levels", "-3"), "levels")],
)
def test_a_bad_argument_fails_before_the_first_gpu_minute(
    aligned: Path,
    tmp_path: Path,
    stubbed_gpu: list[dict[str, Any]],
    extra: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(SystemExit, match=message):
        main([*_argv(aligned, tmp_path / "sweep"), *extra])
    assert stubbed_gpu == []


@pytest.mark.slow
def test_a_tiny_sweep_trains_and_scores_end_to_end(
    aligned: Path, detector_weights: Path, tmp_path: Path
) -> None:
    """The real ultralytics train and val, on CPU, through every seam the stubs stand in for."""
    out = tmp_path / "sweep"
    argv = [
        "--data",
        str(aligned),
        "--dataset",
        "custom",
        "--out",
        str(out),
        "--levels",
        "0",
        "6",
        "--seeds",
        "0",
        "--init-weights",
        str(detector_weights),
        "--epochs",
        "1",
        "--imgsz",
        "64",
        "--batch",
        "2",
        "--workers",
        "0",
        "--device",
        "cpu",
    ]
    assert main(argv) == 0

    rows = _rows(out)
    assert [float(row["misalignment_px"]) for row in rows] == [0.0, 6.0]
    for row in rows:
        assert math.isfinite(float(row["map50"]))
        assert Path(row["weights"]).is_file()
