"""`scripts/margin_resolvability.py` -- the margin gate's server run.

The per-image collector is stubbed with hand-built records, as `test_gate_table.py` stubs
`evaluate_detector`. The untrained fixture detector scores mAP@50 0.0 on any labels, so through
it every interval is [0, 0] and every verdict "resolves" -- a test of the gate against it would
pass without testing anything (margin-gate pitfalls). What is tested here is the wiring: which
pixels each arm reads, that raw thermal is scored once, that a resumed run pays only for missing
rows, and that the verdict is read off every row in the CSV.
"""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from scripts.margin_resolvability import CSV_FILENAME, PAIRS, main

from t2o.data.budget import BudgetError
from t2o.data.manifest import DatasetManifest, ManifestError
from t2o.data.mirror import mirror_labels_onto_infrared
from t2o.metrics.task import ImageStats
from test_resolvability import _arm

THERMAL = "thermal"
# Seed 0's loop is far ahead of both baselines: a wide interval. Seed 1's three arms are one and
# the same records: a zero-width interval on both pairs.
THERMAL_RECORDS = _arm(hit_rate=0.5, seed=7)
RECORDS = {
    THERMAL: THERMAL_RECORDS,
    "loop-s0": _arm(hit_rate=0.9, seed=3),
    "control-s0": _arm(hit_rate=0.3, seed=4),
    "loop-s1": THERMAL_RECORDS,
    "control-s1": THERMAL_RECORDS,
}


def _copy(dataset_root: Path, tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    shutil.copytree(dataset_root, root)
    copied = root / "data.yaml"
    copied.write_text(copied.read_text().replace(f"path: {dataset_root}", f"path: {root}"))
    return copied


@pytest.fixture
def mirrored(dataset_root: Path, tmp_path: Path) -> Path:
    """A copy of the shared fixture with labels on the thermal side, as the custom set has."""
    copied = _copy(dataset_root, tmp_path, "mirrored")
    mirror_labels_onto_infrared(DatasetManifest.load(copied))
    return copied


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    """Stage-3 exports for seeds 0 and 1, both arms, in the server's `runs/e3b-*` layout."""
    root = tmp_path / "runs"
    for arm in ("loop", "control"):
        for seed in (0, 1):
            export = root / f"e3b-{arm}-s{seed}" / "stage3" / "translated"
            for split in ("train", "val"):
                (export / split / "images").mkdir(parents=True)
            (export / "data.yaml").write_text(
                yaml.safe_dump(
                    {
                        "path": str(export),
                        "train": "train/images",
                        "val": "val/images",
                        "names": ["Fuse", "Pole", "Switch", "Transformer"],
                    }
                )
            )
    return root


@pytest.fixture
def judge(tmp_path: Path) -> Path:
    """A stand-in for the reference weights. Never loaded -- the collector is stubbed."""
    path = tmp_path / "best.pt"
    path.write_bytes(b"")
    return path


@pytest.fixture
def collected(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every val run asked for, and answer it from `RECORDS`."""
    calls: list[dict[str, Any]] = []

    def fake_collect(**kwargs: Any) -> dict[str, ImageStats]:
        calls.append(kwargs)
        data_yaml = Path(kwargs["data_yaml"])
        if data_yaml.parent.name == THERMAL:
            return RECORDS[THERMAL]
        # .../e3b-<arm>-s<seed>/stage3/translated/data.yaml
        return RECORDS[data_yaml.parents[2].name.removeprefix("e3b-")]

    monkeypatch.setattr("t2o.metrics.task.collect_image_stats", fake_collect)
    return calls


def _argv(data: Path, runs: Path, judge: Path, out: Path, *seeds: int) -> list[str]:
    return [
        "--data",
        str(data),
        "--out",
        str(out),
        "--loop-template",
        str(runs / "e3b-loop-s{seed}" / "stage3" / "translated"),
        "--control-template",
        str(runs / "e3b-control-s{seed}" / "stage3" / "translated"),
        "--reference-weights",
        str(judge),
        "--resamples",
        "200",
        "--seeds",
        *(str(seed) for seed in seeds),
    ]


def _rows(out: Path) -> list[dict[str, str]]:
    with (out / CSV_FILENAME).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _verdicts(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("verdict: ")]


def _arm_of(call: dict[str, Any]) -> str:
    data_yaml = Path(call["data_yaml"])
    return THERMAL if data_yaml.parent.name == THERMAL else data_yaml.parents[2].name


def test_writes_one_row_per_seed_and_pair(
    mirrored: Path, runs: Path, judge: Path, tmp_path: Path, collected: list[dict[str, Any]]
) -> None:
    out = tmp_path / "gate"
    assert main(_argv(mirrored, runs, judge, out, 0, 1)) == 0

    rows = _rows(out)
    assert [(row["seed"], row["pair"]) for row in rows] == [
        ("0", PAIRS[0]),
        ("0", PAIRS[1]),
        ("1", PAIRS[0]),
        ("1", PAIRS[1]),
    ]
    assert all(row["n"] == "100" and row["pool"] == "40" for row in rows)
    for row in rows[:2]:  # seed 0: the loop is far ahead, so the interval is wide
        assert float(row["low"]) > 0.0
        assert float(row["half_width"]) > 2.0
        assert row["resolves"] == "False"
    for row in rows[2:]:  # seed 1: one set of records on both sides
        assert float(row["half_width"]) == 0.0
        assert row["resolves"] == "True"


def test_raw_thermal_is_scored_once_on_the_thermal_side(
    mirrored: Path, runs: Path, judge: Path, tmp_path: Path, collected: list[dict[str, Any]]
) -> None:
    main(_argv(mirrored, runs, judge, tmp_path / "gate", 0, 1))

    assert sorted(_arm_of(call) for call in collected) == [
        "e3b-control-s0",
        "e3b-control-s1",
        "e3b-loop-s0",
        "e3b-loop-s1",
        THERMAL,
    ]
    assert all(call["weights"] == judge for call in collected)
    thermal = next(Path(c["data_yaml"]) for c in collected if _arm_of(c) == THERMAL)
    lines = thermal.read_text().splitlines()
    val = next(line for line in lines if line.startswith("val:"))
    assert val.endswith(str(Path("val") / "infrared" / "images"))
    # ultralytics honours `path:` literally; the built manifest must not carry a stale one.
    assert not any(line.startswith("path:") for line in lines)


def test_the_verdict_resolves_only_when_every_row_does(
    mirrored: Path,
    runs: Path,
    judge: Path,
    tmp_path: Path,
    collected: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    main(_argv(mirrored, runs, judge, tmp_path / "narrow", 1))
    main(_argv(mirrored, runs, judge, tmp_path / "mixed", 0, 1))

    narrow, mixed = _verdicts(caplog)
    assert narrow.startswith("verdict: RESOLVES")
    assert mixed.startswith("verdict: DOES NOT RESOLVE")


def test_a_resumed_run_skips_written_rows_and_judges_the_whole_csv(
    mirrored: Path,
    runs: Path,
    judge: Path,
    tmp_path: Path,
    collected: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    out = tmp_path / "gate"
    main(_argv(mirrored, runs, judge, out, 0))
    collected.clear()

    # Only seed 1 is new, and seed 1 alone would resolve. Seed 0's wide row is still in the CSV.
    main(_argv(mirrored, runs, judge, out, 0, 1))

    assert sorted(_arm_of(call) for call in collected) == [
        "e3b-control-s1",
        "e3b-loop-s1",
        THERMAL,
    ]
    assert len(_rows(out)) == 4
    assert _verdicts(caplog)[-1].startswith("verdict: DOES NOT RESOLVE")


def test_a_missing_export_fails_before_any_val_run(
    mirrored: Path, runs: Path, judge: Path, tmp_path: Path, collected: list[dict[str, Any]]
) -> None:
    with pytest.raises(ManifestError, match="e3b-loop-s9"):
        main(_argv(mirrored, runs, judge, tmp_path / "gate", 0, 9))
    assert collected == []


def test_an_unlabelled_thermal_side_fails_before_any_val_run(
    dataset_root: Path, runs: Path, judge: Path, tmp_path: Path, collected: list[dict[str, Any]]
) -> None:
    """Every adapted public dataset arrives like this; scored, it would be a silent zero."""
    unmirrored = _copy(dataset_root, tmp_path, "unmirrored")

    with pytest.raises(BudgetError):
        main(_argv(unmirrored, runs, judge, tmp_path / "gate", 0))
    assert collected == []


def test_missing_reference_weights_fail_before_any_val_run(
    mirrored: Path, runs: Path, tmp_path: Path, collected: list[dict[str, Any]]
) -> None:
    with pytest.raises(SystemExit, match="--reference-weights not found"):
        main(_argv(mirrored, runs, tmp_path / "absent.pt", tmp_path / "gate", 0))
    assert collected == []
