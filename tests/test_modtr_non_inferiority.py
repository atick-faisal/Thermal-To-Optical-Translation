"""`scripts/modtr_non_inferiority.py` -- Margin's non-inferiority score.

The per-image collector is stubbed with hand-built records, as `test_margin_resolvability.py`
stubs it: the untrained fixture detector scores mAP@50 0.0 on any labels, so through it every
interval is [0, 0] and every verdict passes (margin-gate pitfalls). What is tested here is the
wiring: which export each arm reads, that each is scored once by the reference judge, what the
CSV holds, which verdict is logged, and that every failure lands before the first val run.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from scripts.modtr_non_inferiority import CSV_FILENAME, MEAN, main

from t2o.data.manifest import ManifestError
from t2o.metrics.task import ImageStats
from test_resolvability import _arm

SEEDS = (0, 1, 3)
LOOP = "e3b-loop"
MODTR = "modtr"
STRONG = {seed: _arm(hit_rate=0.9, seed=10 + seed) for seed in SEEDS}
WEAK = {seed: _arm(hit_rate=0.3, seed=20 + seed) for seed in SEEDS}


def _export(root: Path, run: str, stage: int) -> None:
    export = root / run / f"stage{stage}" / "translated"
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


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    """Final exports for every seed: the loop's at stage 3, ModTr's single stage at stage 0."""
    root = tmp_path / "runs"
    for seed in SEEDS:
        _export(root, f"{LOOP}-s{seed}", stage=3)
        _export(root, f"{MODTR}-s{seed}", stage=0)
    return root


@pytest.fixture
def judge(tmp_path: Path) -> Path:
    """A stand-in for the reference weights. Never loaded -- the collector is stubbed."""
    path = tmp_path / "best.pt"
    path.write_bytes(b"")
    return path


def _stub(
    monkeypatch: pytest.MonkeyPatch, records: dict[str, dict[int, dict[str, ImageStats]]]
) -> list[dict[str, Any]]:
    """Record every val run asked for, and answer it from `records[arm][seed]`."""
    calls: list[dict[str, Any]] = []

    def fake_collect(**kwargs: Any) -> dict[str, ImageStats]:
        calls.append(kwargs)
        # .../<arm>-s<seed>/stage<k>/translated/data.yaml
        arm, _, seed = Path(kwargs["data_yaml"]).parents[2].name.rpartition("-s")
        return records[arm][int(seed)]

    monkeypatch.setattr("t2o.metrics.task.collect_image_stats", fake_collect)
    return calls


@pytest.fixture
def ahead(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The loop far ahead of ModTr on every seed."""
    return _stub(monkeypatch, {LOOP: STRONG, MODTR: WEAK})


@pytest.fixture
def behind(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """ModTr far ahead of the loop on every seed."""
    return _stub(monkeypatch, {LOOP: WEAK, MODTR: STRONG})


def _argv(runs: Path, judge: Path, out: Path, *seeds: int) -> list[str]:
    return [
        "--out",
        str(out),
        "--loop-template",
        str(runs / f"{LOOP}-s{{seed}}" / "stage3" / "translated"),
        "--modtr-template",
        str(runs / f"{MODTR}-s{{seed}}" / "stage0" / "translated"),
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


def test_writes_one_row_per_seed_then_the_seed_averaged_row(
    runs: Path, judge: Path, tmp_path: Path, ahead: list[dict[str, Any]]
) -> None:
    out = tmp_path / "score"
    assert main(_argv(runs, judge, out, *SEEDS)) == 0

    rows = _rows(out)
    assert [row["seed"] for row in rows] == ["0", "1", "3", MEAN]
    assert all(row["pool"] == "40" and row["margin"] == "5.0" for row in rows)
    for row in rows[:-1]:  # the CI is over the seed average, so a seed row has none
        assert float(row["delta"]) > 0.0
        assert row["low"] == row["high"] == row["non_inferior"] == ""
    mean = rows[-1]
    deltas = [float(row["delta"]) for row in rows[:-1]]
    assert float(mean["delta"]) == pytest.approx(sum(deltas) / len(deltas))
    assert float(mean["low"]) > 0.0
    assert float(mean["seed_sd"]) > 0.0
    assert mean["non_inferior"] == "True"


def test_each_export_is_scored_once_by_the_reference_judge(
    runs: Path, judge: Path, tmp_path: Path, ahead: list[dict[str, Any]]
) -> None:
    main(_argv(runs, judge, tmp_path / "score", *SEEDS))

    assert sorted(Path(call["data_yaml"]).parents[2].name for call in ahead) == sorted(
        f"{arm}-s{seed}" for arm in (LOOP, MODTR) for seed in SEEDS
    )
    assert all(call["weights"] == judge for call in ahead)


def test_the_verdict_follows_the_lower_bound(
    runs: Path,
    judge: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    _stub(monkeypatch, {LOOP: STRONG, MODTR: WEAK})
    main(_argv(runs, judge, tmp_path / "ahead", *SEEDS))
    _stub(monkeypatch, {LOOP: WEAK, MODTR: STRONG})
    main(_argv(runs, judge, tmp_path / "behind", *SEEDS))

    passed, failed = _verdicts(caplog)
    assert passed.startswith("verdict: NON-INFERIOR")
    assert failed.startswith("verdict: NOT NON-INFERIOR")
    assert _rows(tmp_path / "behind")[-1]["non_inferior"] == "False"


def test_an_existing_result_is_refused_before_any_val_run(
    runs: Path, judge: Path, tmp_path: Path, behind: list[dict[str, Any]]
) -> None:
    out = tmp_path / "score"
    main(_argv(runs, judge, out, *SEEDS))
    behind.clear()

    with pytest.raises(SystemExit, match="already holds a result"):
        main(_argv(runs, judge, out, *SEEDS))
    assert behind == []


def test_a_missing_modtr_export_fails_before_any_val_run(
    runs: Path, judge: Path, tmp_path: Path, ahead: list[dict[str, Any]]
) -> None:
    """Seed 2 is in the loop's campaign but not ModTr's: the pairing gap the seed list avoids."""
    _export(runs, f"{LOOP}-s2", stage=3)

    with pytest.raises(ManifestError, match="modtr-s2"):
        main(_argv(runs, judge, tmp_path / "score", 0, 2))
    assert ahead == []


def test_missing_reference_weights_fail_before_any_val_run(
    runs: Path, tmp_path: Path, ahead: list[dict[str, Any]]
) -> None:
    with pytest.raises(SystemExit, match="--reference-weights not found"):
        main(_argv(runs, tmp_path / "absent.pt", tmp_path / "score", *SEEDS))
    assert ahead == []
