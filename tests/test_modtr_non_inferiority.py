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

import numpy as np
import pytest
import yaml
from scripts.modtr_non_inferiority import CSV_FILENAME, MEAN, main
from ultralytics.utils.metrics import ap_per_class

from t2o.data.manifest import ManifestError
from t2o.metrics.task import ImageStats, map50_from_stats
from test_resolvability import POOL_SIZE, _arm

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


def _three_class_arm(hit_rate: float, seed: int) -> dict[str, ImageStats]:
    """Image `i` holds one Fuse, Pole or Switch label (`i % 3`), found with probability
    `hit_rate`, plus a false positive of the next class along -- so every Switch-only image
    also carries a Fuse false positive, which a subset that dropped the image would lose."""
    rng = np.random.default_rng(seed)
    arm: dict[str, ImageStats] = {}
    for i in range(POOL_SIZE):
        stem = f"img{i:02d}"
        cls, wrong = float(i % 3), float((i + 1) % 3)
        arm[stem] = ImageStats(
            stem=stem,
            tp=np.repeat(np.array([[rng.random() < hit_rate], [False]]), 10, axis=1),
            conf=rng.uniform(0.0, 1.0, size=2),
            pred_cls=np.array([cls, wrong]),
            target_cls=np.array([cls]),
        )
    return arm


THREE_STRONG = {seed: _three_class_arm(hit_rate=0.9, seed=30 + seed) for seed in SEEDS}
THREE_WEAK = {seed: _three_class_arm(hit_rate=0.3, seed=40 + seed) for seed in SEEDS}
FUSE, POLE, SWITCH = 0, 1, 2


def _mean_ap50(records: dict[str, ImageStats], classes: set[int]) -> float:
    """The mean of ultralytics' own per-class AP@50 over `classes`, from every record."""
    values = list(records.values())
    stats = ap_per_class(
        np.concatenate([r.tp for r in values]),
        np.concatenate([r.conf for r in values]),
        np.concatenate([r.pred_cls for r in values]),
        np.concatenate([r.target_cls for r in values]),
    )
    ap, unique = stats[5], stats[6]
    return float(np.mean([ap[k, 0] for k, c in enumerate(unique) if c in classes]))


@pytest.fixture
def three(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    return _stub(monkeypatch, {LOOP: THREE_STRONG, MODTR: THREE_WEAK})


def test_a_class_subset_scores_the_mean_ap_over_those_classes(
    runs: Path,
    judge: Path,
    tmp_path: Path,
    three: list[dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    out = tmp_path / "score"
    main([*_argv(runs, judge, out, *SEEDS), "--primary-classes", "Fuse", "Pole"])

    rows = _rows(out)
    for row in rows[:-1]:
        seed = int(row["seed"])
        assert float(row["loop_map50"]) == pytest.approx(
            _mean_ap50(THREE_STRONG[seed], {FUSE, POLE})
        )
        assert float(row["modtr_map50"]) == pytest.approx(
            _mean_ap50(THREE_WEAK[seed], {FUSE, POLE})
        )
    assert all(row["classes"] == "Fuse Pole" for row in rows)
    assert _verdicts(caplog)[0].endswith("on Fuse, Pole")


def test_an_image_with_only_excluded_labels_stays_in_the_pool(
    runs: Path, judge: Path, tmp_path: Path, three: list[dict[str, Any]]
) -> None:
    out = tmp_path / "score"
    main([*_argv(runs, judge, out, *SEEDS), "--primary-classes", "Fuse", "Pole"])

    seed_0 = _rows(out)[0]
    assert seed_0["pool"] == str(POOL_SIZE)
    # Dropping the Switch-only images would also drop their Fuse false positives.
    kept_images = [r for r in THREE_STRONG[0].values() if r.target_cls[0] != SWITCH]
    assert float(seed_0["loop_map50"]) != pytest.approx(
        map50_from_stats([_without(r, SWITCH) for r in kept_images])
    )


def _without(record: ImageStats, cls: int) -> ImageStats:
    kept = record.pred_cls != cls
    return ImageStats(
        stem=record.stem,
        tp=record.tp[kept],
        conf=record.conf[kept],
        pred_cls=record.pred_cls[kept],
        target_cls=record.target_cls[record.target_cls != cls],
    )


def test_an_unknown_class_fails_before_any_val_run(
    runs: Path, judge: Path, tmp_path: Path, ahead: list[dict[str, Any]]
) -> None:
    with pytest.raises(SystemExit, match="dog"):
        main([*_argv(runs, judge, tmp_path / "score", *SEEDS), "--primary-classes", "Fuse", "dog"])
    assert ahead == []


def test_the_default_scores_every_class_as_before(
    runs: Path, judge: Path, tmp_path: Path, ahead: list[dict[str, Any]]
) -> None:
    out = tmp_path / "score"
    main(_argv(runs, judge, out, *SEEDS))

    rows = _rows(out)
    for row in rows[:-1]:
        seed = int(row["seed"])
        assert float(row["loop_map50"]) == map50_from_stats(list(STRONG[seed].values()))
        assert float(row["modtr_map50"]) == map50_from_stats(list(WEAK[seed].values()))
    assert all(row["classes"] == "Fuse Pole Switch Transformer" for row in rows)
