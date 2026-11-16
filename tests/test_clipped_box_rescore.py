"""`scripts/clipped_box_rescore.py` -- the val re-score and plan Q3's fix-or-disclose rule.

Both val passes are stubbed: `evaluate_detector` answers from a table of mAPs, and
`collect_image_stats` from hand-built per-image records. No detector is loaded. The weights and
`data.yaml` files are empty placeholders, because the script only checks that they exist.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scripts.clipped_box_rescore import RescoreError, main

from t2o.metrics.task import ImageStats, TaskMetrics


def _records(hits: list[bool]) -> dict[str, ImageStats]:
    """One image per entry: one box, and one prediction that matches it when `hits` says so."""
    return {
        f"img{i}": ImageStats(
            stem=f"img{i}",
            tp=np.full((1, 10), hit),
            conf=np.array([0.9]),
            pred_cls=np.array([0.0]),
            target_cls=np.array([0.0]),
        )
        for i, hit in enumerate(hits)
    }


class Stub:
    """Answers both val passes from `(weights stem, data.yaml stem)`."""

    def __init__(self, map50: dict[tuple[str, str], float]) -> None:
        self.map50 = map50
        # Original trees hold four boxes, all matched. Drop trees lose one matched box.
        self.records = {"original": _records([True] * 4), "drop": _records([True] * 3)}

    @staticmethod
    def _key(kwargs: dict[str, Any]) -> tuple[str, str]:
        return Path(kwargs["weights"]).stem, Path(kwargs["data_yaml"]).stem

    def evaluate(self, **kwargs: Any) -> TaskMetrics:
        value = self.map50[self._key(kwargs)]
        return TaskMetrics(0.0, 0.0, value, value / 2, {}, {})

    def collect(self, **kwargs: Any) -> dict[str, ImageStats]:
        tree = self._key(kwargs)[1]
        return self.records["original" if tree.endswith("original") else "drop"]


def _install(monkeypatch: pytest.MonkeyPatch, map50: dict[tuple[str, str], float]) -> None:
    stub = Stub(map50)
    monkeypatch.setattr("t2o.metrics.task.evaluate_detector", stub.evaluate)
    monkeypatch.setattr("t2o.metrics.task.collect_image_stats", stub.collect)


def _arm(root: Path, name: str, spread: float, thresholds: tuple[str, ...] = ("0.5",)) -> list[str]:
    """One `--arm` argument list, with every file it names created empty."""
    weights = root / f"{name}.pt"
    original = root / f"{name}-original.yaml"
    drops = [root / f"{name}-drop{t}.yaml" for t in thresholds]
    for path in (weights, original, *drops):
        path.write_text("")
    specs = [f"{t}={path}" for t, path in zip(thresholds, drops, strict=True)]
    return ["--arm", name, str(weights), str(spread), str(original), *specs]


def _run(tmp_path: Path, *arms: list[str]) -> list[dict[str, str]]:
    out = tmp_path / "rescore.csv"
    assert main([*(field for arm in arms for field in arm), "--out", str(out)]) == 0
    with out.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_a_small_shift_within_the_spread_is_disclosed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(
        monkeypatch,
        {
            ("a", "a-original"): 0.80,
            ("a", "a-drop0.5"): 0.81,
            ("b", "b-original"): 0.70,
            ("b", "b-drop0.5"): 0.70,
        },
    )

    rows = _run(tmp_path, _arm(tmp_path, "a", 0.02), _arm(tmp_path, "b", 0.02))

    assert [row["verdict"] for row in rows] == ["disclose", "disclose"]
    first = rows[0]
    assert float(first["delta_map50"]) == pytest.approx(0.01)
    assert float(first["delta_map50_95"]) == pytest.approx(0.005)
    assert (first["rank_original"], first["rank_drop"]) == ("1", "1")
    assert (first["rank_changed"], first["exceeds_spread"]) == ("False", "False")


def test_the_dropped_boxes_and_lost_matches_are_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, {("a", "a-original"): 0.8, ("a", "a-drop0.5"): 0.8})

    (row,) = _run(tmp_path, _arm(tmp_path, "a", 0.0))

    # The stub's drop tree is one matched box short of the original.
    assert (row["boxes_dropped"], row["lost_matches"]) == ("1", "1")


def test_a_delta_beyond_the_seed_spread_says_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, {("a", "a-original"): 0.80, ("a", "a-drop0.5"): 0.75})

    (row,) = _run(tmp_path, _arm(tmp_path, "a", 0.02))

    assert (row["exceeds_spread"], row["verdict"]) == ("True", "fix")


def test_a_rank_change_says_fix_for_every_arm_at_that_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(
        monkeypatch,
        {
            ("a", "a-original"): 0.80,
            ("a", "a-drop0.5"): 0.80,
            ("a", "a-drop0.75"): 0.80,
            ("b", "b-original"): 0.79,
            ("b", "b-drop0.5"): 0.79,
            ("b", "b-drop0.75"): 0.82,  # within its loose spread, but it overtakes a
        },
    )
    thresholds = ("0.5", "0.75")

    rows = _run(
        tmp_path, _arm(tmp_path, "a", 0.01, thresholds), _arm(tmp_path, "b", 0.05, thresholds)
    )

    by_threshold = {(row["arm"], row["threshold"]): row for row in rows}
    assert by_threshold[("b", "0.75")]["exceeds_spread"] == "False"
    assert by_threshold[("b", "0.75")]["rank_changed"] == "True"
    assert {row["verdict"] for row in rows if row["threshold"] == "0.75"} == {"fix"}
    assert {row["verdict"] for row in rows if row["threshold"] == "0.5"} == {"disclose"}


def test_arms_with_different_thresholds_are_refused(tmp_path: Path) -> None:
    with pytest.raises(RescoreError, match="every arm needs the same ones"):
        _run(tmp_path, _arm(tmp_path, "a", 0.0, ("0.5",)), _arm(tmp_path, "b", 0.0, ("0.75",)))


def test_a_missing_tree_is_refused_before_any_score(tmp_path: Path) -> None:
    arm = _arm(tmp_path, "a", 0.0)
    (tmp_path / "a-drop0.5.yaml").unlink()

    with pytest.raises(RescoreError, match="does not exist"):
        _run(tmp_path, arm)


def test_a_drop_tree_without_its_threshold_is_refused(tmp_path: Path) -> None:
    arm = _arm(tmp_path, "a", 0.0)
    arm[-1] = arm[-1].partition("=")[2]

    with pytest.raises(RescoreError, match="THRESHOLD=DROP_DATA_YAML"):
        _run(tmp_path, arm)
