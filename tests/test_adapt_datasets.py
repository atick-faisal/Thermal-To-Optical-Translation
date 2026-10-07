"""`scripts/adapt_datasets.py` -- the adapter registry and its raw-folder lookup.

`msrs-day` is the one tree whose raw folder is not named after it: it is a daytime view of
`raw/msrs`. These tests run the script's `main` against the MSRS-shaped fixture from
`test_adapters`.
"""

from __future__ import annotations

from pathlib import Path

from scripts.adapt_datasets import ADAPTERS, RAW_DIRS, main

from test_adapters import TEST_STEMS, TRAIN_STEMS, _build_msrs_raw


def test_msrs_day_is_registered_and_reads_raw_msrs() -> None:
    assert "msrs-day" in ADAPTERS
    assert RAW_DIRS["msrs-day"] == "msrs"


def test_msrs_day_builds_a_daytime_tree_from_raw_msrs(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _build_msrs_raw(raw_root / "msrs")
    dest_root = tmp_path / "processed"

    exit_code = main(
        ["--dataset", "msrs-day", "--raw-root", str(raw_root), "--dest-root", str(dest_root)]
    )

    assert exit_code == 0
    assert not (dest_root / "msrs").exists()
    for split, stems in (("train", TRAIN_STEMS), ("val", TEST_STEMS)):
        written = {
            p.stem for p in (dest_root / "msrs-day" / split / "visible" / "images").iterdir()
        }
        assert written == {stem for stem in stems if stem.endswith("D")}
