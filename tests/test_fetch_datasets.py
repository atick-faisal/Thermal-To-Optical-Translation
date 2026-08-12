"""`scripts/fetch_datasets.py` -- source routing, idempotent skip, and CLI plumbing.

No real network I/O: `subprocess.run` and `huggingface_hub.snapshot_download` are both
monkeypatched. Real fetches are the M0.9 server/Mac-side follow-up, not something this
suite touches (PLAN.md §9's synthetic-fixture discipline extends to "never actually hit the
network in a test").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from scripts.fetch_datasets import (
    SOURCES,
    DatasetSource,
    FetchError,
    build_parser,
    fetch_dataset,
    resolve_sources,
)


def test_resolve_sources_all_returns_every_source() -> None:
    assert resolve_sources(["all"]) == list(SOURCES)


def test_resolve_sources_filters_by_name() -> None:
    assert [s.name for s in resolve_sources(["msrs", "flir"])] == ["msrs", "flir"]


def test_resolve_sources_rejects_unknown_name() -> None:
    with pytest.raises(FetchError, match="unknown dataset 'nope'"):
        resolve_sources(["nope"])


def test_fetch_dataset_routes_git_sources_through_git_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "scripts.fetch_datasets.subprocess.run",
        lambda cmd, check: calls.append(cmd),
    )
    source = DatasetSource(name="msrs", git_url="https://example.invalid/msrs.git")

    dest = fetch_dataset(source, tmp_path)

    assert dest == tmp_path / "msrs"
    assert calls == [["git", "clone", "--depth", "1", source.git_url, str(dest)]]


def test_fetch_dataset_routes_hf_sources_through_snapshot_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: calls.append(kwargs),
    )
    source = DatasetSource(name="flir", hf_repo_id="UserNae3/FLIR_aligned")

    dest = fetch_dataset(source, tmp_path)

    assert dest == tmp_path / "flir"
    assert calls == [
        {"repo_id": "UserNae3/FLIR_aligned", "repo_type": "dataset", "local_dir": dest}
    ]


def test_fetch_dataset_skips_an_already_populated_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("should not be called when the destination already has content")

    monkeypatch.setattr("scripts.fetch_datasets.subprocess.run", _fail)
    source = DatasetSource(name="msrs", git_url="https://example.invalid/msrs.git")
    dest = tmp_path / "msrs"
    dest.mkdir()
    (dest / "README.md").write_text("already fetched")

    result = fetch_dataset(source, tmp_path)

    assert result == dest
    assert (dest / "README.md").read_text() == "already fetched"


def test_build_parser_defaults_to_all_datasets() -> None:
    args = build_parser().parse_args([])
    assert args.dataset == ["all"]
    assert args.dest == Path("dataset/raw")


def test_build_parser_rejects_unknown_dataset_choice() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--dataset", "nope"])
