"""Fidelity metrics: PSNR, SSIM, LPIPS, FID, KID.

Split deliberately along a network boundary rather than a metric boundary.
:class:`FidelityEvaluator` builds all five torchmetrics backbones in its constructor --
including the pretrained AlexNet (LPIPS) and InceptionV3 (FID/KID) weights, fetched over
the network the first time they run and cached under ``~/.cache/torch/`` afterward. There
is no way to construct the class at all without that cost, so *every* test that touches
it is marked ``slow``; the fast suite covers only what is genuinely network-free: the
``MetricsConfig`` shape, the KID subset-size clamp, and ``evaluate_fidelity``'s pool-pairing
helpers -- all pure functions that never reach a backbone.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from PIL import Image

from t2o.config import Config, ConfigError
from t2o.metrics.fidelity import (
    FidelityError,
    FidelityEvaluator,
    _clamp_kid_subset_size,
    _index_by_stem,
    _paired_stems,
    evaluate_fidelity,
)

# --------------------------------------------------------------------------- config


def test_metrics_config_defaults() -> None:
    config = Config.load()
    assert config.metrics.lpips_net == "alex"
    assert config.metrics.kid_subset_size == 50


def test_metrics_config_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"metrics": {"bogus": 1}}))
    with pytest.raises(ConfigError):
        Config.load(path)


@pytest.mark.parametrize(
    "override",
    [{"lpips_net": "vgg"}, {"kid_subset_size": 10}],
)
def test_metrics_config_changes_are_part_of_the_hash(override: dict[str, Any]) -> None:
    baseline = Config.load().config_hash()
    assert Config.load(overrides={"metrics": override}).config_hash() != baseline


# --------------------------------------------------------------------------- KID clamp


def test_kid_subset_size_unchanged_when_within_bounds() -> None:
    assert _clamp_kid_subset_size(configured=10, available=50) == 10
    assert _clamp_kid_subset_size(configured=10, available=10) == 10


def test_kid_subset_size_clamped_when_too_large(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        clamped = _clamp_kid_subset_size(configured=50, available=5)
    assert clamped == 5
    assert "clamping" in caplog.text


# --------------------------------------------------------------------------- evaluator


def _images(n: int, seed: int, size: int = 128) -> torch.Tensor:
    return torch.rand((n, 3, size, size), generator=torch.Generator().manual_seed(seed))


@pytest.mark.slow
def test_identical_images_score_as_a_perfect_match() -> None:
    images = _images(4, seed=0)
    evaluator = FidelityEvaluator(kid_subset_size=4)
    evaluator.update(images, images)
    result = evaluator.compute()

    assert result.psnr == float("inf") or result.psnr > 60.0
    assert result.ssim == pytest.approx(1.0, abs=1e-3)
    assert result.lpips < 1e-4


@pytest.mark.slow
def test_psnr_and_ssim_get_worse_as_noise_increases() -> None:
    target = _images(4, seed=1)

    mild = FidelityEvaluator(kid_subset_size=4)
    mild.update((target + 0.05 * _images(4, seed=2)).clamp(0.0, 1.0), target)
    mild_result = mild.compute()

    severe = FidelityEvaluator(kid_subset_size=4)
    severe.update((target + 0.5 * _images(4, seed=3)).clamp(0.0, 1.0), target)
    severe_result = severe.compute()

    assert mild_result.psnr > severe_result.psnr
    assert mild_result.ssim > severe_result.ssim


@pytest.mark.slow
def test_fid_and_kid_are_finite_on_unrelated_pools() -> None:
    evaluator = FidelityEvaluator(kid_subset_size=4)
    evaluator.update(_images(4, seed=10), _images(4, seed=11))
    result = evaluator.compute()

    assert math.isfinite(result.fid) and result.fid >= 0.0
    assert math.isfinite(result.kid_mean)
    assert math.isfinite(result.kid_std) and result.kid_std >= 0.0


@pytest.mark.slow
def test_reset_clears_accumulated_state() -> None:
    evaluator = FidelityEvaluator(kid_subset_size=4)
    evaluator.update(_images(4, seed=20), _images(4, seed=21))
    evaluator.reset()

    assert evaluator._n_images == 0
    evaluator.update(_images(4, seed=22), _images(4, seed=22))
    assert evaluator.compute().psnr == float("inf") or evaluator.compute().psnr > 60.0


@pytest.mark.slow
def test_mismatched_shapes_are_rejected() -> None:
    evaluator = FidelityEvaluator(kid_subset_size=4)
    with pytest.raises(ValueError, match="differ"):
        evaluator.update(_images(4, seed=0), _images(3, seed=0))


# ------------------------------------------------------------------- evaluate_fidelity


def _write_images(directory: Path, stems: list[str], suffix: str, seed: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    for stem in stems:
        pixels = torch.randint(0, 256, (32, 32, 3), generator=generator, dtype=torch.uint8).numpy()
        Image.fromarray(pixels, mode="RGB").save(directory / f"{stem}{suffix}")


def test_pools_are_paired_by_stem_across_differing_suffixes(tmp_path: Path) -> None:
    """`export.py` always writes .png even when the source visible frame was .jpg, so
    matching on full filename would find nothing at all."""
    stems = [f"frame_{i}" for i in range(3)]
    _write_images(tmp_path / "t", stems, ".png", seed=0)
    _write_images(tmp_path / "r", stems, ".jpg", seed=1)

    common = _paired_stems(_index_by_stem(tmp_path / "t"), _index_by_stem(tmp_path / "r"))

    assert common == ["frame_0", "frame_1", "frame_2"]


def test_disjoint_pools_are_rejected_rather_than_scored_as_empty(tmp_path: Path) -> None:
    _write_images(tmp_path / "t", ["a", "b"], ".png", seed=0)
    _write_images(tmp_path / "r", ["x", "y"], ".png", seed=1)

    with pytest.raises(FidelityError, match="not the same split"):
        _paired_stems(_index_by_stem(tmp_path / "t"), _index_by_stem(tmp_path / "r"))


def test_a_partial_overlap_warns_and_scores_the_intersection(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_images(tmp_path / "t", ["a", "b", "c"], ".png", seed=0)
    _write_images(tmp_path / "r", ["b", "c", "d"], ".png", seed=1)

    with caplog.at_level(logging.WARNING):
        common = _paired_stems(_index_by_stem(tmp_path / "t"), _index_by_stem(tmp_path / "r"))

    assert common == ["b", "c"]
    assert "present in only one pool" in caplog.text


def test_a_missing_directory_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FidelityError, match="image directory not found"):
        _index_by_stem(tmp_path / "nope")


@pytest.mark.slow
def test_identical_pools_on_disk_score_as_a_perfect_match(tmp_path: Path) -> None:
    stems = ["a", "b", "c", "d"]
    _write_images(tmp_path / "t", stems, ".png", seed=7)
    _write_images(tmp_path / "r", stems, ".png", seed=7)

    result = evaluate_fidelity(tmp_path / "t", tmp_path / "r", kid_subset_size=2, batch_size=2)

    assert result.psnr == float("inf") or result.psnr > 60.0
    assert result.ssim == pytest.approx(1.0, abs=1e-4)
    assert result.lpips == pytest.approx(0.0, abs=1e-4)


@pytest.mark.slow
def test_a_degraded_pool_scores_worse_than_a_faithful_one(tmp_path: Path) -> None:
    """The comparison the reward-hacking check actually rests on: worse images must produce
    worse numbers, through the same on-disk path a real run uses."""
    stems = [f"f{i}" for i in range(6)]
    _write_images(tmp_path / "reference", stems, ".png", seed=3)
    _write_images(tmp_path / "faithful", stems, ".png", seed=3)
    _write_images(tmp_path / "degraded", stems, ".png", seed=99)

    faithful = evaluate_fidelity(
        tmp_path / "faithful", tmp_path / "reference", kid_subset_size=3, batch_size=3
    )
    degraded = evaluate_fidelity(
        tmp_path / "degraded", tmp_path / "reference", kid_subset_size=3, batch_size=3
    )

    assert degraded.psnr < faithful.psnr
    assert degraded.ssim < faithful.ssim
    assert degraded.lpips > faithful.lpips
