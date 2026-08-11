"""Fidelity metrics: PSNR, SSIM, LPIPS, FID, KID.

Split deliberately along a network boundary rather than a metric boundary.
:class:`FidelityEvaluator` builds all five torchmetrics backbones in its constructor --
including the pretrained AlexNet (LPIPS) and InceptionV3 (FID/KID) weights, fetched over
the network the first time they run and cached under ``~/.cache/torch/`` afterward. There
is no way to construct the class at all without that cost, so *every* test that touches
it is marked ``slow``; the fast suite covers only what is genuinely network-free: the
``MetricsConfig`` shape and the KID subset-size clamp, which is a pure function.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from t2o.config import Config, ConfigError
from t2o.metrics.fidelity import FidelityEvaluator, _clamp_kid_subset_size

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
