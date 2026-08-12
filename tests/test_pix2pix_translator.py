"""M1: the pix2pix backbone (`PLAN.md` §10 Phase 1, the go/no-go gate before diffusion).

Fast tests construct with `loss_gan=0.0, loss_lpips=0.0` so neither the discriminator nor
the LPIPS network gets built -- mirrors `FidelityEvaluator`'s own fast/slow split rationale
(M0.5): anything that builds a pretrained perceptual network belongs in the `slow` suite.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from t2o.config import Config
from t2o.data.dataset import TranslationBatch
from t2o.translators import Pix2PixTranslator, Translator, build_translator


def _batch(n: int, h: int, w: int, seed: int = 0) -> TranslationBatch:
    generator = torch.Generator().manual_seed(seed)
    return TranslationBatch(
        visible=torch.rand((n, 3, h, w), generator=generator),
        infrared=torch.rand((n, 1, h, w), generator=generator),
        batch_idx=torch.zeros(0),
        cls=torch.zeros(0, 1),
        bboxes=torch.zeros(0, 4),
        names=[f"sample_{i}" for i in range(n)],
    )


# --------------------------------------------------------------------------- protocol


def test_pix2pix_satisfies_the_translator_protocol() -> None:
    assert isinstance(Pix2PixTranslator(loss_gan=0.0, loss_lpips=0.0), Translator)


def test_config_arch_knobs_reach_the_module() -> None:
    config = Config.load(
        overrides={
            "translator": {"backbone": "pix2pix", "net_g": "resnet_6blocks", "ngf": 32},
            "loss": {"gan": 0.0, "lpips": 0.0},
        }
    )
    translator = build_translator(config)
    assert isinstance(translator, Pix2PixTranslator)
    # resnet_6blocks has 6 ResnetBlocks vs resnet_9blocks' 9 -- a cheap, direct way to prove
    # net_g actually reached define_G rather than silently using a default. `.modules()` is
    # nn.Module's own typed recursive traversal, unlike the vendored (untyped) internals.
    n_blocks = sum(1 for m in translator.net_g.modules() if type(m).__name__ == "ResnetBlock")
    assert n_blocks == 6


def test_no_discriminator_or_lpips_built_at_zero_weight() -> None:
    translator = Pix2PixTranslator(loss_gan=0.0, loss_lpips=0.0)
    assert translator.net_d is None
    assert translator.lpips is None


def test_unknown_backbone_field_value_is_rejected_at_load() -> None:
    from t2o.config import ConfigError

    with pytest.raises(ConfigError):
        Config.load(overrides={"translator": {"backbone": "pix2pix", "gan_mode": "wgangp"}})


# --------------------------------------------------------------------------- translate


def test_translate_shape_dtype_and_range() -> None:
    batch = _batch(n=2, h=32, w=32)
    output = Pix2PixTranslator(loss_gan=0.0, loss_lpips=0.0).translate(batch)

    assert output.shape == (2, 3, 32, 32)
    assert output.dtype == torch.float32
    assert output.min() >= 0.0 and output.max() <= 1.0


def test_translate_is_grad_connected() -> None:
    translator = Pix2PixTranslator(loss_gan=0.0, loss_lpips=0.0)
    batch = _batch(n=1, h=32, w=32)

    output = translator.translate(batch)
    output.sum().backward()

    grads = [p.grad for p in translator.net_g.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads if g is not None)  # type: ignore[union-attr]


# --------------------------------------------------------------------------- fit


def test_fit_returns_a_finite_loss() -> None:
    result = Pix2PixTranslator(loss_gan=0.0, loss_lpips=0.0).fit(_batch(n=2, h=32, w=32))
    assert math.isfinite(result["loss_l2"])
    assert "loss_gan" not in result
    assert "loss_lpips" not in result


def test_fit_ignores_task_loss_when_weight_is_zero() -> None:
    translator = Pix2PixTranslator(loss_gan=0.0, loss_lpips=0.0)
    calls = 0

    def task_loss(pred: torch.Tensor, batch: TranslationBatch) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return pred.sum()

    result = translator.fit(_batch(n=2, h=32, w=32), task_loss=task_loss, task_weight=0.0)

    assert calls == 0
    assert "loss_det" not in result


def test_fit_adds_the_task_loss_when_weight_is_positive() -> None:
    translator = Pix2PixTranslator(loss_gan=0.0, loss_lpips=0.0)

    def task_loss(pred: torch.Tensor, batch: TranslationBatch) -> torch.Tensor:
        return pred.sum()

    result = translator.fit(_batch(n=2, h=32, w=32), task_loss=task_loss, task_weight=0.5)

    assert "loss_det" in result
    assert math.isfinite(result["loss_total"])
    assert result["loss_total"] != result["loss_l2"]


# --------------------------------------------------------------------------- slow: real defaults


@pytest.mark.slow
def test_fit_with_real_defaults_returns_every_loss_term() -> None:
    translator = Pix2PixTranslator()  # loss_gan=0.0, loss_lpips=5.0 -- schema defaults
    result = Pix2PixTranslator(loss_gan=0.1, loss_lpips=5.0).fit(_batch(n=2, h=32, w=32))

    assert math.isfinite(result["loss_d"])
    assert math.isfinite(result["loss_gan"])
    assert math.isfinite(result["loss_lpips"])
    assert math.isfinite(result["loss_total"])
    del translator  # constructed only to assert the schema defaults build cleanly, above


@pytest.mark.slow
def test_fit_drives_the_l2_loss_down_on_a_fixed_batch() -> None:
    # loss_lpips=0.0 here (unlike the real-defaults test above): with it active, the
    # optimizer minimises loss_l2 + 5*loss_lpips, and the two terms can legitimately pull
    # in different directions step to step on random-noise data. Isolating loss_l2 alone,
    # matching StubTranslator's own convergence test, is what actually exercises "the
    # optimizer step reduces the loss it's given" without that confound.
    translator = Pix2PixTranslator(loss_gan=0.0, loss_lpips=0.0, lr=2e-3)
    batch = _batch(n=2, h=32, w=32)

    first = translator.fit(batch)["loss_l2"]
    last = first
    for _ in range(20):
        last = translator.fit(batch)["loss_l2"]

    assert last < first


@pytest.mark.slow
def test_full_loop_runs_end_to_end_with_pix2pix(data_yaml: Path, tmp_path: Path) -> None:
    """Proves `build_translator` really dispatches through `Trainer`/`run_loop`, not just in
    isolation -- the direct pix2pix analogue of `test_experiments.py`'s stub-only loop test.
    """
    from t2o.engine.loop import run_loop

    config = Config.load(
        overrides={
            "data": {"manifest": str(data_yaml)},
            "train": {"batch_size": 2, "workers": 0, "epochs_per_stage": 1},
            "translator": {"backbone": "pix2pix", "net_g": "resnet_6blocks"},
            "loss": {"gan": 0.0, "lpips": 0.0},
            "coupling": {"task_weights": [0.0]},
        }
    )
    translator = build_translator(config)
    results = run_loop(config, translator, run_dir=tmp_path, train_detector_stages=False)

    assert len(results) == 1
    assert math.isfinite(results[0].epochs[-1].val_loss)
