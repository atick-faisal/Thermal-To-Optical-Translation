"""Stage 2 of the cascade: the `cyclegan` backbone (cyclegan-cascade Q7, Q10, ADR-01).

Every translator here is small (`resnet_6blocks`, `ngf`/`ndf` 8, 32x32 images) so a `fit` step
costs milliseconds on CPU. The detection term is a toy callable: what is under test is where its
gradient goes, not what a detector makes of it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from t2o.config import Config, ConfigError
from t2o.data.dataset import TranslationBatch
from t2o.translators import CycleGANTranslator, Translator, build_translator

# What the cascade configs must say: unpaired losses only, RGB stage-1 input (ADR-01).
_CYCLEGAN: dict[str, Any] = {
    "translator": {"backbone": "cyclegan", "net_g": "resnet_6blocks", "ngf": 8, "ndf": 8},
    "loss": {"l2": 0.0, "lpips": 0.0, "gan": 1.0},
    "data": {"infrared_channels": 3},
}


def _batch(n: int = 2, size: int = 32, channels: int = 3, seed: int = 0) -> TranslationBatch:
    generator = torch.Generator().manual_seed(seed)
    return TranslationBatch(
        visible=torch.rand((n, 3, size, size), generator=generator),
        infrared=torch.rand((n, channels, size, size), generator=generator),
        batch_idx=torch.zeros(0),
        cls=torch.zeros(0, 1),
        bboxes=torch.zeros(0, 4),
        names=[f"sample_{i}" for i in range(n)],
    )


def _translator(total_batches: int = 100, **kwargs: Any) -> CycleGANTranslator:
    torch.manual_seed(0)
    kwargs = {"net_g": "resnet_6blocks", "ngf": 8, "ndf": 8, **kwargs}
    return CycleGANTranslator(total_batches=total_batches, **kwargs)


def _mean_loss(pred: Tensor, batch: TranslationBatch) -> Tensor:
    return pred.mean()


def _grads(module: nn.Module) -> list[Tensor]:
    grads = [param.grad for param in module.parameters()]
    assert all(grad is not None for grad in grads)
    return [grad for grad in grads if grad is not None]


def _config(**sections: dict[str, Any]) -> Config:
    overrides: dict[str, Any] = {name: dict(section) for name, section in _CYCLEGAN.items()}
    for name, section in sections.items():
        overrides[name] = {**overrides.get(name, {}), **section}
    return Config.load(overrides=overrides)


# --------------------------------------------------------------------------- translate


def test_cyclegan_satisfies_the_translator_protocol() -> None:
    assert isinstance(_translator(), Translator)


@pytest.mark.parametrize("channels", [1, 3])
def test_translate_maps_either_input_width_to_visible_in_unit_range(channels: int) -> None:
    output = _translator(infrared_channels=channels).translate(_batch(channels=channels))

    assert output.shape == (2, 3, 32, 32)
    assert output.dtype == torch.float32
    assert output.min() >= 0.0 and output.max() <= 1.0


def test_translate_is_g_ab_and_grad_connected_to_it_alone() -> None:
    translator = _translator()

    translator.translate(_batch()).sum().backward()

    assert any(grad.abs().sum() > 0 for grad in _grads(translator.net_g_ab))
    assert all(param.grad is None for param in translator.net_g_ba.parameters())


# --------------------------------------------------------------------------- fit


def test_fit_returns_every_cyclegan_term_and_no_detection_at_weight_zero() -> None:
    stats = _translator().fit(_batch(), task_loss=_mean_loss, task_weight=0.0)

    assert {"loss_gan", "loss_cycle", "loss_idt", "loss_d", "loss_total"} <= set(stats)
    assert all(math.isfinite(value) for value in stats.values())
    assert "loss_det" not in stats


def test_identity_off_drops_its_term_from_the_total() -> None:
    stats = _translator(lambda_identity=0.0).fit(_batch())

    assert "loss_idt" not in stats
    assert stats["loss_total"] == pytest.approx(stats["loss_gan"] + stats["loss_cycle"])


def test_identity_weight_multiplies_the_cycle_weight() -> None:
    """Upstream's `lambda_identity` 0.5 is an effective 5 at `lambda_cycle` 10, not 0.5."""
    translator = _translator(lambda_cycle=10.0, lambda_identity=0.5)
    batch = _batch()
    real_a = batch["infrared"] * 2.0 - 1.0
    real_b = batch["visible"] * 2.0 - 1.0
    with torch.no_grad():
        l1 = F.l1_loss(translator.net_g_ab(real_b), real_b) + F.l1_loss(
            translator.net_g_ba(real_a), real_a
        )

    stats = translator.fit(batch)

    assert stats["loss_idt"] == pytest.approx(float(l1) * 5.0, rel=1e-5)


def test_detection_gradient_reaches_g_ab_and_nothing_else() -> None:
    """Same seed, same batch, with and without the detection term: only G_AB may differ.

    G_BA and both discriminators see the same images either way -- the term sits on G_AB's
    output, the discriminators are frozen during the generator step, and their own step reads
    detached fakes -- so their gradients must be identical, not merely close.
    """
    batch = _batch()
    plain = _translator()
    coupled = _translator()

    plain.fit(batch)
    stats = coupled.fit(batch, task_loss=_mean_loss, task_weight=5.0)

    assert "loss_det" in stats
    assert not all(
        torch.equal(a, b)
        for a, b in zip(_grads(plain.net_g_ab), _grads(coupled.net_g_ab), strict=True)
    )
    for name in ("net_g_ba", "net_d_a", "net_d_b"):
        for a, b in zip(_grads(getattr(plain, name)), _grads(getattr(coupled, name)), strict=True):
            assert torch.equal(a, b), name


# --------------------------------------------------------------------------- schedule (Q10)


def _lr(translator: CycleGANTranslator) -> tuple[float, float]:
    return (
        translator.optimizer_g.param_groups[0]["lr"],
        translator.optimizer_d.param_groups[0]["lr"],
    )


def test_learning_rate_holds_for_the_first_half_then_decays_to_zero() -> None:
    translator = _translator(total_batches=4, lr=1.0e-3)
    seen = [_lr(translator)]
    for _ in range(4):
        translator.fit(_batch())
        seen.append(_lr(translator))

    # Steps 0-2 run at the full rate, step 3 at half, and the run ends at zero.
    assert [g for g, _ in seen] == pytest.approx([1.0e-3, 1.0e-3, 1.0e-3, 5.0e-4, 0.0])
    assert [d for _, d in seen] == [g for g, _ in seen]


def test_a_resume_continues_the_decay_and_both_adams() -> None:
    trained = _translator(total_batches=4, lr=1.0e-3)
    for _ in range(3):
        trained.fit(_batch())

    resumed = _translator(total_batches=4, lr=1.0e-3)
    resumed.load_state_dict(trained.state_dict())

    assert _lr(resumed) == pytest.approx((5.0e-4, 5.0e-4))
    for name in ("optimizer_g", "optimizer_d"):
        state = getattr(resumed, name).state_dict()["state"]
        expected = getattr(trained, name).state_dict()["state"]
        assert state.keys() == expected.keys() and state
        assert torch.equal(state[0]["exp_avg"], expected[0]["exp_avg"])


# --------------------------------------------------------------------------- build_translator


def test_build_translator_wires_the_config_into_the_module(data_yaml: Path) -> None:
    config = _config(
        data={"manifest": str(data_yaml)},
        translator={"lambda_cycle": 7.0, "lambda_identity": 0.25},
    )

    translator = build_translator(config)

    assert isinstance(translator, CycleGANTranslator)
    assert (translator.lambda_cycle, translator.lambda_identity) == (7.0, 0.25)
    blocks = sum(1 for m in translator.net_g_ab.modules() if type(m).__name__ == "ResnetBlock")
    assert blocks == 6
    # G_AB reads the three-channel stage-1 input (ADR-01), and G_BA writes it back.
    first_conv = next(m for m in translator.net_g_ab.modules() if isinstance(m, nn.Conv2d))
    assert first_conv.in_channels == 3


@pytest.mark.parametrize(
    ("loss", "match"),
    [
        ({"lpips": 5.0}, "unpaired losses"),
        ({"l2": 1.0}, "unpaired losses"),
        ({"gan": 0.0}, "unpaired losses"),
    ],
)
def test_paired_or_missing_adversarial_loss_is_refused(loss: dict[str, float], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        build_translator(_config(loss=loss))


def test_identity_on_grey_input_is_refused() -> None:
    with pytest.raises(ConfigError, match=r"data\.infrared_channels 3, not 1"):
        build_translator(_config(data={"infrared_channels": 1}))


def test_grey_input_builds_once_identity_is_off(data_yaml: Path) -> None:
    config = _config(
        data={"manifest": str(data_yaml), "infrared_channels": 1},
        translator={"lambda_identity": 0.0},
    )

    assert isinstance(build_translator(config), CycleGANTranslator)


# --------------------------------------------------------------------------- slow: the loop


@pytest.mark.slow
def test_full_loop_runs_end_to_end_with_cyclegan(data_yaml: Path, tmp_path: Path) -> None:
    """`build_translator` -> `run_loop` on unpaired targets, the shape -04's configs take.

    The synthetic tree is grey thermal, so this runs at one channel with identity off; the
    three-channel path is the build test above.
    """
    from t2o.engine.loop import run_loop

    config = _config(
        data={"manifest": str(data_yaml), "infrared_channels": 1, "unpaired_visible": True},
        translator={"lambda_identity": 0.0},
        train={"batch_size": 2, "epochs_per_stage": 1},
        runtime={"workers": 0},
        coupling={"task_weights": [0.0]},
    )
    translator = build_translator(config)
    results = run_loop(config, translator, run_dir=tmp_path, train_detector_stages=False)

    assert len(results) == 1
    assert math.isfinite(results[0].epochs[-1].val_loss)
