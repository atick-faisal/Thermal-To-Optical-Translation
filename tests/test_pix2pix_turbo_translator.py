"""M2a: the one-step diffusion backbone (`PLAN.md` §4, §10 Phase 2a).

Fast tests build a small `AutoencoderKL`/`UNet2DConditionModel` locally rather than
downloading sd-turbo. That is not a mock: the LoRA adapters, the vendored skip forwards, the
one-step scheduler and every gradient path are the real ones -- only the widths are small.
The same reasoning M0.3 applied to the dataset, and it keeps the fast suite offline.

What a tiny model cannot cover is whether sd-turbo's *own* config still works against
current diffusers/transformers/peft. That is the one `slow` test, and it skips itself unless
the checkpoint is already in the HuggingFace cache.

`loss_lpips=0.0` throughout the fast tests: building the LPIPS network downloads pretrained
weights, which `tests/test_pix2pix_translator.py` already keeps out of the fast suite.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor
from transformers import CLIPTextConfig, CLIPTextModel

from t2o.config import Config
from t2o.data.dataset import TranslationBatch
from t2o.translators import Pix2PixTurboTranslator, Translator, build_translator
from t2o.translators.pix2pix_turbo import (
    SD_TURBO,
    SDTurboComponents,
    TurboTranslatorError,
    skip_connection_channels,
)

_CROSS_ATTENTION_DIM = 16
_PROMPT_TOKENS = 8


class _Tokenizer:
    """Stands in for the CLIP tokenizer: the wrapper only calls it once, at construction."""

    model_max_length = _PROMPT_TOKENS

    def __call__(self, prompt: str, max_length: int, **_: Any) -> Any:
        class _Encoded:
            input_ids = torch.zeros((1, max_length), dtype=torch.long)

        return _Encoded()


def _tiny_components() -> SDTurboComponents:
    # Splatted through `dict[str, Any]`: diffusers annotates these tuple parameters as
    # one-element `Tuple[str]`, so any real block list is a type error at the call site.
    vae_kwargs: dict[str, Any] = {
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ("DownEncoderBlock2D",) * 4,
        "up_block_types": ("UpDecoderBlock2D",) * 4,
        "block_out_channels": (4, 8, 16, 16),
        "layers_per_block": 1,
        "latent_channels": 4,
        "norm_num_groups": 2,
    }
    unet_kwargs: dict[str, Any] = {
        "in_channels": 4,
        "out_channels": 4,
        "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D"),
        "up_block_types": ("CrossAttnUpBlock2D", "UpBlock2D"),
        "block_out_channels": (8, 16),
        "layers_per_block": 1,
        "cross_attention_dim": _CROSS_ATTENTION_DIM,
        "attention_head_dim": 2,
        "norm_num_groups": 2,
    }
    text_encoder = CLIPTextModel(
        CLIPTextConfig(
            vocab_size=16,
            hidden_size=_CROSS_ATTENTION_DIM,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=_PROMPT_TOKENS,
            bos_token_id=0,
            eos_token_id=1,
        )
    )
    return SDTurboComponents(
        tokenizer=_Tokenizer(),
        text_encoder=text_encoder,
        vae=AutoencoderKL(**vae_kwargs),
        unet=UNet2DConditionModel(**unet_kwargs),
        scheduler=DDPMScheduler(),
    )


def _translator(**kwargs: Any) -> Pix2PixTurboTranslator:
    kwargs.setdefault("loss_lpips", 0.0)
    kwargs.setdefault("loss_gan", 0.0)
    return Pix2PixTurboTranslator(_tiny_components(), prompt="a thermal frame", **kwargs)


def _batch(n: int = 2, h: int = 48, w: int = 64, seed: int = 0) -> TranslationBatch:
    generator = torch.Generator().manual_seed(seed)
    return TranslationBatch(
        visible=torch.rand((n, 3, h, w), generator=generator),
        infrared=torch.rand((n, 1, h, w), generator=generator),
        batch_idx=torch.zeros(0),
        cls=torch.zeros(0, 1),
        bboxes=torch.zeros(0, 4),
        names=[f"sample_{i}" for i in range(n)],
    )


# ------------------------------------------------------------------ derived skip widths


def test_skip_widths_reproduce_upstreams_hardcoded_convs() -> None:
    # `pix2pix_turbo.py:40-43` writes these four literally, for sd-turbo's own VAE, whose
    # block_out_channels is exactly [128, 256, 512, 512] (confirmed against the checkpoint).
    assert skip_connection_channels([128, 256, 512, 512]) == (
        (512, 512),
        (256, 512),
        (128, 512),
        (128, 256),
    )


def test_skip_widths_refuse_a_vae_that_is_not_four_blocks() -> None:
    with pytest.raises(TurboTranslatorError, match="4-block"):
        skip_connection_channels([64, 128, 256])


def test_the_tiny_vaes_skip_convs_are_wired_to_its_own_widths() -> None:
    translator = _translator()
    # block_out_channels (4, 8, 16, 16) -> ins (16, 8, 4, 4), outs (16, 16, 16, 8).
    widths = [
        (conv.in_channels, conv.out_channels)
        for conv in (
            getattr(translator.vae.decoder, f"skip_conv_{index}").base_layer
            for index in range(1, 5)
        )
    ]
    assert widths == [(16, 16), (8, 16), (4, 16), (4, 8)]


# ------------------------------------------------------------------------------ contract


def test_turbo_satisfies_the_translator_protocol() -> None:
    assert isinstance(_translator(), Translator)


def test_translate_returns_the_protocols_shape_and_range() -> None:
    translator = _translator()
    generated = translator.translate(_batch(n=2, h=48, w=64))
    assert generated.shape == (2, 3, 48, 64)
    assert float(generated.detach().min()) >= 0.0
    assert float(generated.detach().max()) <= 1.0


def test_a_size_not_divisible_by_sixty_four_survives_and_comes_back_unpadded() -> None:
    # 48 is 480's stand-in: the dataset is 640x480 and 480 % 64 == 32, so without the
    # internal pad the UNet's skip concat shape-mismatches on a full frame. `train.crop`
    # cannot fix that -- validation and export always run whole images.
    generated = _translator().translate(_batch(n=1, h=48, w=64))
    assert generated.shape[-2:] == (48, 64)


def test_translate_stays_grad_connected_to_the_adapters() -> None:
    translator = _translator()
    translator.translate(_batch(n=1)).sum().backward()
    lora = [p for name, p in translator.named_parameters() if "lora" in name and p.requires_grad]
    assert lora
    assert any(param.grad is not None and float(param.grad.abs().sum()) > 0.0 for param in lora)


# ----------------------------------------------------------------------- what trains


def test_only_the_adapters_conv_in_and_skip_convs_train() -> None:
    translator = _translator()
    trainable = {name for name, param in translator.named_parameters() if param.requires_grad}
    assert trainable
    assert all(
        "lora" in name or "unet.conv_in" in name or "skip_conv" in name for name in trainable
    )
    # The frozen base is the point: it is both the reason 850 pairs are enough and DRaFT's
    # own lever against reward hacking (PLAN.md §4).
    assert not any(name.endswith("unet.conv_out.weight") for name in trainable)


def test_the_detection_gradient_reaches_an_adapter() -> None:
    translator = _translator()

    def task_loss(generated: Tensor, _: TranslationBatch) -> Tensor:
        return generated.mean()

    stats = translator.fit(_batch(n=1), task_loss=task_loss, task_weight=2.0)
    assert "loss_det" in stats


def test_no_task_term_at_weight_zero() -> None:
    stats = _translator().fit(_batch(n=1), task_loss=lambda g, _: g.mean(), task_weight=0.0)
    assert "loss_det" not in stats


def test_fit_reports_the_same_loss_keys_the_other_backbones_do() -> None:
    stats = _translator().fit(_batch(n=1))
    assert set(stats) == {"loss_l2", "loss_total"}


def test_no_discriminator_or_lpips_built_at_zero_weight() -> None:
    translator = _translator()
    assert translator.net_d is None
    assert translator.lpips is None


def test_the_vendored_patchgan_is_built_at_positive_gan_weight() -> None:
    translator = _translator(loss_gan=1.0)
    assert translator.net_d is not None
    stats = translator.fit(_batch(n=1))
    assert {"loss_d", "loss_gan"} <= set(stats)


# ------------------------------------------------------------------------ checkpoints


def test_the_checkpoint_holds_only_what_trains_and_round_trips() -> None:
    translator = _translator()
    reduced = translator.state_dict()
    full = torch.nn.Module.state_dict(translator)

    assert set(reduced) < set(full)
    assert all("lora" in k or "unet.conv_in" in k or "skip_conv" in k for k in reduced)
    # How much smaller is a property of the real checkpoint, not of a toy whose LoRA rank is
    # a large fraction of its own width -- the slow test below carries that number.
    assert sum(v.numel() for v in reduced.values()) < sum(v.numel() for v in full.values())

    translator.load_state_dict(reduced)


def test_a_checkpoint_from_a_different_model_is_refused() -> None:
    translator = _translator()
    corrupted = dict(translator.state_dict())
    corrupted["unet.some_layer_that_does_not_exist.weight"] = torch.zeros(1)
    with pytest.raises(TurboTranslatorError, match="does not define"):
        translator.load_state_dict(corrupted)


def test_the_caption_is_not_a_checkpoint_entry() -> None:
    # It is a non-persistent buffer: recomputable from the prompt, and the prompt is part of
    # experiment identity already (`Pix2PixTurboTranslatorConfig.prompt`).
    assert "caption" not in torch.nn.Module.state_dict(_translator())


# --------------------------------------------------------------------------- one step


def test_the_scheduler_denoises_from_the_distilled_timestep() -> None:
    # Not `set_timesteps(1)`: that yields [999] only under sd-turbo's own "trailing" spacing
    # and [0] under diffusers' default. DDPMScheduler() here is the default one, so this
    # would be [0] if the timestep were not named explicitly.
    assert list(_translator().scheduler.timesteps) == [999]


def test_evaluation_is_deterministic_although_training_samples_the_posterior() -> None:
    translator = _translator()
    batch = _batch(n=1)

    translator.eval()
    with torch.no_grad():
        first = translator.translate(batch)
        second = translator.translate(batch)
    assert torch.equal(first, second)

    translator.train()
    torch.manual_seed(0)
    with torch.no_grad():
        sampled = translator.translate(batch)
    assert not torch.equal(first, sampled)


# ------------------------------------------------------------------------- config wiring


def test_config_knobs_reach_the_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("t2o.translators.load_sd_turbo", lambda _: _tiny_components())
    config = Config.load(
        overrides={
            "translator": {
                "backbone": "pix2pix_turbo",
                "prompt": "an overhead line",
                "lora_rank_unet": 2,
                "lora_rank_vae": 1,
            },
            "loss": {"gan": 0.0, "lpips": 0.0},
        }
    )
    translator = build_translator(config)
    assert isinstance(translator, Pix2PixTurboTranslator)
    # peft names the adapter weights `lora_A.default.weight` with shape (r, in_features), so
    # the rank is directly readable -- proof the config value reached LoraConfig.
    ranks = {
        name.split(".")[0]: param.shape[0]
        for name, param in translator.named_parameters()
        if name.endswith("lora_A.default.weight")
    }
    assert ranks["unet"] == 2


def test_a_blank_prompt_is_rejected_at_config_load() -> None:
    with pytest.raises(Exception, match="blank"):
        Config.load(overrides={"translator": {"backbone": "pix2pix_turbo", "prompt": "  "}})


# ------------------------------------------------------------------------------- slow


def _sd_turbo_is_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(SD_TURBO, "unet/diffusion_pytorch_model.safetensors")
    return isinstance(cached, str)


@pytest.mark.slow
@pytest.mark.skipif(not _sd_turbo_is_cached(), reason="sd-turbo not in the HuggingFace cache")
def test_the_real_sd_turbo_checkpoint_still_assembles() -> None:
    """The only check that sd-turbo's own config survives current diffusers/transformers/peft.

    A tiny locally-built VAE/UNet proves the wiring; it cannot prove that `add_adapter`,
    `latent_dist`, `scaling_factor` and `sched.step().prev_sample` still mean what upstream's
    2023-era pinned stack meant by them. ~15s on CPU at this size.
    """
    from t2o.translators.pix2pix_turbo import load_sd_turbo

    translator = Pix2PixTurboTranslator(
        load_sd_turbo(), prompt="a photo", loss_lpips=0.0, loss_gan=0.0
    )
    batch = _batch(n=1, h=96, w=128)

    translator.eval()
    with torch.no_grad():
        generated = translator.translate(batch)
    assert generated.shape == (1, 3, 96, 128)

    translator.train()
    assert "loss_total" in translator.fit(batch)

    # 9.5M trainable parameters against sd-turbo's ~1.3B: the reason a stage checkpoint is
    # tens of MB rather than the ~2.5GB an unreduced state_dict would write twice per stage.
    trainable = sum(p.numel() for p in translator.parameters() if p.requires_grad)
    assert trainable < 20_000_000
