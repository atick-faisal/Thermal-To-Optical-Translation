"""Uniform wrapper per backbone over `third_party/`. See `Translator` for the contract."""

from __future__ import annotations

import torch
from torch import nn

from t2o.config import (
    AmpDtype,
    Config,
    ConfigError,
    Pix2PixTranslatorConfig,
    Pix2PixTurboTranslatorConfig,
    StubTranslatorConfig,
)
from t2o.seeding import seed_everything
from t2o.translators.pix2pix import Pix2PixTranslator
from t2o.translators.pix2pix_turbo import Pix2PixTurboTranslator, load_sd_turbo
from t2o.translators.protocol import Translator
from t2o.translators.stub import StubTranslator

__all__ = [
    "Pix2PixTranslator",
    "Pix2PixTurboTranslator",
    "StubTranslator",
    "Translator",
    "build_translator",
]

# `TrainConfig.amp_dtype` is a config-layer enum; only the backbones that actually autocast
# need the torch dtype behind it, so the mapping lives at the one place they are constructed.
_AMP_DTYPES = {
    AmpDtype.BFLOAT16: torch.bfloat16,
    AmpDtype.FLOAT16: torch.float16,
    AmpDtype.FLOAT32: torch.float32,
}


def build_translator(config: Config) -> nn.Module:
    """Construct the backbone named by `config.translator.backbone`.

    One `isinstance` branch per `TranslatorConfig` union member (PLAN.md invariant 3): a new
    backbone adds one branch here and nothing else in the dispatch changes -- M2a's
    pix2pix-turbo cost exactly that. The final `raise` is what keeps the next addition
    (M2b's LBBDM) a one-branch diff rather than a silent fallthrough.

    Seeds every RNG (`t2o.seeding.seed_everything`: torch, CUDA, numpy, `random`) from
    `config.train.seed` before constructing the backbone. `train.seed` is part of
    `config_hash()` on the premise that a seed is scientific (M0.2); that premise only holds
    if it actually determines the translator's initial weights, and this is the one place
    every caller (cli.py, tests) already goes through to get one. No other choke point
    exists: `Trainer` never constructs a translator, it only trains one.
    """
    seed_everything(config.train.seed)
    if isinstance(config.translator, StubTranslatorConfig):
        return StubTranslator(hidden_channels=config.translator.hidden_channels)
    if isinstance(config.translator, Pix2PixTranslatorConfig):
        return Pix2PixTranslator(
            net_g=config.translator.net_g,
            net_d=config.translator.net_d,
            ngf=config.translator.ngf,
            ndf=config.translator.ndf,
            gan_mode=config.translator.gan_mode,
            lr=config.train.lr,
            loss_l2=config.loss.l2,
            loss_lpips=config.loss.lpips,
            loss_gan=config.loss.gan,
        )
    if isinstance(config.translator, Pix2PixTurboTranslatorConfig):
        return Pix2PixTurboTranslator(
            components=load_sd_turbo(config.translator.pretrained),
            prompt=config.translator.prompt,
            lora_rank_unet=config.translator.lora_rank_unet,
            lora_rank_vae=config.translator.lora_rank_vae,
            net_d=config.translator.net_d,
            ndf=config.translator.ndf,
            gan_mode=config.translator.gan_mode,
            lr=config.train.lr,
            loss_l2=config.loss.l2,
            loss_lpips=config.loss.lpips,
            loss_gan=config.loss.gan,
            amp=config.train.amp,
            amp_dtype=_AMP_DTYPES[config.train.amp_dtype],
        )
    raise ConfigError(f"no translator wired up for backbone {config.translator.backbone!r}")
