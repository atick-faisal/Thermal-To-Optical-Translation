"""Uniform wrapper per backbone over `third_party/`. See `Translator` for the contract."""

from __future__ import annotations

import torch
from torch import nn

from t2o.config import Config, ConfigError, Pix2PixTranslatorConfig, StubTranslatorConfig
from t2o.translators.pix2pix import Pix2PixTranslator
from t2o.translators.protocol import Translator
from t2o.translators.stub import StubTranslator

__all__ = [
    "Pix2PixTranslator",
    "StubTranslator",
    "Translator",
    "build_translator",
]


def build_translator(config: Config) -> nn.Module:
    """Construct the backbone named by `config.translator.backbone`.

    One `isinstance` branch per `TranslatorConfig` union member (PLAN.md invariant 3): a new
    backbone adds one branch here and nothing else in the dispatch changes. The final `raise`
    is unreachable today (two union members) but is what makes the next backbone's addition
    (M2a's pix2pix-turbo) a one-branch diff rather than a silent fallthrough.

    Seeds the global torch RNG from `config.train.seed` before constructing the backbone.
    `train.seed` is part of `config_hash()` on the premise that a seed is scientific (M0.2);
    that premise only holds if it actually determines the translator's initial weights, and
    this is the one place every caller (cli.py, tests) already goes through to get one. No
    other choke point exists: `Trainer` never constructs a translator, it only trains one.
    """
    torch.manual_seed(config.train.seed)
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
    raise ConfigError(f"no translator wired up for backbone {config.translator.backbone!r}")
