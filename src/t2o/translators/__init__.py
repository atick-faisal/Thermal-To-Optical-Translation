"""Uniform wrapper per backbone over `third_party/`. See `Translator` for the contract."""

from __future__ import annotations

from torch import nn

from t2o.config import Config, ConfigError, StubTranslatorConfig
from t2o.translators.protocol import Translator
from t2o.translators.stub import StubTranslator

__all__ = [
    "StubTranslator",
    "Translator",
    "build_translator",
]


def build_translator(config: Config) -> nn.Module:
    """Construct the backbone named by `config.translator.backbone`.

    One `isinstance` branch per `TranslatorConfig` union member (PLAN.md invariant 3): a new
    backbone adds one branch here and nothing else in the dispatch changes. The final `raise`
    is unreachable today (one union member) but stops being unreachable the moment M1 adds
    `pix2pix`, so it stays rather than being deferred until it can actually fire.
    """
    if isinstance(config.translator, StubTranslatorConfig):
        return StubTranslator(hidden_channels=config.translator.hidden_channels)
    raise ConfigError(f"no translator wired up for backbone {config.translator.backbone!r}")
