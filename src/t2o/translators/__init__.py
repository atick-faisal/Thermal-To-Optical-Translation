"""Uniform wrapper per backbone over `third_party/`. See `Translator` for the contract."""

from __future__ import annotations

from t2o.translators.protocol import Translator
from t2o.translators.stub import StubTranslator

__all__ = [
    "StubTranslator",
    "Translator",
]
