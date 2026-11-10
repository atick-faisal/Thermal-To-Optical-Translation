"""Shared config machinery: the strict base model, the error type, and the YAML merge.

PLAN.md §13 records a deliberate deviation from ``../Clean-SeAFusion/src/seafusion/config.py``:
pydantic v2 replaces that module's hand-rolled ``_build``/``_coerce``/``_to_plain``
reflection, keeping its two best behaviours -- unknown-key rejection and ``snapshot()``.

Unknown keys are rejected rather than ignored. A silently-dropped typo in a training config
costs a full run to discover, so the trade is heavily in favour of failing fast.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

CONFIG_FILENAME = "config.yaml"


class ConfigError(ValueError):
    """Raised for an unknown key, a malformed value, or an impossible combination."""


class ConfigBase(BaseModel):
    """Base for every config section.

    ``extra="forbid"`` is the load-bearing setting; ``frozen=True`` makes sections value
    objects, so nothing downstream can mutate the config a run was hashed and snapshotted
    from. ``validate_default=True`` means a default that violates its own validator fails
    at import time rather than only when someone happens to override it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class RunConfigBase(ConfigBase):
    """A config that is one whole run: loaded, hashed and snapshotted the same way every time.

    Shared by the experiment :class:`~t2o.config.schema.Config` and the cascade's
    :class:`~t2o.config.schema.RefineConfig`, so both reject unknown keys, hash the same way
    and leave the same ``config.yaml`` behind. A subclass carries a ``runtime`` section; the
    hash leaves it out.
    """

    @classmethod
    def load(
        cls,
        path: Path | str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> Self:
        """Build a config from an optional YAML file plus optional nested overrides.

        Overrides win over the file, which wins over the declared defaults.
        """
        merged: dict[str, Any] = {}
        if path is not None:
            loaded = yaml.safe_load(Path(path).read_text()) or {}
            if not isinstance(loaded, Mapping):
                raise ConfigError(f"{path}: expected a mapping at the top level")
            merged = deep_merge(merged, loaded)
        if overrides:
            merged = deep_merge(merged, overrides)
        try:
            return cls.model_validate(merged)
        except ValidationError as exc:
            raise as_config_error(exc) from exc

    def to_dict(self) -> dict[str, Any]:
        """The resolved config as YAML-safe primitives: Path -> str, enum -> str."""
        return self.model_dump(mode="json")

    def config_hash(self) -> str:
        """Short digest identifying the *experiment*, stored in checkpoints to catch drift.

        `runtime` is excluded wholly: device, run name and W&B flags do not change what is
        measured, so two invocations of one experiment must agree here. Diverges from
        ``../Clean-SeAFusion/src/seafusion/engine/fusion_trainer.py:60``, which hashed
        everything and therefore warned on a renamed run.
        """
        payload = self.model_dump(mode="json", exclude=self._hash_exclude())
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def _hash_exclude(self) -> dict[str, Any]:
        """What :meth:`config_hash` leaves out, in pydantic's nested ``exclude`` form.

        ``runtime`` always. A subclass adds a field that arrived after runs were hashed, while
        it holds the value those runs implicitly had, so adding a knob orphans no checkpoint.
        """
        return {"runtime": True}

    def snapshot(self, run_dir: Path) -> Path:
        """Write the fully resolved config into the run directory, for reproducibility.

        Includes `runtime`, unlike the hash -- provenance wants to know which machine.
        """
        target = Path(run_dir)
        target.mkdir(parents=True, exist_ok=True)
        destination = target / CONFIG_FILENAME
        destination.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return destination


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``update`` over ``base``.

    Nested rather than top-level so a YAML file that sets one key in a section does not
    wipe that section's other defaults.
    """
    merged = dict(base)
    for key, value in update.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def as_config_error(exc: ValidationError) -> ConfigError:
    """Render a pydantic ``ValidationError`` as a ``ConfigError`` with dotted locations.

    Every error is reported, not just the first, so a config with three typos fails once
    instead of three runs in a row.
    """
    lines = [f"{_location(error['loc'])}: {error['msg']}" for error in exc.errors()]
    plural = "s" if len(lines) > 1 else ""
    return ConfigError(f"invalid config ({len(lines)} error{plural}):\n  " + "\n  ".join(lines))


def _location(loc: tuple[int | str, ...]) -> str:
    """Format a pydantic error location as a dotted config path.

    Discriminated unions insert the matched tag as a segment, so a bad key under the stub
    translator reports as ``translator.stub.nope`` rather than ``translator.nope``. Kept
    as-is: naming the union member the key was checked against is worth more than a path
    that pastes straight back into the YAML.
    """
    return ".".join(str(part) for part in loc) or "<root>"
