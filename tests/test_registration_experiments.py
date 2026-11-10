"""C2's registration-error axis, translation half -- `experiments/registration/*.yaml`.

Each level is read against a zero point: the custom set's against e3b (`e3_pix2pix_loop.yaml`),
FLIR's against its own de-rolled 0 px file (ADR-02). A level file that drifted from its parent
on anything but the level would put that drift on the curve's x-axis without a word, so these
tests pin every file to its parent cell. FLIR's parent is e3f, whose data and detectors
`experiments/modtr_flir.yaml` writes out and `test_modtr_experiments.py` already checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from t2o.config import Config
from t2o.data.calibration import load_calibration
from t2o.data.misalignment import REGISTRATION_LEVELS_PX

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
REGISTRATION_DIR = EXPERIMENTS_DIR / "registration"
PIX2PIX_LOOP = EXPERIMENTS_DIR / "e3_pix2pix_loop.yaml"
MODTR_FLIR = EXPERIMENTS_DIR / "modtr_flir.yaml"
FLIR_CALIBRATION = Path("calibration/flir.json")

# The custom set's 0 px is the existing aligned e3b runs (plan Q4), so it has no file here.
CUSTOM_LEVELS = REGISTRATION_LEVELS_PX[1:]
FLIR_LEVELS = REGISTRATION_LEVELS_PX


def _path(dataset: str, level: float) -> Path:
    return REGISTRATION_DIR / f"{dataset}_{level:02.0f}px.yaml"


CUSTOM_PATHS = [_path("custom", level) for level in CUSTOM_LEVELS]
FLIR_PATHS = [_path("flir", level) for level in FLIR_LEVELS]
ALL_PATHS = CUSTOM_PATHS + FLIR_PATHS


def _science_without_the_axis(config: Config) -> dict[str, Any]:
    """Everything `config_hash()` covers, minus the two fields this axis sets."""
    dumped = config.model_dump(exclude={"runtime"})
    del dumped["data"]["misalignment_px"], dumped["data"]["calibration"]
    return dumped


def _flir_parent() -> dict[str, Any]:
    """e3f: the pix2pix loop with FLIR's corpus and detectors."""
    parent = _science_without_the_axis(Config.load(PIX2PIX_LOOP))
    modtr_flir = _science_without_the_axis(Config.load(MODTR_FLIR))
    return {**parent, "data": modtr_flir["data"], "detector": modtr_flir["detector"]}


def test_the_directory_holds_exactly_the_planned_levels() -> None:
    """Plan Q2's levels: no stray file runs, and none is missing from the runbook."""
    assert sorted(REGISTRATION_DIR.glob("*.yaml")) == sorted(ALL_PATHS)


@pytest.mark.parametrize(
    ("path", "level"),
    [pytest.param(_path("custom", lv), lv, id=f"custom-{lv:g}") for lv in CUSTOM_LEVELS]
    + [pytest.param(_path("flir", lv), lv, id=f"flir-{lv:g}") for lv in FLIR_LEVELS],
)
def test_each_file_sets_its_own_level_at_the_first_aligned_seed(path: Path, level: float) -> None:
    config = Config.load(path)

    assert config.data.misalignment_px == level
    assert config.train.seed == 0


@pytest.mark.parametrize("path", CUSTOM_PATHS, ids=lambda p: p.stem)
def test_custom_levels_differ_from_e3b_only_by_the_level(path: Path) -> None:
    """e3b's runs are the 0 px point, so everything but the level must be e3b's."""
    config = Config.load(path)

    assert config.data.calibration is None
    assert _science_without_the_axis(config) == _science_without_the_axis(Config.load(PIX2PIX_LOOP))


@pytest.mark.parametrize("path", FLIR_PATHS, ids=lambda p: p.stem)
def test_flir_levels_are_e3f_de_rolled(path: Path) -> None:
    """ADR-02: every FLIR level, 0 px included, de-rolls by the checked-in calibration."""
    config = Config.load(path)

    assert config.data.calibration == FLIR_CALIBRATION
    assert _science_without_the_axis(config) == _flir_parent()


def test_the_calibration_the_flir_files_name_is_checked_in() -> None:
    """The server resolves it repo-relative; a renamed file would fail only at startup there."""
    assert load_calibration(REPO_ROOT / FLIR_CALIBRATION).magnitude_px() > 0


def test_every_level_is_a_distinct_run() -> None:
    """A copied file whose level or name was not changed would overwrite another level's run."""
    configs = [Config.load(path) for path in [*ALL_PATHS, PIX2PIX_LOOP]]

    assert len({config.config_hash() for config in configs}) == len(configs)
    assert len({config.runtime.name for config in configs}) == len(configs)
