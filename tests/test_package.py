from __future__ import annotations

import pytest

import t2o
from t2o import cli


def test_package_imports() -> None:
    assert t2o.__version__


def test_entry_point_target_is_callable() -> None:
    # `[project.scripts] t2o = "t2o.cli:main"` is resolved lazily, so a broken target
    # fails only when someone runs the command. Catch it here instead.
    assert callable(cli.main)


def test_version_flag_exits_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
