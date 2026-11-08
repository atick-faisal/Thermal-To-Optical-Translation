"""The cascade gate's stage-2 campaign -- `experiments/cascade_pisa_sr.yaml` (CASCADE-GATE-07).

The file is one `RefineConfig` for every run of the campaign; the server runbook supplies the
source, checkpoint and output per run. These tests read it through the same parser and override
table `t2o refine` uses, so a renamed flag or a drifted field fails here and not on GPU1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from t2o.cli import _REFINE_OVERRIDES, build_parser, overrides_from_args
from t2o.config import AmpDtype, RefineConfig

CASCADE = Path(__file__).resolve().parent.parent / "experiments" / "cascade_pisa_sr.yaml"


def _load_as_the_runbook_does(stage: str, source: str) -> RefineConfig:
    args = build_parser().parse_args(
        [
            "refine",
            "--config",
            str(CASCADE),
            "--source",
            f"{stage}/{source}",
            "--checkpoint",
            "runs/cascade-gate/weights/pisa_sr.pkl",
            "--run-dir",
            stage,
            "--name",
            f"cascade-{source}",
            "--device",
            "cuda:1",
        ]
    )
    return RefineConfig.load(args.config, overrides_from_args(args, _REFINE_OVERRIDES))


def test_the_file_pins_adr01s_setting_on_val() -> None:
    config = RefineConfig.load(CASCADE)

    assert config.refiner.dtype is AmpDtype.FLOAT32
    assert config.source.split == "val"


@pytest.mark.parametrize("source", ["translated", "erased-translated"])
def test_each_refine_lands_beside_its_source_in_the_same_stage(source: str) -> None:
    """`--write-back` files scores in the run whose `stage<N>` sits above the scored tree,
    so the cascade's scores land in the same run and stage as stage 1's."""
    config = _load_as_the_runbook_does("runs/e3t-loop-s3/stage3", source)

    assert config.source.export == Path("runs/e3t-loop-s3/stage3") / source
    assert config.runtime.path == Path("runs/e3t-loop-s3/stage3") / f"cascade-{source}"
    assert config.runtime.device == "cuda:1"


def test_val_and_erased_val_differ_only_in_their_source() -> None:
    stage = "runs/e3b-loop-s2/stage3"
    val = _load_as_the_runbook_does(stage, "translated").model_dump(exclude={"runtime"})
    erased = _load_as_the_runbook_does(stage, "erased-translated").model_dump(exclude={"runtime"})

    assert {**val, "source": None} == {**erased, "source": None}
    assert val["source"]["split"] == erased["source"]["split"]
