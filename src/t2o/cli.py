"""``t2o`` command line interface.

Subcommands wrap the engine entry points built in M0.8: `train` (one stage of
`engine.trainer.Trainer`), `loop` (the full `engine.loop.run_loop` alternation), `export`
(`engine.export.export_translated`), and `evaluate` (`metrics.task.evaluate_detector`,
standalone -- PLAN.md §1 rules out an inference service, so this only ever scores an
existing detector checkpoint, never runs a translator). This module is the sole caller of
``logging.basicConfig`` (PLAN.md §13); library modules only ever take
``logging.getLogger(__name__)``.

Heavy submodules (torch, ultralytics) are imported inside each subcommand function rather
than at module level, so `t2o --version` and `t2o --help` stay fast and importable without a
GPU-capable torch build.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from t2o import __version__
from t2o.config import Config
from t2o.imaging import Normalize

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

# Maps --flag onto its config path. Keeping this table explicit means a new flag cannot
# silently land on the wrong section, and `argparse` stays the single source of truth for
# parsing. A path is (section, ..., key); any depth is walked by `overrides_from_args`.
# Deliberately partial -- fields tweaked rarely (translator.hidden_channels, train.crop,
# metrics.*, coupling.reward_target, ...) stay --config-file-only.
_OVERRIDES: dict[str, tuple[str, ...]] = {
    "device": ("runtime", "device"),
    "seed": ("train", "seed"),
    "run_dir": ("runtime", "run_dir"),
    "name": ("runtime", "name"),
    "wandb": ("runtime", "wandb"),
    "data": ("data", "manifest"),
    "annotation_fraction": ("data", "annotation_fraction"),
    "epochs": ("train", "epochs_per_stage"),
    "batch_size": ("train", "batch_size"),
    "lr": ("train", "lr"),
    "workers": ("train", "workers"),
    "task_weights": ("coupling", "task_weights"),
    "grad_scale": ("coupling", "grad_scale"),
    "in_loop_weights": ("detector", "in_loop", "weights"),
    "eval_init_weights": ("detector", "evaluation", "init_weights"),
    "detector_epochs": ("detector", "evaluation", "epochs"),
    "detector_batch": ("detector", "evaluation", "batch"),
    "normalize": ("export", "normalize"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="t2o", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    # `--version`/`--help` still short-circuit before this is checked, same as they always
    # have -- so `t2o --version` stays reachable with no subcommand given.
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("train", "train the translator for a single stage"),
        ("loop", "run the full alternating translator/detector stage loop"),
        ("export", "translate a dataset into a YOLO dataset the detector can train on"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        _add_config_arguments(sub)

    train = subparsers.choices["train"]
    train.add_argument("--resume", action="store_true", help="resume from translator_last.pt")
    train.add_argument("--stage", type=int, default=0, help="stage index, selects task_weight")

    loop = subparsers.choices["loop"]
    loop.add_argument(
        "--no-detector",
        action="store_true",
        help="translator only; skips export and detector fine-tuning every stage",
    )
    loop.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted loop run; skips stages already in run_dir/metrics.json",
    )

    export = subparsers.choices["export"]
    export.add_argument("--checkpoint", type=Path, required=True, help="translator checkpoint")
    export.add_argument("--out", type=Path, required=True, help="output dataset root")

    evaluate = subparsers.add_parser(
        "evaluate", help="score a detector checkpoint against a dataset split (E1/E2)"
    )
    evaluate.add_argument("--weights", type=Path, required=True, help="detector checkpoint")
    evaluate.add_argument("--data", type=Path, required=True, help="data.yaml to validate against")
    evaluate.add_argument("--imgsz", type=int, default=640)
    evaluate.add_argument("--batch", type=int, default=16)
    evaluate.add_argument("--device", help="null/auto, 'cpu', '0', or '0,1'")

    return parser


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="YAML config; flags below override it")
    # Resolved by engine.trainer.resolve_device -- a single torch.device, never a
    # comma-joined multi-GPU list (PLAN.md §3 rules out DDP on Windows entirely; use
    # CUDA_VISIBLE_DEVICES for a second GPU instead).
    parser.add_argument("--device", help="null/auto, 'cpu', '0', or 'cuda:0'")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--name", help="run name under run_dir")
    parser.add_argument("--wandb", action="store_true", default=None)
    parser.add_argument(
        "--data",
        type=Path,
        help="the dataset's data.yaml; it declares root, splits, nc, names and modality tokens",
    )
    parser.add_argument("--annotation-fraction", type=float, help="E8 low-annotation fraction")
    parser.add_argument("--epochs", type=int, help="translator epochs per stage")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--task-weights", type=float, nargs="+", help="the staged lambda_det ramp")
    parser.add_argument("--grad-scale", type=float)
    parser.add_argument("--in-loop-weights", type=Path, help="frozen in-loop detector weights")
    parser.add_argument(
        "--eval-init-weights", type=Path, help="evaluation detector bootstrap weights"
    )
    parser.add_argument("--detector-epochs", type=int, help="evaluation detector epochs per stage")
    parser.add_argument("--detector-batch", type=int)
    parser.add_argument("--normalize", choices=[m.value for m in Normalize])


def overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Fold the CLI flags that were actually supplied into a nested override mapping."""
    overrides: dict[str, Any] = {}
    for flag, path in _OVERRIDES.items():
        value = getattr(args, flag, None)
        if value is None:
            continue
        if isinstance(value, Path):
            value = str(value)
        node = overrides
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return overrides


def config_from_args(args: argparse.Namespace) -> Config:
    return Config.load(getattr(args, "config", None), overrides_from_args(args))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
    )

    if args.command == "evaluate":
        return _run_evaluate(args)

    config = config_from_args(args)
    return _COMMANDS[args.command](args, config)


def _run_train(args: argparse.Namespace, config: Config) -> int:
    from t2o.coupling.schedule import build_detection_loss, weight_for_stage
    from t2o.data.manifest import DatasetManifest
    from t2o.engine.trainer import Trainer
    from t2o.tracking import RunTracker
    from t2o.translators import build_translator

    manifest = DatasetManifest.load(config.data.manifest)
    translator = build_translator(config)
    stage = min(args.stage, len(config.coupling.task_weights) - 1)
    task_loss = build_detection_loss(
        config.coupling, stage, config.detector.in_loop.weights, manifest.nc
    )

    with RunTracker(config) as tracker:
        trainer = Trainer(
            config,
            manifest,
            translator,
            run_dir=config.runtime.path,
            tracker=tracker,
            task_loss=task_loss,
            task_weight=weight_for_stage(config.coupling, stage),
        )
        if args.resume:
            trainer.resume()
        else:
            config.snapshot(config.runtime.path)
        history = trainer.train()

    logger.info(
        "stage %d done: %d epochs, final val_loss=%.5f", stage, len(history), history[-1].val_loss
    )
    return 0


def _run_loop(args: argparse.Namespace, config: Config) -> int:
    from t2o.engine.loop import run_loop
    from t2o.tracking import RunTracker
    from t2o.translators import build_translator

    translator = build_translator(config)
    with RunTracker(config) as tracker:
        results = run_loop(
            config,
            translator,
            run_dir=config.runtime.path,
            tracker=tracker,
            train_detector_stages=not args.no_detector,
            resume=args.resume,
        )

    for result in results:
        detector_note = f" | mAP50 {result.detector.map50:.4f}" if result.detector else ""
        logger.info(
            "stage %d (task_weight=%s): %d epochs%s",
            result.stage,
            result.task_weight,
            len(result.epochs),
            detector_note,
        )
    return 0


def _run_export(args: argparse.Namespace, config: Config) -> int:
    import torch

    from t2o.engine.export import export_translated
    from t2o.engine.trainer import resolve_device
    from t2o.translators import build_translator

    translator = build_translator(config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    translator.load_state_dict(state["translator"])

    device = resolve_device(config.runtime.device)
    translator.to(device)

    data_yaml = export_translated(translator, config, args.out, device=device)
    logger.info("wrote %s", data_yaml)
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    from t2o.metrics.task import evaluate_detector

    metrics = evaluate_detector(
        args.weights,
        args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    for name, ap50 in sorted(metrics.per_class_ap50.items()):
        logger.info("  %-20s AP50=%.4f", name, ap50)
    return 0


_COMMANDS = {
    "train": _run_train,
    "loop": _run_loop,
    "export": _run_export,
}


if __name__ == "__main__":
    raise SystemExit(main())
