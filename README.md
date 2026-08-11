# Thermal→Visible Translation with Detection-in-the-Loop

Research code for one question:

> Does closing a training-time detection-consistency loop around a diffusion-based
> thermal→visible translator improve downstream power-line component detection, without
> hallucinating or erasing safety-relevant components?

This repo is an instrument for producing one defensible results table and defending it. It
is not a product — no inference service, no deployment tooling, no UI.

- **[RESEARCH_FINDINGS.md](RESEARCH_FINDINGS.md)** — what we are proving and why
- **[PLAN.md](PLAN.md)** — how the code gets built
- **[TASKS.md](TASKS.md)** — the working checklist

**Status: Phase 0 (instrument).** No research claims yet.

## Install

```sh
uv sync --extra cpu     # macOS / CPU development
uv sync --extra gpu     # CUDA training server
```

Exactly one of `cpu`/`gpu` must be chosen — they are declared as conflicting extras with
explicit PyTorch indices, which is what lets both resolve from a single `uv.lock`.

## Workflow

Code is written and smoke-tested on macOS (CPU/MPS); training happens on a native Windows
Server 2022 box with 2×A100. There is no shared filesystem — the only transport is
`git push` here, `git pull` there.

Consequences, accommodated by design rather than fought (PLAN.md §3):

- **No DDP.** NCCL does not exist on Windows. The second A100 is throughput, not scale —
  one experiment per GPU. Every method must train on a single 40GB card.
- **No bash launchers.** Python is invoked directly through our own config layer.
- **Spawn, not fork.** Dataset classes stay importable at module level, everything
  picklable, entry points guarded. This is the most common source of silent hangs here.
- **No symlinks.** Absolute config paths instead.
- Assume no triton, no xformers, no `torch.compile`.

### Data is never committed

`dataset/` is gitignored with no exceptions. The thermal/visible pairs are unpublished
research data and this remote is public, so **a fresh clone contains no images** — you
supply your own under `dataset/` and point `data.yaml` at it.

### Smoke-fixture discipline

Training data and detector weights live only on the server. Since nothing can be committed,
the test suite generates **synthetic pairs at test time** with `tmp_path_factory` — random
images plus hand-written YOLO labels in the same layout as the real thing. That makes the
suite runnable on a bare clone, and lets a test construct the exact pathology it is probing
(out-of-range class id, unpairable filename, degenerate box) rather than hoping real data
contains one.

Every component must run end-to-end on that synthetic fixture, on CPU, in seconds, as a
pytest. Nothing is pushed without the smoke suite passing:

```sh
uv run ruff check && uv run pyright && uv run pytest -m "not slow"
```

## Layout

```
src/t2o/
  config/        pydantic schemas, YAML loading, config hashing, snapshotting
  data/          manifest, pairing, dataset, labels, adapters per public dataset
  metrics/       fidelity (PSNR/SSIM/LPIPS/FID/KID), task (mAP), faithfulness
  translators/   uniform wrapper per backbone over vendored third_party/
  detection/     in-loop detector + evaluation detector, strictly separated
  coupling/      detection-consistency loss and its schedule
  engine/        trainer, loop, export, checkpointing
experiments/     experiment config YAMLs — tracked. An experiment IS a config file.
runs/            run outputs — ignored. Configs go to the server; results do not come back.
third_party/     vendored at pinned commits, never edited in place
```

## Licensing

Third-party components carry constraints worth noting before any downstream use:
**sd-turbo** is under the Stability AI Non-Commercial Research Community License, **CUT**
bundles NVIDIA StyleGAN2 code (non-commercial), and **ultralytics** is AGPL-3.0 (network
copyleft).
