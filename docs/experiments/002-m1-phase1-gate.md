# 002 — M1 Phase 1 go/no-go: pix2pix with and without the detection loss

**Date:** 2026-08-13 · **Task:** M1 · **Machine:** Windows server, 2× A100 40 GB · **Wall clock:** not recorded
**git_sha:** not recorded — results committed in cf55092 (adapted arm) and 7297601 (zero-shot gate) · **W&B:** not recorded · **Log:** none saved — transcribed from TASKS.md 1048-1138 and 1230-1359 @ pre-spec-migration

## Command

As written in `TASKS.md`. The `<data.yaml>` and `<.pt>` paths actually passed on the server
are not recorded.

```
uv run t2o loop --config experiments/pix2pix_baseline.yaml --data <real data.yaml> \
    --in-loop-weights <visible-trained.pt> --eval-init-weights <visible-trained.pt> \
    --device cuda:0
uv run t2o loop --config experiments/pix2pix_loop.yaml --data <real data.yaml> \
    --in-loop-weights <visible-trained.pt> --eval-init-weights <visible-trained.pt> \
    --device cuda:0
```

The zero-shot gate: validation passes only, no retraining, against the exports at
`runs/<name>/stage*/translated/data.yaml`:

```
uv run t2o evaluate --weights <optical.pt> --data <thermal data.yaml> --device 0
uv run t2o evaluate --weights <optical.pt> --data <real paired data.yaml> --device 0
uv run t2o evaluate --weights <optical.pt> \
    --data runs/pix2pix-baseline/stage0/translated/data.yaml --device 0
uv run t2o evaluate --weights <optical.pt> \
    --data runs/pix2pix-loop/stage<N>/translated/data.yaml --device 0   # N = 0..3
```

## Configuration

- Translator: pix2pix, `resnet_9blocks` generator (the paper's `unet_256` needs input divisible
  by 256; 640x480 is not). Vendored `models/networks.py` at `2a7afba`.
- `experiments/pix2pix_baseline.yaml`: one stage, `task_weights: [0.0]`. It never builds the
  in-loop detector (`coupling/schedule.py`).
- `experiments/pix2pix_loop.yaml`: four stages, `task_weight` 0.0 / 1.0 / 2.0 / 3.0, 100 epochs
  per stage, warm-started stage to stage.
- The two files differ only in `coupling.task_weights` and `runtime.name`, which
  `test_pix2pix_experiments.py::test_baseline_and_loop_configs_differ_only_by_design` pins.
  Both use `loss.gan: 1.0` (the schema default is `0.0`) and `train.lr: 2.0e-4`, which is
  pix2pix's own value.
- `train.epochs_per_stage` and `batch_size` were a starting point, not a measured optimum.
- The arms are not matched on total epoch budget: the loop arm runs 4x `epochs_per_stage`.
- Two evaluation arms per stage:
  - **adapted:** `DetectorResult.map50`, the evaluation detector fine-tuned for 50 epochs on
    that stage's translated train images, then validated on its translated val;
  - **zero-shot:** `detector.reference`, the unadapted visible-trained `optical_best_n_1.pt`
    (`yolo11n`), never fine-tuned on anything this project produced, run through
    `t2o evaluate` (`metrics.task.evaluate_detector`).
- One optical `.pt` was passed as both `in_loop.weights` and `eval_init_weights`, so the
  zero-shot judge of stages 1–3 is the checkpoint that supplied their training gradient.
- Evaluation set: the frozen custom val split, 153 images / 423 instances, four classes.

## Results

Adapted arm (fine-tuned detector, in-domain), as recorded in cf55092:

| stage | task_weight | mAP50 |
| --- | --- | --- |
| baseline | 0.0 | 0.9199 |
| 0 | 0.0 | 0.9053 |
| 1 | 1.0 | 0.9191 |
| 2 | 2.0 | 0.8984 |
| 3 | 3.0 | 0.9132 |

Zero-shot arm (`optical_best_n_1.pt`, identical val split), as recorded in 7297601:

| arm | mAP50 | mAP50-95 | Fuse | Pole | Switch | Transformer |
| --- | --- | --- | --- | --- | --- | --- |
| raw thermal (floor) | 0.1887 | 0.0829 | 0.0377 | 0.5551 | 0.0047 | 0.1573 |
| pix2pix baseline, λ_det=0 | **0.7751** | 0.5244 | 0.8582 | 0.9592 | 0.4320 | 0.8512 |
| loop stage 0, λ_det=0 | 0.7622 | 0.5031 | 0.8199 | 0.9526 | 0.3966 | 0.8799 |
| loop stage 1, λ_det=1 | 0.8218 | 0.5784 | 0.8874 | 0.9558 | 0.5478 | 0.8961 |
| loop stage 2, λ_det=2 | 0.8332 | 0.5536 | 0.8729 | 0.9670 | 0.5781 | 0.9149 |
| loop stage 3, λ_det=3 | **0.8696** | 0.5841 | 0.8942 | 0.9695 | 0.7127 | 0.9019 |
| real visible (ceiling) | 0.9213 | 0.6530 | 0.9448 | 0.9624 | 0.8226 | 0.9554 |

No results file was saved; there is nothing to re-render from.

## Findings

### F03 — The Phase 1 gate passes on the uncontaminated λ_det = 0 arm, on all four classes

The bar was "translated mAP beats raw-thermal mAP on at least one class". The λ_det=0
baseline never constructs the in-loop detector, so it cannot be contaminated by it. It scores
0.7751 zero-shot mAP50 against the raw-thermal floor's 0.1887: **+0.586 mAP50**, with every
class improving. Fuse (0.0377 → 0.8582) and Switch (0.0047 → 0.4320) go from unusable to
usable. C4's premise ("does translation beat direct thermal detection") is answered
affirmatively without needing the loop at all.

### F04 — λ_det raises zero-shot mAP50 monotonically, but the judge is contaminated, so the gain is not yet reportable

Against the epoch-matched control (loop stage 0, λ=0, 0.7622): λ=1 → +0.060, λ=2 → +0.071,
λ=3 → **+0.107**. Against the baseline arm, stage 3 is +0.095. The noise floor between two runs
of the same configuration (baseline vs. loop stage 0, identical computation) is 0.0129 on
this arm, so stage 3's gain is ~8x noise and past PLAN.md §16's "+2–4 mAP50 over the strongest
baseline" criterion. The gain is concentrated in Switch, the hardest class and the one raw
thermal fails on completely (0.0047): 0.3966 → 0.7127. Stage 3 reaches 94% of the
real-visible ceiling, up from 83%.

The λ_det > 0 stages are graded by the same checkpoint that supplied their training gradient,
and a monotone gain in λ_det is exactly the shape reward hacking produces. Two pieces of
evidence would separate them: an independently-trained visible detector as judge, and fidelity
metrics (if LPIPS/FID degrade as λ_det rises while zero-shot mAP climbs, that is reward
hacking; if they hold, the gain is real). Until then the honest claim is the gate itself (F03).

### F05 — The adapted (fine-tuned) arm is saturated and cannot rank the arms

Every stage lands between 0.8984 and 0.9199. A same-domain-trained detector already scores
above 0.9 mAP50 on raw thermal (record 001, F01), so ~0.92 on translated images says a YOLO
fine-tune converges on this dataset whatever domain it is shown. The baseline and loop
stage 0 are the same computation (same seed, same 100 epochs, `task_weight=0`) and returned
0.9199 and 0.9053, a 1.5-point gap with no experimental difference behind it; every loop
stage falls within that band. The adapted arm ranks stage 3 (0.9132) *below* the baseline
(0.9199), where the zero-shot arm ranks it +0.095 above: the two arms invert the ordering.
Anything measured on the adapted arm alone should be treated as uninformative. It also needs
exactly the thermal-domain annotations E8's low-annotation framing exists to avoid.

## Provenance caveats

- F04's λ_det > 0 numbers come from a contaminated judge: `optical_best_n_1.pt` is both the
  in-loop detector and the zero-shot judge. `engine/loop.py` warns about this at run start.
  Only the λ_det = 0 rows are clean.
- The raw-thermal floor (0.1887) was re-measured as part of this gate evaluation and replaces
  M0.10's ad-hoc "< 0.05". That correction is recorded in record 001 (F02 and its caveats), not
  minted again here.
- The 0.0129 noise floor is one pair of identical-config runs, not a variance estimate.
- Single seed, one detector checkpoint, 153 val images, no confidence interval.
- The two arms are not matched on total epoch budget; a budget-matched ablation was left to
  E3.
- "+2–4 mAP50 over the strongest baseline" is the legacy `PLAN.md §16` criterion, not
  `docs/goal.md`'s Margin, which is measured on the held-out test split with a bootstrap CI.
- `runtime.wandb: false` in both tracked files, with opt-in by `--wandb`; whether it was
  passed is not recorded.

## Next

Separate a real λ_det gain from reward hacking before any λ_det > 0 number is reported:
fidelity metrics on these runs' exports (M1.1, record 003), then an independently-trained
visible detector as judge (M1.2, record 004). Prediction: if the gain is real, LPIPS/FID hold
as λ_det rises, and an independent judge still ranks stage 3 above the λ=0 control.
