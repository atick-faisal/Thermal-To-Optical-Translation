# 004 — M1.2 step 1: an independent `yolo11s` judge re-scores M1, and E3's design

**Date:** 2026-08-13 · **Task:** M1.2 · **Machine:** Windows server, 2× A100 40 GB · **Wall clock:** not recorded
**git_sha:** not recorded — results committed in e495fff (judge training) and 72ca7db (re-score + gate decision) · **W&B:** not recorded · **Log:** none saved — transcribed from TASKS.md 1511-1680 @ pre-spec-migration

## Command

As written in `TASKS.md`. The `<data.yaml>` paths actually passed on the server are not
recorded.

```
uv run t2o train-detector --data <real paired data.yaml> --init-weights yolo11s.pt \
    --epochs 100 --seed 1 --out runs/reference-yolo11s --device cuda:0

# then, with the printed weights path:
uv run t2o evaluate --weights runs/reference-yolo11s/weights/best.pt --data <thermal data.yaml> --device 0
uv run t2o evaluate --weights runs/reference-yolo11s/weights/best.pt --data <real paired data.yaml> --device 0
uv run t2o evaluate --weights runs/reference-yolo11s/weights/best.pt \
    --data runs/pix2pix-baseline/stage0/translated/data.yaml --device 0
uv run t2o evaluate --weights runs/reference-yolo11s/weights/best.pt \
    --data runs/pix2pix-loop/stage<N>/translated/data.yaml --device 0     # N = 0..3
```

## Configuration

### The judge

- `yolo11s` (a different architecture from the in-loop `yolo11n`), seed 1 (not `train.seed`'s
  0), 100 epochs, trained on the **visible train split only** and validated on visible val. It
  never saw anything this project produced.
- Trained by the new standalone `t2o train-detector` subcommand. `engine/detector_stage.py::
  train_detector` takes explicit parameters instead of a `Config`, with two callers:
  `engine/loop.py` (evaluation detector) and `cli train-detector` (reference detector). The
  roles stay separated by who calls it with which weights, not by a role flag.
- Re-scoring is validation passes only, no retraining, through the same
  `metrics.task.evaluate_detector` path M1 used, on the same 153-image / 423-instance val split.
  The old judge is `yolo11n`, the checkpoint that supplied the in-loop gradient (F04).

### E3's design, fixed here, run in records 005–007

Three defects block M1's λ_det claim (F07): stage 3 ran 400 warm-started epochs against a
100-epoch λ=0 comparator; one `yolo11n` checkpoint was both the in-loop detector and the
judge; and n = 1. E3 removes all three. Scope is the **pix2pix arm only**; the
`pix2pix-turbo` cell waits on M2a.

| arm | `coupling.task_weights` | translator epochs | in-loop detector |
| --- | --- | --- | --- |
| control | `[0, 0, 0, 0]` | 400, warm-started | **never constructed** |
| loop | `[0, 1, 2, 3]` | 400, warm-started | constructed from stage 1 |

Everything else — warm-start, four exports, four evaluation-detector fine-tunes, the seed —
is identical per pair. **The only difference in the entire computation is λ_det.** The
control needs no new machinery: `build_detection_loss` returns `None` at weight 0
(`coupling/schedule.py`), so an all-zero ramp is the same code path with the coupling term
absent, four times over.

**Stage 0 is a free null control.** Both arms are λ=0 there, so the paired stage-0 difference
at each seed measures run-to-run noise *from inside the experiment itself*, at no extra GPU
cost. If the stage-3 effect is not clearly larger than the stage-0 difference, E3 is negative
and gets reported that way.

**Six seeds, and the number is not arbitrary.** The comparison is paired (seed *i*'s control
vs seed *i*'s loop), and the assumption-free test on paired data is an exact sign-flip
permutation over 2ⁿ assignments. Smallest attainable two-sided p: 0.25 at n=3, 0.0625 at
n=5, **0.031 at n=6** — n=6 is the first size where a distribution-free two-sided test can
clear 0.05 *at all*, whatever the effect size. At ~6h per 4-stage run that is ~72 GPU-hours.
Falling back to 3 seeds is allowed but then the writeup says "consistent across 3 seeds", not
"significant".

## Results

Judge training, `yolo11s` seed 1, on the 153-image / 423-instance visible val split, as
recorded in e495fff: **P 0.8931 R 0.9030 mAP50 0.9364 mAP50-95 0.6822**.

Re-scoring under both judges, as recorded in 72ca7db:

| arm | old mAP50 | **new mAP50** | Δ | new mAP50-95 | Fuse | Pole | Switch | Transformer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw thermal (floor) | 0.1887 | **0.1552** | −0.034 | 0.0675 | 0.0321 | 0.3949 | 0.0130 | 0.1810 |
| baseline, λ=0, 100 ep | 0.7751 | **0.7851** | +0.010 | 0.5082 | 0.8515 | 0.9514 | 0.5279 | 0.8095 |
| loop stage 0, λ=0, 100 ep | 0.7622 | **0.7260** | −0.036 | 0.4806 | 0.8172 | 0.9440 | 0.4128 | 0.7300 |
| loop stage 1, λ=1, 200 ep | 0.8218 | **0.8244** | +0.003 | 0.5503 | 0.8791 | 0.9679 | 0.5336 | 0.9169 |
| loop stage 2, λ=2, 300 ep | 0.8332 | **0.8106** | −0.023 | 0.5421 | 0.8596 | 0.9724 | 0.5127 | 0.8978 |
| loop stage 3, λ=3, 400 ep | 0.8696 | **0.8470** | −0.023 | 0.5593 | 0.8852 | 0.9696 | 0.6175 | 0.9156 |
| real visible (ceiling) | 0.9213 | **0.9366** | +0.015 | 0.6831 | 0.9746 | 0.9719 | 0.8507 | 0.9492 |

| comparison | Δ mAP50 | vs. the 0.059 noise floor |
| --- | --- | --- |
| stage 3 vs. the **baseline** arm (λ=0) | +0.062 | ~1× — indistinguishable from one noise draw |
| stage 3 vs. **loop stage 0** (λ=0, epoch-matched) | +0.121 | ~2×, but confounded by 400 vs 100 epochs |

No results file was saved; there is nothing to re-render from.

## Findings

### F10 — M1's gate is confirmed by a detector that supplied no gradient to anything

Baseline 0.7851 against the thermal floor's 0.1552 is **+0.630 mAP50** under the independent
judge, all four classes improving (Fuse 0.0321 → 0.8515, Pole 0.3949 → 0.9514, Switch
0.0130 → 0.5279, Transformer 0.1810 → 0.8095). Under the old judge the same comparison was
+0.586 (F03). Nothing about the headline claim depended on the contaminated judge.

### F11 — No evidence the old judge was trained on val: M1's 0.9213 ceiling stands

The old `yolo11n`'s recorded provenance is "the dataset with ultralytics defaults", which does
not confirm a held-out split. If it had memorised val, it would score real visible *above* a
judge trained on train only. It does the opposite: the new train-split-only judge scores
**0.9364** where the old one scored 0.9213. The gap runs the other way, and is the size
expected from `yolo11s` being the larger model. M1's ceiling row (record 001) stands.

### F12 — Self-grading was real but small, about 2 points; λ_det's effect is not monotone

The λ>0 arms lose ~0.023 on average when the judge changes; the λ=0 arms move −0.036 and
+0.010, in both directions. The in-loop checkpoint inflated the arms it had trained, but by
less than run-to-run noise (F13), consistent with F06's no-reward-hacking result.

Monotonicity was partly an artifact. Old: 0.7622 → 0.8218 → 0.8332 → 0.8696, strictly
increasing. New: 0.7260 → 0.8244 → **0.8106** → 0.8470 — stage 2 dips below stage 1. The
overall direction survives; the clean ramp does not. Any writeup must show the new column and
must not describe λ_det's effect as monotone.

The gain lives in Switch: 0.4128 → 0.6175 from stage 0 to stage 3, the hardest class and the
one raw thermal fails on completely (0.0130). Pole is saturated at ~0.95 in every translated
arm and will not separate anything.

### F13 — The noise floor is 0.059 mAP50, not 0.013: at n = 1, λ_det's gain cannot be separated from run variance

Baseline and loop stage 0 are the same computation at the same seed. Under the old judge they
differed by 0.0129 (F04's noise floor); under the new one, by **0.0591** — 4.6× larger.
Against it, stage 3 beats the baseline arm by +0.062 (~1×, one noise draw) and loop stage 0
by +0.121 (~2×, but confounded by 400 vs 100 epochs). Both λ=0 comparators are the same
condition, and they disagree by more than the effect being claimed. This is the situation
E3's design anticipated, and it turns the six-seed budget-matched campaign from rigour into
necessity. The 0.059 floor is what the aggregator's stage-0 null control has to beat.

### F14 — ultralytics resolves a relative `project` against a machine-global `runs_dir`, not the cwd, so outputs land in another repository

The first server run of `--out runs/reference-yolo11s` from this repo wrote to
`D:\Atick\GitHub\Thermal-Image-Research\runs\detect\runs\reference-yolo11s`, not to
`runs/reference-yolo11s` under the cwd. `cfg/__init__.py::get_save_dir` appends a relative
`project` under `SETTINGS["runs_dir"]/<task>`, and that setting is frozen to whichever git
root was current the first time ultralytics wrote its `settings.json`; the doubled `runs` is
the `runs_dir / task / project` composition. Fixed in e495fff by resolving the path inside
`train_detector`, which covers the loop's per-stage detector directories too.

**M1's numbers are unaffected**: `_resolve_weights` reads the trainer's real `save_dir`, so
stage-to-stage warm-starting always chained the correct checkpoint — only the files sat in the
wrong repository. `evaluate_detector` still writes ultralytics' throwaway `val/` scratch dirs
there; nothing reads them, so they are left alone.

## Provenance caveats

- The new judge's real-visible mAP50 appears twice: **0.9364** at the end of its training run
  (e495fff) and **0.9366** in the re-score table (72ca7db). F11 quotes 0.9364 as the source
  does. The source does not explain the 0.0002 gap; it is not reconciled here.
- F11 is an inference, not a test: the old and new judges differ in architecture (`yolo11n` vs
  `yolo11s`) as well as training split, so "the gap is the size expected from the larger
  model" is the source's reading, not a measured control.
- The old-judge column is lifted from records 001 and 002, not re-measured here.
- Single seed per arm, 153 val images, no confidence interval. The 0.059 noise floor is one
  pair of identical-config runs, not a variance estimate.
- The M1.2 preamble (TASKS.md 1511-1540) states outcomes from steps 6–8 — the +0.0070 null at
  `grad_scale: 1.0e-2`, the 2.3% dose, the +0.0512 (p = 0.031) result at 0.15, and the
  effective λ_det of 0.15/0.30/0.45. It was written after those steps ran. Those numbers
  belong to records 005–007 and are not lifted here.
- F14's path and mechanism were observed on the server with the ultralytics version pinned at
  the time; the version is not recorded.

## Next

Make "differ only by seed" true rather than merely likely (M1.2 step 2: seed every RNG, add
`worker_init_fn`), then run E3 as designed above with `detector.reference.weights` pointing at
this judge. Prediction: the stage-3 paired difference has to clear the stage-0 null across six
seeds; if it does not, E3 is reported negative.
