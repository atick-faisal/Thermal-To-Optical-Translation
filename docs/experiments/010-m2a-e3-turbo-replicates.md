# 010 — M2a step 5: E3 on pix2pix-turbo at `grad_scale: 0.15`, cut from six paired seeds to three after four OOM episodes

**Date:** 2026-09-23 · **Task:** M2a · **Machine:** Windows server, 2× A100 40 GB · **Wall clock:** 24.1 h per stage (spread 20–30 h), ~96 h per complete run; launched 2026-08-25, n = 3 complete 2026-09-23, crashes included
**git_sha:** not recorded — the runs spanned b7b1627, 5a771e4, 975af9c and 32446bc; results committed in 07a1a92 · **W&B:** group `e3-turbo-g015`; each resumed run appears there twice under one name · **Log:** none saved — transcribed from TASKS.md 3072-3605 @ pre-spec-migration

## Command

As written in `TASKS.md`, in PowerShell since the server is native Windows. The `$DATA` and
`$OPTICAL` paths actually passed are not recorded. Twelve runs at the calibrated dose, `e3t-`
prefix so `runs/e3-*` and `runs/e3b-*` stay unambiguous (`pair_runs` refuses a mixed glob anyway).
**Split across the two cards by seed, never by arm** — a pair must stay on one card or the paired
difference absorbs whatever differs between the GPUs. Shell B starts a minute after shell A, so
ultralytics' machine-global `settings.json` first-touch write lands once. Both shells must carry
**identical** `$DATA`/`$OPTICAL`: those are machine-specific paths rather than config, so a
difference between the shells is a confound `test_control_and_loop_configs_differ_only_by_design`
cannot see.

```powershell
# shell A -- seeds 0,1,2 on cuda:0
foreach ($s in 0,1,2) { foreach ($arm in 'control','loop') {
  uv run t2o loop --config "experiments/e3_turbo_$arm.yaml" `
    --data $DATA --in-loop-weights $OPTICAL --eval-init-weights $OPTICAL `
    --seed $s --name "e3t-$arm-s$s" --group e3-turbo-g015 --wandb --device cuda:0
} }

# shell B -- identical but for the seed range and the card
foreach ($s in 3,4,5) { foreach ($arm in 'control','loop') {
  uv run t2o loop --config "experiments/e3_turbo_$arm.yaml" `
    --data $DATA --in-loop-weights $OPTICAL --eval-init-weights $OPTICAL `
    --seed $s --name "e3t-$arm-s$s" --group e3-turbo-g015 --wandb --device cuda:1
} }
```

This is the first attempt's launcher. The relaunches added a failure check (F65) and an allocator
environment variable; their exact commands are not recorded.

The readout, naming the six runs rather than globbing (F71). `aggregate`'s first line must say
`6 runs, stages [0, 1, 2, 3]` before anything below it is read:

```powershell
$RUNS = 'runs/e3t-control-s0','runs/e3t-loop-s0','runs/e3t-control-s1',
        'runs/e3t-loop-s1','runs/e3t-control-s3','runs/e3t-loop-s3'

uv run t2o aggregate --runs $RUNS --stage 3 `
  --metric zero_shot.map50 fidelity.lpips --csv runs/e3t-tidy.csv
uv run python scripts/loss_share.py --runs runs/e3t-loop-s0 runs/e3t-loop-s1 runs/e3t-loop-s3

foreach ($arm in 'control','loop') { foreach ($s in 0,1,3) {
  uv run t2o faithfulness --translated "runs/e3t-$arm-s$s/stage3/translated" `
    --data $DATA --weights runs/reference-yolo11s/weights/best.pt --write-back --device cuda:0
} }
uv run t2o aggregate --runs $RUNS --stage 3 `
  --metric faithfulness.false_object_rate faithfulness.missed_object_rate `
           faithfulness.detection_consistency
```

## Configuration

**What step 5 was.** E3's strong arm: the λ_det = 0 control against the [0, 1, 2, 3] loop on
`Pix2PixTurboTranslator`, at record 009's calibration — `grad_scale: 0.15`, `train.batch_size: 2`,
full 640×512 frames, bf16 autocast (`train.amp` is honoured, `pix2pix_turbo.py:383`), 100
translator epochs per stage, aggregated with `t2o aggregate` exactly as the pix2pix campaign was.
The pix2pix campaign (record 007) is the control it is read against. C2 is scored with the
**reference `yolo11s`**, exactly as on pix2pix: it is what makes the two cells comparable, and
this backbone back-props through the entire one-step generator with far more capacity to find a
genuine hack. pix2pix's clean C2 (record 008) is a baseline for this campaign, not a guarantee
about it.

**Six seeds became three — decided before any stage-3 number was read.** n = 6 was
pre-registered. The budget argued for n = 3 once the real throughput (F62) and the crash rate were
known: n = 6 would cost ~580 GPU-h (~12 days on two cards) from the relaunch, n = 3 ~276 GPU-h
(~6 days). Seeds 3, 0, 1 were named in advance as the cheapest pair plus two. Reducing n costs
statistical power and nothing else: `grad_scale`, `epochs_per_stage` and the reference judge are
untouched, so turbo-minus-pix2pix stays a pure backbone contrast. **Taken 2026-09-23**, on these
terms: the sign-flip floor at n = 3 is p = 0.25, so turbo is reported as *consistent across three
seeds* and never as *significant*; pix2pix's n = 6 / p = 0.031 result (F35) stays the headline;
and the mismatch between the two campaigns' n is a stated limitation. By then the queues' `break`
left four runs at 4/4 (`e3t-control-s3`, `e3t-loop-s3`, `e3t-loop-s0`, `e3t-control-s1`) and two
at 3/4 (`e3t-control-s0`, `e3t-loop-s1`), each with its partner complete — so finishing those two
completed exactly pairs s0, s1 and s3. Seeds 2, 4 and 5 would have needed six more runs, ~576
GPU-h, at a crash rate that had so far been 100%.

**Pre-registered, before the numbers arrived.** The endpoint is unchanged: the paired stage-3
zero-shot mAP50 difference, exact sign-flip test, n = 3 (only the floor moves, to p = 0.25).
Beyond it:

- **(a)** the loop arm's sd ran 0.69 / 0.52 / 0.41× the control's on pix2pix's three faithfulness
  metrics (F52) and 2.4× tighter on stage-3 mAP50 (F42) — untested there, a real prediction here;
- **(b)** `grad_scale` is identical across the two campaigns, so the turbo-minus-pix2pix contrast is
  a backbone contrast and nothing else.

Before those, on 2026-09-14, one prediction was written down from the first complete run alone —
Observation B (F70) — so that what followed would read as a test rather than a story fitted to
three seeds after the fact.

**The endpoint does not move.** Stage 3 is pre-registered and stage 0 is turbo's best control
number. Reporting stage 0 because Observation B makes stage 3 look unflattering is the same move
as re-tuning `grad_scale` after seeing a mAP result, which step 4 refused and PLAN.md §8 forbids.

**The memory saga, in order** — each step changes allocation only, never numerics, RNG or config,
so runs before and after it belong to one campaign:

1. **First attempt, from 2026-08-25.** Three weeks, twelve runs started, none finished — 18 of 48
   stages recorded (F65). All twelve OOM'd inside `total.backward()` (`pix2pix_turbo.py:345`),
   never at a stage boundary. Read at the time as fragmentation, not capacity.
2. **`PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"`** — withdrawn: a silent no-op on
   Windows (F63). `e3t-loop-s0` OOM'd again on 2026-09-15 on cuda:1 during stage 3, in
   `vae.decode` (`pix2pix_turbo.py:394`).
3. **`max_split_size_mb:512`** on card B; card A was left on the ineffective setting, since killing
   it forces the same Adam-resetting resume a crash would. Card B failed again the same day, now in
   FID with a cuSOLVER error (F64).
4. **b7b1627** — `torch.cuda.empty_cache()` once per epoch in `Trainer.train` and once before
   `evaluate_fidelity`. Fixed the cuSOLVER failure; did not stop the OOMs — both cards crashed
   again some hours after the relaunch, 2026-09-23 (F66, F67).
5. **5a771e4** — per-epoch `max_memory_allocated`, `max_memory_reserved` and card total, to the
   console and W&B under `stage{n}/memory/*` (`trainer.py::_peak_memory`). Headroom logged
   instead of discovered by crashing.
6. **975af9c** — `Trainer._train_epoch` calls `torch.cuda.empty_cache()` every
   `CACHE_RELEASE_STEPS` = 50 steps (`trainer.py:72`, `:309-310` at the tag), between two `fit()`
   calls, where the graph is gone and most segments are fully free. Not a guarantee: the run still
   needs ~32 GB of a 39.70 GB card.

The two stranded runs resumed for the last few epochs of stage 3: `e3t-control-s0` from epoch 97
and `e3t-loop-s1` from epoch 95, ~25 and ~50 minutes of training, then ~4 h of stage boundary
each, in parallel — "n = 3 is about five hours away, not the ~24 h the stage-level estimate
assumed".

**Not fixed during the campaign, deliberately.** Checkpointing the optimizers, `batch_size: 1`,
the `[512, 512]` crop and UNet gradient checkpointing (PLAN.md §15's own named ladder) would all
have made the resumed runs differ from the finished ones by machinery as well as by seed, and
orphaned the four finished runs. They are what a *restarted* campaign should be built on, with
reserved peak as the acceptance number.

## Results

As recorded in 07a1a92 and the commits before it. Per-run rows are in `runs/e3t-tidy.csv` on the
server; the source carries no console block for the campaign, only the tables below.

**Throughput**, wall clock between consecutive `stage*/translator_last.pt` mtimes across all 13
completed intervals of the first attempt:

| unit | measured |
|---|---|
| 100 translator epochs (batch 2, full 640×512, bf16 autocast) | ~20 h |
| one stage boundary (export → zero-shot → fidelity → 50-epoch adapted fine-tune) | ~4 h |
| **one stage** | **24.1 h** (spread 20–30 h) |
| one complete 4-stage run | **~96 h = 4 days** |
| twelve runs | **~1 150 GPU-h ≈ 24 days on two cards, crash-free** |

**Allocator state at the four OOM episodes:**

| episode | allocated | reserved-but-unallocated | failed request |
| --- | --- | --- | --- |
| first attempt, all twelve runs | 32.05 GiB | 6.92 GiB | 320 MiB |
| after `expandable_segments` (a silent no-op here) | 32.44 GiB | 6.63 GiB | 160 MiB |
| after `max_split_size_mb:512` | — | — | failed again |
| after `b7b1627`'s per-epoch `empty_cache` | 32.06 / 32.13 GiB | 6.83 / 6.93 GiB | 320 / 160 MiB |

**Resume scars, read from `metrics.json` before the last relaunch.** A resumed stage records only
the epochs it ran after the resume, so a list shorter than 100 is a crash scar:

| run | epochs per stage | scarred stages | **stage-3 steps under reset moments** |
| --- | --- | --- | --- |
| `e3t-control-s3` | `[56, 100, 100, 99]` | 0, 3 | 1 epoch |
| `e3t-loop-s3` | `[100, 2, 100, 100]` | 1 | none |
| `e3t-loop-s0` | `[100, 100, 100, 0]` | 3 | **none** |
| `e3t-control-s1` | `[100, 100, 100, 100]` | none — never resumed | none |
| `e3t-control-s0` | `[100, 1, 100]` | 1 | stage 3 in flight |
| `e3t-loop-s1` | `[100, 2, 100]` | 1 | stage 3 in flight |

| pair | control post-reset epochs in stage 3 | loop post-reset epochs in stage 3 |
| --- | --- | --- |
| s3 | 1 | 0 |
| s0 | 2 | 0 |
| s1 | 0 | 4 |

**First complete run, `e3t-control-s3`** — finished 2026-09-14, λ_det = 0, 4/4 stages:

| stage | `zero_shot.map50` | `mAP50-95` | `fidelity.lpips` | `fidelity.fid` | recorded epochs |
| --- | --- | --- | --- | --- | --- |
| 0 | **0.8889** | 0.5974 | 0.2735 | 75.23 | 56 |
| 1 | 0.8570 | 0.5871 | 0.2674 | 81.46 | 100 |
| 2 | 0.8569 | 0.5938 | 0.2576 | 81.43 | 100 |
| 3 | **0.8293** | 0.5615 | 0.2642 | 82.12 | 99 |

**The campaign, seeds 0, 1, 3 × two arms, all six runs 4/4.** Paired zero-shot mAP50, loop minus
control:

| stage | Δ mAP50 (s0 / s1 / s3) | mean | p | 95% CI |
| --- | --- | --- | --- | --- |
| 0 (null) | +.0276 / +.0256 / **−.0601** | −0.0023 | 1.000 | [−.0601, +.0276] |
| 1 | +.0339 / +.0383 / +.0101 | +0.0274 | 0.250 | [+.0100, +.0383] |
| 2 | +.0358 / +.0298 / +.0048 | +0.0235 | 0.250 | [+.0048, +.0359] |
| **3** | +.0338 / +.0121 / +.0459 | **+0.0306** | **0.250** | [+.0121, +.0459] |

Trajectory sensitivity statistic at stage 3: +0.0329, p = 0.75, CI [−.0135, +.1060]; per seed
+.0063 / −.0135 / **+.1060**.

`fidelity.lpips` at stage 3: raw paired contrast −0.0197 (3/3); stage-0 null −0.0117 (3/3), per-seed
offsets −.0028 / −.0221 / −.0102; trajectory **−0.0080, p = 0.500**, CI crossing zero.

Control-arm `zero_shot.map50` per run:

```
control-s0  .8303 → .8384 → .8294 → .8184     up, down, down     net −.0119
control-s1  .8604 → .8468 → .8455 → .8501     down, down, up     net −.0103
control-s3  .8889 → .8570 → .8569 → .8293     monotone           net −.0596
```

Arm mean .8599 → .8474 → .8439 → .8326.

**C2 at stage 3**, scored by the reference `yolo11s`, loop minus control, against the pix2pix cell
(F49, F50):

| metric | pix2pix (n=6) | turbo (n=3) |
| --- | --- | --- |
| false-object rate ↓ | −0.0289, p=.156 | −0.0193, p=.500 |
| missed-object rate ↓ | −0.0370, **p=.031** | −0.0386, **p=.250** (floor) |
| detection-consistency ↑ | +0.0291, **p=.031** | +0.0319, **p=.250** (floor) |

Missed-object rate 0.1552 → 0.1166, 3/3 seeds.

**Dose and GAN loss by stage**, pix2pix from record 007's campaign:

| | stage 1 | stage 2 (clean both arms) | stage 3 |
| --- | --- | --- | --- |
| pix2pix `loss_gan` | 1.80 | 1.75 | 1.82 |
| pix2pix detector share | 10.0% | **16.1%** | 19.8% |
| turbo `loss_gan` (loop) | 2.47 | 3.31 | 4.51 |
| turbo detector share | 8.8% | **12.3%** | 12.9% |

Control-arm `loss_gan` (`--terms-only`), stages 0–3: 1.17 → 2.27 → 3.26 → 4.44. At the clean stage
2, per control seed: 1.55 / 3.81 / 4.42.

**Endpoint exposure to the AdamW reset**, the campaign result's own audit:

```
                 stage:  0    1    2    3          endpoint exposure
e3t-control-s0         100    1  100    2          2 epochs  (reset @98, capped by the stage)
e3t-control-s1         100  100  100  100          clean
e3t-control-s3          56  100  100   99          ~2-3 epochs  (reset @1)
e3t-loop-s0            100  100  100    0          none      (no training after resume)
e3t-loop-s1            100    2  100    4          ~3 epochs (reset @96)
e3t-loop-s3            100    2  100  100          clean
```

**sd ratios, loop / control, stage 3:** mAP50 0.71×, LPIPS 0.40×, missed-object 0.54×,
detection-consistency 0.25×, false-object 1.07×.

## Findings

### F62 — Turbo costs 24.1 h a stage and ~96 h a 4-stage run; the ~72 GPU-h budget the campaign launched against was pix2pix's, and 16× low (legacy: M2a step 5, throughput)

The plan carried pix2pix's "~6h per 4-stage run" over unchanged, which is the whole reason the
first attempt consumed three weeks and finished nothing. Twelve crash-free runs would cost ~1 150
GPU-h, ~24 days on two cards. `train.amp` is `true` at `bfloat16` and honoured, so 24 h is the
honest cost of this configuration, not a missing-mixed-precision bug. **A backbone's throughput is
not inherited with its `grad_scale`**: time one stage of any new backbone before committing a
campaign to it. Step 4's two 25-epoch probes (record 009) had the answer in them, and their wall
clock was never read.

### F63 — `expandable_segments:True` is silently unsupported on Windows: torch 2.12.1 accepts the variable, prints `expandable_segments not supported on this platform`, and keeps the native allocator (legacy: M2a step 5, the withdrawn allocator fix)

`torch.cuda.memory_snapshot()` reports `is_expandable` as `[False]`. The crash-free 24 h after the
relaunch on it was luck, not the fix: the crash came back on 2026-09-15 with 32.44 GiB allocated,
6.63 GiB reserved-but-unallocated and a 160 MiB request failing — the first attempt's fingerprint
to within 0.3 GiB. The one-liner that caught it, to run before crediting any allocator setting on
this server:

```powershell
uv run python -c "import torch; torch.zeros(1, device='cuda:1'); print([s.get('is_expandable') for s in torch.cuda.memory_snapshot()])"
```

### F64 — A run resumed mid-stage 3 died in FID with `CUSOLVER_STATUS_INTERNAL_ERROR` from `cusolverDnCreate`; read as memory, not numerics, and fixed by releasing the cache before `evaluate_fidelity` (b7b1627) (legacy: M2a step 5, the cuSOLVER/FID failure)

The resumed `e3t-loop-s0` finished stage 3's training, export and zero-shot pass, then failed in
`loop.py:212` → torchmetrics `_compute_fid` → `torch.linalg.eigvals`. The likely reading — "though
no authoritative source states it" — is that cuSOLVER allocates its handle outside torch's caching
allocator, torch releases its cache only for its own failed allocations, and a day of training
leaves that cache holding the card. It surfaced only now because PyTorch keeps the handle for the
whole process: an uninterrupted run creates it at stage 0's FID, but this process was resumed
mid-stage 3, so its first `eigvals` came after a full day of cache growth. The error's own
suggestion, `preferred_linalg_library`, was rejected because it changes how FID is computed
mid-campaign. The failure has not recurred since b7b1627.

### F65 — A crashed `uv run` returns and `foreach` moves to the next seed, so the first attempt started all twelve runs and completed none: 18 of 48 stages recorded, every run stranded at 1–3 stages (legacy: M2a step 5, the launcher's failure check)

A relaunch must read `metrics.json` after each run and `break` on fewer than four stages, or a
card spends days on runs that cannot finish. The relaunch's check did its job on 2026-09-15: card
B stopped rather than starting `e3t-control-s1` behind a broken run. **Recovery is cheap because
`--resume` is stage- and epoch-granular**: every stranded stage-1 checkpoint sat at epoch 97–99, so
resuming re-trained 1–2 epochs rather than 100.

### F66 — The turbo configuration never fit: four crashes across three allocator configurations and two cards show a steady ~32.1 GiB allocated plus ~6.8 GiB reserved-but-unallocated, ~39.0 of a 39.70 GiB card, about 98% (legacy: M2a step 5, the fourth episode)

That overhead does not drift across the episodes; it is a steady state, not a leak. So the
diagnosis carried since the first attempt — "fragmentation, not capacity" — is wrong, or at best
half true. Step 4 measured 34.14 GB of `max_memory_allocated`, `--no-detector`, one stage, and
recorded ~5.8 GB of headroom (F55). It never measured `max_memory_reserved`, and the allocator
serves an allocation out of reserved memory, so reserved is the quantity that decides whether the
next one raises. **The calibration measured the wrong quantity, and every run that finished did so
by winning a coin flip repeatedly.** Step 4's own closing note called the boundary "an argument,
not a measurement"; the risk fired three times and was read as fragmentation each time, "because an
allocator flag is a cheaper hypothesis than a wrong margin". `e3t-control-s1` ran all four stages
without a single crash: the configuration can fit, it just does not reliably fit.

### F67 — Allocator settings cannot fix this on Windows: the ~6.8 GB is trapped in segments that each still hold a live tensor, which only `expandable_segments` addresses (legacy: M2a step 5, allocator settings exhausted)

Torch already releases every fully free segment before it raises an OOM, yet the tracebacks report
~6.8 GB reserved-but-unallocated at the moment of the raise — so that memory is not releasable.
`cudaFree` works per segment, a segment with one survivor cannot be returned, and the holes around
the survivors are too small and scattered to serve a contiguous 320 MiB request: *intra-segment*
fragmentation. That explains all three failed flags at once: `garbage_collection_threshold`
releases exactly the set the emergency path already releases; `max_split_size_mb` limits splitting
but not survivorship; `expandable_segments` — one growable virtual segment — is the setting that
addresses it, and Windows does not have it (F63). **b7b1627 is credited for the wrong thing**: it
fixed the cuSOLVER failure (F64), but a per-epoch `empty_cache()` cannot help a crash that lands
mid-epoch, since fragmentation rebuilds across an epoch's hundreds of steps. The lever taken is a
release every 50 steps, between `fit()` calls (975af9c). This finding is an argument from the
tracebacks and the allocator's documented behaviour, not a measurement; see the
`garbage_collection_threshold` caveat below.

### F68 — Every resume silently resets both of turbo's `AdamW` optimizers, which the checkpoint does not hold; at the endpoint the exposure is 1–4 epochs of 400, falling on both arms (legacy: M2a step 5, the optimizer reset)

`Pix2PixTurboTranslator` owns two `AdamW` optimizers (`pix2pix_turbo.py:263` for the LoRA generator,
`:276` for the PatchGAN) and `trainer.py`'s module docstring accepts that momentum does not survive
a resume — a note written when the CPU stand-in `StubTranslator` was the only backbone. A
crashed-and-resumed run is therefore not the same experiment as a clean one, and this campaign
resumed repeatedly. **What was wrong was that it went unrecorded.** Measured on 2026-09-23: the
reset reaches the readout only through optimizer steps after a resume in the stage that produces the
reported checkpoint, which is stage 3. Four of the six runs carry an effectively uncontaminated
endpoint; `e3t-loop-s0` resumed with its checkpoint already at the final epoch, so `range(100, 100)`
was empty and it took zero steps on reset moments. The disturbance does not fall systematically on
one arm — s3 and s0 disturb the control, s1 the loop. Stated as a limitation; nothing is re-run for
it. F79 is the campaign result's re-reading of this audit.

### F69 — Turbo's control arm starts far above pix2pix's: stage-0 zero-shot mAP50 0.8889 against pix2pix's control mean 0.7579 ± .0372, +0.131 or ~3.5 sd (legacy: M2a step 5, observation A)

A pretrained one-step generator produces detection-legible visible frames before any coupling is
applied. Stage 3 keeps a smaller version of the lead — 0.8293 against 0.7975 ± .0330, about 1 sd —
and `fidelity.lpips` 0.2642 against 0.2905 ± .0054, roughly 5 sd better. The LPIPS gap is the
sturdier of the two, since pix2pix's control sd there was only ±.0054. **Weak evidence**: n = 1,
control arm only, and `e3t-control-s3` is the most-disturbed run in the campaign (two resumes,
stage 0 at epoch 44 and stage 3).

### F70 — Prediction, recorded 2026-09-14 from one run: turbo's control arm declines monotonically under warm-started training, 0.8889 → 0.8570 → 0.8569 → 0.8293, with FID 75.23 → 82.12 and LPIPS flat; `e3t-control-s0` and `e3t-control-s1` will show the same decline (legacy: M2a step 5, observation B)

More fidelity training would drift away from detection-legible structure without buying a matching
perceptual gain. pix2pix's control did not do this — it dipped at stage 1 and rose to stage 2
(0.7579 → 0.7519 → 0.8071) before its own stage-3 dip. Written down with n = 1 so it could not be
fitted afterwards; the two truncated stages of the run it came from are the two that carry the
trend's endpoints. If it replicated, it was to be reported alongside the stage-3 endpoint, never in
place of it. **It did not**: F73.

### F71 — Globbing `runs/e3t-*` produces a plausible, wrong n = 6 table over stage 0 alone: the six first-attempt stumps collapse `aggregate`'s `common_stages` to `[0]`, and `pair_runs`' unpaired-seed guard stays quiet (legacy: M2a step 5, "name the six runs; do not glob")

The stumps — `control-s2` [0,1], `control-s4` [0], `control-s5` [0], `loop-s2` [0], `loop-s4` [0],
`loop-s5` [0,1] — are paired by arm for seeds 2, 4 and 5, so the run **succeeds**. That happened on
2026-09-23. Since f6ee35e, a `--stage` the runs do not all reach is fatal
(`cli.py::_run_aggregate`), but the glob is still the thing to avoid.

### F72 — E3 replicates on turbo: stage-3 paired zero-shot mAP50 +0.0306, p = 0.250 (the n = 3 floor), CI [+.0121, +.0459], 3/3 seeds, against a level stage-0 null of −0.0023, p = 1.000 (legacy: M2a step 5, campaign result finding 1)

p = 0.25 is 2/2³, the only p this design can reach. Stronger than that conveys: the loop arm wins
**9 of 9** paired comparisons across the three coupled stages, and 2 of 3 at stage 0 — the pattern
a real effect with an honest null produces. Against pix2pix, whose stage-0 null drew −0.0397 (F40)
and whose loop arm started four points behind to finish five ahead, here the arms start level.
Reported as *consistent across three seeds*, never significant: pix2pix's n = 6 / p = 0.031 (F35)
carries the significance claim, and this is backbone-transfer corroboration.

### F73 — Observation B is falsified per run: only `control-s3`, the run it was derived from, is monotone, and its decline (−.0596) is five times the other two (−.0119, −.0103) (legacy: M2a step 5, campaign result finding 2)

The arm *mean* (.8599 → .8474 → .8439 → .8326) looks monotone only because averaging smooths the
wobbles while `control-s3` dominates the trend. What survives is weaker and still worth reporting:
**all three control runs finish below where they started**, so warm-started fidelity training does
cost detection legibility on this backbone. The monotonicity was one run's history — precisely what
the prediction was written to test. It did its job; the finding does not get promoted.

### F74 — The stage-3 trajectory statistic (+0.0329, p = 0.75, CI [−.0135, +.1060]) is one pair's baseline draw, not a contradiction of F72 (legacy: M2a step 5, campaign result finding 3)

Its per-seed values are +.0063 / −.0135 / **+.1060**: the whole statistic is the s3 pair, which drew
a −0.0601 stage-0 imbalance and regressed to the mean by stage 3. The raw paired contrast at the
same stage is +.0338 / +.0121 / +.0459 — homogeneous, 3/3. The pre-registered rule
(`aggregate.py::aggregate`) says the raw contrast governs when the stage-0 draw is level, and for
mAP50 it is level (p = 1.000). Applied mechanically, not chosen after the fact.

### F75 — `fidelity.lpips` is null on turbo: the raw stage-3 contrast is −0.0197 (3/3), but the stage-0 null is −0.0117 (3/3), so the trajectory governs, −0.0080, p = 0.500, CI crossing zero (legacy: M2a step 5, campaign result finding 4)

The same pre-registered rule as F74, read the other way, because the two metrics' stage-0 draws
differ. No LPIPS benefit is claimed. **Not a device confound**, which the stage-0 loss tables first
suggested (every control loss term 7–15% above the loop's): the per-seed offsets span an 8× range
(−.0028 / −.0221 / −.0102), which a hardware difference would not produce; a sign-consistent null is
a 1-in-4 event at n = 3; and each pair is pinned to one card. `control-s3` does not explain it either
— it has the **lowest** stage-0 `loss_lpips` in the arm (0.7821 against `control-s1`'s clean
1.0832). Chance draw, recorded and closed.

### F76 — C2 comes back clean on turbo, with pix2pix's effect sizes: false-object rate −0.0193 (p = .500), missed-object −0.0386 and detection-consistency +0.0319 (both at the p = .250 floor); the mechanism is recall, not precision (legacy: M2a step 5, campaign result finding 5)

The pre-registered discriminator was "false objects flat or falling"; they fell. Missed objects drop
0.1552 → 0.1166, a 25% relative reduction, 3/3 seeds: the translator renders real objects more
legibly and does not invent new ones, which is coherent with a mAP50 gain. Same caveat as on
pix2pix (F50), for the same pre-registered reason: false-object rate is the only one of the three
independent of the gain, and it is the one that does not reach the floor. Directionally
favourable, not established. `reward_target` stays null on this backbone too (F53).

### F77 — No dose-response on turbo (0 → +.0274 → +.0235 → +.0306), because the realised dose plateaued: the detector's share runs 8.8 / 12.3 / 12.9%, below the 20–30% band, as loop `loss_gan` inflates 2.47 → 3.31 → 4.51 (legacy: M2a step 5, campaign result finding 6)

Against pix2pix's monotone 0 → +.0280 → +.0357 → +.0512 (F36). pix2pix's GAN loss is flat across
the ramp (1.80 / 1.75 / 1.82), so tripling `w` nearly doubles the detector's share (10.0 → 19.8%);
turbo's quadruples, outrunning `w`. **The dose stopped rising and the gain stopped rising with it.**
Falsifiable: a `grad_scale` that tracks GAN inflation rather than sitting constant should restore
the dose-response — the concrete experiment this campaign earned, carried to M2b. **The inflation
is backbone-intrinsic, not caused by coupling**: the control arm has no detection term and inflates
just as hard (1.17 → 2.27 → 3.26 → 4.44), and at stage 2 the arms are within 1% (3.26 vs 3.31).
Consequence for pre-registered reading (b): `grad_scale` and λ_eff are identical across the two
campaigns, but identical configuration does not mean identical realised dose — 12.3% against 16.1%
at the same nominal λ. The contrast is a backbone contrast, and the backbone's own adversarial
dynamics are part of what differs. Reading (b) stands, with that stated.

### F78 — Loss-space figures at n = 3 are weak, and only stage 2 is clean in both arms; control `loss_gan` at that stage spans 1.55 / 3.81 / 4.42 across three seeds of one arm (legacy: M2a step 5, campaign result finding 7)

`epoch_means` weights each run equally regardless of length and drops a 0-epoch run silently while
the `runs` column still says 3, so the resumed stages' loss rows are pooled from stumps: loop stage 1
is {100, 2, 2}, loop stage 3 is {0, 4, 100}, control stage 3 is {2, 100, 99}. Stage 2 is
{100, 100, 100} in both arms and is the only row that should be quoted. A 3× spread at identical
settings means pooled loss means here describe rather than measure.

### F79 — The resume audit comes out balanced: two control runs carry ~2–3 degraded endpoint epochs against one loop run's ~3, on opposite pairs, out of 100 at a decayed LR — it cannot systematically favour either arm (legacy: M2a step 5, campaign result finding 8)

`loop.py:174` builds a fresh `Trainer` per stage, but the optimizers are the *translator's* and
`run_loop` holds one translator across all four stages — the warm start. So both `AdamW` states
accumulate over the whole run, and a resume resets everything built so far, not the current stage's
share. (The source notes an earlier draft had this backwards, reasoning from the fresh `Trainer`.)
β₁ = 0.9 recovers in ~10 steps and β₂ = 0.999 in ~1000, so with hundreds of steps per epoch the
damage is the first two or three epochs after the reset, wherever in the stage it landed. s0
handicaps control, s1 handicaps loop, s3 roughly neither. **M2a's open caveat is closed.**
`e3t-control-s1` is clean across all four stages and is the anchor to read the others against. The
largest single disturbance, `control-s3`'s stage-0 reset at epoch 44, is not what produced the
stage-0 offset (F75). Fixed for the next campaign, not this one: e210b13 checkpoints both `AdamW`
states via `get_extra_state()` / `set_extra_state()`, at ~2× the trainable parameters per
checkpoint; it changes no numerics and cannot repair a run that already resumed, so these six stand.

### F80 — Pre-registered reading (a) partially replicates: loop/control sd ratios at stage 3 are 0.71× (mAP50), 0.40× (LPIPS), 0.54× (missed-object), 0.25× (detection-consistency), but 1.07× on false-object rate (legacy: M2a step 5, campaign result finding 9)

All but one sit in pix2pix's 0.69 / 0.52 / 0.41 band (F52). The exception, false-object rate, is
also the metric with the weakest effect in both campaigns. So the regularisation pattern replicates
on the metrics tied to the gain and does not on the one independent metric. At n = 3 an sd is a
2-dof estimate: a direction, not a measurement, and no variance test is run on it.

## Provenance caveats

- **The source's internal line references are stale at the tag.** "the budget fallback taken at
  line 3203" and "the decision at line 3203" point at the open-decision paragraph, which starts at
  line 3200. "~6h per 4-stage run' (line 1502)" is at line 1564 (record 004's source range).
- **Cross-references are re-keyed to F-IDs**: "M1.2 step 8's table" is record 007's Results;
  "M1.2 step 8 finding 8" and pix2pix's "2.4× tighter" are F42; pix2pix's 0.69 / 0.52 / 0.41 sd
  ratios are F52; "finding 13 there" is F50; pix2pix's C2 column is F49/F50; its dose-response is
  F36; its stage-0 −0.0397 is F40; "finding 4" inside finding 8 is F75. The pix2pix `loss_gan`
  1.80 / 1.75 / 1.82 and shares 10.0 / 16.1 / 19.8% match record 007's `loss_share.py` block
  (1.8033 / 1.7464 / 1.8211; F38).
- **F77's stage-1 and stage-3 turbo shares and `loss_gan` are pooled from resume stumps** (F78):
  only the stage-2 column is clean in both arms. The source quotes all three anyway and marks stage
  2 as the clean one. The turbo `loss_share.py` output itself is not in the source — only the
  derived table.
- **The two resume audits disagree in detail.** The pre-result table counts post-resume epochs in
  stage 3 (`control-s3` 1, `loop-s1` 4); F79's counts recovery epochs after the reset
  (`control-s3` ~2–3, `loop-s1` ~3). Both are lifted as written; F79 is the later reading and both
  conclude the same: small, and not systematic by arm.
- **Observation A's comparators cross n**: one turbo control run against the pix2pix control arm's
  six-run mean ± sd from record 007.
- **F67 is reasoned, not measured.** A probe of `garbage_collection_threshold:0.8` came back
  identical to baseline (`reserved 27.80 GiB, allocated 27.79 GiB` both ways), but the source marks
  it inconclusive and not the evidence: `reserved ≈ allocated` shows the test allocation forced the
  emergency cache flush in both arms, which masks what the flag does differently. The
  `max_split_size_mb:512` episode has no allocator numbers in the source.
- **`e3t-loop-s0`'s stage-3 `epochs` list is empty in `metrics.json`**, so `loss_share.py` pools
  fewer than 100 epochs for it; W&B holds the full curves. `t2o aggregate` never reads `epochs`.
- **W&B duplicates resumed runs.** `wandb.init` passes no `id`/`resume` (`tracking.py:57` in the
  source; the call sits at `:57-62` at the tag), so each resumed run appears twice in
  `e3-turbo-g015` under one name: two segments of one run, not two seeds.
- **`git_sha` spans code changes.** The first attempt ran before b7b1627 (2026-09-15); 5a771e4,
  975af9c and 32446bc ("release the cache before the detector fine-tune too") landed on 2026-09-23
  before the last relaunch. The source describes the per-epoch release, the peak logging and the
  per-50-step release as numerically inert; it does not mention 32446bc. Which run segment ran at
  which SHA is not recorded.
- **Code anchors were checked at the tag, not re-executed**: `pix2pix_turbo.py:263` and `:276` are
  the two `AdamW` constructors at 07a1a92 (they moved to `:269` / `:282` at the tag, after e210b13
  added `get_extra_state`); `CACHE_RELEASE_STEPS = 50` is `trainer.py:72`; the levelness rule is in
  `src/t2o/analysis/aggregate.py` (`TrajectoryResult`, `:125-132`).
- **`runs/e3t-tidy.csv` is on the server only.** Unlike E8 and E9's tidy files it was never copied
  to `docs/results/`.
- **Date is the source's.** It dates the campaign result 2026-09-23; 07a1a92 committed it at 18:11.
  The first attempt began 2026-08-25, the first complete run finished 2026-09-14.
- **Nothing later is spliced in.** `git blame` at the tag shows lines 3072–3605 written by
  b8962eb, 18fb4a1, b7b1627, 08cb614, 6f39412, 249606d, f6ee35e, 07a1a92 and e210b13 (2026-09-13
  to 2026-09-23), plus a few older lines from c54afbb, 3e20cf6 and 476ef8c. The code-comment
  correction at `trainer.py`'s `empty_cache()` site and the config headers stay in the repo and are
  not migrated.

## Next

The source carries two items to M2b explicitly:

- **Checkpoint both `AdamW` states** — since landed in e210b13 (F79).
- **A `grad_scale` that tracks GAN inflation** rather than sitting constant should restore the
  dose-response (F77).

It states three more without assigning them to a section:

- **A restarted turbo campaign is built on `batch_size: 1`, the `[512, 512]` crop or UNet gradient
  checkpointing** (PLAN.md §15's ladder), with reserved peak as the acceptance number. Watch
  `stage{n}/memory/reserved_gib`: near 39 GiB means stop rather than spend another day on a coin
  flip.
- **n = 6 on turbo** needs seeds 2, 4 and 5 in both arms — six runs, ~576 GPU-h at the measured
  throughput.
- **Time one stage of any new backbone before its campaign** (F62).
