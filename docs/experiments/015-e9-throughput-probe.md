# 015 — E9 step 4: the corpus-cap seam, a per-epoch clock, and the throughput probe — the FLIR cell projects to 77–103 GPU-h, and host RAM, not VRAM, decides concurrency

**Date:** 2026-09-28 · **Task:** M3 E9 · **Machine:** Windows server, 2× A100 — the probe solo on `cuda:0`; the concurrency measurements on step 5's run 1 (`e3f-control-s0`) · **Wall clock:** probe A 1,284.2 s, B 1,191.2 s, C 1,384.5 s, back to back 10:45–12:10 (+03:00)
**git_sha:** not recorded — the seam landed in 5bc815e, the probe result in c7977a3, the concurrency budget in 707f31a, 99b2898 and 109515f · **W&B:** not recorded · **Log:** `runs/flir-probe-report-2026-09-28.txt` (probe A–C); the concurrency budget is none saved — transcribed from TASKS.md 4962-5056 @ pre-spec-migration

## Command

Three epochs, not one, so `seconds` gives a median instead of a single startup-contaminated sample.
Each run prices one complete stage: a translator epoch, the export, the zero-shot pass, FID, and a
50-epoch detector fine-tune.

```powershell
$FLIR  = "dataset/processed/flir/data.yaml"
$JUDGE = "runs/reference-flir-yolo11s/weights/best.pt"

# A -- control, val loss over FLIR's full 1,013
uv run t2o loop --config experiments/e3_pix2pix_control.yaml `
    --data $FLIR --max-train-images 600 `
    --in-loop-weights yolo11n.pt --eval-init-weights yolo11n.pt --reference-weights $JUDGE `
    --epochs 3 --task-weights 0.0 --seed 0 --name flir-probe-control --device cuda:0

# B -- identical but for the val-loss cut, at the custom set's 153
uv run t2o loop --config experiments/e3_pix2pix_control.yaml `
    --data $FLIR --max-train-images 600 --val-loss-images 153 `
    --in-loop-weights yolo11n.pt --eval-init-weights yolo11n.pt --reference-weights $JUDGE `
    --epochs 3 --task-weights 0.0 --seed 0 --name flir-probe-val153 --device cuda:0

# C -- the coupled arm at the ramp's HEAVIEST weight, not its lightest
uv run t2o loop --config experiments/e3_pix2pix_loop.yaml `
    --data $FLIR --max-train-images 600 --val-loss-images 153 `
    --in-loop-weights yolo11n.pt --eval-init-weights yolo11n.pt --reference-weights $JUDGE `
    --epochs 3 --task-weights 3.0 --seed 0 --name flir-probe-loop --device cuda:0
```

The host-RAM probe, to be taken during C's detector stage:

```powershell
Get-Process python | Measure-Object WorkingSet64 -Sum
```

## Configuration

**Step 4 (a), the seam — done 2026-09-28.** ~150 lines across six files, 12 new tests; suite
**511 passed / 4 skipped**, ruff and pyright clean. The table had promised ~20 lines; two things
traced afterwards (F121, F122) changed the step, and that is recorded rather than absorbed.

| field | caps | read by |
| --- | --- | --- |
| `data.max_train_images` | the training corpus | `Trainer.train_dataset` **and** the export |
| `data.val_loss_images` | the per-epoch val *loss* only | `Trainer.val_dataset`, nothing else |
| `data.subset_seed` | which images either cut draws | both |

The cut goes through `data/dataset.py::capped_subset`, which draws through the existing
`annotated_subset` rather than shuffling again, so a budget manifest and this cut select
**identical** images at one `(count, seed)`. A cap larger than the split **raises**: silently
training on 400 when the config says 600 would confound the one variable the knob exists to pin.

**`config_hash` changes for every existing config.** Correct — corpus size changes what is
measured — and there is precedent at the `workers` move. No test pins a hash literal, and a resume
mismatch **warns** rather than raising, so nothing already on disk breaks.

**No new experiment configs.** The FLIR cell runs `experiments/e3_pix2pix_{control,loop}.yaml` with
CLI overrides, exactly as the custom campaign already does with
`--data`/`--in-loop-weights`/`--eval-init-weights`. That keeps
`test_control_and_loop_configs_differ_only_by_design` guarding one pair rather than two, and
`Config.snapshot` still records the fully resolved config per run.

**`--task-weights 3.0` for C on purpose.** FLIR's box density is far above the custom set's — 24,732
cars and 13,094 people across the corpus — and the coupling term runs `v8DetectionLoss` over every
one of them, so the surcharge is priced at its **maximum** rather than carried over as the custom
cell's measured +6.8% (F90).

**`yolo11n.pt` for the in-loop and bootstrap weights makes these three runs unreportable as
results.** Step 2b's FLIR-visible `yolo11n` did not exist yet. The probe measures seconds, and
seconds do not depend on which checkpoint was loaded — but **no mAP may be read off
`runs/flir-probe-*`**, and none of it is a gate number.

**How the columns are read.** Epoch seconds come off the new `EpochStats.seconds`; the detector term
off ultralytics' own cumulative `time` column at epoch 50; the run span off mtimes, where
**`config.yaml` is the run's start, not its end** — `engine/loop.py:145` snapshots it before the
first epoch, and `metrics.json` is written last. The boundary is the interval between
`stage0/translator_last.pt` and `metrics.json`. "Rest" is boundary minus detector: export,
zero-shot and FID.

## Results

As recorded in c7977a3, 707f31a, 99b2898 and 109515f. The tables are the source's, verbatim; the
probe's are re-checked against the saved report below them.

**Step 4 (b), the probe (2026-09-28)** — three runs on **one card, solo**:

| probe | median `seconds` | run span | boundary | of which detector | rest |
| --- | --- | --- | --- | --- | --- |
| A — control, val 1,013 | 55.077 s | 1,284.2 s | 1,116.9 s | 700.4 s | 416.5 s |
| B — control, val 153 | 43.524 s | 1,191.2 s | 1,059.2 s | 653.6 s | 405.6 s |
| C — loop λ=3, val 153 | 51.156 s | 1,384.5 s | 1,229.3 s | 812.9 s | 416.4 s |

Derived in the source: run start to `translator_last.pt` exceeds the sum of the three `seconds` by
**1.56–2.13 s**; **A − B = 11.553 s/epoch** (21.0% of an epoch); **C − B = 7.632 s/epoch**
(+17.5%); boundary **1,135 s** a stage. Detector mAP50s, read only as a noise check: A **0.3073**,
B **0.3551**, C **0.3494**.

The projection, `4 × (100·t_epoch + t_boundary)` per run:

| quantity | val capped at 153 | val left at 1,013 |
| --- | --- | --- |
| control run, flat stages | 6.10 h | 7.38 h |
| loop run, flat stages | 6.73 h | 8.02 h |
| **twelve-run cell, flat** | **77.0 GPU-h** | 92.4 GPU-h |
| twelve-run cell, +20%/stage carried | **103.3 GPU-h** | 124.0 GPU-h |

**From the saved report**, verbatim lines (`runs/flir-probe-report-2026-09-28.txt`; paths
shortened to the run directory by `…`, nothing else changed):

```
================ flir-probe-control ================
2026-09-28T10:45:04.4836290+03:00          1 KB  …\flir-probe-control\config.yaml
2026-09-28T10:47:51.8196880+03:00     65,017 KB  …\flir-probe-control\stage0\translator_last.pt
2026-09-28T11:06:28.6828384+03:00          2 KB  …\flir-probe-control\metrics.json
        "seconds": 55.077484499663115
        "seconds": 55.52917249966413
        "seconds": 54.83268739935011
      "map50": 0.30621540387060653,
50,700.385,1.58028,1.07282,1.13526,0.41714,0.27914,0.30734,0.15008,1.8214,1.37652,1.3551,3.725e-05,3.725e-05,3.725e-05
================ flir-probe-val153 ================
2026-09-28T11:12:44.5837092+03:00          1 KB  …\flir-probe-val153\config.yaml
2026-09-28T11:14:56.5914949+03:00     65,017 KB  …\flir-probe-val153\stage0\translator_last.pt
2026-09-28T11:32:35.8092749+03:00          2 KB  …\flir-probe-val153\metrics.json
        "seconds": 43.52431140001863
        "seconds": 43.05109839979559
        "seconds": 43.87270940002054
      "map50": 0.35972011226719597,
50,653.627,1.48161,0.98187,1.09283,0.4475,0.31727,0.35508,0.17586,1.7542,1.25313,1.30649,3.725e-05,3.725e-05,3.725e-05
================ flir-probe-loop ================
2026-09-28T11:46:55.6048324+03:00          1 KB  …\flir-probe-loop\config.yaml
2026-09-28T11:49:30.7398153+03:00     65,017 KB  …\flir-probe-loop\stage0\translator_last.pt
2026-09-28T12:10:00.0656022+03:00          2 KB  …\flir-probe-loop\metrics.json
    "task_weight": 3.0,
        "seconds": 51.58576110005379
        "seconds": 50.26779139973223
        "seconds": 51.156035100109875
      "map50": 0.36968060391722657,
50,812.891,1.444,0.95655,1.07413,0.69849,0.32521,0.34935,0.17376,1.7381,1.22342,1.2924,3.725e-05,3.725e-05,3.725e-05
```

The `map50` line is `metrics.json`'s `detector` block; the `50,…` line is the detector's
`results.csv` at epoch 50 (`epoch,time,…,metrics/mAP50(B),…`). Every timing cell of the probe table
re-derives from these lines. Each run's listing holds 600 exported train images and 1,013 val.

**Step 5's concurrency budget** — what was measured, from three sides:

| measured | value | where |
| --- | --- | --- |
| the box | **128 GiB RAM, 64 logical CPUs** | step 2, `TotalVisibleMemorySize` |
| GPU peak, one loop run, translator epochs | **22.59 GiB reserved** (14.91 allocated) of 39.70 | step 5 run 1 |
| host peak, one loop run, translator epochs, `workers 16` | **~45 GB** | step 4 (b) |
| host peak, one detector run, 4,129 images, `--workers 8` | **~24 GB** across 24 processes, ≈1 GB each | step 2b |
| two detector runs at `workers 16` | **crashed** — 96 loader processes, 128 GiB exhausted | step 2 |

**The launch rule, pre-registered and run solo** — measured on run 1 (`e3f-control-s0`), stage-0
detector fine-tune, 2026-09-28 19:5x:

| phase | processes | host RAM |
| --- | --- | --- |
| translator epochs | 18 — main + wandb + 16 t2o workers | **14.5 GB**, sawtoothing to 3.2 GB |
| between epochs | 2–3 — main + wandb only | 3.2 GB |
| **stage-0 detector** | **50** — main + wandb + 16 train + 32 val | **52.4 GB** |

`nvidia-smi` agreed with the 22.59 GiB at 23 GB of 40.

Re-render the probe without re-running: `Get-ChildItem runs/flir-probe-* -Recurse` for the mtimes,
and each run's `metrics.json` and `stage0/detector/stage0/results.csv`.

## Findings

### F121 — FLIR's val split is 6.6× the custom set's and the loop scores it 400 times a run, but the per-epoch val loss feeds nothing, so it can be capped without touching a reported number (legacy: M3 E9 step 4, finding 1)

The custom set is **600 train / 153 val**; FLIR-aligned is **4,129 / 1,013**. Capping train alone
leaves the val side 6.6× larger, and `Trainer.train()` calls `self._validate()` **every epoch** —
**400 full val passes** in a four-stage run at `epochs_per_stage: 100`. Weighting a train step at ~3×
a val step, an epoch costs `3·600 + 153 = 1953` units on the custom set against
`3·600 + 1013 = 2813` on FLIR, **~1.44×**, which would put the cell nearer **~180 GPU-h** than F90's
~126. **That was arithmetic, not a measurement** (F125 measures it).

What the pass feeds: `translator_best.pt` is written by `trainer.py` and read by **no loop, no
export and no script** — `run_loop` warm-starts from the live translator and resumes from
`translator_last.pt`. The val loss is a W&B curve and a checkpoint nobody loads.

### F122 — Nothing in the repo recorded a per-epoch wall clock; `EpochStats.seconds` now does, so the FLIR campaign prices itself from `metrics.json` (legacy: M3 E9 step 4, finding 2)

`EpochStats` carried `epoch`, `train_losses`, `val_loss` and no duration; the tracker logged no
time; `metrics.json` had no timestamps. That is why step 0 had to rebuild pix2pix's per-stage cost
from 72 `translator_last.pt` mtimes across 24 runs (F89, F90), and why step 2 was cheaper only
because ultralytics writes a cumulative `time` column of its own. `seconds` is now logged at INFO,
tracked to W&B and written into `metrics.json`. It is **defaulted, not required**, because
`engine/loop.py` rebuilds every entry with `EpochStats(**epoch)` and a required field would make
every earlier run unresumable; `tests/test_loop.py::_without_timings` strips it before the
determinism assertions, because a wall clock is **provenance, not a measurement**.

### F123 — The cap has to reach the export and the val cut must not: exporting all 4,129 while the translator trained on 600 would give the FLIR cell a 6.9× larger detector budget than the custom cell it is matched against (legacy: M3 E9 step 4, "what the seam is")

The exported train split is what the evaluation detector fine-tunes on, so `max_train_images`
reaches the export. `val_loss_images` is deliberately **not** called `max_val_images`: it never
reaches the exported val split, so zero-shot mAP, FID and the detector's own val read every val pair
— the same 1,013 the gate scored. `tests/test_export.py::test_export_caps_train_and_leaves_val_whole`
pins that asymmetry. `subset_seed` is held apart from `train.seed`, as `annotation_seed` is: the cell
sweeps `train.seed` 0..5, and a corpus that redrew with it would add corpus identity as a second
difference between seeds. `annotation_fraction` could not serve — it withholds **boxes, not
images**.

### F124 — The new clock is complete: run start to `translator_last.pt` exceeds the sum of the three `seconds` by only 1.56–2.13 s, so `EpochStats.seconds` covers ~99% of in-stage translator time (legacy: M3 E9 step 4(b), item 0)

The residue is Trainer construction, the pair check over the whole 4,129-image split, and the
checkpoint write. Step 5 can be priced from `metrics.json` alone, against step 0's mtime archaeology
(F89, F122).

### F125 — FLIR's full 1,013-image val pass is 21.0% of an epoch (A − B = 11.553 s), not the 30–40% the arithmetic predicted; `--val-loss-images 153` still saves 15.4 GPU-h across the cell and goes into step 5 (legacy: M3 E9 step 4(b), item 1)

The arithmetic (F121) over-predicted, but the direction held and the knob pays for itself:
92.4 → 77.0 GPU-h on the flat projection.

### F126 — The coupling surcharge is +17.5% per coupled epoch (C − B = 7.632 s) and flat in λ, so one λ=3 probe prices stages 1–3; per complete run it is +10.4%, against the custom cell's +6.8% (legacy: M3 E9 step 4(b), item 2)

`build_detection_loss` returns `None` only at weight 0; at λ = 1, 2 and 3 the in-loop detector's
forward and backward are the identical graph and only the scalar multiplier differs. Per run the
surcharge dilutes to +10.4% because stage 0 is λ=0 in both arms and the four boundaries are shared.
~1.5× F90's +6.8% — FLIR's box density showing up exactly where it was predicted to.

### F127 — The stage boundary is 1,135 s and a real measurement: export + zero-shot + FID agree to ±1.3% across the three runs (416.5 / 405.6 / 416.4 s), while the fixed 50-epoch detector varies ±11% (653.6–812.9 s) (legacy: M3 E9 step 4(b), item 3)

The non-detector work is identical in all three (1,613 images exported, the same 1,013-image val
scored), and so is its time. The detector is the same fixed job every time, so its spread is machine
contention.

### F128 — The FLIR cell projects to 77.0 GPU-h flat and 103.3 with step 0's +20%/stage carried — down from ~126 GPU-h, with the ~180 h fear dead — and the band is a floor, not a forecast (legacy: M3 E9 step 4(b), "row 5 moves from ~126 GPU-h to a band of 77–103")

Two caveats travel with the band. Only **stage 0** was probed, so F89's +20%-per-stage growth is
*carried*, not measured on FLIR. And the probe ran **solo**, while F90's 126 h was measured two-up
with the `e3b-*` campaign six-deep on each card, so every interval it timed carried contention these
runs never saw. Two-up shares 64 logical CPUs between 96 loader processes and slows both runs by an
unmeasured amount, so two cards buy less than 2×.

### F129 — The probe rules out step 0's explanation for the +20%-a-stage growth: the export, and so the detector's fine-tune set, is the same fixed corpus every stage (legacy: M3 E9 step 4(b), item 3b)

F89 floated "the adapted detector's fine-tune set grows with each stage's export". It does not:
`export_translated` writes the same 600 train and the whole val every stage, in the custom cell as
well as this one, and the fine-tune is a fixed 50 epochs over it. The growth remains unexplained,
which is why F128 carries it as an upper bound.

### F130 — Host RAM is ~45 GB per run at `workers 16`, and it — not VRAM — decides whether the cell runs two-up; step 2's ≳1.3 GiB per-process lower bound was off by ~35× (legacy: M3 E9 step 4(b), item 4)

Two concurrent runs need ~90 GB resident, so the decision turns on the box's total memory. **The
lever is `runtime.workers`**, result-neutral and outside `config_hash` since M1.2 step 2b
(augmentation draws from a per-sample generator keyed on `(seed, epoch, index)`); lowering it changes
throughput and nothing measured. The VRAM figure first recorded beside this one is withdrawn (F131).

### F131 — VRAM is not irrelevant: the translator peaks at 22.59 GiB reserved (14.91 allocated) of 39.70, so the 3.42 GB first recorded was sampled outside a translator step and is withdrawn; two runs on one card are ruled out (legacy: M3 E9 step 5 concurrency budget, "Correction — VRAM is not irrelevant")

Step 5's run 1 logged it at epoch 9 (`trainer.py:228`), and `reserved` is the number that decides
(F66). Nothing changed between the probe and this run — same config, same `batch_size: 8` — so 3.42 GB
was most likely sampled at the boundary, where the heaviest thing on the card is a `yolo11n`
fine-tune at batch 16. One run per card leaves ample margin; two on one card would need ~45 GiB of a
40 GiB card, so the fallback if two-up fails is one run at a time.

### F132 — The detector stage inside a loop run is the host-RAM peak: 52.4 GB across 50 processes at `workers 16`, against 14.5 GB during translator epochs — step 2's crash configuration exactly, four times a run, at ≈1.03 GB a worker (legacy: M3 E9 step 5 concurrency budget, "the launch rule was pre-registered, run solo, and it fired")

50 is `workers` + `workers * 2` capped at `os.cpu_count() // device_count()` = 32, both pools
resident at once (`engine/loop.py:245` hands `runtime.workers` straight to `train_detector`).
Subtracting the 3.2 GB base leaves 48 workers in 49.2 GB, **≈1.03 GB each** — between step 2b's
measured ≈1 GB and the crash's implied ≳1.3 GB (F97), so the per-process cost is now a measurement
from three independent runs rather than a bound. The mosaic buffer, `min(ni, batch*8, 1000)` =
**128 decoded images**, does not shrink with the 600-image corpus. Translator epochs sawtooth
because neither `t2o` loader sets `persistent_workers`: the train and val pools never coexist.

### F133 — The cell runs two-up at `workers 8` (26 processes, ~28 GB a run; ~56 GB and 52 processes on 64 CPUs for two); two-up at `workers 16` was rejected unattempted, and the translator pays nothing for the cut (legacy: M3 E9 step 5 concurrency budget, "two-up at `workers 8` is what the cell runs")

At `workers 16`, two-up is ~105 GB of 128 GiB and 100 processes on 64 logical CPUs (1.56×
oversubscription) — **both** of step 2's failure conditions at once (F96, F97). At `workers 8`, both
ratios sit under 1.0, corroborated by step 2b's 24 GB across 24 processes. The translator consumes
**12.7 images/s** at `batch_size: 8`; 8 workers supply 1.6 images/s each, nowhere near binding, so the
cost of halving `workers` falls only on the detector, ~11 minutes of a ~96-minute stage. `--batch`
is not free to move and is not touched.

### F134 — A and B are the same experiment differing only by an inert val cut, and their detector mAP50s differ by 0.048 — a second, independent estimate of the noise floor, from FLIR, just under M1.2's 0.059 (legacy: M3 E9 step 4(b), item 5)

`val_loss_images` reaches no exported split and, per `engine/trainer.py:193`, cannot perturb the
translator either (the val loader "neither shuffles nor augments, so no worker there consumes a
random number"). So A's 0.3073 against B's 0.3551 is entirely detector-side nondeterminism, against
F13's 0.0591. C's 0.3494 landing between the two controls is the cleanest demonstration that none of
these mAPs is reportable.

### F135 — `--resume` is safe on a fresh run directory, so every run in the cell carries it and a crash costs at most one epoch (legacy: M3 E9 step 5 concurrency budget, "run 1 was stopped one stage in and resumed rather than restarted")

The translator checkpoints `translator_last.pt` every epoch, `loop.py:188` resumes the first
unrecorded stage from it, and `_load_existing_results` returns `[]` when there is no `metrics.json`.
Run 1 was stopped after one stage to change `workers` and resumed rather than restarted.

## Provenance caveats

- **Commits that wrote this range** (`git blame` at the tag): 5bc815e (2026-09-28 09:58) step 4 (a),
  findings 1–2, the seam, the commands and the empty table (4729–4887); c7977a3 (13:38) the filled
  table and items 0–5; 707f31a (17:18) the concurrency-budget table and "the unmeasured term";
  99b2898 (18:39) the VRAM correction and the host-RAM reconciliation; 109515f (20:19) the launch
  rule's result, two-up at `workers 8`, the resume and the solo caveat.
- **F131–F133 and F135 come from step 5's run 1, not from the probe.** They sit under step 4 in the
  source, between the probe and step 5's header, which is why they are here. Run 1 itself, and the
  rest of step 5, are record 016's.
- **The withdrawn 3.42 GB claim no longer exists at the tag.** c7977a3's item 4 read "GPU peak was
  **3.42 GB**, so two runs on one card is nowhere near the VRAM limit"; 99b2898 rewrote that sentence
  into the withdrawal F130 now carries. The original survives only in `git show c7977a3:TASKS.md`.
- **The ~45 GB has two contradictory sampling stories in the source.** 99b2898 says the probe's
  sample "almost certainly landed in the boundary" (~2 min of translator against ~18 of boundary);
  707f31a's paragraph, left in place below it, says it "was sampled during translator epochs". The
  budget table's row says "translator epochs". The report holds no RAM figure, so neither is
  checkable; read F130's ~45 GB as unattributed to a phase.
- **The detector mAP50s are the epoch-50 `results.csv` row, not `metrics.json`.** `metrics.json`'s
  `detector.map50` (the `best.pt` evaluation) gives A 0.3062, B 0.3597, C 0.3697. On those, A − B is
  0.0535 (still under 0.059) but **C is above both controls**, not between them — F134's last
  sentence holds only on the epoch-50 reading. `best.pt`'s epoch is not in the report (head and tail
  only).
- **Step 2b's ~24 GB across 24 processes appears first in this range.** Record 013 states the host-RAM
  peak was never measured; where and when step 2b's figure was taken is not recorded.
- **Unit mix in "~23 GB would be left".** The source gives ~105 GB of 128 GiB and "the ~23 GB that
  would be left"; 128 GiB is 137.4 GB, so the remainder is ~32.6 GB. F133 does not cite the remainder.
- **"47 s over 600 images" is not a probe number.** B's median is 43.5 s and C's 51.2 s; their mean is
  47.3 s. Whether 47 s is that mean or step 5 run 1's epoch is not stated.
- **Derived numbers re-checked from the report:** all three medians, spans, boundaries, detector
  times and rests; the 1.56–2.13 s residues (1.56 / 1.90 / 2.13); A − B 11.553 s (20.98%); C − B
  7.632 s (17.54%); the boundary mean 1,135.1 s; the projection table to its last digit, with +20%/stage
  read as compounding 1.2^k per stage (×1.342 on the run); 15.4 GPU-h = 92.4 − 77.0; +10.4% = 6.73 /
  6.10; ≈1.03 GB = 49.2 / 48; 1.56× = 100 / 64.
- **Code anchors, checked at the tag:** `engine/loop.py:145` (`config.snapshot`), `:188` (resume),
  `:245` (`workers=`), `engine/trainer.py:193` (the val-loader comment), `:228` (peak memory),
  `build_detection_loss` (`coupling/schedule.py:21`, `None` at weight ≤ 0), `_load_existing_results`
  (`[]` with no `metrics.json`), no loader passing `persistent_workers`, and `translator_best.pt`
  read by nothing under `src/` or `scripts/`. **Two anchors had drifted:** the source's
  `trainer.py:236` for the per-epoch checkpoint is a W&B comment at the tag (the write is `:336`), and
  `cli.py:256` for `--workers` was correct at 109515f but is `:270` at the tag. The ultralytics
  anchors (`min(ni, batch*8, 1000)`, the val pool's `workers * 2`) are the source's, not re-checked.
- **The two-up plan's 1.0 ratios are arithmetic.** The source calls ~56 GB and 52 processes
  corroborated by step 2b, but two concurrent loop runs at `workers 8` were not measured in this range.
- **Cross-references, re-keyed:** the ~126 GPU-h, the +6.8% and the two-up `e3b-*` timing are F90;
  the +20%/stage and its floated explanation are F89; the 0.059 noise floor is F13; step 2's crash and
  ≳1.3 GiB are F96, F97; the open host-RAM question is F120; `reserved` versus allocated is F66; the
  `workers` decision of M1.2 step 2b is a no-run record SPEC-MIGRATION-19 will mint. "Row 5" is the
  E9 table's step-5 cost row, which goes to the roadmap.

## Next

Step 5 (record 016): the twelve-run FLIR cell. Posted by this record:

- **The cell runs two-up at `--workers 8`, with `--val-loss-images 153` and `--resume`** (F125, F133,
  F135). `--batch` stays.
- **Budget 77–103 GPU-h as a floor, not a forecast** (F128): measured solo, and two cards buy less
  than 2×.
- **The campaign prices itself from `metrics.json`** (F122, F124), so the +20%-a-stage growth that
  F128 carries and F129 leaves unexplained is measured on FLIR rather than assumed.
