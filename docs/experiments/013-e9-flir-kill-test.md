# 013 — E9 steps 1–3 and 2b: FLIR's thermal labels mirrored, both FLIR detectors trained, and the kill-test scored — WEAK PASS at +0.2123

**Date:** 2026-09-27 · **Task:** M3 E9 · **Machine:** Windows server, 2× A100 — one run at a time on `cuda:0` (64 logical CPUs, 128 GiB RAM) · **Wall clock:** judge 1.43 h; in-loop `yolo11n` 2.29 h wall, 1.36 h clean; gate table "minutes", not timed
**git_sha:** not recorded — the mirror landed in 866e357, the gate driver in accdd8d, the crash diagnosis in e0c4a27 and 2b47d10, the judge and gate result in 569840f, step 2b's result in 707f31a · **W&B:** not recorded · **Log:** none saved for steps 1–3 — transcribed from TASKS.md 4147-4477 @ pre-spec-migration; step 2b's ultralytics `results.csv` saved as `runs/inloop-flir-yolo11n-results-2026-09-28.csv`

## Command

Step 1, the thermal-label mirror, once per dataset:

```powershell
uv run python scripts/mirror_thermal_labels.py --data dataset/processed/flir
```

Step 2, the judge — the relaunch that counts:

```powershell
uv run t2o train-detector --data dataset/processed/flir/data.yaml `
    --init-weights yolo11s.pt --epochs 100 --seed 1 --workers 8 `
    --out runs/reference-flir-yolo11s --device cuda:0
```

Step 2b, the in-loop detector:

```powershell
uv run t2o train-detector --data dataset/processed/flir/data.yaml `
    --init-weights yolo11n.pt --epochs 100 --seed 0 --workers 8 `
    --out runs/inloop-flir-yolo11n --device cuda:0
```

Step 3, the kill-test:

```powershell
uv run python scripts/gate_table.py --data dataset/processed/flir/data.yaml `
    --weights runs/reference-flir-yolo11s/weights/best.pt `
    --primary-classes bicycle car person --out runs/gate/flir --device 0
```

The first, crashed launch of step 2 ran both detectors together, one per card, at `--workers 16`;
its exact command lines are not in the source.

## Configuration

**The sequence, this record's rows** (the source's table, which record 012 posted; its rows 0, 3b,
4 and 5 belong to records 012, 014, 015 and 016 and were filled in by later commits):

| # | step | cost | gates |
| --- | --- | --- | --- |
| 1 | mirror FLIR `visible/labels` → `infrared/labels`, with a test | done — 79 lines + 5 tests | step 3 |
| 2 | train the FLIR judge (`yolo11s`, 100 ep); the in-loop `yolo11n` deferred to step 5 | **1.43 h measured** — done 2026-09-27; `best.pt` is **epoch 46**, not 100 | step 3 |
| 3 | **kill-test** — the FLIR gate table, floor against ceiling | minutes — done 2026-09-27: headroom **+0.2123**, **WEAK PASS** | *everything* |
| 2b | the in-loop `yolo11n` (FLIR visible, seed 0) | **1.36 h clean / 2.29 h wall** — done 2026-09-28; `best.pt` is **epoch 63**; one epoch stalled 56 min | step 5 |

**Step 1 — the mirror** (`t2o.data.mirror` + `scripts/mirror_thermal_labels.py`; 5 tests, suite
458 passed / 4 skipped, pyright clean). It runs on an *already-adapted* tree rather than inside the
adapters, because `adapt_datasets.py` short-circuits on a populated destination and re-extracting
FLIR's ~1.4 GB archive to add a few thousand text files would be absurd. E8 had already built the
guard rail this unblocks (`data/budget.py::_require_labels_beside`, whose message literally says
*"Mirror the labels onto this modality first"*). The mirror **refuses a non-empty
`infrared/labels` unless forced** — the custom set carries labels on both sides, and if its thermal
ones are drawn on the thermal frame in their own right, a blind copy would replace better boxes with
worse on the dataset every headline rests on. The 5.90 px caveat and its direction of bias (F93)
are written into the module docstring. Expected `train=4129, val=1013` — confirmed, on a real
4,129/1,013 tree.

**Step 2 — the judge's relaunch settings**, each a response to the crash (F96, F97):

- **One run at a time, and the `yolo11n` in-loop detector deferred.** It gates step 5, step 5 is
  gated on step 3's verdict, so under a sub-0.15 headroom it is never needed at all. Running it
  first time round to keep card 1 busy traded a free idle GPU against the run that actually gates
  the decision.
- **`--workers 8`** → 8 train + 16 val, up to 24 processes in one process tree instead of 96 across
  two. `--workers 4` (4 + 8 = 12) is the fallback and costs little: at ~44 s/epoch the GPU wants
  ~6.3 it/s of 640×512 crops, and mosaic mostly hits the in-worker buffer, not disk.
- **The crashed run directories deleted before relaunching.** `train_detector` passes
  `exist_ok=True` (`detector_stage.py:162`) and `save_metrics` opens `results.csv` with mode `"a"`,
  writing a header only when the file is absent (`engine/trainer.py:919-927`). Relaunching into the
  same `--out` appends — epochs 1,2,3,1,2,3,… with a `time` column that restarts.
- **`--batch 16` left at its default**, matching the custom judge's recipe (record 004) rather than
  changing a second variable at the same time.
- **`--seed 1` for the judge** — invariant 7, and the reason `cli.py:187` defaults `--seed 1` rather
  than inheriting `train.seed`.
- **Pre-flight: `path:`.** The command hands the *source* `data.yaml` straight to ultralytics (F95);
  if `head -1 dataset/processed/flir/data.yaml` does not name this machine's real dataset root, fix
  that line first.
- **A relative `--out` is safe.** `train_detector` already resolves a relative `project` to an
  absolute path (`engine/detector_stage.py:138-146`), the bug that sent the custom judge into a
  different repository (record 004).
- **Cost measured, not inherited**, from ultralytics' cumulative `time` column in `results.csv`.

**Step 2b** was deferred out of step 2 and bought once step 3's WEAK PASS made it worth buying.
(Local numbering — not M1.2's step 2b, which is the `workers` fix.)

**Step 3 — the gate driver** (`scripts/gate_table.py` + `tests/test_gate_table.py`, 12 tests; suite
470 passed / 4 skipped, ruff and pyright clean). Two validation passes, no training. It prints the
gate table as markdown, writes `runs/gate/flir/gate.csv`, and prints the **pre-registered verdict**:
`KILL_THRESHOLD = 0.15` and `STRONG_THRESHOLD = 0.40` are module constants
(`scripts/gate_table.py:60-61`) citing record 012's rule table, so the band is fixed by the file
rather than chosen by whoever reads the number.

**Both arms are built with `write_budget_manifest` at fraction 1.0, differing only in
`infrared`** — the construction `annotation_sweep.py:328-330` uses for E8's arm A0, with the same
"fraction 1.0 only to obtain a val path; nothing trains on it" rationale. Three things that buys:

1. `_require_labels_beside` (`data/budget.py:51`) **refuses a thermal side with no labels**, so
   step 1 is verified before a GPU starts rather than scored as an unlabelled split.
2. What it emits carries resolved `train`/`val` and **no `path:` at all** (`budget.py:128-130`), so
   neither arm depends on that field being current. Passing the source manifest for the ceiling arm
   is what failed locally (F95).
3. One construction with one flag flipped is what makes ceiling and floor differ in the pixels and
   in nothing else.

**Primary mean: 3-class** (bicycle / car / person), dog stated separately (F94).

## Results

As recorded in e0c4a27, 2b47d10, 569840f and 707f31a. No console output survives for steps 2–3;
these are the source's distilled tables, verbatim. Step 2b's rows were checked against its saved
CSV (see Provenance caveats).

**Step 2, the first launch (2026-09-27), crashed.** The `cuda:1` run raised, inside the validator's
`get_stats` at epoch 3:

```
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 2.83 MiB for an array with shape (296307, 10) and data type bool
```

The `cuda:0` judge hung with no error at all. `GPU_mem` was 2.56G of 40.

Four ultralytics facts (8.4.117) that compound, none visible from the command line:

| # | fact | source |
| --- | --- | --- |
| 1 | the **validation** loader is built at `workers * 2` | `models/yolo/detect/train.py:100` |
| 2 | both loaders are `InfiniteDataLoader`s that spawn workers in `__init__` and, via `_RepeatSampler`, never shut them down — so **both pools are resident for the whole run**, the train pool included during every validation pass | `data/build.py:75`; `engine/trainer.py:281,291` |
| 3 | each pool is capped at `os.cpu_count() // device_count()`, and `device_count()` is **2 regardless of `--device cuda:0`** — a halving that assumes one process per card and does nothing to stop two processes taking it twice | `data/build.py:363` |
| 4 | every worker is a full Windows `spawn` import of torch, and each *train* worker also holds a mosaic buffer of `min(ni, batch*8, 1000)` = **128 decoded images** (~1 MB each at 640×512); `pin_memory` is on and the val batch is doubled to 32 | `data/base.py:131`; `engine/trainer.py:293` |

The server, measured 2026-09-27: **64 logical CPUs and 128 GiB of RAM**
(`[Environment]::ProcessorCount` = 64; `TotalVisibleMemorySize` = 134,216,648 KB), so the per-pool
cap is 64 / 2 = **32**:

| `--workers` | train pool | val pool (`workers * 2`, capped at 32) | processes |
| --- | --- | --- | --- |
| 8 — the relaunch | 8 | 16 | **24** |
| 16 — what crashed | 16 | 32 | **48 per run, 96 across two** |

The crashed run's first four `results.csv` rows: 66.0 / 62.7 / 45.1 / 44.0 s per epoch.

**Step 2, the run that counts (2026-09-27)** — `runs/reference-flir-yolo11s`:

| row | epoch | `time` | mAP50 | mAP50-95 | precision | recall |
| --- | --- | --- | --- | --- | --- | --- |
| best (`best.pt`) | **46** | 2403.77 s = 0.67 h | **0.5525** | **0.2765** | 0.7791 | 0.5108 |
| last | 100 | 5137.57 s = 1.43 h | 0.5079 | 0.2486 | 0.6321 | 0.5066 |

`lr/pg0` at epoch 46: 6.93e-4. `time` is monotone across all 100 rows, 52.3 s/epoch to row 46 and
50.6 s/epoch after.

**Step 2b (2026-09-28)** — `runs/inloop-flir-yolo11n`:

| row | epoch | `time` | mAP50 | mAP50-95 | precision | recall |
| --- | --- | --- | --- | --- | --- | --- |
| best (`best.pt`) | **63** | 6353.56 s | **0.5028** | **0.2417** | 0.6901 | 0.4537 |
| last | 100 | 8234.91 s = 2.29 h | 0.4884 | 0.2287 | 0.8035 | 0.4335 |

Epoch 9's `time` delta is **3,370.8 s against a 48.8 s median**; all ninety-nine others fall inside
43.4–87.9 s. Without that row the run is **4,881 s = 1.36 h**. Epoch 9's train losses
1.74816 / 1.07415 / 1.20188; its mAP50 0.4143.

**The two detectors side by side:**

| | judge | in-loop |
| --- | --- | --- |
| run | `runs/reference-flir-yolo11s` | `runs/inloop-flir-yolo11n` |
| architecture / seed | `yolo11s` / **1** | `yolo11n` / **0** |
| `best.pt` | epoch 46, mAP50 **0.5525** | epoch 63, mAP50 **0.5028** |

**Step 3's plumbing check**, before the real run — the driver against a randomly-initialised 4-class
`yolo11s` on CPU, on the real FLIR tree (the mAP is meaningless by construction): both arms ran, and
the thermal arm read **1,013 images / 8,601 instances**, against 8,604 counted by hand.

**Step 3, the gate — FLIR-aligned, 2026-09-27.** `runs/gate/flir/gate.csv`, one visible-trained
judge (`reference-flir-yolo11s/weights/best.pt`, epoch 46), two validation passes over the same
1,013 scenes:

| arm | mAP50 | mAP50 (3-class) | mAP50-95 | bicycle | car | dog | person |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ceiling (visible) | 0.5523 | **0.6566** | 0.2767 | 0.4922 | 0.8176 | 0.2396 | 0.6599 |
| floor (thermal) | 0.3397 | **0.4442** | 0.1492 | 0.3627 | 0.6033 | 0.0260 | 0.3667 |
| **headroom** | +0.2126 | **+0.2123** | +0.1275 | +0.1295 | +0.2143 | +0.2135 | +0.2932 |

Verdict printed against the pre-registered band: 0.15 ≤ 0.2123 < 0.40 → **WEAK PASS**.

Re-render on the server without retraining: rerun step 3's command (two validation passes); the
detector tables come from each run's `results.csv`.

## Findings

### F95 — ultralytics honours a `data.yaml`'s `path:` literally, while `t2o`'s loader falls through a stale one: a moved tree passes `t2o` and dies in ultralytics (legacy: M3 E9 step 1, "that tolerance does not generalise")

`manifest.py::_resolve_root` tries each candidate root and takes the first that actually contains
the declared train split, so a `path:` pointing nowhere falls through to the manifest's own
directory — the mirror script is unaffected. Ultralytics instead honours `path:` and raises
`FileNotFoundError`. `write_manifest_yaml` writes `path:` absolute at adapt time, so any command that
hands a *source* `data.yaml` to ultralytics — `t2o evaluate --data …/data.yaml` and `t2o
train-detector`, i.e. step 2 — fails at once on a tree that has moved since it was adapted. Against
the hygiene item as first recorded: **non-blocking for the data layer, blocking for ultralytics.**
Hit locally while building step 3's driver.

### F96 — Step 2's first launch died of host memory, not GPU memory: a 2.83 MiB numpy allocation failed while the GPU held 2.56 GB of 40 (legacy: M3 E9 step 2, "2.83 MiB is the whole diagnosis")

A process that cannot allocate 2.83 MiB is out of *host* memory (or past Windows' commit limit); a
CUDA OOM raises `torch.OutOfMemoryError`, not a numpy error. The array itself is ordinary: 296,307
predictions × 10 IoU thresholds, ≈292 per val image against `max_det=300`, exactly what an epoch-3
detector produces. Nothing about the data or the GPU was wrong.

### F97 — Four ultralytics loader facts put two `--workers 16` runs at 96 loader processes on 64 CPUs and 128 GiB, bounding each process at ≳1.3 GiB; one run at a time is the load-bearing fix (legacy: M3 E9 step 2, "what actually exhausted RAM")

The val pool is `workers * 2`, both pools stay resident, and the per-pool cap is halved by a
`device_count()` that ignores `--device` (Results). 96 workers on 64 logical CPUs is also 1.5× CPU
oversubscription, so the `cuda:0` hang had two causes — a main process waiting on starved or dead
workers waits forever. 96 workers plus two main processes exhausted 128 GiB, so each averaged
**≳1.3 GiB**. At that rate one `--workers 16` run (48 processes, ≈64 GiB) would probably have
squeezed through and two never could: **one run at a time** is the fix, and the lower worker count
is only the margin. Acts on M1.2 step 2b's "16 workers runs clean" (`TASKS.md:1763`), which held for
one process on `t2o`'s own single-pool loader at `batch_size: 2` and did not transfer to two
ultralytics detector runs. The arithmetic is pinned beside the `workers` parameter in
`engine/detector_stage.py`.

### F98 — The FLIR judge costs 1.43 h for 100 epochs; the ~6–15 GPU-h estimate was 4–10× high, and the crashed run's first rows projected it 14% low (legacy: M3 E9 step 2, "measure the cost, do not inherit an estimate")

Measured: `time` = 5137.57 s at epoch 100. Against record 012's estimate of ~6–15 GPU-h, scaled from
E8's `yolo11n` cells and never timed. The crashed run's first four rows (66.0 / 62.7 / 45.1 / 44.0 s,
the first two carrying startup and the label-cache build) projected **≈1.25 h**. Pessimistic this
time, so the error cost nothing. It also supplies the wall clock the custom judge never had.

### F99 — The judge's `best.pt` is epoch 46 at 0.5525 mAP50; its last 54 epochs made it worse by 8.1% mAP50 and 10.1% mAP50-95, so "0.508" is the final epoch, not the judge (legacy: M3 E9 step 2, "`best.pt` is epoch 46, not epoch 100")

Epoch 46: 0.5525 / 0.2765; epoch 100: 0.5079 / 0.2486, for 0.76 h of GPU time. Nothing downstream is
harmed — `train_detector` reports and every later step loads `weights/best.pt` — and the gate table
reproduces 0.5523 on the same split. **What does not follow is that `--epochs 50` would do the same
job:** ultralytics decays the LR across the *total* epoch count, and `lr/pg0` at epoch 46 was
6.93e-4, mid-decay. A 50-epoch run is a different schedule, not a prefix of this one, and could land
either side of 0.5525. `time` is monotone across all 100 rows, so the relaunch did not hit the
append trap and the wall clock is real.

### F100 — The in-loop `yolo11n` costs 1.36 h, not the 2.29 h on the clock: one epoch stalled 3,370.8 s against a 48.8 s median, so per-run cost is read from medians, not totals (legacy: M3 E9 step 2b, "one epoch ate 0.93 h of it")

Epoch 9's delta is 69× the median; all ninety-nine others fall inside 43.4–87.9 s. **A pause, not
slow compute:** epoch 9's train losses (1.74816 / 1.07415 / 1.20188) sit on the trend between
epochs 8 and 10, and its mAP50 of 0.4143 was the run's best so far. Not the append trap either —
`time` is monotone and exactly one row is anomalous. A wall-clock-only reading carries a **68%
error** into step 5's projection, and one stall of this size among 48 stage boundaries would be
indistinguishable from real cost — so step 5 reads medians. Against a pricing of ~0.6–0.9 h, clean
it is 1.36 h; what record 012 got right is "cheaper than the judge", 48.8 s an epoch against the
judge's 52.3.

### F101 — The in-loop `best.pt` is epoch 63, and its last 37 epochs gave back 2.9% mAP50 and 5.4% mAP50-95 — the judge's late-epoch pattern again, milder (legacy: M3 E9 step 2b, "`best.pt` is epoch 63, not 100")

Epoch 63: 0.5028 / 0.2417; epoch 100: 0.4884 / 0.2287. Against the judge's −8.1% / −10.1% over its
last 54 epochs (F99). Harmless for the same reason: everything downstream loads `weights/best.pt`.

### F102 — Invariant 7 holds structurally — the judge and the in-loop detector differ in architecture and seed — but their 0.0497 mAP50 gap is below the 0.059 noise floor, so "the judge is stronger" is not shown (legacy: M3 E9 step 2b, "invariant 7 holds structurally, and the margin is smaller than it looks")

`yolo11s` / seed 1 at 0.5525 against `yolo11n` / seed 0 at 0.5028. Different architecture *and*
seed, so `engine/loop.py`'s "in-loop is not the judge" warning stays a real separation. The gap is
measured against M1.2's 0.059 mAP50 noise floor (F13). The invariant asks for a *different* model,
not a better one, so nothing rests on the gap.

### F103 — The mirror landed: on the real FLIR tree the gate's thermal arm read 1,013 images and 8,601 instances, against 8,604 counted by hand (legacy: M3 E9 step 3, "verified end to end on the real FLIR tree")

The 3-instance difference is three duplicate label lines that ultralytics drops and logs. That
reconciliation, against record 012's per-split count, is the evidence step 1 did what it claims; the
driver's plumbing ran on a randomly-initialised `yolo11s`, so its mAP is meaningless by
construction.

### F104 — The kill-test passes weakly on FLIR-aligned: 3-class headroom +0.2123 (ceiling 0.6566, floor 0.4442), only +0.062 over the 0.15 kill line (legacy: M3 E9 step 3, "GATE DECISION — FLIR-aligned, 2026-09-27: WEAK PASS")

Read off the band record 012 pre-registered, not chosen after the fact: 0.15 ≤ 0.2123 < 0.40 →
*"WEAK PASS — transfers weakly. Still run the cell: a smaller gain where thermal is legible is honest
cross-dataset evidence."* A pass, but not a comfortable one.

### F105 — FLIR's raw-thermal floor is 0.4442, 2.4× the custom set's 0.1887: a raised floor, not a lowered ceiling, compresses the headroom from +0.733 to +0.212, as predicted before the number existed (legacy: M3 E9 step 3, "the prediction written before the number existed was correct")

Against record 012's posted prediction ("night pedestrians and warm engine blocks are highly legible
in thermal — expect a much higher floor") and the custom set's bracket (F02). Same construction on
both datasets; the kill-test moved the headroom from +0.733 to +0.212.

### F106 — The 3-class guard did not bind: dog's own headroom (+0.2135) lands within 0.0012 of the 3-class figure, so 4-class and 3-class agree to 0.0003 (legacy: M3 E9 step 3, "the 3-class guard was right to build and did not bind")

4-class +0.2126 against 3-class +0.2123; the verdict is the same either way. Ex ante, a 0.4 AP50
swing on 13 instances could have moved a 4-class mean by 0.1, the width of the decision band (F94),
so the guard was correct insurance that happened not to be needed. Dog's two numbers (0.2396
ceiling, 0.0260 floor) are noise on 13 instances in both arms.

### F107 — `person` has the largest headroom (+0.2932), the class thermal is supposedly best at: the floor measures a visible-trained judge's domain gap, not thermal information content (legacy: M3 E9 step 3, "`person` has the largest headroom")

Against car +0.2143 and bicycle +0.1295. A human reads a thermal pedestrian instantly; an RGB-trained
YOLO does not. That gap is what translation claims to close, so this is the supportive direction —
worth stating in the paper.

### F108 — `bicycle` is the weak class at +0.1295, below the kill line on its own, and the compression comes from a low ceiling (0.4922), not a high floor (legacy: M3 E9 step 3, "`bicycle` is the weak class")

Ceiling 0.4922 against car 0.8176 and person 0.6599; floor 0.3627. Unheated metal frames are
thermally dim and small in frame. If the cell's gain concentrates in `car` and `person` and leaves
`bicycle` flat, this row predicted it.

### F109 — With only +0.062 of margin, the mirrored labels' misalignment confound is load-bearing; the mAP50-95 / mAP50 ratio (0.501 ceiling, 0.439 floor, a 12.3% relative drop) is mild evidence it is not the main driver (legacy: M3 E9 step 3, "the misalignment confound is now load-bearing, not cosmetic")

The direction was pre-registered (F93): mirrored labels inherit the 5.90 px residual, which at IoU
0.5 penalises small boxes and pushes the **floor down**, flattering translation — so a large enough
share of this headroom being label error would move the verdict toward KILL. A ~6 px offset attacks
high-IoU AP far harder than AP50, so if it dominated the floor the ratio would fall much further
than 12%. Suggestive, not conclusive: the two arms also differ in the pixels, which is the point of
the experiment. The clean test is the de-roll and re-gate (Next).

## Provenance caveats

- **Step 2b's saved CSV disagrees with the source in three places**, all lifted unchanged above:
  - The stall-free total: dropping epoch 9's 3,370.8 s from 8,234.91 s gives **4,864 s (1.35 h)**,
    not the 4,881 s (1.36 h) recorded. How 4,881 was computed is not stated.
  - The run's **mAP50 peaks at epoch 76 (0.5153)**, not epoch 63. Epoch 63 is still `best.pt`:
    ultralytics 8.4.117 selects on mAP50-95 alone (`ultralytics/utils/metrics.py:1009`, weights
    `[0, 0, 0, 1]`), and epoch 63 is the mAP50-95 peak. A `best` row in this record is that peak,
    so F101's "gave back 2.9% mAP50" is measured from `best.pt`, not from the run's mAP50 maximum.
  - "The last 37 epochs cost 0.50 h clean": the table's own `time` values give
    8,234.91 − 6,353.56 = 1,881 s = **0.52 h**.
- **The judge's `best` row presumably follows the same mAP50-95 rule**, but its `results.csv` was not
  saved, so whether epoch 46 is also its mAP50 peak cannot be checked.
- **Two gate-table headrooms disagree with their own rows in the fourth decimal**: 3-class
  0.6566 − 0.4442 = 0.2124 against +0.2123, dog 0.2396 − 0.0260 = 0.2136 against +0.2135. Read as
  differences of unrounded `gate.csv` values; `gate.csv` itself is not on the Mac.
- **"14% low"** (F98) is 1.43 / 1.25 = 1.14 — the measurement is 14% above the projection; the
  projection is 12.6% below the measurement.
- **The gate decision block was annotated after the fact.** "Superseded in size, not in verdict, by
  step 3b" and the sequence table's 3b row were written by f19525e (2026-09-28), after step 3b ran;
  the block otherwise is 569840f (2026-09-27). The annotation is left out of the findings: F104 is
  stated as believed on 2026-09-27, and its later fate belongs in the ledger.
- **Twelve commits wrote this range** (`git blame` at the tag): 5a7d67c (2026-09-24) the sequence
  skeleton; 400f1f2 (09-26) step 0's recap; 866e357 (09-26) step 1; accdd8d (09-26) step 3's driver
  and step 2's command; e0c4a27 and 2b47d10 (09-27, morning) the crash diagnosis and the server's
  CPU/RAM figures; 569840f (09-27) the judge and the gate; dd266e6 (09-27) "decided: do it first";
  f19525e and c7977a3 (09-28), 707f31a (09-28) step 2b, 123410b (10-01) — the later ones writing
  rows and annotations for records 014–016.
- **M1.2 step 2b's `workers` finding is cited by legacy label only** (F97, `TASKS.md:1751-1764`). It
  becomes a no-run record under SPEC-MIGRATION-19, which mints its F-ID after this one; the
  two-ended link is left for SPEC-MIGRATION-19 and the census (SPEC-MIGRATION-20), as record 008 did.
- **The host-RAM peak was never measured.** The probe (`Get-Process python | Measure-Object
  WorkingSet64 -Sum`, at the end of epoch 1) needs the processes alive, and the judge had exited
  before anyone looked. F97's ≳1.3 GiB is a lower bound derived from a crash. The source deferred
  the measurement to step 2b or step 4's probe, whichever came first; step 2b's text does not report
  one.
- **The epoch-9 stall's cause is not recorded** and is unrecoverable.
- **"Locally" and the plumbing check's machine are not stated.** F95 was "hit locally while building
  step 3's driver"; F103's check ran "on CPU, on the real FLIR tree". The Mac holds no dataset, so
  the second is read as the server.
- **The crashed launch's exact commands are not recorded** — only "both detectors … on both cards at
  `--workers 16`".
- **The source's internal line references, resolved at the tag.** "lines 1576-1590" is record 004's
  relative-`project` bug and "lines 1600-1601" the custom judge's recipe; "line 1751" and "Line 1763"
  are M1.2 step 2b (no-run, SPEC-MIGRATION-19). "line 4001" (the custom judge's missing wall clock)
  is at 3960, and "line 3958" ("cheaper than the judge") at 3965 — both record 012's blocker 2.
- **"Invariant 7"** is `PLAN.md §5`'s Invariants item 7, "Three detector roles, never conflated."
- **Code anchors were read at accdd8d, not executed:** `engine/detector_stage.py:138-146` and `:162`,
  `cli.py:187`, `data/budget.py:51` and `:128-130`, `scripts/annotation_sweep.py:328-330`,
  `scripts/gate_table.py:60-61`. The ultralytics anchors in the four-facts table are the source's,
  against 8.4.117, not re-checked.
- **Cross-references, re-keyed:** the custom set's 0.1887 floor and +0.733 headroom are F02; the
  0.059 noise floor is F13; the 5.90 px residual and its bias are F93; dog's 13 instances are F94;
  the ~6–15 GPU-h estimate and the rule table are record 012's. The ~0.6–0.9 h pricing F100 is
  measured against appears nowhere else at the tag — 707f31a, which states it, only added lines —
  so its origin is not recorded.

## Next

Step 3b (record 014): de-roll the thermal labels with `calibration/flir.json`, then re-gate —
CPU-minutes against the cell's ~126 GPU-h. **Decided: do it first**, inverting the standing plan
(de-roll only if the cell comes back weak or null), which was written when the headroom was expected
to be large. Posted by this record:

- **It is the clean test of F109's confound**, which the mAP50-95 / mAP50 ratio only suggests.
- **The in-loop detector is ready for step 5:** `--in-loop-weights` and `--eval-init-weights` take
  `runs/inloop-flir-yolo11n/weights/best.pt`; `--reference-weights` stays the judge.
- **The host-RAM peak** is still owed to step 4's 1-epoch probe.
- **For the cell:** if its gain concentrates in `car` and `person` and leaves `bicycle` flat, F108
  predicted it.
