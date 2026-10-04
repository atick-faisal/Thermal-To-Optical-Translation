# 011 — E8: the annotation-budget sweep, direct thermal against translation at `N` ∈ {10 … 600}, with arms C and D topped up to six translator seeds

**Date:** 2026-09-24 · **Task:** M3 E8 · **Machine:** Windows server, 2× A100 40 GB (the sweep ran on `cuda:0`) · **Wall clock:** not recorded — the design estimated ~48 fine-tunes at ~10 min mean ≈ 8 GPU-hours; the C/D top-up took "minutes"
**git_sha:** not recorded — the driver landed in 225113e; results committed in 03335a9, the top-up in 95c18db · **W&B:** none — `scripts/annotation_sweep.py` has no W&B logging · **Log:** none saved — transcribed from TASKS.md 3633-3914 @ pre-spec-migration

## Command

As written in `TASKS.md`, in PowerShell since the server is native Windows. `$DATA` on the server
is `D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml` (step 0, below). The sweep,
all five arms at the default budgets and seeds {0, 1, 2}:

```powershell
uv run python scripts/annotation_sweep.py --data $DATA --out runs/e8 --device cuda:0
```

The C/D top-up, which resumes past the nine C/D rows already written and adds seeds 3–5 of the same
twelve exports:

```powershell
uv run python scripts/annotation_sweep.py --data $DATA --out runs/e8 --arms C D --seeds 0 1 2 3 4 5 --device cuda:0
```

## Configuration

**Why this and not another E3 cell.** Scoring the project against `RESEARCH_FINDINGS.md`
§10 after the turbo campaign closed: **causality, stability and faithfulness are all
satisfied** — twice over for causality (pix2pix n=6 p=0.031, turbo n=3 9/9). **Margin and
consistency are at zero.** Every campaign since M1.2 step 7 served a criterion that step 8
had already met. That is over-convergence, not divergence, and the fix is to spend the next
block on a criterion that is not yet met.

E8 is the cheapest of the two by a wide margin, and PLAN.md §11 already calls it *"likely the
headline, not the fallback"* while §15 predicts *direct thermal detection wins at full
annotation*. Until it runs, the paper does not know which claim it is defending.

**The budget lesson from M2a step 5 applies to the choice of instrument.** pix2pix is ~6 h per
4-stage run against turbo's ~96 h (line 1502 vs line 3105) — 16×. *[Corrected by E9 step 0: a
pix2pix run is 10.5 h measured, so the real ratio is 9×. The decision it justifies is unchanged.]*
**pix2pix is the exploration instrument; turbo is the confirmation instrument**, and turbo stays
frozen until a cheap instrument reports back. E8 is cheaper still: it is almost entirely
*detector* fine-tunes over exports that already exist, with no translator training at all.

**Question.** At what target-domain annotation budget does translation beat training a
detector directly on thermal?

**x-axis.** `N` annotated images, `N ∈ {0, 10, 25, 50, 100, 200, 400, 600}` (600 = the full
custom train split). Three seeds per point. These are **error bars on a curve, not a paired
significance test** — no p-value is claimed from the curve itself.

**Arms.** All four score the same 153-image / 423-instance val split against the same labels.

| arm | detector | trained on | annotation cost |
| --- | --- | --- | --- |
| **A. Direct thermal** (the §8 baseline) | yolo11n from COCO | `N` annotated **thermal** images | `N` |
| **B. Translated + adapted** | yolo11n from COCO | `N` annotated **translated** images (λ=0 export) | `N` |
| **C. Translated + zero-shot, λ=0** | the reference judge, visible-trained, never sees thermal | — | **0** |
| **D. Translated + zero-shot, λ>0** | same judge | — (`N` went to the *translator*) | `N` |

- **A vs B** is paired by construction: identical init, epochs, seed, budget and val protocol,
  differing only in the pixels. The exact sign-flip test applies per budget.
- **A's crossover with C** is the headline number. C is flat and needs **zero** target-domain
  annotations — M1's gate measured it at 0.7751 against the raw-thermal floor's 0.1887. The `N`
  at which A catches up *is* the claim.
- **D vs A at matched `N`** states C1's practical value: the same annotations spent on the
  translator's loop rather than on the detector (0.8696 at N=600).

**Annotation accounting, stated up front rather than found by a reviewer.** Arms A and B cost
**the same** annotations — training on translated frames still requires those scenes labelled.
Only arm C is genuinely annotation-free. Writing the accounting down is what makes the claim
defensible; a table that implies B is cheap would not survive review.

**Fixed-epoch caveat.** Epochs are held constant across budgets so A and B stay comparable. At
N=10 that overfits. The *comparison* is fair; the absolute numbers at small `N` are not optimal,
and the paper must say so rather than present them as tuned.

**This is not `DataConfig.annotation_fraction` inside E3.** PLAN.md §16 records why: that knob
gates only the batch's `cls`/`bboxes` in the coupling term, and the λ=0 control never reads
annotations at all — so lowering it makes the two arms *more* alike. E8 cuts the **detector's**
budget instead. The two cuts share one selector (below) precisely so the x-axis means one thing.

**Cost.** ~48 YOLO fine-tunes (2 trained arms × 8 budgets × 3 seeds) at ~10 min mean ≈ **8
GPU-hours**. Arms C and D are `t2o evaluate` calls on existing exports — minutes.

**The seam.** `data/dataset.py::_annotated_subset` promoted to public `annotated_subset`, and new
`data/budget.py::write_budget_manifest` building an **ultralytics** manifest (`train:` → a text
file of image paths) at a given `(fraction, seed)`, with `infrared=True` switching to the thermal
side of the same pairs via `Pairing`. One selector serves both the translator and detector cuts,
so a budget means the same thing on both. `train_detector` needed no change —
`detector_stage.py:111` hands `data_yaml` straight to ultralytics. Two guards:

- A budget manifest is **rejected by `DatasetManifest.load`** (it requires split *directories*).
  Deliberate: a detector-side budget must never reach the translator's data layer. Symlinked
  directories would have loaded on both sides and PLAN.md §3 rules symlinks out on native Windows
  anyway.
- `infrared=True` **fails loudly if that modality has no labels beside it**. Every adapted public
  dataset labels the visible side only (`data/adapters/common.py`), so arm A there would train on
  a split ultralytics reads as entirely unlabelled — it trains, it reports, and the mAP describes
  nothing. The custom pairs do carry both sides, which is exactly what makes the difference easy
  to miss.

**Step 0 (read-only, server).** Every artifact E8 needs is on disk; **nothing has to be retrained
or re-exported**, which is what makes the sweep an ~8 GPU-hour job rather than a campaign.

- **The paired dataset is not under the repo.** `$DATA` on the server is
  `D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml`, and `dataset/` at the repo
  root does not exist there at all. The driver must therefore take `--data` like every other
  command in this project and never assume `dataset/yolo_rgbt`.
- **Arm A is buildable.** 600/600 train and 153/153 val images-to-labels on **both** modalities,
  so the infrared guard passes on the real data, not only on the fixture. `write_budget_manifest`
  was run against `$DATA` at `fraction = 50/600` on both arms and selected N = 50 from
  `train\visible\images` and `train\infrared\images` respectively — the seam is confirmed end to
  end on the machine that will run it.
- **All twelve stage-3 exports survive**, `runs/e3b-{control,loop}-s{0..5}`, each 600 train / 153
  val with label counts matching image counts exactly, and each with `translator_last.pt` still
  beside it as a rebuild path that was not needed. Consequence for the design: arm B **pairs export
  seed `s` with detector seed `s`** for s ∈ {0,1,2}, so its error bars carry translator variance
  instead of treating one export as though it were the translator. Pinning a single export would
  have made the bars understate spread, and the writeup would have had to say so.
- `runs/reference-yolo11s/weights/best.pt` present (arms C and D's judge). `yolo11n.pt` is **not**
  at the repo root — ultralytics auto-downloads it, so the run needs network on first touch.
  1016 GB free on `D:`.
- **Label resolution verified** rather than assumed: ultralytics 8.4.117's `img2label_paths`
  substitutes `os.sep + "images" + os.sep` and takes the *last* occurrence, so the Windows `\`
  separator is handled and both the source layout (`{split}/{visible,infrared}/{images,labels}`)
  and the export layout (`{split}/{images,labels}`) resolve with zero missing labels.

**The sweep driver**, `scripts/annotation_sweep.py`, following the `scripts/loss_share.py`
precedent. `t2o evaluate` and `t2o train-detector` only log, so the driver calls `train_detector` /
`evaluate_detector` in-process and appends a tidy CSV row per cell. Three decisions in it that the
design above did not settle:

- **A fifth arm, `A0`** — the reference judge on raw thermal, the zero-annotation floor the
  headline is quoted against (M1 measured 0.1887). Arms A and B fine-tune from COCO `yolo11n`,
  which has no class correspondence with this vocabulary and so cannot be scored at `N=0` at all;
  the judge can. The `detector` column records `finetuned` vs `reference` so that discontinuity is
  visible in the CSV rather than hidden in a curve. `A0` has no seed dependence — one judge, one
  split — so it is written once at `seed = -1` instead of duplicated across seeds to look
  symmetric.
- **One seed does two jobs**: it picks *which* `N` images are annotated and it seeds the
  detector's training. Splitting them would report error bars that exclude the luck of the draw,
  and at `N=10` that luck is most of the variance.
- **`--control-template` / `--loop-template`** carry a `{seed}` placeholder rather than taking a
  list of export directories, so the detector seed is paired with the *export* seed structurally.
  A list of six paths against three seeds would misalign silently and produce error bars that
  look like translator variance and are not.

Cells run cheapest-first (zero-shot anchors, then ascending `N`) and each row is flushed as it
completes, so an interrupted sweep keeps its GPU time and a re-run resumes. Smoked end to end
through real ultralytics on CPU before shipping: it trains on exactly `N` images, validates
against the **full** untouched val split, and resolves labels on the thermal side — the failure
this most needed ruling out, since an unlabelled split trains and reports a number that means
nothing.

**Resolved defaults**, read from the driver at the tag since the commands pass none of them: 50
epochs, `imgsz` 640 and batch 16 (E3's own `detector.evaluation` settings), 16 workers;
`yolo11n.pt` init for A and B; the judge at `runs/reference-yolo11s/weights/best.pt` for A0, C
and D; C reads `runs/e3b-control-s{seed}/stage3/translated` and D
`runs/e3b-loop-s{seed}/stage3/translated` — the stage-3 exports of record 007's campaign at
`grad_scale: 0.15`. B fine-tunes on the same λ=0 exports as C.

## Results

As recorded in 03335a9 and 95c18db. The source carries no console block, only the tables below.
The per-seed rows are tracked at **`docs/results/e8-tidy.csv`**, 55 rows: all 49 cells of the
sweep in one pass, no failed cell, plus the top-up's six. The server copy lives under gitignored
`runs/e8/`; the tracked copy drops the driver's `weights` and `data` columns, which hold
server-local paths under `runs/` and point at nothing here.

**The curve.** mAP50 on the fixed 153-image / 423-instance val split, mean ± sd over seeds
{0,1,2}. A and B are paired per seed: same budget, same scenes, same COCO init, same 50 epochs,
differing only in the pixels.

| `N` | **A** direct thermal | **B** translated, fine-tuned | A − B | seeds agreeing |
| --- | --- | --- | --- | --- |
| 10 | 0.1090 ± 0.0190 | 0.1412 ± 0.0105 | −0.032 | 3/3 for **B** |
| 25 | 0.3564 ± 0.0809 | 0.3556 ± 0.0248 | +0.001 | 1/3 — a tie |
| 50 | 0.6210 ± 0.0824 | 0.5621 ± 0.0394 | +0.059 | 3/3 for **A** |
| 100 | 0.7511 ± 0.0125 | 0.6920 ± 0.0383 | +0.059 | 3/3 for **A** |
| 200 | 0.8447 ± 0.0053 | 0.8046 ± 0.0154 | +0.040 | 3/3 for **A** |
| 400 | 0.9025 ± 0.0070 | 0.8641 ± 0.0160 | +0.038 | 3/3 for **A** |
| 600 | 0.9300 ± 0.0096 | 0.8978 ± 0.0047 | +0.032 | 3/3 for **A** |

**Flat anchors**, C and D at **n=6** after the top-up: **A0** (judge on raw thermal, 0
annotations) **0.1552**, a single cell; **C** (judge on the λ=0 export, 0 annotations)
**0.7975 ± 0.0330**; **D** (judge on the λ>0 export, `N`=600 spent inside the *translator*)
**0.8487 ± 0.0140**.

**The crossover.** A passes **C at `N ≈ 150`** and **D at `N ≈ 214`**, by linear interpolation
inside the bracketing budgets (100–200 and 200–400). **Quote the interval, not the point**: the
sweep measures at neither `N`, and the dominant uncertainty is the flat line's spread across
translator seeds, not the interpolation. Reading C ± 1 sd back through A's curve gives
**`N` ∈ [114, 185]** (± 1 sem: [135, 164]); for D, `N` ∈ [189, 262].

**The C/D top-up**, twelve C/D rows, no duplicate `(arm, seed)` — the resume skipped the nine
already written and added six in minutes:

| | n=3 | **n=6** |
| --- | --- | --- |
| C mAP50 | 0.8049 ± 0.0438 (sem 0.0253) | **0.7975 ± 0.0330 (sem 0.0135)** |
| D mAP50 | 0.8572 ± 0.0110 (sem 0.0064) | **0.8487 ± 0.0140 (sem 0.0057)** |
| A × C crossover | `N` ≈ 157, ±1 sd [111, 214] | **`N` ≈ 150, ±1 sd [114, 185]** |

At n=6 the paired D − C difference is **+0.051218 → +0.0512, 6/6 seeds, p = 0.031**.

No report command exists. Every number above recomputes from `docs/results/e8-tidy.csv` (see
Provenance caveats).

## Findings

### F81 — Annotation-free translation is worth ≈150 annotated thermal images, interval [114, 185], a quarter of the 600-image train split; spending the same annotations inside the translator's loop (arm D) raises that to ≈214 (legacy: M3 E8, finding 1)

The comparison is arm A, `yolo11n` fine-tuned on `N` annotated thermal images, against the flat
zero-annotation line of arm C, the reference judge on the λ=0 export (0.7975 ± 0.0330, n = 6). A
passes C between `N` = 100 (0.7511) and `N` = 200 (0.8447), and D (0.8487) between 200 and 400
(0.9025). So C1's coupling is itself worth another ~64 annotations. The interval, not the point,
is what to quote: the sweep measures at neither `N`.

### F82 — `PLAN.md` §15's prediction is confirmed: direct thermal wins at full annotation, and criterion 1 is measured, not met — the margin over the §8 baseline is +0.642 at `N`=0 and −0.133 at `N`=600 (legacy: M3 E8, finding 2)

A reaches 0.9300 at `N`=600 against C's 0.7975 and D's 0.8487. Any claim must name the regime, and
the honest sentence is *"below ~150 target-domain annotations, translation beats direct thermal
detection"* — not *"translation beats direct thermal detection"*.

### F83 — Arm B does not work: training on translated λ=0 frames is worse than training on thermal at every budget ≥50, 3/3 seeds each, by +0.032 to +0.059 mAP50 — "translate, then fine-tune" is off the table (legacy: M3 E8, finding 3)

The finding that most changes the paper. A and B are paired per seed — same budget, scenes, init
and epochs, differing only in the pixels. Translation's value is **entirely** in annotation-free
transfer (arm C); as a preprocessing step in front of a supervised detector it costs accuracy.
B's win at `N`=10 is not evidence against this — F84, both arms are broken there.

### F84 — `N`=10 is degenerate and must never be read as "what 10 images buy": both trained arms land below the zero-annotation raw-thermal floor, 0.109 and 0.141 against A0's 0.155, at precision ≈0.01 with recall ≈0.45 (legacy: M3 E8, finding 4)

The detector is firing on everything. The fixed-epoch caveat pre-registered in the design bit
exactly where it was predicted to. The A-crosses-A0 point at `N ≈ 13` sits inside this zone and is
not quotable.

### F85 — The error bars behave, and no p-value is claimed: A's spread collapses from ±0.082 at `N`=50 to ±0.010 at `N`=600, and at n = 3 the exact sign-flip test floors at p = 0.25, so the reportable statistic is sign consistency, 3/3 at every budget ≥50 (legacy: M3 E8, finding 5)

The collapse is why the crossover bracket is wide while the endpoint is tight. Sign consistency
rather than significance is exactly what the design pre-registered for the curve.

### F86 — A free validity check, passed: A0's 0.1552 reproduces the gate re-check's clean-judge thermal floor to four decimals, through a different code path and months later, and C (0.7975) and D (0.8487) land on that table's single-seed λ=0 (0.7851) and stage-3 (0.8470) rows (legacy: M3 E8, finding 6)

The table is record 004's `yolo11s` re-score of M1 (F10). Nothing drifted between campaigns.

### F87 — C vs D reproduces E3's headline exactly, which makes it a reproduction, not evidence: at n = 6 the paired D − C difference is +0.0512, 6/6 seeds, p = 0.031, identical to the recorded stage-3 endpoint to every digit that entry carries (legacy: M3 E8, finding 7)

It is the same computation: the same judge (`runs/reference-yolo11s/weights/best.pt`, set in
`experiments/e3_pix2pix_*.yaml`), the same six exports, reached through the sweep driver instead
of the campaign's own metric path. Both halves matter. It is **worth recording**: the stored
exports still score exactly as the campaign recorded months ago, so nothing in the export or
evaluation path has drifted and the artifacts every downstream arm reuses are sound. And it is
**not to be counted twice**: E8 contributes nothing new to the causality criterion, and a paper that
reported +0.0512 from E3 (F35) and again from E8 would be reporting one result as two.

### F88 — The crossover's width is arm C's spread across translator seeds, not arm A's resolution: topping C up from three seeds to six nearly halved its standard error (0.0253 → 0.0135) and narrowed the interval from a span of 103 images to 71, while the point barely moved (157 → 150) (legacy: M3 E8, tighten the crossover)

The obvious move was to infill A's budgets (`--arms A --budgets 125 150 175`, ~1 GPU-hour), and it
was the wrong one. A is already tight where it matters (±0.0125 at `N`=100, ±0.0053 at `N`=200);
C was the moving target at ±0.0438 across three translator seeds, and reading C ± 1 sd (0.761 to
0.849) back through A's curve put the crossover anywhere from `N` ≈ 111 to `N` ≈ 214. Extra A
budgets would have bought a *falsely precise* crossover against a line not yet pinned down. Seeds
3–5 of the same twelve exports were already on disk, so the fix cost minutes, not GPU-hours.
Infilling arm A's budgets is **not** worth doing: the remaining width is still C's spread, not A's
resolution.

## Provenance caveats

- **Arm A trains on registration-transferred labels, which the source does not say.** Per
  `docs/goal.md` Constraints, the custom set's boxes were drawn on the visible images and moved to
  thermal with the sister project's registration. So arm A is registration-based label transfer,
  not a registration-free direct-thermal baseline. Separately, the reference judge for A0, C and D
  was trained on the visible train split (record 004), the visible halves of the same 600 pairs, so
  E8 cannot separate "visible labels from these pairs" from "visible labels from anywhere". Both
  are disclosures; no finding's claim is changed.
- **The design's "0.8696 at N=600" for arm D is M1's single loop run under the in-loop judge**
  (record 003, F06), which F04 shows self-grades. The sweep measured D under the reference judge:
  0.8487 at n = 6.
- **The source's internal line references, resolved at the tag.** "~6 h per 4-stage run … (line
  1502)" is at line 1564, the same stale reference record 010 notes. "line 3105" opens record 010's
  throughput paragraph; the ~96 h row is line 3113 (F62). "line 1636" is the raw-thermal row of
  record 004's judge table (F10). "line 1535" is the step-8 row of M1.2's status table and "line
  2427" opens record 007's paired table, whose stage-3 cell is line 2431 (F35).
- **Other cross-references, re-keyed.** The `annotation_fraction` argument "PLAN.md §16 records" is
  F22 (record 005). "M1's gate measured it at 0.7751" is F03. "The budget lesson from M2a step 5"
  is F62. "E9 step 0"'s 10.5 h is record 012's. "(pix2pix n=6 p=0.031, turbo n=3 9/9)" are F35 and
  F72.
- **Two later splices sit inside the range.** Lines 3647–3649, the bracketed E9 step 0 correction,
  were written by 400f1f2 (2026-09-26) and are kept verbatim above. Lines 3909–3913, the 2026-10-01
  addendum to the deferred criterion-1 row, were written by 123410b. That row (3899–3913) is not
  part of this record; it goes to `docs/roadmap.md` under E8 (Q13). Every other line is from
  8d30e45, b9ea7c3, 225113e, 03335a9, b2a0238 and 95c18db (2026-09-23 to 2026-09-24).
- **The deferred row's margins disagree with F82's.** Row 3899 quotes +0.650 at `N`=0 and −0.125 at
  `N`=600, against F82's +0.642 and −0.133. The row was written in b2a0238, before the top-up
  (95c18db), from n = 3's C of 0.8049: 0.8049 − 0.1552 = 0.6497 and 0.9300 − 0.8049 = 0.1251. Both
  lifted as written; F82's are the n = 6 values.
- **F84's "recall ≈0.45" is the source's rounding.** Per seed at `N`=10 the tidy CSV has precision
  0.006–0.019 and recall 0.405–0.586; B seed 2's 0.586 is the outlier.
- **Re-checked against `docs/results/e8-tidy.csv` for this record, not re-run.** Every mean ± sd in
  the curve and anchor tables, both sem columns, every A − B mean and seed count, D − C's +0.051218
  (per seed +.0420 / +.0034 / +.1116 / +.0793 / +.0417 / +.0294), and every crossover and interval
  endpoint (including `N ≈ 13`) recompute from the 55 rows to the digits shown.
- **Code anchors were read at the tag, not executed.** The resolved defaults are
  `annotation_sweep.py:73-76` and `:241-281`. `detector_stage.py:111` is `train_detector`'s
  signature, at 8d30e45 and at the tag (`src/t2o/engine/`). The ultralytics `img2label_paths`
  behaviour is the source's reading of 8.4.117.
- **Date is 03335a9's commit date.** The source names no run date. The driver landed in 225113e
  (2026-09-23 19:42), so the 49-cell pass ran between then and 03335a9 (2026-09-24 15:45); the
  top-up was committed in 95c18db at 16:09 the same day.

## Next

The source settles its own follow-up: infilling arm A's budgets is not worth doing (F88). It
leaves one decision open, the deferred edit to `RESEARCH_FINDINGS.md` §10 criterion 1, which waits
on a second dataset. That row lives in `docs/roadmap.md` under E8, and the public-dataset cell it
waits on is E9 (records 012–016).
