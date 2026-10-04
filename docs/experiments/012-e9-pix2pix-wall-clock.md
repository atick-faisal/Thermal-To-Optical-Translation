# 012 — E9 pricing and step 0: the FLIR cell's three blockers costed, and pix2pix's wall clock measured from checkpoint mtimes

**Date:** 2026-09-26 · **Task:** M3 E9 · **Machine:** Windows server — a read of `runs/` checkpoint mtimes, no GPU · **Wall clock:** not recorded
**git_sha:** not recorded — pricing committed in 5a7d67c, step 0's command fixed in ef32bb4, its result in 400f1f2, the per-split class counts in accdd8d · **W&B:** none — the local mtime fallback was used; the `e3-pix2pix-g015` group's Runtime column was not read · **Log:** none saved — transcribed from TASKS.md 3915-4146 @ pre-spec-migration

## Command

Step 0, as written in `TASKS.md` (ef32bb4's corrected form), in PowerShell from the repo root on
the server:

```powershell
$rows = Get-ChildItem runs -Recurse -Filter translator_last.pt | ForEach-Object {
  [pscustomobject]@{ Run = $_.Directory.Parent.Name; Stage = $_.Directory.Name; Done = $_.LastWriteTime }
}
$out = foreach ($g in ($rows | Sort-Object Run, Stage | Group-Object Run)) {
  $prev = $null
  foreach ($r in $g.Group) {
    $h = ''
    if ($prev) { $h = [math]::Round(($r.Done - $prev).TotalHours, 2) }
    [pscustomobject]@{ Run = $r.Run; Stage = $r.Stage; Done = $r.Done; Hours = $h }
    $prev = $r.Done
  }
}
$out | Format-Table -AutoSize
$h = @($out | Where-Object { $_.Hours -ne '' } | Select-Object -ExpandProperty Hours | Sort-Object)
"intervals=$($h.Count)  median=$($h[[int]($h.Count/2)]) h  min=$($h[0]) h  max=$($h[-1]) h"
```

The per-boundary split, the cross-run `stage0` recovery and the turbo re-derivation have no
command in the source; nor does the per-split class count.

## Configuration

**Why this ran before the cell was planned.** M0.9's dataset audit left three blockers against the
FLIR-aligned cell. Pricing them **changed the shape of the cell**:

- **Blocker 1 costs nothing** — the measurement already exists in a sibling repo, and it is far
  better characterised than our note said.
- **Blocker 2 is not a blocker.** It is the entry fee for answering blocker 3 — and pricing it
  surfaced a hard gap nobody had on a list: **FLIR carries no thermal labels at all**, so blocker
  3's measurement is *impossible today* (F93).
- **Blocker 3 is a minutes-long kill-test that gates everything else**, so it runs first.
- And the cell's own cost rests on a throughput figure this repo **never measured** and was once
  wrong about by 16× — step 0.

**Blocker 1 — cross-modal misalignment. 0 GPU-h, already measured** in
`../Thermal-Image-Registration`. The sister repo's numbers, cited, not re-measured:

| fact | value | source |
| --- | --- | --- |
| FLIR-aligned residual, three matchers | roma **5.90**, eloftr 5.65, splg 5.69 px `epe_mean` | sibling `TASKS.md:232-234` (P1-1a) |
| the pipeline's own noise floor, under a full 30° warp | **~0.2 px**, `sr_3px` 1.00 in all eight cells | sibling `TASKS.md:316` (P1-1b, sweep B) |
| structure of FLIR's residual | a **fixed −1.18° camera roll**; p10 10.23 / p50 10.66 / p90 11.56 px, 48 of 50 pairs above 10 px | sibling `TASKS.md:331`, `:355` (P1-1b, sweep A) |
| checked-in correction | `calibration/flir.json` — a corner field, n = 1,013 val pairs, element-wise median of three matchers, spread 1.23 px, magnitude 9.32 px | sibling repo, `calibration/` |
| composed, end to end on flir val | `epe_median` **5.48 → 2.25 px**, `sr_3px` **0.00 → 0.30** | sibling `TASKS.md:795-806` |
| irreducible afterwards | ~4–5 px of per-pair scatter no calibration removes | sibling P1-1c/d |

Three matchers spanning three decades of compute cost agreeing, against a 0.2 px pipeline floor, is
what makes this a property of **the data** rather than of a matcher. Why it bites here: pix2pix
trains against the paired visible frame with `l2: 1.0` + `lpips: 5.0`
(`experiments/e3_pix2pix_control.yaml`). A *systematic* roll rotates the supervision target
relative to the input, and the generator's only way to satisfy that is to blur. The custom pairs
are registered, so **this confound exists on the public cell and nowhere else** — exactly the
thing that would leave a weak FLIR result uninterpretable.

**Decision: pre-register the residual as a stated caveat; de-roll only if the cell survives the
kill-test and then comes back weak or null.** Nothing is paid to correct a cell that may not run.
`calibration/flir.json` is the *accepted* constant — the sibling repo keeps `calibration/rejected/`
for ones that failed, so the distinction is real and worth re-checking before relying on it.

**Blocker 2 — no detector weights. ~1 unattended GPU-day, plus one ~15-line seam.**

1. `runs/reference-flir-yolo11s`, the judge: `t2o train-detector --init-weights yolo11s.pt
   --epochs 100 --seed 1` on FLIR visible (4,129 train / 1,013 val). The custom equivalent (600
   train images, mAP50 0.9364, F11) has no recorded wall clock; scaling E8's yolo11n cells gives
   **~6–15 GPU-h, an estimate, not a measurement**, to be timed when it runs.
2. `runs/inloop-flir-yolo11n`, the in-loop detector: same data, `yolo11n.pt`, seed 0. Needed
   because COCO `yolo11n.pt` in-loop trips `detection/frozen.py:84-95`'s nc-mismatch warning (4
   FLIR classes against 80 COCO) — not fatal, semantically wrong. Cheaper than the judge.
3. Mirror `visible/labels` → `infrared/labels` by stem (F93).

**Blocker 3 — does the premise transfer? Minutes, and it gates everything.** Reproduce M1's gate
table on FLIR: two `t2o evaluate` calls against the new judge, one on visible val (the ceiling),
one on the mirrored thermal val (the floor). Custom headroom was **0.9213 − 0.1887 = +0.733**
(F02), and the per-class row says why — Fuse 0.0377, Switch 0.0047, i.e. thermally invisible
components. FLIR's classes are bicycle / car / dog / person; night pedestrians and warm engine
blocks are highly legible in thermal. **Expect a much higher floor.** That is the whole reason
this is a kill-test and not a formality. The decision rule, recorded before the number exists:

| FLIR headroom (ceiling − floor) | reading | action |
| --- | --- | --- |
| **≥ 0.40** | the premise transfers | run the cell at 600 matched pairs |
| **0.15 – 0.40** | transfers weakly | still run it — "a smaller gain where thermal is legible" is honest criterion-2 evidence, and arguably a better paper than a second large win |
| **< 0.15** | translation has nothing to buy here | **do not spend the ~126 GPU-h** (step 0's measured figure, up from the ~72 first estimated)**.** The floor measurement *is* the finding, and it is direct evidence for the hypothesis behind the deferred §10 criterion-1 decision |

**Also pre-registered: 3-class primary** (bicycle / car / person), with dog stated separately
(F94).

**Decision: run the cell on a seeded 600-pair matched subset.** FLIR-aligned is 4,129 train pairs
against the custom set's 600 — 6.9× the translator's epoch cost. The cell then costs what the
custom one did, and corpus size is *held fixed* instead of becoming a second variable a reviewer
can attribute the result to — criterion 2 is about the method being consistent, not about dataset
scale. This needs a small new seam, `DataConfig.max_train_images` plus its own seed, because
`annotation_fraction` gates **annotations, not image count**: `config/schema.py:62` and
`data/dataset.py:196` (at 5a7d67c) show `visible_paths` is every image in the directory and
`annotation_fraction` only decides which of them count as labelled.

**Step 0's method.** `translator_last.pt` is rewritten every epoch (`engine/trainer.py:308`), so
its mtime is when that stage's *training* ended. Consecutive mtimes therefore bracket [stage *N*'s
boundary work + stage *N*+1's 100 epochs] — the same unit record 010 calls "one stage" (24.1 h =
~20 h training + ~4 h boundary), so the two are directly comparable. `stage0` has no predecessor
and shows blank. **Read the median, not the mean.** Two shells ran concurrently by seed (0–2 on
`cuda:0`, 3–5 on `cuda:1`), so these intervals include cross-card CPU and dataloader contention —
which is the realistic figure, since the FLIR cell would run the same way. But an interval that is
wildly large is idle wall clock or an interruption, not compute, and the mtime method cannot tell
the difference.

## Results

As recorded in 400f1f2 and accdd8d. The source carries no console output; these are its distilled
tables, verbatim.

**24 completed pix2pix runs on disk — `e3-*` (the pre-`g015` launch) and `e3b-*` (the campaign) —
72 stage intervals.** The two launches agree to within noise, so they are pooled.

| interval | n | median | min | max |
| --- | --- | --- | --- | --- |
| `stage0` → `stage1` | 24 | **2.40 h** | 2.19 | 2.58 |
| `stage1` → `stage2` | 24 | **2.83 h** | 2.66 | 3.06 |
| `stage2` → `stage3` | 24 | **3.37 h** | 3.16 | 3.62 |
| all intervals pooled | 72 | 2.83 h | 2.19 | 3.62 |

`stage0`, recovered from the gap between one run's `stage3` and the next run's `stage0` on the same
card ([previous run's final eval + this run's `stage0`]): **n = 10, median 1.79 h, range 1.66–1.85
h**.

| unit | measured |
| --- | --- |
| one complete 4-stage **control** run | **10.18 h** |
| one complete 4-stage **loop** run | **10.75 h** |
| the coupling's own cost | **+0.57 h/run, +6.8%** |
| the twelve-run cell | **~126 GPU-h ≈ 63 h wall clock on two cards ≈ 2.6 days** |

Per-card wall clock predicted from the parts: **62.8 h**. Observed `e3b-*` campaign span from
estimated launch to last checkpoint: **63.0 h**.

Turbo, re-derived from the same disk with the six 130–502 h interruption intervals discarded:
median **23.46 h** per stage.

FLIR instance counts per split (they reconcile exactly with the totals car 24,732, person 13,094,
bicycle 2,926, dog 108):

| split | bicycle | car | dog | person |
| --- | --- | --- | --- | --- |
| train | 2,566 | 20,608 | **95** | 8,987 |
| val | 360 | 4,124 | **13** | 4,107 |

Re-render on the server by re-running the command above; the `stage0` recovery and the turbo
filter are applied by hand.

## Findings

### F89 — A pix2pix stage is not a constant: interval medians are 2.40 / 2.83 / 3.37 h, about +20% a stage, with `stage0` recovered at 1.79 h (legacy: M3 E9 step 0, "a stage is not a constant")

The spread within a boundary is ±8% (n = 24 each), which is what a clean unattended campaign looks
like — so these are compute, not idle wall clock. Why it grows is not established here; the
natural explanation is that the adapted detector's fine-tune set grows with each stage's export,
and it is worth *not* assuming a flat per-stage cost when probing FLIR in step 4.

### F90 — A complete 4-stage pix2pix run costs 10.18 h control and 10.75 h loop, so the twelve-run cell is ~126 GPU-h; the accounting closes to 0.2 h over 63 (legacy: M3 E9 step 0, the measured unit costs)

The coupling's own cost is +0.57 h/run, +6.8%. Predicting per-card wall clock from the parts gives
62.8 h; the observed `e3b-*` campaign span is 63.0 h. A 0.2 h agreement over 63 h means nothing
material is missing from the accounting.

### F91 — "~6 h per 4-stage run" was low by 1.7×, and "~72 GPU-h" by 1.75×; at full corpus the FLIR cell is ~870 GPU-h ≈ 18 days on two cards, not ~496 h / 10 days (legacy: M3 E9 step 0, the estimate corrected)

Against record 004's design figure (`TASKS.md:1564`), asserted and never measured. Nothing already
spent changes — the campaigns ran and finished. What changes is every *forward* estimate that was
a multiple of that figure, starting with this cell. The 600-matched decision holds and gets more
attractive: the matched cell is ~126 GPU-h ≈ 2.6 days, still cheap enough that the kill-test
remains the only real gate.

### F92 — pix2pix:turbo is 9.0× per complete run (10.5 h against ~94 h), not 16×; the same method re-derives turbo at 23.46 h a stage against the 24.1 h recorded (legacy: M3 E9 step 0, the instrument ratio)

The 16× (F62; record 011's choice of instrument) divided turbo's *measured* run by pix2pix's
*estimated* one. The mtime method reproduces turbo's figure, a number it was not fitted to, to
within 3% — the real check on it. The standing decision does not move — turbo stays frozen,
pix2pix stays the exploration instrument — but the honest figure is 9×.

### F93 — FLIR carries no thermal labels on disk, so the raw-thermal floor cannot be measured until they are mirrored, and mirrored labels bias that floor down (legacy: M3 E9 pricing, blocker 2 item 3)

`dataset/processed/flir/{train,val}/infrared/labels` **does not exist** — not empty, absent.
`data/adapters/common.py:129` and `:144` (`write_label` / `write_label_lines`) both hardcode
`split_root / "visible" / LABELS_SEGMENT`, and the pairing layer resolves a label from the
*visible* path, so one copy of the labels is by design sufficient for everything built so far. It
is not sufficient for a raw-thermal evaluation. Fix: mirror by stem — precisely what
`tests/test_budget.py:23-39`'s `thermal_labelled` fixture already does in-memory. **The direction
of the bias, pre-registered:** mirrored labels inherit the 5.90 px residual. At mAP50 (IoU 0.5) a
~6 px offset on a typical FLIR car or person box is a small IoU penalty — it pushes the thermal
floor **down**, which *flatters* translation.

### F94 — FLIR's dog class has 13 val instances against 4,124 cars, and a 0.4 AP swing on it moves the 4-class mAP50 by 0.1 — the width of the kill-test's decision band (legacy: M3 E9 pricing, class imbalance)

A different case from the custom set's: `Connector` has zero instances and costs nothing (F09),
because `metrics/task.py::_extract_per_class_ap` omits zero-instance classes and ultralytics
averages only over classes present. **Dog is not empty**, so none of that protection applies: it
is averaged in at full weight, 25% of a 4-class mean, on far too few instances for a stable AP.
Reported 4-class, the kill-test could be flipped by a single dog detection — which is why
`scripts/gate_table.py` computes the 3-class primary mean itself instead of leaving it as
arithmetic on a terminal log.

## Provenance caveats

- **"Ran locally" is read as the mtime fallback on the server.** The source says "Ran locally (W&B
  was not needed)"; `runs/` lives only on the server, and the Mac holds no run outputs.
- **Four commits wrote this range** (`git blame` at the tag): 5a7d67c (2026-09-24) the pricing;
  ef32bb4 (2026-09-26 21:43) step 0's command and how-to-read, after the original "matched nothing
  on the server"; 400f1f2 (21:53) the result, plus a splice of "~126 GPU-h" into the rule table's
  `< 0.15` row and the "1.75× low" lead-in; accdd8d (23:13) the per-split counts. The rule table's
  thresholds are 5a7d67c's; its "~126" is the later splice.
- **The control/loop split is not tabulated.** The pooled medians (1.79 + 2.40 + 2.83 + 3.37) sum
  to 10.39 h, between 10.18 and 10.75; the per-arm intervals behind those two figures are not
  recorded.
- **`stage0`'s n = 10 is inferred from on-card ordering**, not read directly; each value carries
  the previous run's final eval. How the "estimated launch" behind the 63.0 h span was estimated is
  not stated.
- **The turbo re-derivation discards six intervals by judgment** (130–502 h, the first turbo
  attempt's idle gaps).
- **Rows to ignore in that output**, recorded so the next reader does not chase them: `Run = runs`
  (the `dataloader-check*` and `vram-probe` directories sit directly under `runs/`, so the walk
  reads `runs` as their parent — this is also where the nonsensical −0.42 h and 268 h intervals
  come from); `e3-probe-g015`, `turbo-probe-g015`, `turbo-probe-g075`, `vram-probe` and
  `pix2pix-loop-test`, all reduced-epoch probes; and `e3t-*`, whose 130–502 h intervals are the
  first turbo attempt's idle gaps.
- **Blocker 1's numbers are the sister repo's**, cited from its `TASKS.md`, not re-checked here.
- **Code anchors were valid at 5a7d67c / ef32bb4; some moved by the tag.** `config/schema.py:62`
  and `data/dataset.py:196` shift once `max_train_images` lands; `engine/trainer.py:308` is
  `torch.save(state, self.run_dir / LAST_CHECKPOINT)` at ef32bb4. Read, not executed.
- **The source's internal line references, resolved at the tag.** "line 2364" and "lines 2364,
  2381" are the `--group e3-pix2pix-g015` launch lines 2368, 2385 and 2392 (record 007). "lines
  3100-3110", "3112-3115", "line 3105" and "line 3114" are record 010's throughput paragraph,
  3104–3118: 24.1 h at 3112, ~96 h at 3113, "wrong by 16×" at 3118 (F62). "line 3639" is the 16×
  in record 011's configuration, at 3647. "lines 1459-1463" and "line 1564" resolve as written.
- **Cross-references, re-keyed:** "step 1 above" (0.9364) is F11; M1's gate-table headroom is F02;
  the zero-instance protection is F09.
- **The judge's "~6–15 GPU-h" and blocker 2's "~1 GPU-day" are estimates**, so flagged in the
  source.

## Next

The kill-test, E9 steps 1–3 (record 013): train both FLIR detectors, mirror the labels, score
ceiling and floor. Posted before it runs:

- **The floor will be much higher than the custom set's 0.1887**, because FLIR's classes are
  thermally legible.
- **Headroom decides the cell** by the rule table above: ≥ 0.40 run, 0.15–0.40 run, < 0.15 stop.
- **The mirrored-label floor is biased down** (F93), so a weak headroom is, if anything, overstated.
- **3-class is the primary mean** (F94).
