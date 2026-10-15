---
slug: msrs-budget-sweep
title: msrs-day annotation budget sweep
status: shipped
created: 2026-10-15
---

# msrs-day annotation budget sweep

## Summary

On msrs-day the translation loop scores 0.3592 mAP50 on the two passive classes (car_stop,
color_cone). That is well above its no-loop control and the raw-thermal floor, but 0.2036 below a
detector trained on all 536 labelled thermal images (record 024, F168–F169). The passive-class
claim is now closed as negative (`docs/roadmap.md`, Drafting criteria). What is still open is
goal C2's annotation-budget axis: **how many labelled thermal images does a thermal-trained
detector need to match what the loop gives for free?** The custom set answered this in record 011
(E8): annotation-free translation was worth ≈150 thermal images, and ≈214 with the loop (F81).
This feature re-asks the same question on msrs-day, so C2's map has a second dataset on that
axis, including the passive classes, whatever the answer turns out to be.

## Tech Stack / Approach

**Reuse the E8 driver, `scripts/annotation_sweep.py`, mostly through its options.** It already
does everything the sweep needs:

- one tidy CSV, appended cell by cell, so an interrupted sweep can resume;
- the cheapest cells first;
- one seed that both picks the N images and seeds training (`:26-29`);
- a startup check on budgets and exports (`:291-307`).

msrs-day needs only options:

- `--data dataset/processed/msrs-day/data.yaml`;
- `--control-template runs/e3md-control-s{seed}/stage3/translated`;
- `--loop-template runs/e3md-loop-s{seed}/stage3/translated`;
- `--reference-weights runs/reference-msrs-day-yolo11s-s1/weights/best.pt`;
- `--budgets` set explicitly, because the default 600 is refused on a 536-image split (`:291-293`).

**The one code change is a two-class score.** As it stands, `Row` (`:79-94`) holds only the
all-class `map50`, and `_train_cell` takes that number from `train_detector`'s `DetectorResult`
(`src/t2o/engine/detector_stage.py:79-87`), which reports no per-class AP. Without a two-class
score the sweep cannot be read against the loop's 0.3592. The fix copies the pattern
`scripts/gate_table.py` already uses (`:95`, `:129-138`):

- **New option:** `--primary-classes`.
- **New `Row` fields:** `primary_map50` and `per_class_ap50`.
- **Trained cells are re-scored** with `evaluate_detector` (`src/t2o/metrics/task.py:114-160`)
  on the cell's `best.pt`. This is the same single evaluation path that scored the thermal
  detector's 0.5628 (invariant 1, `docs/design.md` §5). It adds one validation pass per cell,
  about a minute.
- **The mean comes from `primary_mean`** (`task.py:57`). The gate table, the campaign aggregator
  and this sweep then compute the headline one way.

**Training-set instance counts are logged per cell.** Each budget manifest's label files are
counted per class and written to the row. A small draw of contiguous video frames can miss a
passive class entirely, and a reader of the CSV has to be able to see that.

**Tests:** extend `tests/test_annotation_sweep.py`. It runs on synthetic pairs from
`tmp_path_factory`, on CPU.

**The server run is not a task.** It happens after the code tasks and comes back as a log for
`/log-experiment`. Two cautions for that run:

- **Never run two sweeps at once.** Every budget cell rewrites the shared
  `train/infrared/labels.cache` (ultralytics `data/dataset.py:282`), so the server runs one sweep
  on one card.
- **Check the inputs exist before launch.** The twelve `e3md` exports and the mirrored infrared
  labels were present for record 024's faithfulness pass on 2026-10-15.

**Built-in reproductions are validity checks, not evidence** (F86, F87):

- A at N = 536 should land near T's 0.5628.
- A0 should land near J's 0.0526.
- D should land near the loop's 0.3592.

**Label caveat, carried into the record.** msrs-day's thermal boxes are the visible-side boxes
mirrored onto thermal with no correction (`mirror_thermal_labels.py`). The residual misalignment
has a median EPE of 4.63 px (record 024's caveats), which is large against car_stop's roughly
4×7 px boxes. T and arm A share this error, so the sweep is consistent with itself. But arm A is
"thermal with mirrored labels", not label-free ground truth. This echoes the custom set's
label-transfer caveat in a weaker form.

## Open Questions

## Resolved

### Q1: Which score defines "matches the loop"?

- [x] Two-class mAP50 as the headline, all-class as a secondary column (recommended) — the
      two-class score is msrs-day's pre-registered endpoint (record 024 §8), and the passive
      classes are what C2's headroom map asks about. All-class costs nothing extra and lets the
      result sit beside E8's all-class curve.
- [ ] All-class mAP50 only — no code change at all, and it compares directly with E8. But the
      car class dominates it (the loop already recovers 91% of T−J on car, F170), so it would
      answer a question nobody asked about the passive classes.
- [ ] Two-class only — the narrowest reading. Losing all-class makes the cross-dataset comparison
      with record 011 impossible, for no saving.

### Q2: Which arms run?

- [x] A, A0, C and D; drop B (recommended) — A is the curve, A0 the floor, and C and D are the
      two levels it is read against. B (fine-tune on translated images) lost to thermal at every
      N ≥ 50 on the custom set (F83), and that answer is not what C2 asks. Dropping B halves the
      trained cost.
- [ ] All five, as in E8 — a full replication of record 011 on a second dataset. It roughly
      doubles the GPU time (about +1–2 GPU-h) to re-test a claim that is not on the goal's
      criteria list.
- [ ] A and A0 only, and read the loop's 0.3592 off record 024 — the cheapest option, but it
      loses the in-sweep reproduction that proves nothing drifted between the campaign and the
      sweep (F86's role).

### Q3: Which budgets?

- [x] 25, 50, 100, 200, 536 (recommended) — dense where the crossover probably sits (25–100 if
      the custom-set curve shape holds), and the top point reproduces T. N = 10 is left out
      because it is degenerate at a fixed epoch count (F84).
- [ ] 10, 25, 50, 100, 200, 400, 536 — the same shape as E8, so the two curves overlay point for
      point. It adds a known-degenerate point and a 400 that sits far above any plausible
      crossover.
- [ ] 25, 50, 75, 100, 150, 200, 536 — denser around the likely crossover, which narrows its
      interval. Two extra budgets at about 0.03–0.06 GPU-h each. Worth it only if Q2 keeps
      C and D, because the crossover is read against them.

### Q4: How many seeds per arm?

- [x] A at 3 seeds (0–2); C and D at 6 seeds (0–5) via a second, zero-shot-only invocation
      (recommended) — mirrors record 011, where 3 → 6 seeds on C nearly halved the crossover's
      width (F88). C and D take minutes, and all six msrs-day exports exist.
- [ ] Every arm at 3 seeds (0–2) — one invocation and the simplest command. It reads only half
      the loop's exports, so the levels A is matched against are noisier than they need to be.
- [ ] Every arm at 6 seeds (0–5) — six seeds on A permit an exact sign-flip test at each budget
      (floor p = .031). It doubles A's cost for significance that E8 chose not to chase (F85
      reports sign consistency, not p).

### Q5: Init weights and training length?

- [x] COCO `yolo11s.pt`, 100 epochs, batch 16, fixed across budgets (recommended) — exactly how
      T was trained (record 022), so A at N = 536 reproduces 0.5628 and the curve ends on a
      known number. Fixed epochs keep cells comparable, which was E8's choice too.
- [ ] Fixed iterations (epochs scaled by 536/N) — fairer to small budgets, which F84 showed are
      starved at a fixed epoch count. But no existing number anchors it, and small-N cells
      become as expensive as the full-N cell (about 0.2 GPU-h each).
- [ ] The script defaults, `yolo11n.pt` and 50 epochs — matches E8's custom-set settings, but
      breaks the link to T, so the top of the curve would not reproduce anything.

### Q6: How are the N images picked?

- [x] Plain seeded random, as in E8, with per-class training instance counts logged per cell
      (recommended) — no change to the selector the translator shares (`annotated_subset`,
      `src/t2o/data/dataset.py:134-153`). A draw that misses a passive class shows up in the
      CSV instead of being hidden.
- [ ] A class-stratified sampler that guarantees passive instances in every draw — cleaner low-N
      cells, but either it changes the shared selector (and so the translator's own subset) or
      it adds a second selector, which breaks E8's "one budget quantity across arms".
- [ ] Random, but redraw until both passive classes appear — a small change, but it quietly
      biases small budgets toward frames with passive objects, so they look better than a real
      random annotation would.

## Out of Scope

- **Beating T, or closing the passive gap.** The passive-class claim is closed as negative, and
  goal.md concedes supervised thermal at full annotation. This sweep measures the gap and does
  not chase it.
- **A new translator run, a dose change, or a turbo or backbone run.** The sweep only reads the
  twelve existing `e3md` exports.
- **Fixing the mirrored-label misalignment.** Registration is the sister project's problem
  (goal.md, Non-Goals). The residual error is disclosed, not corrected.
- **The other datasets' budget curves.** Only msrs-day is swept here. The custom set already has
  its curve (record 011).
- **A sweep registry or config file per dataset.** The sweep stays flag-driven, as E8 was.
