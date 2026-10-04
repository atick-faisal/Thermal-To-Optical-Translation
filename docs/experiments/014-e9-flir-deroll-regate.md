# 014 — E9 step 3b: FLIR's thermal labels de-rolled by the sibling's calibration, then the kill-test re-gated — WEAK PASS stands at +0.2066

**Date:** 2026-09-28 · **Task:** M3 E9 · **Machine:** Windows server, 2× A100 — the de-roll on CPU, the gate on `cuda:0` · **Wall clock:** de-roll 1.8 s for 5,142 label files; gate not timed
**git_sha:** not recorded — the de-roll landed in dd266e6, the stale-cache fix in 7761a30, the result in f19525e · **W&B:** not recorded · **Log:** none saved — transcribed from TASKS.md 4478-4728 @ pre-spec-migration

## Command

Labels first, then the gate; the de-roll is CPU-only and takes seconds.

```powershell
uv run python scripts/mirror_thermal_labels.py --data dataset/processed/flir `
    --calibration calibration/flir.json --force
Remove-Item dataset/processed/flir/*/visible/labels.cache -ErrorAction SilentlyContinue
uv run python scripts/gate_table.py --data dataset/processed/flir/data.yaml `
    --weights runs/reference-flir-yolo11s/weights/best.pt `
    --primary-classes bicycle car person --calibration calibration/flir.json `
    --out runs/gate-derolled --device cuda:0
```

`--force` is required because the tree is already plain-mirrored. The two runs stay side by side as
`runs/gate/gate.csv` against `runs/gate-derolled/gate.csv`, both rows stamped with the calibration
digest — a floor is only comparable to another floor measured against the same boxes, and the two
label states are the same filenames in the same directory. Restoring the uncorrected labels is the
same command without `--calibration`, which removes the provenance marker too, so it can never
outlive what it describes.

The mirror deletes the *thermal* side's stale cache itself. The middle line clears the **visible**
one, which the mirror never touches and which is legitimately current — its labels have not moved.
Removing it anyway is what turns "the ceiling reproduced at 0.5523 / 0.6566" into a real end-to-end
check of the pipeline rather than a check that a pickle can be unpickled twice. **Both trees were
de-rolled and both caches cleared before the 2026-09-28 run.**

## Configuration

**Why this runs before the cell.** The standing decision was *de-roll only if the cell comes back
weak or null* (record 012's blocker 1), written when the headroom was expected to be ~+0.7. At
**+0.062 of margin** (F104) the cheap measurement comes first: CPU-minutes against the ~126 GPU-h it
gates. Step 3b (a), the de-roll, ran 2026-09-27; step 3b (b), the re-gate, 2026-09-28.

**The constant** — `calibration/flir.json`, a byte-for-byte transcription of the sibling repo's
*accepted* constant at `git_sha 739796f`:

| field | value |
| --- | --- |
| `digest` | `210142740cdba163` |
| pairs | n = 1,013 FLIR val |
| estimator | element-wise median of roma / superpoint-lightglue / eloftr |
| `spread_px` | 1.23 |
| `magnitude_px` | 9.3158 |
| measured at | 640×512 |

Transcribed rather than read across the filesystem because **the sibling repo is not on the training
server**. `tests/test_calibration.py` asserts the digest against a hardcoded literal, so a typo in
one corner cannot pass as a calibration.

**Every FLIR frame is exactly 640×512** — all 4,129 + 1,013 pairs, both modalities, counted. The
constant therefore applies with no rescaling, and `ResidualCalibration.validate_for`'s shape guard is
a real check rather than a formality: a 640×512 corner field applied at 1280×1024 is silently *half*
the offset it claims to be.

**Which label state a tree holds** is recorded in `{train,val}/infrared/LABELS_PROVENANCE.json`,
written beside `labels/` rather than inside it because ultralytics globs that directory. The gate
refuses to start unless it matches `--calibration` (F111).

**The direction** is `visible_to_infrared()`, i.e. `inv(H)` — named for the direction it travels
rather than for the sibling's reference/moving convention (F112). A test pins the sign by asserting
FLIR's bottom-left corner moves **left** by 13.76 px.

**`--box-transform`** chooses between mapping all four box corners and re-bounding (`corners`, the
default and the geometrically correct reading) and mapping only the centre (`centre`, which preserves
width and height exactly). Under FLIR's roll the two differ by ~1 px on a box. Running both costs
CPU-seconds, and the pair agreeing is the cheapest available evidence that the choice does not carry
the result — the only reason the second mode exists.

**The 4-point homography solve is numpy-only, not `cv2.getPerspectiveTransform`**: opencv is not a
declared dependency, it arrives transitively through ultralytics, and a direct dependency to save
twelve lines is not a trade worth making. `tests/test_calibration.py` asserts the solve against cv2
when cv2 happens to import and against analytic cases regardless.

**A box the de-roll carries off the frame is dropped, not clamped to a degenerate target** — the rule
`adapters/common.py::voc_to_yolo_lines` already follows — and logged at WARNING.

**Unchanged from record 013:** the judge (`runs/reference-flir-yolo11s/weights/best.pt`, epoch 46),
the 1,013 val scenes, the 3-class primary mean (bicycle / car / person), and the band
`KILL_THRESHOLD = 0.15` / `STRONG_THRESHOLD = 0.40`.

## Results

As recorded in dd266e6, 7761a30 and f19525e. No console output survives; these are the source's
distilled tables, verbatim.

**Step 3b (a), the de-roll (2026-09-27)** — CPU only, 1.8 s for 5,142 label files.

**The first re-gate (2026-09-27) was invalid.** Its `gate.csv` came back **bit-identical** to the
uncorrected one: floor 0.3397, 3-class floor 0.4442, every per-class AP50 matching to the last digit.
Val label bytes against the plain copy:

| | files | content changed | byte size changed |
| --- | --- | --- | --- |
| `flir/val/infrared/labels` against the plain copy | 1,013 | **1,013** | **0** |

326,952 bytes of val labels before, 326,952 after. The val cache was written 2026-09-26 23:04, the
labels 2026-09-27 17:10. Train's total moved by **exactly 38 bytes** (1,225,728 → 1,225,690). No
de-rolled number was produced by this run.

**The direction sweep** — FLIR val's visible frame warped by `H(α)`, the published corner field
scaled by α, scored against the raw thermal frame over 60 pairs:

| α | mutual information | gradient NCC |
| --- | --- | --- |
| −1.50 | 0.6680 | 0.2911 |
| **−1.00** | **0.6833** | **0.3346** |
| −0.50 | 0.6644 | 0.2992 |
| 0.00 (raw pair) | 0.6335 | 0.2558 |
| +1.00 | 0.6070 | 0.2196 |

α = −1 improved mutual information on **59 of 60** pairs; α = +1 on 8. The constant's displacement
over a 17×17 grid: median **5.21 px** / mean 5.52 / max 13.49; left edge +13.5 px, right edge
−12.0 px, centre ~0.8 px.

**How far the de-roll moves the boxes** — IoU between each plain-mirrored box and its de-rolled
replacement, over the tree on disk:

| split | metric | bicycle | car | dog | person | 3-class |
| --- | --- | --- | --- | --- | --- | --- |
| val | median IoU | 0.832 | 0.851 | 0.850 | **0.773** | **0.822** |
| val | share below IoU 0.5 | 6.4% | 1.0% | 0.0% | **9.7%** | **5.4%** |
| train | median IoU | 0.760 | 0.824 | 0.713 | 0.716 | 0.796 |
| train | share below IoU 0.5 | 11.8% | 3.0% | 14.7% | 17.8% | 7.9% |

Box counts: val 8,604 → 8,604, **zero dropped**. Train dropped **one** box of 32,256 (0.003%):
`FLIR_04106.txt`'s last `person`, 6 px wide at x ∈ [0, 6], carried to x ∈ [−13.8, −7.8].

**Pre-registered, before the re-gate ran** (verbatim):

1. **The band does not move.** `gate_table.py`'s `KILL_THRESHOLD = 0.15` and
   `STRONG_THRESHOLD = 0.40` are not touched, re-read or reinterpreted.
2. **The de-rolled headroom governs.** It is the less-flattering of the two measurements, so it
   replaces +0.2123 as the number the cell decision is read off. Stated now, not after seeing it.
3. **Expected direction:** better-placed boxes → the floor **rises** → the headroom **falls**,
   concentrated in `person`.
4. **Expected size: small.** AP50 needs only IoU ≥ 0.5, and a perfectly-placed thermal prediction
   already scored a median **0.822** against the uncorrected label. So the de-roll can only convert
   the **5.4%** of val boxes below that threshold plus those near the margin — a single-digit-point
   floor rise, concentrated in `person`. The WEAK PASS most likely survives.
5. **The refutation condition, named in advance:** if the 3-class floor rises above
   **0.5066**, i.e. 0.6566 − 0.15 — a rise of **+0.0624** — the verdict flips to **KILL** and the
   cell does not run. The ceiling cannot move, so this is a one-sided test.
6. **The ceiling must reproduce at 0.5523 / 0.6566.** It reads visible images and visible labels,
   which this step does not touch. A ceiling that moves means the pipeline changed rather than the
   data, and invalidates the comparison rather than adding to it.

**Step 3b (b), the re-gate (2026-09-28)** — `runs/gate-derolled/gate.csv`, same judge, same images,
thermal labels de-rolled by `210142740cdba163`:

| arm | mAP50 | mAP50 (3-class) | mAP50-95 | bicycle | car | dog | person |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ceiling (visible) | 0.5523 | **0.6566** | 0.2767 | 0.4922 | 0.8176 | 0.2396 | 0.6599 |
| floor (thermal, de-rolled) | 0.3435 | **0.4499** | 0.1514 | 0.3674 | 0.6051 | 0.0242 | 0.3772 |

**Headroom +0.2066** on the 3-class mAP50 (+0.2088 all-class), against a kill line of 0.15 — a margin
of **+0.0566**. Ceiling to full precision: 0.5523263239785735 / 0.6565721600344453.

Against step 3's floor (record 013):

| | bicycle | car | **person** | dog | 3-class | recall |
| --- | --- | --- | --- | --- | --- | --- |
| step 3 (plain mirror) | 0.3627 | 0.6033 | 0.3667 | 0.0260 | 0.4442 | 0.2832 |
| step 3b (de-rolled) | 0.3674 | 0.6051 | **0.3772** | 0.0242 | **0.4499** | 0.2912 |
| delta | +0.0047 | +0.0018 | **+0.0106** | −0.0018 | **+0.0057** | +0.0080 |

Re-render on the server without retraining: rerun the gate command above (two validation passes).

## Findings

### F110 — The first re-gate scored a stale ultralytics label cache, not the de-rolled labels: ultralytics keys `labels.cache` on file sizes and paths, and the de-roll keeps every val label file the same length (legacy: M3 E9 step 3b, "the first re-gate was invalid — ultralytics scored a stale label cache")

The bit-identical `gate.csv` is the diagnosis rather than a coincidence — 8,604 boxes moving by a
median 5.2 px cannot leave 17 significant figures untouched. ultralytics decides whether a split's
`labels.cache` is current with `ultralytics/data/utils.py::get_hash`, which hashes the **sum of the
files' sizes** and the path strings — never their contents — and `YOLODataset.get_labels` reuses the
pickle whenever that hash matches. **The de-roll collides with that hash by construction:**
`deroll_label_lines` re-emits every line in the same `:.6f` form `adapters/common.py::voc_to_yolo_lines`
writes, `:.6f` on a normalised coordinate is always exactly 8 characters, and the class index is
unchanged. So 1,013 val files changed content and 0 changed size; the cache was a day stale with its
hash still matching.

Two details confirm the mechanism rather than merely fit it. Train's total moved by exactly 38 bytes,
the length of the one dropped `FLIR_04106` line, so *train's* cache would have rebuilt — only val
collided perfectly, and val is the one split the gate scores. And **the ceiling reproducing at
0.5523 / 0.6566 was never evidence of anything**, because that arm read its own equally-stale
`val/visible/labels.cache`. The pre-registration was untouched, because no de-rolled number was ever
produced; nothing about the de-roll itself was wrong either.

**The fix:** `t2o.data.labels.invalidate_label_cache` deletes `<labels_dir>.cache`, and
`mirror_labels_onto_infrared` calls it on both write paths *before* writing, so a crash halfway
through cannot leave a cache claiming to describe the directory. A test pins the byte-size collision
itself, so removing that call can never look harmless.

### F111 — `gate.csv`'s calibration-digest column recorded that a flag was passed, not that the tree was in that state; it is now verified against the tree in both directions (legacy: M3 E9 step 3b, "`gate.csv`'s digest column became a verified fact")

The invalid run stamped `210142740cdba163` on a row whose boxes were not de-rolled.
`gate_table.py::_verified_digest` now reads the val thermal side's `LABELS_PROVENANCE.json` and
refuses to start on a mismatch, in both directions — a `--calibration` the tree does not carry, and a
de-rolled tree scored without one. Same shape as the α-sign trap (F112): a number that looks exactly
like a correction, so it gets a check rather than a comment.

### F112 — The visible → thermal point map is `inv(H)`, the inverse of what the sibling's wording reads as; applying `H` would have doubled the misalignment while looking exactly like a correction (legacy: M3 E9 step 3b, "the direction was measured, and it is the inverse of what the source's wording says")

The sibling documents `corner_shift` as carrying a *reference* corner to where the *moving* modality
lands (`cmreg/gt/calibration.py:67-69`), with `moving` defaulting to thermal
(`cmreg/config/schema.py:330`), which reads as visible → thermal = `H`. Measured instead: both mutual
information and gradient NCC — two criteria that share nothing with the matcher EPE the constant was
estimated from — peak at **α = −1.0** (0.6833 / 0.3346 against 0.6335 / 0.2558 for the raw pair and
0.6070 / 0.2196 at α = +1), smoothly and unimodally, and α = −1 improved mutual information on
**59 of 60** pairs where α = +1 improved it on 8. A wrong sign produces a plausible number in either
direction, hence the test pinning it.

### F113 — The constant is confirmed twice over, independently: its median displacement (5.21 px) matches roma's 5.90 px EPE, and the α optimum at −1.0 says the published magnitude needs no refitting (legacy: M3 E9 step 3b, "two independent confirmations of the constant fell out of this")

Over a 17×17 grid the field moves points a median **5.21 px** / mean 5.52 / max 13.49, against
roma's `epe_mean` of **5.90 px** from a completely different instrument (F93's residual). α = −1.0
being the optimum to within the sweep's resolution says the magnitude fits our tree as published. The
field is strongly x-dependent (left edge +13.5 px, right edge −12.0 px, centre ~0.8 px) — the
roll/anisotropic-scale ambiguity the sibling cites as the reason a corner field is published and
*not* "−1.18° of roll".

### F114 — The de-roll moves val boxes to a median IoU of 0.822 against the plain mirror, 5.4% below IoU 0.5, `person` hit hardest (0.773, 9.7%); box counts reconcile exactly (legacy: M3 E9 step 3b, "how far it actually moves the boxes — measured on the written labels")

Per class on val: bicycle 0.832 / 6.4%, car 0.851 / 1.0%, dog 0.850 / 0.0%, **person 0.773 / 9.7%**;
train moves further (3-class 0.796 / 7.9%). `person` carries the largest headroom (+0.2932, F107),
so it is the specific risk this step exists to price. Val keeps all 8,604 boxes, so the gate's floor
loses nothing. Train drops one of 32,256: `FLIR_04106`'s 6 px `person` at the left edge, carried
off-frame to x ∈ [−13.8, −7.8] — genuinely outside the thermal camera's field of view, the offset
being the reason. The WARNING it is logged at is the guard working.

### F115 — The de-rolled kill-test is still a WEAK PASS: 3-class headroom +0.2066 (ceiling 0.6566, floor 0.4499), a margin of +0.0566 over the 0.15 kill line; the floor came in 0.0567 short of the 0.5066 refutation line (legacy: M3 E9 step 3b, "WEAK PASS stands, headroom +0.2066")

Read off the unmoved band (pre-registered item 1). By item 2 the de-rolled headroom governs and
replaces F104's +0.2123 as the number the cell decision is read off. All-class +0.2088. **The verdict
is unchanged, and step 4 proceeds.**

### F116 — The ceiling reproduced bit-identically while the floor moved, so the pipeline did not change; the floor rose, confirming the correction's direction a third time (legacy: M3 E9 step 3b, "all three pre-registered readings land")

Ceiling 0.5523263239785735 / 0.6565721600344453, every digit, with the visible cache cleared
(Command). Because the floor *did* move this time, that identity is now a consistency check rather
than a cache artefact (F110). The floor rose (+0.0057 3-class, recall 0.2832 → 0.2912), as item 3
predicted — a third confirmation of the direction, independent of the α sweep (F112) and of the
matcher EPE (F113).

### F117 — `person` is where the floor moved: +0.0106, six times car's move and 62% of the 3-class rise, exactly the class whose labels the audit said move most (legacy: M3 E9 step 3b, "`person` is where it happened")

Against bicycle +0.0047 and car +0.0018; `person`'s labels had the lowest median IoU (0.773) and the
largest share below 0.5 (9.7%) in the audit (F114). `dog` fell by 0.0018 on 13 instances, which is
noise at that count.

### F118 — The size was over-predicted: the floor rose 0.57 of a point, not single digits, so label misalignment accounts for 2.7% of the measured headroom and was never load-bearing — the floor's loss is objects the judge cannot see, not boxes it places a few pixels off (legacy: M3 E9 step 3b, "the size was over-predicted, and that is the finding")

Item 4 predicted a single-digit-point rise; the rise is +0.0057. Blocker 1 is now priced rather than
feared. The reason is in the recall column: the thermal floor sits at **0.29 recall against 0.51
precision**, so what the judge loses on thermal frames is overwhelmingly *objects it cannot see at
all*. A 5 px label correction can only rescue detections already near the IoU 0.5 margin, and at a
median IoU of 0.822 between the two label sets there were few of those. F109's mAP50-95 / mAP50
reading was mild evidence of this; F93's predicted downward bias is real, and small.

### F119 — The floor is honest and the gap is real, but the de-roll does not fix the cell's own confound: pix2pix trains on rolled image pairs, which needs the thermal images warped, not the labels (legacy: M3 E9 step 3b, "this does *not* fix the cell's own confound")

+0.2066 is a genuine modality gap on a public dataset, not an artefact of borrowing visible-frame
boxes — still well under the custom set's +0.733 (F02), which remains the thing E9 exists to report.
But pix2pix trains against the paired visible frame with `l2: 1.0` + `lpips: 5.0`, and a systematic
roll between input and target is satisfied by blurring. Warping the images is a separate and much
larger job. What step 3b buys is an honest floor, not an aligned training set.

### F120 — Two concurrent E3 runs at `runtime.workers: 16` would each build step 2's two resident ultralytics worker pools in every detector stage; whether that clears the host's 128 GiB is unknown and free to measure (legacy: M3 E9 step 3b, "carried forward to step 5 — a host-RAM ceiling")

The twelve-run cell was going to run two arms concurrently, one per card, and E3's configs carry
`runtime.workers: 16`, which `engine/loop.py:245` hands straight to `train_detector` — the exact
combination that failed in step 2 (F97). Stated rather than extrapolated: M1's two completed server
runs did survive at `workers: 16`, but their profile differs (`t2o`'s single-pool loader at
`batch_size: 2` for most of the run). The budget is **128 GiB across both cards**, minus whatever the
two translators hold, at ≳1.3 GiB a detector worker (F97). Step 2's `Measure-Object WorkingSet64`
probe answers it with no extra GPU time; take that number before committing ~126 GPU-h to a pair of
runs that could die on day two.

## Provenance caveats

- **Commits that wrote this range** (`git blame` at the tag): dd266e6 (2026-09-27 17:15) the order
  reversal, the constant, the direction sweep, the audit, the pre-registration, the commands and
  "also worth knowing"; 7761a30 (2026-09-27 17:54) the stale-cache diagnosis, its fix and the
  digest check (4484–4530), plus the visible-cache line of the commands; f19525e (2026-09-28 09:15)
  the step header, the result and the "both trees" line. **F120's paragraph is older than step 3b**: e0c4a27 and 2b47d10 (2026-09-27, 09:39 and 09:47)
  wrote it with step 2's crash diagnosis, before the de-roll existed. It sits under step 3b's "Also
  worth knowing" by position, which is why it is here.
- **The cache's write time does not match record 013's dates.** The source calls 2026-09-26 23:04 "the
  *first* gate run", but record 013 dates step 3's gate 2026-09-27 (569840f). Whether the cache came
  from step 3's plumbing check, or the server clock differs, is not recorded.
- **Derived numbers that differ from their own rows, read as differences of unrounded values** (the
  `gate.csv` files are not on the Mac): 3-class headroom 0.6566 − 0.4499 = 0.2067 against +0.2066;
  `person`'s delta 0.3772 − 0.3667 = 0.0105 against +0.0106. "2.7% of the measured headroom" is
  0.0057 / 0.2123 (2.68%); against the de-rolled +0.2066 it is 2.76%. The margin (+0.0566) and the
  shortfall to 0.5066 (0.0567) both check against the rows.
- **The floor's 0.51 precision appears only in prose.** Neither gate table carries a precision column;
  it is lifted as stated.
- **The direction sweep's machine and date are not stated.** It reads FLIR frames, which the Mac does
  not hold, so it is read as the server; dd266e6 dates it no later than 2026-09-27. The 60 pairs'
  selection is not recorded.
- **`--box-transform centre` was offered as a cross-check, but no `centre` result is recorded.** The
  source says running both "costs CPU-seconds"; whether it was run is not stated.
- **The source's internal line reference, resolved at the tag.** "(line 1794)" for E3's
  `workers: 16` is M1.2 step 2b's `workers` decision at 1794–1797 (no-run, SPEC-MIGRATION-19).
- **Code anchors were not executed.** `engine/loop.py:245` is `workers=config.runtime.workers` at the
  tag. The sibling anchors (`cmreg/gt/calibration.py:67-69`, `cmreg/config/schema.py:330`) and the
  ultralytics ones (`data/utils.py::get_hash`, `YOLODataset.get_labels`) are the source's, not
  re-checked.
- **Cross-references, re-keyed:** the +0.2123 headroom and 0.4442 floor are F104; `person`'s +0.2932
  is F107; the mAP50-95 / mAP50 ratio is F109; the 5.90 px residual and its bias are F93; the custom
  set's +0.733 is F02; step 2's crash, the ≳1.3 GiB a worker and the WorkingSet64 probe are F96, F97
  and record 013's caveats; "blocker 1" and the ~126 GPU-h are record 012's.

## Next

Step 4 (record 015): the corpus-cap seam, a per-epoch clock, and the throughput probe. Posted by this
record:

- **The cell's decision number is +0.2066** (F115), not +0.2123; the WEAK PASS band still says run it.
- **Take the host-RAM peak** (F120) during the probe, before committing ~126 GPU-h to two concurrent
  runs.
- **The cell's training-pair misalignment is still open** (F119): an image warp, not a label fix, and
  a much larger job.
