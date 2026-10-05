---
slug: msrs-passive-gate
title: MSRS passive-class headroom gate
status: planned
created: 2026-10-05
---

# MSRS passive-class headroom gate

## Summary

The claim is that translation helps a colour-trained detector on objects that blend into the
thermal background. FLIR could not test it: all three of its scored classes (bicycle, car,
person) are warm, and the loop only reached the raw-thermal floor (record 016). MSRS ships
segmentation masks for thermally passive classes. Two of them are common enough to score:
`car_stop` (406 train-day / 132 test-day blobs) and `color_cone` (264 / 102).

This feature turns those masks into YOLO boxes and builds a daytime-only labelled MSRS tree.
That makes the existing kill-test (`scripts/gate_table.py`, record 013's recipe) runnable on
MSRS per class. The gate then decides, for a few GPU-hours, whether a ~117 GPU-h campaign is
worth it.

The feature also fixes a train/val leak found during reconnaissance. 71 of the 80 `detection/`
pairs are pixel-identical (mean absolute error 0.0) to MSRS frames, and 23 of those are val
frames. The current adapter merges `detection/` into train, so `splits/msrs.json` (1,163 train)
holds 23 labelled copies of val. No record has scored MSRS yet, so no result needs retracting.

## Tech Stack / Approach

**Data layer only.** The gate, judge training and reporting already exist and are reused
unchanged.

- **Leak fix.** `src/t2o/data/adapters/msrs.py:45-75` stops reading `detection/`. Train becomes
  the 1,083 `train/` frames. `splits/msrs.json` is re-frozen by `scripts/freeze_splits.py`, and
  its hash changes. The 361-frame val split is unchanged.
- **Mask → box.** A small pure function maps a `Segmentation_labels/*.png` mask (480×640 uint8
  class indices; order in `dataset/raw/msrs/visualize.py`) to YOLO lines. It takes connected
  components per kept class with `cv2.connectedComponentsWithStats`, which returns each
  region's box and area in one call (OpenCV already arrives with ultralytics). It drops
  components under a 10 px minimum area (Q3) and writes `cls cx cy w h`. Ultralytics'
  `convert_segment_masks_to_yolo_seg` was read and rejected: it writes polygons, keeps all
  eight classes as `pixel − 1`, and filters only contours under 3 points. The written labels go through
  `t2o.data.adapters.common.write_label`'s conventions. Class ids are remapped explicitly to the
  kept class list, not passed through.
- **Two trees, one adapter.** The plain `msrs` tree keeps every frame, now mask-labelled
  (1,083 train / 361 val). The `msrs-day` tree keeps frames whose stem ends in `D` (536 train /
  179 val), built by the same adapter with a day-only switch. Night was measured
  before deciding: the median visible luma is 26 at night against 99 by day, so night gives
  neither a usable visible ceiling nor a translator target.
- **Registration.** Through `ADAPTERS` in `scripts/adapt_datasets.py:36`, one adapter per
  dataset, with the source and split reasoning in the module docstring (the `adapters/m3fd.py`
  idiom). `scripts/freeze_splits.py:42` freezes any new processed tree under its own name.
- **Thermal labels.** Mirrored after adaptation by `scripts/mirror_thermal_labels.py`, never
  inside the adapter (record 013's reasoning). MSRS is pre-registered, so a plain mirror needs
  no calibration.
- **Tests.** The MSRS-shaped fixture in `tests/test_adapters.py:41-67` gains synthetic
  `Segmentation_labels` masks. The tests cover the remap, the area floor, day filtering and the
  absence of `detection/` stems. The existing `skipif`-guarded real-data test pattern is kept.
- **The gate itself is a server run, not a task.** It follows record 013's recipe:
  1. `mirror_thermal_labels.py`.
  2. Train a new visible yolo11s judge with `t2o train-detector --init-weights yolo11s.pt
     --epochs 100 --seed 1`. The FLIR judge cannot be reused: the class set differs
     (invariant 7).
  3. `gate_table.py --primary-classes car_stop color_cone`.

**Server rebuild.** The adapter skips a populated destination (`dest_already_populated`), so
the server's existing `dataset/processed/msrs/` must be deleted before re-adapting. Delete any
`labels.cache` too: it is keyed on sizes and paths, not contents. The server also needs the raw
`Segmentation_labels/` folders, which `dataset/` being git-ignored does not carry.

**goal.md touch-point.** `docs/goal.md:123` lists "MSRS (1,163 / 361)". The leak fix makes that
1,083 / 361, and a day tree is a further variant. Editing goal.md is the human's conversation,
not a task in this feature.

## Open Questions

## Resolved

### Q1: How is the daytime-only subset represented?

- [x] A separate processed tree, `msrs-day`, frozen as `splits/msrs-day.json` (recommended) —
      every consumer (judge training, gate, a later campaign) reads it as an ordinary dataset
      with no filter plumbing, and the frozen data contract covers it as is
- [ ] One labelled `msrs` tree with all frames, plus a day filter at manifest-build time — keeps
      a single frozen split, but `train-detector`, `gate_table.py` and the campaign path would
      each need a stem filter, and an unfiltered run silently mixes in night
- [ ] Day-only becomes the only `msrs` tree — no second dataset to track, but it drops night
      MSRS for good, including any later "where translation stops paying" use on the
      lighting axis

### Q2: Which mask classes become detection classes?

- [x] `car`, `person`, `bike`, `car_stop`, `color_cone` (recommended) — the two passive targets
      plus warm classes in the same frames, so the gate shows passive vs warm headroom side by
      side; the classes dropped are too rare or not box-shaped
- [ ] All eight mask classes — nothing discarded, but `bump` (41 / 15) and `guardrail` (27 / 8)
      are too rare to score, and `curve` and `guardrail` are long strips whose boxes mostly
      cover background
- [ ] `car_stop` and `color_cone` only — the purest passive test, but it loses the in-dataset
      warm comparison, and a judge with two tiny classes may train poorly

### Q3: What is the blob → box policy?

*Revisited after the first pick of 30 px.* That floor was a judgement call, not a measurement, so the
daytime masks were measured (train + test, 8-connectivity blobs) and the small ones viewed on the
visible images:

| Class | Blobs | 1–9 px | 10–29 px |
| --- | --- | --- | --- |
| car | 892 | 13 | 4 |
| person | 1,827 | 53 | 52 |
| bike | 479 | 42 | 19 |
| car_stop | 535 | 0 | 11 |
| color_cone | 363 | 4 | 23 |

Blobs of 1–9 px are crumbs: specks on the edge of a person or a bicycle wheel, not objects.
Car stops of 10–29 px are real distant bollards (about 4×7 px boxes). Cones of 10–29 px are
mostly real distant cones, with a few fragments of a larger cone. MSRS is a single resolution
(480×640), so a pixel floor means the same thing in every frame; what varies is distance, which
is why the floor sits below the smallest real far-away object rather than at a size guess.

- [ ] Connected components (8-connectivity), dropping blobs under a fixed minimum area of
      about 30 px, with the value fixed in code and tested (recommended) — removes mask speckle
      and annotation crumbs while keeping real cones (test-day median 162 px); touching cones
      merging into one box is accepted and noted
- [x] Connected components with a 10 px floor, the value cited to this table in the code
      (recommended after measuring) — removes the crumb cluster, which is almost all warm-class
      fragments, and keeps all 11 small car stops and the 23 small cones; a 30 px floor drops
      them, about 2% of car stops and 7% of cones, which are the classes under test
- [ ] Connected components with no area floor — simplest, but single-pixel annotation fragments
      become ground-truth boxes the judge is penalised for missing
- [ ] Dilate before labelling, to rejoin an occluded object's fragments — fewer split cars, but
      it also merges neighbouring cones and people, and its kernel size is a second tuning knob
- [ ] A larger floor such as 400 px — cleaner boxes, but it discards about two thirds of the
      cones and half the car stops, which are exactly the objects under test

### Q4: What does the leak fix do with the 9 `detection/` pairs that have no twin?

- [x] Drop `detection/` entirely (recommended) — masks now label every train frame, and the 9
      leftovers carry only `person, bicycle, car` labels; merged in, they would teach the judge
      that unlabelled cones and car stops are background
- [ ] Keep the 9 unique pairs — 9 more training frames, at the cost of a second label source
      with a different class order (`classes.txt` is person/bicycle/car; masks are car/person/
      bike) and missing passive-class labels

## Out of Scope

- **Editing `docs/goal.md`.** The MSRS count (1,163 → 1,083 after the leak fix) and whether
  `msrs-day` counts toward Consistency's datasets are the human's conversation.
- **The server runs.** The judge training and the gate are runs between tasks; their result
  becomes a record via `/log-experiment`, not a task row.
- **The campaign.** Translator and loop runs on MSRS wait for the gate's verdict and get their
  own feature.
- **Night MSRS as a benchmark.** Unless Q1 picks otherwise, night frames stay in the plain
  `msrs` tree and nothing here scores them.
- **Splitting the power-line set into faulty / non-faulty.** Set aside by the human: switches
  and fuses are scored together.
- **Other passive-object datasets** (FMB, DroneVehicle, InfraParis, VITLD). These are
  considered only if this gate fails.
