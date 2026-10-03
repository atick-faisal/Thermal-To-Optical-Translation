# 001 — E1 reference bracket

**Date:** 2026-08-12, visible-trained row re-measured 2026-08-13 · **Task:** M0.10 (re-measure: M1) · **Machine:** Windows server, 2× A100 40 GB · **Wall clock:** not recorded
**git_sha:** not recorded — results committed in 2a6b17d (bracket) and 7297601 (M1 re-measure) · **W&B:** not recorded · **Log:** none saved — transcribed from TASKS.md 990-1019 and 1080-1105 @ pre-spec-migration

## Command

Not recorded. The source says only that M1's re-measure went through `t2o evaluate`
(`metrics.task.evaluate_detector`) on the frozen val split.

## Configuration

- Detectors: the existing `.pt` weights, one trained on thermal and one on visible. The
  visible-trained one is `optical_best_n_1.pt` (`yolo11n`), never fine-tuned on anything
  this project produced. The thermal-trained checkpoint's name is not recorded.
- Evaluation set: the frozen custom val split, 153 images / 423 instances, four classes
  (Fuse, Pole, Switch, Transformer).
- Axes: {detector trained on thermal, on visible} × {evaluated on raw thermal, on real
  visible}. The thermal-trained detector was not run on visible.

## Results

Bracket, as it stands in `TASKS.md` (visible-trained row corrected at M1):

| detector trained on \ evaluated on | thermal | visible |
| --- | --- | --- |
| thermal | **> 0.9 mAP50** | — |
| visible | **0.1887 mAP50** | **0.9213 mAP50** |

The visible-trained row per class, from M1's gate evaluation (`optical_best_n_1.pt`,
`t2o evaluate`, same val split):

| arm | mAP50 | mAP50-95 | Fuse | Pole | Switch | Transformer |
| --- | --- | --- | --- | --- | --- | --- |
| raw thermal (floor) | 0.1887 | 0.0829 | 0.0377 | 0.5551 | 0.0047 | 0.1573 |
| real visible (ceiling) | 0.9213 | 0.6530 | 0.9448 | 0.9624 | 0.8226 | 0.9554 |

No results file was saved; there is nothing to re-render from.

## Findings

### F01 — In-domain detection is excellent on both modalities; the domain gap, not the sensor, is the problem

The thermal-trained detector on thermal scores > 0.9 mAP50, as does the visible-trained detector on
visible (0.9213). Thermal frames are not fundamentally information-poor for this task: the
open question behind C4/E1 ("does translation even help over direct thermal detection")
already has a clear directional answer — the signal is there, the *domain gap* is the problem,
not the sensor.

### F02 — A visible-trained detector collapses on raw thermal: 0.1887 against 0.9213 on visible, very unevenly across classes

The same `optical_best_n_1.pt` drops from 0.9213 mAP50 on real visible to 0.1887 on raw,
untranslated thermal, a gap of 0.733. Pole survives at 0.5551 — a pole's silhouette is
thermally obvious — while Switch is 0.0047 and Fuse 0.0377, i.e. gone. This is the floor
M1's gate ("translated mAP must beat raw-thermal mAP on at least one class, or stop")
measures against. The realistic deployment baseline is the detector you can actually label
data for (visible) run on the raw sensor frame, not a thermal-trained detector, which needs
exactly the thermal annotations E8's low-annotation framing exists to avoid depending on.

## Provenance caveats

- The visible-trained row was first recorded on 2026-08-12 from an ad-hoc measurement as
  "< 0.05" (thermal) and "> 0.9" (visible). M1's gate evaluation re-ran it through
  `metrics.task.evaluate_detector` and got 0.1887 / 0.9213; `TASKS.md` was then edited in
  place, so the table above is the corrected one. The paper must not report < 0.05.
- The thermal-trained cell, "> 0.9 mAP50", is still the original ad-hoc figure. It was never
  re-measured through `evaluate_detector`, and there is no exact number behind it. It is not
  E8's thermal-supervised 0.93 at 600 labels, which is a different detector.
- The visible-trained detector was trained on "the dataset with ultralytics defaults", which
  does not by itself confirm that val was held out. M1.2 step 1 (record 004) addresses this.
- Every number here is from one detector checkpoint, single seed, on 153 val images. No
  confidence interval.

## Next

M1's gate: translated mAP must beat the 0.1887 raw-thermal floor on at least one class.
The bar is low — a good sign for M1's feasibility.
