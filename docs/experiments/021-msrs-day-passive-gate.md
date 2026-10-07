# 021 — msrs-day passive-class gate: V, T and J yolo11s at three seeds each — PASS, both passive classes GO

**Date:** 2026-10-07 · **Task:** MSRS-PASSIVE-GATE-06 · **Machine:** Windows server, 2× A100 (`cuda:0` only, A100-SXM4-40GB) · **Wall clock:** gate 18.5 min (15:31:16 → 15:49:48, nine validation passes); the six training runs are not in the saved files
**git_sha:** 8360dd9 (gate run; no `-dirty` check, see caveats) · **W&B:** not recorded · **Log:** `logs/2026-10-07-msrs-passive-gate-gate-table.txt`; per-pass rows in `logs/2026-10-07-msrs-passive-gate.csv` (a copy of `runs/gate/msrs-day/gate.csv`)

## Command

The log does not echo the commands; these are the server runbook's, which the run followed.

```powershell
# six trainings, one at a time; <arm> ∈ {reference (visible manifest), thermal (thermal manifest)}
uv run t2o train-detector --data runs/gate/msrs-day/manifests/<visible|thermal>/data.yaml `
    --init-weights yolo11s.pt --epochs 100 --seed <1|2|3> --workers 8 `
    --out runs/<reference|thermal>-msrs-day-yolo11s-s<seed> --device cuda:0

uv run python scripts/gate_table.py --data dataset/processed/msrs-day/data.yaml `
    --weights runs/reference-msrs-day-yolo11s-s{1,2,3}/weights/best.pt `
    --thermal-weights runs/thermal-msrs-day-yolo11s-s{1,2,3}/weights/best.pt `
    --primary-classes car_stop color_cone --out runs/gate/msrs-day --device 0
```

## Configuration

- **Arms.** V (ceiling) is a `yolo11s` trained on visible and scored on visible val. T is the same
  recipe trained on the mirrored thermal labels and scored on thermal val. J (floor) is each V
  checkpoint scored on thermal val.
- **Seeds.** 1, 2 and 3 for both V and T (plan.md Q7). T's seeds are unpaired with V's, so the gaps
  are differences of seed means and carry no spread.
- **Data.** `msrs-day`, mask-labelled: 536 train frames (fraction 1.0, manifest seed 0), and 179
  val frames with 976 instances in all nine passes on both modalities, so the plain mirror carried
  every label.
- **Classes.** `car, person, bike, car_stop, color_cone`; primary `car_stop color_cone`.
- **Calibration digest.** Empty: MSRS is pre-registered, so the thermal labels are a plain mirror.
- **Bands, fixed before any number existed.** KILL when V − J < 0.15 (`gate_table.py:80`); PASS
  when V − J ≥ 0.40 (`:81`); passive GO when V − T ≥ 0.10 (`:85`, plan.md Q6).

## Results

```
2026-10-07 15:33:24,059 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8698  R 0.7104  mAP50 0.7841  mAP50-95 0.5199
2026-10-07 15:35:29,114 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8531  R 0.7281  mAP50 0.7985  mAP50-95 0.5200
2026-10-07 15:37:30,227 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8268  R 0.7421  mAP50 0.7934  mAP50-95 0.5147
2026-10-07 15:39:32,320 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.4304  R 0.1736  mAP50 0.1412  mAP50-95 0.0589
2026-10-07 15:41:34,915 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.3108  R 0.1469  mAP50 0.1203  mAP50-95 0.0526
2026-10-07 15:43:38,302 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.2973  R 0.1348  mAP50 0.1064  mAP50-95 0.0448
2026-10-07 15:45:41,438 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7972  R 0.5919  mAP50 0.6531  mAP50-95 0.3828
2026-10-07 15:47:44,677 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7473  R 0.6209  mAP50 0.6496  mAP50-95 0.3802
2026-10-07 15:49:48,264 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7781  R 0.6024  mAP50 0.6401  mAP50-95 0.3792
2026-10-07 15:49:48,270 INFO    __main__: gate table:
| arm | mAP50 | mAP50 (2-class) | mAP50-95 | car | person | bike | car_stop | color_cone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ceiling (visible) | 0.7920 � 0.0073 | **0.8251 � 0.0025** | 0.5182 � 0.0030 | 0.9361 � 0.0058 | 0.7159 � 0.0103 | 0.6577 � 0.0321 | 0.8905 � 0.0113 | 0.7597 � 0.0147 |
| floor (thermal) | 0.1226 � 0.0175 | **0.0526 � 0.0149** | 0.0521 � 0.0071 | 0.0763 � 0.0119 | 0.3620 � 0.0257 | 0.0697 � 0.0225 | 0.0679 � 0.0252 | 0.0373 � 0.0046 |
| thermal-trained (thermal) | 0.6476 � 0.0067 | **0.5628 � 0.0218** | 0.3808 � 0.0019 | 0.9038 � 0.0109 | 0.6509 � 0.0157 | 0.5578 � 0.0196 | 0.6772 � 0.0136 | 0.4483 � 0.0378 |
| V - T (sensor gap) | +0.1444 | **+0.2623** | +0.1374 | +0.0323 | +0.0650 | +0.0999 | +0.2132 | +0.3114 |
| T - J (domain gap) | +0.5250 | **+0.5102** | +0.3287 | +0.8275 | +0.2889 | +0.4881 | +0.6094 | +0.4110 |
2026-10-07 15:49:48,270 INFO    __main__: headroom on 2-class mAP50 (car_stop, color_cone): 0.8251 - 0.0526 = +0.7725  [all-class: +0.6693]
2026-10-07 15:49:48,270 INFO    __main__: PASS -- the premise transfers. Run the cell.
2026-10-07 15:49:48,270 INFO    __main__: on 2-class mAP50: sensor gap V - T +0.2623, domain gap T - J +0.5102
2026-10-07 15:49:48,270 INFO    __main__: car_stop: V - T +0.2132 -> passive GO -- colour shows what thermal does not; this class tests the passive claim.
2026-10-07 15:49:48,270 INFO    __main__: color_cone: V - T +0.3114 -> passive GO -- colour shows what thermal does not; this class tests the passive claim.
2026-10-07 15:49:48,270 INFO    __main__: written: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\gate\msrs-day\gate.csv
```

The CSV holds the per-pass rows. They are in this order: V s1–s3 on visible, then J s1–s3, then T
s1–s3. The nine `eval` lines above follow the same order.

Re-render on the server without retraining: rerun the gate command above with
`runs/*-msrs-day-yolo11s-s*` intact (nine validation passes, ~18 min).

## Findings

### F158 — msrs-day clears the kill test with +0.7725 headroom on the two passive classes, 3.7× FLIR's

- **Primary (2-class) mAP50:** the ceiling V is 0.8251 ± 0.0025 and the floor J is 0.0526 ± 0.0149.
- **V − J:** +0.7725, 1.9× the 0.40 PASS line.
- **Per seed:** each V checkpoint passes on its own, at +0.7539, +0.7824 and +0.7810.
- **Against FLIR:** FLIR's de-rolled 3-class headroom was +0.2066, a WEAK PASS (F115).
- **All-class headroom:** +0.6693. It is lower because person's floor is 0.3620, the only floor
  above 0.08. A visible-trained detector reads thermal people at 51% of its visible AP, but cars at
  8%.

### F159 — Both passive classes are passive GO, and no single seed could have flipped it

- **V − T against plan.md Q6's pre-registered 0.10:**

  | Class | V − T | Multiple of the 0.10 line | Worst V seed minus best T seed |
  | --- | --- | --- | --- |
  | car_stop | +0.2132 | 2.1× | +0.1928 |
  | color_cone | +0.3114 | 3.1× | +0.2616 |

- **Seed safety:** both worst-versus-best margins are above 0.10. A one-seed gate, the design Q7
  rejected, would have reached the same verdict.
- **2-class V − T:** +0.2623.

### F160 — The sensor gap is a passive-class gap: V − T averages +0.2623 on the passive classes against +0.0657 on the warm ones, 4.0×

- **Same detectors, frames and validation passes** as the passive figures.
- **Warm-class V − T:** car +0.0323, person +0.0650, bike +0.0999.
- **First of its kind:** this is the first per-class T on any dataset in this repo, and the pattern
  is the one the passive hypothesis predicts.
- **Bike sits on the line:** it is 0.0001 under 0.10, and its worst V seed beats its best T seed by
  only +0.0469. Its "domain-gap only" status cannot be told from passive GO.
- **Clearly below the line:** only car and person. The gate marks only primary classes, so bike's
  position changes no verdict.

### F161 — Two thirds of the passive headroom is domain gap

- **The 2-class +0.7725 splits** into T − J +0.5102 (66%) and V − T +0.2623 (34%).
- **Two bars for the campaign:**
  - Lifting the passive classes from J's 0.0526 up to T's 0.5628 is the zero-label route only.
  - Testing the passive claim needs the loop above 0.5628, towards V's 0.8251.

## Provenance caveats

- **The training logs were not read here.** The six trainings' wall clock, `best.pt` epochs and
  training SHA are absent from the files this record was written from. Those logs are being logged
  as a record of their own. The `git 8360dd9` line covers the gate run only.
- **The SHA has no `-dirty` check.** The runbook wrote `git rev-parse --short HEAD`, so a modified
  server tree would read the same.
- **`best.pt` is selected on the same 179-frame val the gate scores.** This is reasoned from
  ultralytics' fitness-on-val selection, not checked.
  - V and T each pick their best epoch on the split they are scored on. J does not: it reuses V's
    checkpoint, which was picked on visible val.
  - So V and T are somewhat optimistic, and T − J more so than V − T.
  - Record 013's judge carries the same property.
- **The plain mirror assumes MSRS is pixel-registered.** That is not measured on this tree.
  - car_stop boxes reach down to about 4×7 px (plan.md Q3).
  - A residual offset would depress T and J on exactly the passive classes, and so inflate V − T
    there.
- **The spreads are training-seed spread only, not val sampling.** Each passive class has only
  ~100–130 val instances.
- **`±` prints as `�` in the log.** This is a console encoding fault on the server, despite
  `[Console]::OutputEncoding` being UTF-8, and ultralytics' `ping: 0.1±0.0` lost its `±` entirely.
  All 24 mean ± std cells were re-derived from the CSV and match.

## Next

- **The `msrs-day` campaign is its own feature** (plan.md, Out of Scope).
- **The dose is unmeasured on `msrs-day`.** `grad_scale: 0.15` was calibrated on the custom set and
  FLIR (F144); whether it lands in the 20–30% band here is unknown.
- **Q8 still owes T arms on the other datasets:** FLIR (`yolo11s`), custom (`yolo11n`) and M3FD
  (both arms).
- **Prediction, from F160:** FLIR's three classes are all warm, so its T arm will give
  V − T < 0.10 on every class.
