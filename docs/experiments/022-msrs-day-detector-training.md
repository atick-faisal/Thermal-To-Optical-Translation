# 022 — msrs-day V and T yolo11s training: the six 100-epoch runs behind record 021's gate

**Date:** 2026-10-07 · **Task:** MSRS-PASSIVE-GATE-06 · **Machine:** Windows server, 2× A100 (`cuda:0` only, A100-SXM4-40GB) · **Wall clock:** 1.33 h of training by ultralytics' clock (V 0.201 / 0.239 / 0.234 h, T 0.201 / 0.230 / 0.224 h); run ends 13:43:32 → 14:59:09, first start not logged
**git_sha:** 8360dd9, all six (no `-dirty` check) · **W&B:** not recorded · **Log:** `logs/2026-10-07-msrs-passive-gate-train-{reference,thermal}-msrs-day-yolo11s-s{1,2,3}.txt`

## Command

The logs do not echo the command; this is the server runbook's, which the run followed. Each
trainer argument line confirms `data`, `seed`, `epochs=100`, `workers=8`, `batch=16`,
`deterministic=True` and `patience=100`.

```powershell
# one at a time on cuda:0; <arm> ∈ {reference (visible manifest), thermal (thermal manifest)}
uv run t2o train-detector --data runs/gate/msrs-day/manifests/<visible|thermal>/data.yaml `
    --init-weights yolo11s.pt --epochs 100 --seed <1|2|3> --workers 8 `
    --out runs/<reference|thermal>-msrs-day-yolo11s-s<seed> --device cuda:0
```

## Configuration

- **Arms.** V trains on the visible manifest and T on the thermal one, seeds 1–3 each, with an
  identical recipe otherwise. The optimizer is `optimizer=auto`, which resolved to
  `AdamW(lr=0.001111, momentum=0.9)` in every run.
- **Thermal labels.** T s1 wrote a fresh `infrared/labels.cache`. Train has 517 labelled frames and
  19 backgrounds (536 in all); val has 175 and 4 (179 in all). These match plan.md's 19 / 4
  daytime label-less negatives.
- **Validation.** 179 frames and 976 instances at every epoch, in all six runs.
- **One launch each.** Every log opens with a single `git 8360dd9` line and holds exactly 100 epoch
  blocks, so each run ran once, with no relaunch.

## Results

Each log's final-validation tail, verbatim, from `100 epochs completed` to the end of the file.

`logs/2026-10-07-msrs-passive-gate-train-reference-msrs-day-yolo11s-s1.txt`:

```
100 epochs completed in 0.201 hours.
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-msrs-day-yolo11s-s1\weights\last.pt, 19.2MB
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-msrs-day-yolo11s-s1\weights\best.pt, 19.2MB

Validating D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-msrs-day-yolo11s-s1\weights\best.pt...
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.5it/s 0.7s
                   all        179        976       0.87      0.712      0.788      0.519
Speed: 0.1ms preprocess, 0.7ms inference, 0.0ms loss, 1.2ms postprocess per image
2026-10-07 13:43:32,157 INFO    t2o.engine.detector_stage: detector reference-msrs-day-yolo11s-s1: P 0.8701  R 0.7120  mAP50 0.7877  mAP50-95 0.5188
2026-10-07 13:43:32,157 INFO    t2o.cli: detector weights: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-msrs-day-yolo11s-s1\weights\best.pt
2026-10-07 13:43:32,157 INFO    t2o.cli:   mAP50 0.7877  mAP50-95 0.5188
```

`logs/2026-10-07-msrs-passive-gate-train-reference-msrs-day-yolo11s-s2.txt`:

```
100 epochs completed in 0.239 hours.
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-msrs-day-yolo11s-s2\weights\last.pt, 19.2MB
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-msrs-day-yolo11s-s2\weights\best.pt, 19.2MB

Validating D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-msrs-day-yolo11s-s2\weights\best.pt...
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.9it/s 1.0s
                   all        179        976      0.852      0.728      0.799      0.519
Speed: 0.2ms preprocess, 0.8ms inference, 0.0ms loss, 1.8ms postprocess per image
2026-10-07 13:59:31,231 INFO    t2o.engine.detector_stage: detector reference-msrs-day-yolo11s-s2: P 0.8516  R 0.7275  mAP50 0.7994  mAP50-95 0.5185
2026-10-07 13:59:31,237 INFO    t2o.cli: detector weights: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-msrs-day-yolo11s-s2\weights\best.pt
2026-10-07 13:59:31,237 INFO    t2o.cli:   mAP50 0.7994  mAP50-95 0.5185
```

`logs/2026-10-07-msrs-passive-gate-train-reference-msrs-day-yolo11s-s3.txt`:

```
100 epochs completed in 0.234 hours.
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-msrs-day-yolo11s-s3\weights\last.pt, 19.2MB
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-msrs-day-yolo11s-s3\weights\best.pt, 19.2MB

Validating D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-msrs-day-yolo11s-s3\weights\best.pt...
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.6it/s 1.1s
                   all        179        976      0.829      0.743      0.792      0.516
Speed: 0.2ms preprocess, 1.0ms inference, 0.0ms loss, 1.8ms postprocess per image
2026-10-07 14:15:10,582 INFO    t2o.engine.detector_stage: detector reference-msrs-day-yolo11s-s3: P 0.8287  R 0.7426  mAP50 0.7922  mAP50-95 0.5162
2026-10-07 14:15:10,582 INFO    t2o.cli: detector weights: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-msrs-day-yolo11s-s3\weights\best.pt
2026-10-07 14:15:10,582 INFO    t2o.cli:   mAP50 0.7922  mAP50-95 0.5162
```

`logs/2026-10-07-msrs-passive-gate-train-thermal-msrs-day-yolo11s-s1.txt`:

```
100 epochs completed in 0.201 hours.
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\thermal-msrs-day-yolo11s-s1\weights\last.pt, 19.2MB
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\thermal-msrs-day-yolo11s-s1\weights\best.pt, 19.2MB

Validating D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\thermal-msrs-day-yolo11s-s1\weights\best.pt...
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 8.3it/s 0.7s
                   all        179        976      0.797      0.591      0.656      0.382
Speed: 0.1ms preprocess, 0.8ms inference, 0.0ms loss, 1.2ms postprocess per image
2026-10-07 14:28:49,627 INFO    t2o.engine.detector_stage: detector thermal-msrs-day-yolo11s-s1: P 0.7969  R 0.5913  mAP50 0.6562  mAP50-95 0.3819
2026-10-07 14:28:49,632 INFO    t2o.cli: detector weights: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\thermal-msrs-day-yolo11s-s1\weights\best.pt
2026-10-07 14:28:49,632 INFO    t2o.cli:   mAP50 0.6562  mAP50-95 0.3819
```

`logs/2026-10-07-msrs-passive-gate-train-thermal-msrs-day-yolo11s-s2.txt`:

```
100 epochs completed in 0.230 hours.
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\thermal-msrs-day-yolo11s-s2\weights\last.pt, 19.2MB
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\thermal-msrs-day-yolo11s-s2\weights\best.pt, 19.2MB

Validating D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\thermal-msrs-day-yolo11s-s2\weights\best.pt...
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 6.4it/s 0.9s
                   all        179        976      0.747       0.62       0.65      0.378
Speed: 0.2ms preprocess, 0.8ms inference, 0.0ms loss, 1.5ms postprocess per image
2026-10-07 14:44:10,770 INFO    t2o.engine.detector_stage: detector thermal-msrs-day-yolo11s-s2: P 0.7467  R 0.6204  mAP50 0.6502  mAP50-95 0.3783
2026-10-07 14:44:10,779 INFO    t2o.cli: detector weights: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\thermal-msrs-day-yolo11s-s2\weights\best.pt
2026-10-07 14:44:10,779 INFO    t2o.cli:   mAP50 0.6502  mAP50-95 0.3783
```

`logs/2026-10-07-msrs-passive-gate-train-thermal-msrs-day-yolo11s-s3.txt`:

```
100 epochs completed in 0.224 hours.
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\thermal-msrs-day-yolo11s-s3\weights\last.pt, 19.2MB
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\thermal-msrs-day-yolo11s-s3\weights\best.pt, 19.2MB

Validating D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\thermal-msrs-day-yolo11s-s3\weights\best.pt...
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 7.1it/s 0.8s
                   all        179        976      0.787      0.591       0.64      0.379
Speed: 0.2ms preprocess, 0.9ms inference, 0.0ms loss, 1.4ms postprocess per image
2026-10-07 14:59:09,510 INFO    t2o.engine.detector_stage: detector thermal-msrs-day-yolo11s-s3: P 0.7874  R 0.5905  mAP50 0.6403  mAP50-95 0.3795
2026-10-07 14:59:09,510 INFO    t2o.cli: detector weights: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\thermal-msrs-day-yolo11s-s3\weights\best.pt
2026-10-07 14:59:09,510 INFO    t2o.cli:   mAP50 0.6403  mAP50-95 0.3795
```


The per-epoch validation lines above each tail are in the same logs. This table is derived from them
and from the tails, alongside record 021's gate score for the same file:

| run | train h | best epoch (mAP50-95) | epoch-100 mAP50 | `best.pt` mAP50, train val | `best.pt` mAP50, gate (021) |
| --- | --- | --- | --- | --- | --- |
| V s1 | 0.201 | 90 | 0.781 | 0.7877 | 0.7841 |
| V s2 | 0.239 | 90 | 0.793 | 0.7994 | 0.7985 |
| V s3 | 0.234 | 100 | 0.793 | 0.7922 | 0.7934 |
| T s1 | 0.201 | 88 | 0.657 | 0.6562 | 0.6531 |
| T s2 | 0.230 | 95 (tied 97, 98) | 0.653 | 0.6502 | 0.6496 |
| T s3 | 0.224 | 89 (tied 93, 97) | 0.640 | 0.6403 | 0.6401 |

Re-render without retraining: the per-epoch rows are in each run's
`runs/<arm>-msrs-day-yolo11s-s<seed>/results.csv` on the server.

## Findings

### F162 — A 100-epoch `yolo11s` on msrs-day costs 0.20–0.24 h, 6–7× less than FLIR's judge

- **Cost.** All six runs together took 1.33 h by ultralytics' clock.
- **Launch to end.** Runs 2–6 span 75.6 min between consecutive end stamps, about 15 min a run
  including startup and final validation. That is at the bottom of the runbook's 15–20 min estimate.
- **Comparison.** FLIR's `yolo11s` judge took 1.43 h for 100 epochs (F98).
- **For planning.** A three-seed T arm on a dataset this size is about 0.7 GPU-h.

### F163 — `best.pt` comes late, and selecting it on val buys at most +0.008 all-class mAP50 over the epoch-100 weights

- **The best epoch.** Selection is by mAP50-95 alone: ultralytics 8.4.117's fitness weights are
  `[0, 0, 0, 1]` (`utils/metrics.py:1009`). The best epoch falls between 88 and 100 in all six
  runs.
- **The gain.** Best minus epoch 100 is +0.004 / +0.006 / 0.000 for V and −0.004 / +0.002 / +0.008
  for T.
- **Comparison.** FLIR's judge peaked at epoch 46, and its last 54 epochs lost 8.1% (F99). Nothing
  decays here.
- **What this settles.** Record 021 cautioned that `best.pt` is selected on the val split the gate
  scores. This bounds that gain, at all-class mAP50, to 26× under the smallest passive V − T
  (+0.2132).

### F164 — ultralytics' post-training val and `gate_table`'s re-score agree to within 0.0036 mAP50 on the same `best.pt`

- **The deltas.** Gate minus training val is −0.0036 / −0.0009 / +0.0012 for V and −0.0031 /
  −0.0006 / −0.0002 for T.
- **Comparison.** The largest is 16× under the 0.059 judge-to-judge noise floor (F13).
- **The cause is unchecked.** Training val ran in 6 batches and the gate in 12; whether that causes
  the deltas was not checked.

## Provenance caveats

- **F163 is all-class only.** Training runs with `verbose=False`, so no per-class AP is logged, and
  the selection gain on the 2-class primary is unmeasured.
- **T s2's and T s3's ties are at the log's three-decimal precision.** ultralytics compares full
  precision, so the true best epoch is one of the tied ones.
- **The V runs read an existing visible `labels.cache`.** No `Scanning … labels` line appears in
  their logs. The cache is presumably from the stopped first launch on the same rebuilt tree, and
  its 976 val instances match thermal's fresh cache. Its origin was not checked.
- **Only some classes start with pretrained head rows.** All six logs print
  `Remapped 2/5 cls head rows from pretrained weights by class name`. By reading
  `ultralytics/nn/tasks.py:375-399` (not by observation), the two are `car` and `person`; COCO's
  `bicycle` does not match `bike`. So F160's warm-vs-passive contrast sets classes that partly
  start pretrained against classes that never do. The effect is untested.
- **The run time is not set by modality.** V s1 and T s1 both took 0.201 h against 0.224–0.239 h
  for the rest; the split is unexplained.
- **The git_sha has no `-dirty` check**, as in record 021.

## Next

F162 sizes plan.md Q8's remaining T arms (FLIR `yolo11s`, custom `yolo11n`, M3FD). No new
prediction.
