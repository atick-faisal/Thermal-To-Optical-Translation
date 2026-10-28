# 032 — FLIR's 3-class Margin (ADR-06): the loop is not non-inferior, and the whole CI lies below −δ

**Date:** 2026-10-27 · **Task:** MODTR-BASELINE-10 · **Machine:** VSIBMEOT05 (`cuda:1`) · **Wall clock:** 1 h 37 min 3 s (17:48:26 → 19:25:29 +03:00): 16 min 58 s scoring the six exports, then 79 min 36 s of bootstrap
**git_sha:** `pre-spec-migration-193-g477c9f7` (`git describe --always --dirty`, so a clean tree at `477c9f7`) · **W&B:** none (no W&B line in the log) · **Log:** `logs/2026-10-27-modtr-baseline-06-non-inferiority-flir-3class.txt`, with `logs/2026-10-27-modtr-non-inferiority-flir-3class.csv`

## Command

From the provenance block, as run:

```
uv run python scripts/modtr_non_inferiority.py --loop-template runs/e3f-loop-s{seed}/stage3/translated --modtr-template runs/modtr-flir-s{seed}/stage0/translated --reference-weights runs/reference-flir-yolo11s/weights/best.pt --primary-classes bicycle car person --out runs/modtr-non-inferiority-flir-3class --device cuda:1
```

This is record 031's non-inferiority command with `--primary-classes bicycle car person` added and
a new `--out`. The script's default seeds, 0, 1 and 3, fill what the command leaves out. The log's
`data.yaml` lines confirm them.

## Environment

```
=== PROVENANCE ===
captured: 2026-10-27 17:48:26 +03:00
hostname: VSIBMEOT05
os: Microsoft Windows Server 2022 Standard 10.0.20348 (build 20348)
cpu: AMD EPYC 7763 64-Core Processor
cpu_logical_cores: 64
ram_total: 128.0 GB
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16; NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16
cuda_driver: 13.0
git_sha: pre-spec-migration-193-g477c9f7
working_dir: D:\Atick\GitHub\Thermal-To-Optical-Translation
command: uv run python scripts/modtr_non_inferiority.py --loop-template runs/e3f-loop-s{seed}/stage3/translated --modtr-template runs/modtr-flir-s{seed}/stage0/translated --reference-weights runs/reference-flir-yolo11s/weights/best.pt --primary-classes bicycle car person --out runs/modtr-non-inferiority-flir-3class --device cuda:1
python: 3.12.7
torch: 2.12.1+cu130
cuda_torch_build: 13.0
cudnn: 92000
cuda_available: True
gpu0_torch: NVIDIA A100-SXM4-40GB, 39.7 GiB, sm 8.0
=== END PROVENANCE ===
```

## Configuration

Nothing was trained. This run re-scores record 031's exports on the endpoint ADR-06 set for FLIR.

| Axis | Value | Where it is evidenced |
| --- | --- | --- |
| Endpoint | **3-class mAP@50 (bicycle, car, person)**, FLIR's primary since F94. ADR-06 set it after record 031's 4-class result was seen | `--primary-classes bicycle car person`; the CSV's `classes` column; `on bicycle, car, person` in the verdict line |
| Class restriction | each image's `ImageStats` restricted to the three classes before the bootstrap. Every image stays in the pool | `scripts/modtr_non_inferiority.py:119-131` (`_restrict`), `:287-288`; `pool` 1013 in the CSV |
| Non-inferiority test | ADR-02 as restated by ADR-06: paired 95% bootstrap over the val images on seed-averaged mAP@50, 10,000 resamples, bootstrap seed 0. Passes when the lower bound is above −5 points | `seed-averaged bootstrap: 3 seed(s) x 1013 images x 10000 resamples`; the CSV's `margin`, `resamples` and `bootstrap_seed` columns |
| Scorer | FLIR's reference `yolo11s`, `runs/reference-flir-yolo11s/weights/best.pt` | the command |
| Scored split | all 1,013 FLIR val images | `1013 images` on every evaluation |
| Loop arm | e3f loop (pix2pix, record 016), stage-3 exports, seeds 0, 1, 3 | the log's `data.yaml` lines |
| ModTr arm | record 031's exports, `runs/modtr-flir-s{0,1,3}/stage0/translated` | the log's `data.yaml` lines |

## Results

From the end of the provenance block to the finish stamp. The per-batch
`Class Images Instances Box(P ...` progress lines (287 of them), the ANSI colour escapes and the
blank lines are stripped. Everything else is verbatim:

```
2026-10-27 17:48:52,197 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s0\stage3\translated
2026-10-27 17:48:52,197 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s1\stage3\translated
2026-10-27 17:48:52,203 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s3\stage3\translated
2026-10-27 17:48:52,203 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\translated
2026-10-27 17:48:52,208 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\translated
2026-10-27 17:48:52,212 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\translated
2026-10-27 17:48:52,213 INFO    __main__: seed 0: scoring the loop and ModTr
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,348 parameters, 0 gradients, 21.4 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 1027.2546.5 MB/s, size: 612.7 KB)
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s0\stage3\translated\val\labels.cache... 1013 images, 0 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 1013/1013  0.0s
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s0\stage3\translated\val\images\FLIR_09042.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s0\stage3\translated\val\images\FLIR_09055.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s0\stage3\translated\val\images\FLIR_09653.png: 1 duplicate labels removed
                   all       1013       8601      0.531      0.342      0.375      0.179
Speed: 1.7ms preprocess, 1.5ms inference, 0.0ms loss, 1.3ms postprocess per image
2026-10-27 17:51:35,484 INFO    t2o.metrics.task: captured per-image stats for 1013 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s0\stage3\translated\data.yaml
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,348 parameters, 0 gradients, 21.4 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 1363.9136.1 MB/s, size: 538.9 KB)
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\translated\val\labels.cache... 1013 images, 0 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 1013/1013  0.0s
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\translated\val\images\FLIR_09042.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\translated\val\images\FLIR_09055.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\translated\val\images\FLIR_09653.png: 1 duplicate labels removed
                   all       1013       8601      0.512      0.372      0.413       0.19
Speed: 1.3ms preprocess, 1.3ms inference, 0.0ms loss, 1.2ms postprocess per image
2026-10-27 17:54:21,483 INFO    t2o.metrics.task: captured per-image stats for 1013 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\translated\data.yaml
2026-10-27 17:54:21,535 INFO    __main__: seed 1: scoring the loop and ModTr
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,348 parameters, 0 gradients, 21.4 GFLOPs
val: Fast image access  (ping: 0.10.2 ms, read: 1269.4499.8 MB/s, size: 606.4 KB)
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s1\stage3\translated\val\labels.cache... 1013 images, 0 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 1013/1013  0.0s
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s1\stage3\translated\val\images\FLIR_09042.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s1\stage3\translated\val\images\FLIR_09055.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s1\stage3\translated\val\images\FLIR_09653.png: 1 duplicate labels removed
                   all       1013       8601      0.483      0.314      0.356      0.169
Speed: 1.4ms preprocess, 1.7ms inference, 0.0ms loss, 1.3ms postprocess per image
2026-10-27 17:57:01,361 INFO    t2o.metrics.task: captured per-image stats for 1013 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s1\stage3\translated\data.yaml
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,348 parameters, 0 gradients, 21.4 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 1415.7291.3 MB/s, size: 528.1 KB)
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\translated\val\labels.cache... 1013 images, 0 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 1013/1013  0.0s
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\translated\val\images\FLIR_09042.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\translated\val\images\FLIR_09055.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\translated\val\images\FLIR_09653.png: 1 duplicate labels removed
                   all       1013       8601      0.758      0.365       0.41      0.197
Speed: 1.6ms preprocess, 1.6ms inference, 0.0ms loss, 1.5ms postprocess per image
2026-10-27 17:59:50,631 INFO    t2o.metrics.task: captured per-image stats for 1013 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\translated\data.yaml
2026-10-27 17:59:50,694 INFO    __main__: seed 3: scoring the loop and ModTr
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,348 parameters, 0 gradients, 21.4 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 1069.2396.1 MB/s, size: 620.4 KB)
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s3\stage3\translated\val\labels.cache... 1013 images, 0 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 1013/1013  0.0s
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s3\stage3\translated\val\images\FLIR_09042.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s3\stage3\translated\val\images\FLIR_09055.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s3\stage3\translated\val\images\FLIR_09653.png: 1 duplicate labels removed
                   all       1013       8601      0.471      0.285      0.319      0.152
Speed: 1.8ms preprocess, 1.5ms inference, 0.0ms loss, 1.4ms postprocess per image
2026-10-27 18:02:48,446 INFO    t2o.metrics.task: captured per-image stats for 1013 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s3\stage3\translated\data.yaml
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,348 parameters, 0 gradients, 21.4 GFLOPs
val: Fast image access  (ping: 0.20.0 ms, read: 925.8118.9 MB/s, size: 574.4 KB)
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\translated\val\labels.cache... 1013 images, 0 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 1013/1013  0.0s
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\translated\val\images\FLIR_09042.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\translated\val\images\FLIR_09055.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\translated\val\images\FLIR_09653.png: 1 duplicate labels removed
                   all       1013       8601      0.749      0.366        0.4      0.188
Speed: 1.5ms preprocess, 1.4ms inference, 0.0ms loss, 1.3ms postprocess per image
2026-10-27 18:05:50,983 INFO    t2o.metrics.task: captured per-image stats for 1013 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\translated\data.yaml
2026-10-27 19:25:26,595 INFO    t2o.analysis.resolvability: seed-averaged bootstrap: 3 seed(s) x 1013 images x 10000 resamples, delta CI [-9.87, -7.27] mAP points
2026-10-27 19:25:27,086 INFO    __main__: wrote D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-non-inferiority-flir-3class\modtr-non-inferiority.csv
2026-10-27 19:25:27,086 INFO    __main__: per-seed deltas (loop - ModTr): s0 -7.32, s1 -7.81, s3 -10.55; sd 1.74 mAP points
2026-10-27 19:25:27,086 INFO    __main__: verdict: NOT NON-INFERIOR -- lower bound -9.87 <= -5.0 mAP points (delta -8.56) on bicycle, car, person
=== FINISHED 2026-10-27 19:25:29 +03:00 ===
```

The six ultralytics `all` rows, in log order. These are **4-class**: the class restriction applies
to the script's per-image stats, not to ultralytics' own summary line. The run label at the start
of each row is added here, from the `captured per-image stats ... from <path>` line that follows it
in the log. The rest of each row is verbatim:

```
e3f-loop-s0     all       1013       8601      0.531      0.342      0.375      0.179
modtr-flir-s0   all       1013       8601      0.512      0.372      0.413       0.19
e3f-loop-s1     all       1013       8601      0.483      0.314      0.356      0.169
modtr-flir-s1   all       1013       8601      0.758      0.365       0.41      0.197
e3f-loop-s3     all       1013       8601      0.471      0.285      0.319      0.152
modtr-flir-s3   all       1013       8601      0.749      0.366        0.4      0.188
```

All six rows match record 031's rows to every printed digit.

`logs/2026-10-27-modtr-non-inferiority-flir-3class.csv`, verbatim:

```
seed,pool,loop_map50,modtr_map50,delta,low,high,half_width,seed_sd,margin,non_inferior,resamples,bootstrap_seed,loop_data,modtr_data,weights,classes
0,1013,0.477354271266942,0.5505445635163094,-7.319029224936741,,,,,5.0,,10000,0,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s0\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\translated\data.yaml,runs\reference-flir-yolo11s\weights\best.pt,bicycle car person
1,1013,0.4668874675809838,0.544990049013342,-7.810258143235821,,,,,5.0,,10000,0,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s1\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\translated\data.yaml,runs\reference-flir-yolo11s\weights\best.pt,bicycle car person
3,1013,0.4250702294869515,0.5305715353530983,-10.550130586614682,,,,,5.0,,10000,0,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s3\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\translated\data.yaml,runs\reference-flir-yolo11s\weights\best.pt,bicycle car person
mean,1013,0.45643732277829246,0.5420353826275832,-8.559805984929081,-9.867190158761293,-7.2664671325177235,1.3003615131217847,1.7410831334544394,5.0,False,10000,0,runs/e3f-loop-s{seed}/stage3/translated,runs/modtr-flir-s{seed}/stage0/translated,runs\reference-flir-yolo11s\weights\best.pt,bicycle car person
```

Re-render the verdict without re-training, on the server: the command above with a new `--out`,
since the script refuses to overwrite its CSV.

## Findings

### F210 — FLIR's Margin fails on its 3-class primary too: loop − ModTr −8.56 mAP@50 points, 95% CI [−9.87, −7.27], every value below −δ

ADR-06 moved FLIR's Margin endpoint to 3-class mAP@50 and kept everything else in ADR-02. The pass
rule is unchanged: the lower bound of the seed-averaged paired 95% CI on loop − ModTr must be above
−5 points. On 3-class it is −9.87. The comparison is record 031's 4-class verdict on the same
exports, the same judge and the same bootstrap seed:

| endpoint | seed-averaged delta | 95% CI | half-width | per-seed deltas (s0, s1, s3) | seed sd |
| --- | --- | --- | --- | --- | --- |
| 4-class (record 031, F205) | −5.80 | [−7.00, −3.65] | 1.67 | −3.87, −5.42, −8.10 | 2.14 |
| **3-class (this run, ADR-06)** | **−8.56** | **[−9.87, −7.27]** | **1.30** | −7.32, −7.81, −10.55 | 1.74 |

- **The verdict does not change: NOT NON-INFERIOR.** The switch ADR-06 made after seeing the
  4-class result rescues nothing, as its consequences section anticipated.
- **Now the whole CI lies below −δ.** On 4-class the upper end, −3.65, was above −5, so the loop
  was "not shown non-inferior" and no more. On 3-class the upper end is −7.27. At 95%, the loop
  trails ModTr by more than the 5-point margin on FLIR's own primary endpoint.
- **Every seed's gap widened.** The per-seed deltas moved by −3.44, −2.39 and −2.45 points.
- **The loop's 3-class values are 0.4774, 0.4669 and 0.4251. ModTr's are 0.5505, 0.5450 and
  0.5306.** ModTr is ahead at every seed, by at least 7.32 points.
- **The custom set is unaffected.** It keeps its all-class endpoint (ADR-06), so record 030's F200
  stands as written.

### F211 — Record 031's prediction holds: ModTr's `yolo11s` 3-class mean is 0.5420 against the loop's 0.4564, and the dog AP that could have reversed it is 0.0049, not ≥ 0.26

Record 031 posted a prediction before this run: ModTr's 3-class seed mean would be above the loop's
0.4564. It is 0.5420. This run's loop mean, 0.4564, reproduces the figure record 031 took from
record 026.

F206 said the 3-class gap could reach zero only if ModTr's `yolo11s` dog AP averaged at least
0.26. No log prints `yolo11s` per-class AP, so the dog AP is backed out by arithmetic:
dog = 4 × (4-class mAP) − 3 × (3-class mAP). The 4-class values are record 031's CSV, and the
3-class values are this run's. The formula holds because `map50_from_stats` takes the plain mean of
`ap_per_class` over the classes in the targets (`src/t2o/metrics/task.py:256-262`), and all four
classes are present in FLIR's val split.

| arm | dog AP s0 / s1 / s3 (derived) | mean | 4-class → 3-class lift |
| --- | --- | --- | --- |
| e3f loop | 0.0664 / 0.0220 / 0.0009 | 0.0298 | +10.67 points |
| ModTr | 0.0018 / 0.0044 / 0.0085 | **0.0049** | +13.43 points |

- **The derivation checks out against an independent source.** The loop's derived 0.0298 equals
  the `yolo11s` dog AP that F206 cites from record 026's per-class columns.
- **Dog does not carry the gap under `yolo11s`.** Under RT-DETR, F206 had already shown this. Both
  arms score dog near zero, so dropping it lifts both, and lifts ModTr more.
- **The 2.76-point widening splits into two parts.** Write the 3-class gap as
  (4 × 4-class gap − loop-minus-ModTr dog gap) / 3. Rescaling a gap from four classes to three moves
  it from −5.80 to −7.73, which is −1.93 points. Removing the loop's small dog lead (0.0298 against
  0.0049, so +2.49 points of AP) costs a further −0.83.
- **The two judges now agree on FLIR's primary endpoint.** The `yolo11s` 3-class gap is −8.56. The
  RT-DETR 3-class gap is −10.83 (F206, a point estimate without a CI).

### F212 — The 3-class interval is the tighter one: half-width 1.30 against 4-class's 1.67, and seed sd 1.74 against 2.14

The comparison is record 031's 4-class run, which used the same 1,013 images, seeds, judge,
resamples and bootstrap seed (the table under F210).

- **The half-width fell 22%** (1.67 → 1.30) and the seed sd fell 19% (2.14 → 1.74).
- **It fell even though the gap grew by about half** (−5.80 → −8.56).
- **Rescaling alone would have pushed the other way.** Dividing by three classes instead of four
  scales any per-class difference up by 4/3. So the dog class contributed more resampling noise
  than its quarter share. That is consistent with F94's reason for making 3-class FLIR's primary,
  but this run does not isolate dog's contribution, so the mechanism is a reading of the numbers,
  not a measurement of it.

## Provenance caveats

- **The endpoint is post hoc.** ADR-06 was written after record 031's 4-class FAIL was seen. Two
  things weigh against reading it as goalpost-moving. First, 3-class was FLIR's primary before
  ModTr existed (F94). Second, record 031 posted a directional prediction before this run, and the
  switch made the loop's result worse, not better. Both verdicts are reported, as ADR-06 requires.
- **The log's task segment says `-06-`, but the run belongs to MODTR-BASELINE-10.** The filename
  otherwise follows the `logs/<date>-<task>-<what>.txt` convention. `-10` added `--primary-classes`
  (ADR-06), and `-06` owns the campaign configs. It is cited here as saved.
- **The CSV carries no task segment in its name** (`2026-10-27-modtr-non-inferiority-flir-3class.csv`),
  like record 031's. It is the server's
  `runs/modtr-non-inferiority-flir-3class/modtr-non-inferiority.csv`, copied back and renamed.
- **The dog APs in F211 are derived, not logged.** They come from arithmetic across two CSVs,
  checked against one independent value (the loop's 0.0298).
- **Every ultralytics `all` row is 4-class.** Only the CSV and the bootstrap line are 3-class. The
  six rows equal record 031's. That is the same exports re-scored, not second evidence.
- **Every label scan read the `labels.cache` that record 031's evaluations created**
  (`labels.cache... 0.0s`). No label changed between the two runs, so the cache is current.
- **Inherited from record 031, unchanged:**
  - the arms are not compute-matched (pix2pix 4 × 100 epochs at `grad_scale` 0.15, against ModTr's
    339 detection-only epochs);
  - FLIR's pix2pix arms carry record 016's roll confound (F119), which cannot reach ModTr;
  - the loop's exports come from record 016's campaign at an earlier SHA;
  - three val images lose one duplicate label each in both arms, so the pairing holds.

## Next

FLIR's Margin is now settled on both endpoints. It fails on 4-class (F205). On the 3-class primary
the loop trails ModTr by more than δ (F210). Plan Q7 still has the custom set deciding Margin, where
the loop passes (F200). FLIR is its public counterpart.

- **The paper has to report both FLIR verdicts and the order they were decided in** (ADR-06). It
  also has to say why the two datasets disagree. The leading reading is still the roll confound
  (F119), which reaches pix2pix's reconstruction objective and not ModTr's. This run does not test
  that.
- **No further Margin run on FLIR is suggested by this result.** A third endpoint would be the
  goalpost-moving ADR-06 was careful to avoid.
