# 040 — Registration-error axis, degradation report: on custom, image CIs resolve the zero-shot drop at 3, 12 and 24 px and the adapted arm's only at 24 px; on FLIR's 3-class primary the collapsed 0 px sits below every other level, and label transfer leads translation by more than it did all-class

**Date:** 2026-11-15 · **Task:** REGISTRATION-AXIS-07 · **Machine:** VSIBMEOT05, 2× A100 (custom on `cuda:0`, FLIR on `cuda:1`, run concurrently) · **Wall clock:** custom 48 min 33 s (13:46:19 → 14:34:52 +03:00); FLIR 6 h 05 min 46 s (13:46:33 → 19:52:19 +03:00)
**git_sha:** `pre-spec-migration-287-gf041f87` (both; clean, no `-dirty` suffix) · **W&B:** none (no W&B line in either log) · **Log:** `logs/2026-11-15-REGISTRATION-AXIS-07-report-custom.txt`, `logs/2026-11-15-REGISTRATION-AXIS-07-report-flir.txt`; results CSVs `logs/2026-11-15-REGISTRATION-AXIS-07-report-custom.csv`, `logs/2026-11-15-REGISTRATION-AXIS-07-report-flir.csv`

This is the read-out of goal C2's registration-error axis (legacy C4). It trains nothing. It
re-scores the stage-3 exports of record 037's translation runs per image, and takes record 036's
label-transfer sweep as it stands. Every translation point gets an image-bootstrap CI of its own
mAP50, and a paired image-bootstrap CI of its delta to the 0 px run, both scored on the same draw.
Record 037 asked for both: the custom zero-shot curve needed image CIs before any level below 24 px
could be called a drop (F229), and FLIR's translation half needed its 3-class primary so both FLIR
halves share one metric.

## Command

```
uv run python scripts/registration_report.py --dataset custom --sweep-csv runs/registration-custom/registration-tidy.csv --runs runs/regc-* runs/e3b-loop-s0 --seed-runs runs/e3b-loop-s* --out runs/registration-report --device cuda:0
```

```
uv run python scripts/registration_report.py --dataset flir --sweep-csv runs/registration-flir/registration-tidy.csv --runs runs/regf-* --primary-classes bicycle car person --out runs/registration-report --device cuda:1
```

## Environment

custom:

```
=== PROVENANCE ===
captured: 2026-11-15 13:46:19 +03:00
hostname: VSIBMEOT05
os: Microsoft Windows Server 2022 Standard 10.0.20348 (build 20348)
cpu: AMD EPYC 7763 64-Core Processor
cpu_logical_cores: 64
ram_total: 128.0 GB
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16; NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16
cuda_driver: 13.0
git_sha: pre-spec-migration-287-gf041f87
working_dir: D:\Atick\GitHub\Thermal-To-Optical-Translation
command: uv run python scripts/registration_report.py --dataset custom --sweep-csv runs/registration-custom/registration-tidy.csv --runs runs/regc-* runs/e3b-loop-s0 --seed-runs runs/e3b-loop-s* --out runs/registration-report --device cuda:0
python: 3.12.7
torch: 2.12.1+cu130
cuda_torch_build: 13.0
cudnn: 92000
cuda_available: True
gpu0_torch: NVIDIA A100-SXM4-40GB, 39.7 GiB, sm 8.0
=== END PROVENANCE ===
```

FLIR:

```
=== PROVENANCE ===
captured: 2026-11-15 13:46:33 +03:00
hostname: VSIBMEOT05
os: Microsoft Windows Server 2022 Standard 10.0.20348 (build 20348)
cpu: AMD EPYC 7763 64-Core Processor
cpu_logical_cores: 64
ram_total: 128.0 GB
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16; NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16
cuda_driver: 13.0
git_sha: pre-spec-migration-287-gf041f87
working_dir: D:\Atick\GitHub\Thermal-To-Optical-Translation
command: uv run python scripts/registration_report.py --dataset flir --sweep-csv runs/registration-flir/registration-tidy.csv --runs runs/regf-* --primary-classes bicycle car person --out runs/registration-report --device cuda:1
python: 3.12.7
torch: 2.12.1+cu130
cuda_torch_build: 13.0
cudnn: 92000
cuda_available: True
gpu0_torch: NVIDIA A100-SXM4-40GB, 39.7 GiB, sm 8.0
=== END PROVENANCE ===
```

## Configuration

| Axis | custom | FLIR |
| --- | --- | --- |
| Label-transfer half | record 036's sweep CSV, 5 levels × seeds 0–2, all-class mAP50 | record 036's sweep CSV, 5 levels × seeds 0–2, `primary_map50` (bicycle, car, person) |
| Translation runs | `regc-{03,06,12,24}px-s0` and, as 0 px, `e3b-loop-s0` (record 037), stage 3 | `regf-{00,03,06,12,24}px-s0` (record 037), stage 3 |
| Zero-shot judge | each run's config snapshot: the custom reference `yolo11s` | each run's config snapshot: the FLIR reference `yolo11s` |
| Adapted arm | the run's own fine-tuned `yolo11n` on its translated val | same |
| Classes scored | all (4) | `--primary-classes bicycle car person`, applied before every translation mAP |
| Seed yardstick | `--seed-runs runs/e3b-loop-s*`, 6 aligned seeds | none |
| Bootstrap | 10,000 resamples, n-out-of-n over 153 val images, `--bootstrap-seed` 0 (default) | 10,000 resamples over 1,013 val images, seed 0 |
| Collapse flag | stage-3 FID > 200 (`--collapse-fid` default) | same |
| Zero-shot check | each re-score must reproduce `metrics.json`'s `zero_shot.map50` or the script stops | same |
| Config hash | not printed | not printed |

The ultralytics `all` rows below are the all-class val pass. On FLIR that is four classes with dog;
the result block's FLIR translation numbers are the 3-class primary, computed from the same
per-image records.

## Results

custom, each val pass's scoring line and its ultralytics `all` row, then the result block:

```
2026-11-15 13:46:47,344 INFO    __main__: custom: 5 label-transfer level(s), 5 translation run(s) at 0, 3, 6, 12, 24 px, 10000 resamples
2026-11-15 13:46:47,344 INFO    __main__: 0 px zero_shot: scoring runs\e3b-loop-s0\stage3\translated\data.yaml
                   all        153        423      0.855      0.783      0.855      0.568
2026-11-15 13:49:44,254 INFO    __main__: 0 px adapted: scoring runs\e3b-loop-s0\stage3\translated\data.yaml
                   all        153        423      0.926       0.85      0.914      0.628
2026-11-15 13:52:42,114 INFO    __main__: 3 px zero_shot: scoring runs\regc-03px-s0\stage3\translated\data.yaml
                   all        153        423      0.813      0.711      0.768      0.474
2026-11-15 13:55:48,203 INFO    __main__: 3 px adapted: scoring runs\regc-03px-s0\stage3\translated\data.yaml
                   all        153        423      0.904      0.871      0.924      0.641
2026-11-15 13:58:49,188 INFO    __main__: 6 px zero_shot: scoring runs\regc-06px-s0\stage3\translated\data.yaml
                   all        153        423      0.846      0.765      0.826      0.555
2026-11-15 14:01:50,530 INFO    __main__: 6 px adapted: scoring runs\regc-06px-s0\stage3\translated\data.yaml
                   all        153        423      0.917      0.837      0.905       0.63
2026-11-15 14:04:55,706 INFO    __main__: 12 px zero_shot: scoring runs\regc-12px-s0\stage3\translated\data.yaml
                   all        153        423      0.857      0.718      0.797      0.511
2026-11-15 14:07:56,969 INFO    __main__: 12 px adapted: scoring runs\regc-12px-s0\stage3\translated\data.yaml
                   all        153        423      0.913      0.822      0.884      0.585
2026-11-15 14:11:03,918 INFO    __main__: 24 px zero_shot: scoring runs\regc-24px-s0\stage3\translated\data.yaml
                   all        153        423      0.785      0.697      0.738      0.425
2026-11-15 14:14:12,964 INFO    __main__: 24 px adapted: scoring runs\regc-24px-s0\stage3\translated\data.yaml
                   all        153        423      0.918      0.767      0.838      0.463
2026-11-15 14:17:19,580 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\regc-24px-s0\stage3\translated\data.yaml
2026-11-15 14:20:20,921 INFO    t2o.analysis.resolvability: paired image bootstrap: 153 of 153 images x 10000 resamples, delta CI [-13.12, -3.74] mAP points
2026-11-15 14:22:29,394 INFO    t2o.analysis.resolvability: paired image bootstrap: 153 of 153 images x 10000 resamples, delta CI [-6.76, 0.36] mAP points
2026-11-15 14:24:39,709 INFO    t2o.analysis.resolvability: paired image bootstrap: 153 of 153 images x 10000 resamples, delta CI [-10.68, -1.49] mAP points
2026-11-15 14:26:46,451 INFO    t2o.analysis.resolvability: paired image bootstrap: 153 of 153 images x 10000 resamples, delta CI [-15.20, -7.68] mAP points
2026-11-15 14:29:14,408 INFO    t2o.analysis.resolvability: paired image bootstrap: 153 of 153 images x 10000 resamples, delta CI [-0.80, 3.60] mAP points
2026-11-15 14:31:08,995 INFO    t2o.analysis.resolvability: paired image bootstrap: 153 of 153 images x 10000 resamples, delta CI [-3.13, 1.38] mAP points
2026-11-15 14:32:55,929 INFO    t2o.analysis.resolvability: paired image bootstrap: 153 of 153 images x 10000 resamples, delta CI [-5.83, 0.09] mAP points
2026-11-15 14:34:50,218 INFO    t2o.analysis.resolvability: paired image bootstrap: 153 of 153 images x 10000 resamples, delta CI [-10.30, -4.35] mAP points
2026-11-15 14:34:50,229 INFO    __main__: === custom: mAP50, all classes against misalignment px ===
2026-11-15 14:34:50,229 INFO    __main__: label_transfer     0 px  0.9311  sd 0.0040 (n=3)  delta +0.0000
2026-11-15 14:34:50,229 INFO    __main__: label_transfer     3 px  0.9303  sd 0.0017 (n=3)  delta -0.0008
2026-11-15 14:34:50,229 INFO    __main__: label_transfer     6 px  0.9301  sd 0.0030 (n=3)  delta -0.0010
2026-11-15 14:34:50,229 INFO    __main__: label_transfer    12 px  0.9158  sd 0.0053 (n=3)  delta -0.0153
2026-11-15 14:34:50,229 INFO    __main__: label_transfer    24 px  0.8582  sd 0.0206 (n=3)  delta -0.0729
2026-11-15 14:34:50,229 INFO    __main__: zero_shot          0 px  0.8548  CI [79.60, 90.30] pts  0 px point  FID 88.10
2026-11-15 14:34:50,229 INFO    __main__:                         aligned seed set: sd 0.0140 over 6 seeds (not in the CI above)
2026-11-15 14:34:50,229 INFO    __main__: zero_shot          3 px  0.7682  CI [69.13, 83.97] pts  delta -0.0866 [-13.12, -3.74] pts  FID 97.71
2026-11-15 14:34:50,229 INFO    __main__: zero_shot          6 px  0.8256  CI [74.82, 88.54] pts  delta -0.0292 [-6.76, +0.36] pts  FID 97.13
2026-11-15 14:34:50,229 INFO    __main__: zero_shot         12 px  0.7967  CI [71.94, 85.94] pts  delta -0.0581 [-10.68, -1.49] pts  FID 133.52
2026-11-15 14:34:50,229 INFO    __main__: zero_shot         24 px  0.7383  CI [67.12, 80.10] pts  delta -0.1165 [-15.20, -7.68] pts  FID 140.79
2026-11-15 14:34:50,229 INFO    __main__: adapted            0 px  0.9145  CI [86.97, 94.90] pts  0 px point  FID 88.10
2026-11-15 14:34:50,229 INFO    __main__:                         aligned seed set: sd 0.0050 over 6 seeds (not in the CI above)
2026-11-15 14:34:50,229 INFO    __main__: adapted            3 px  0.9242  CI [89.21, 95.48] pts  delta +0.0097 [-0.80, +3.60] pts  FID 97.71
2026-11-15 14:34:50,229 INFO    __main__: adapted            6 px  0.9053  CI [85.78, 94.21] pts  delta -0.0092 [-3.13, +1.38] pts  FID 97.13
2026-11-15 14:34:50,229 INFO    __main__: adapted           12 px  0.8844  CI [83.61, 92.78] pts  delta -0.0301 [-5.83, +0.09] pts  FID 133.52
2026-11-15 14:34:50,229 INFO    __main__: adapted           24 px  0.8377  CI [78.73, 88.77] pts  delta -0.0768 [-10.30, -4.35] pts  FID 140.79
2026-11-15 14:34:50,229 INFO    __main__: translation CIs are over val images at one training seed; they carry no seed noise
2026-11-15 14:34:50,229 INFO    __main__: Neither 0 px is truly zero. The custom set's thermal boxes were moved from visible by the sister
2026-11-15 14:34:50,229 INFO    __main__: repo's registration, so every level sits on that error. FLIR keeps about 2.25 px of median
2026-11-15 14:34:50,229 INFO    __main__: per-pair scatter after calibration (record 012).
2026-11-15 14:34:50,229 INFO    __main__: wrote D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-report\custom-registration-curve.csv
=== FINISHED 2026-11-15 14:34:52 +03:00 ===
```

FLIR, the same lines:

```
2026-11-15 13:47:02,449 INFO    __main__: flir: 5 label-transfer level(s), 5 translation run(s) at 0, 3, 6, 12, 24 px, 10000 resamples
2026-11-15 13:47:02,449 INFO    __main__: 0 px zero_shot: scoring runs\regf-00px-s0\stage3\translated\data.yaml
                   all       1013       8601      0.578      0.119      0.121     0.0581
2026-11-15 13:50:07,922 INFO    __main__: 0 px adapted: scoring runs\regf-00px-s0\stage3\translated\data.yaml
                   all       1013       8601      0.473      0.343      0.374      0.187
2026-11-15 13:53:13,354 INFO    __main__: 3 px zero_shot: scoring runs\regf-03px-s0\stage3\translated\data.yaml
                   all       1013       8601      0.599      0.308      0.342      0.166
2026-11-15 13:56:23,860 INFO    __main__: 3 px adapted: scoring runs\regf-03px-s0\stage3\translated\data.yaml
                   all       1013       8601      0.509      0.419       0.43      0.215
2026-11-15 13:59:29,929 INFO    __main__: 6 px zero_shot: scoring runs\regf-06px-s0\stage3\translated\data.yaml
                   all       1013       8601      0.419      0.251       0.26      0.125
2026-11-15 14:02:40,943 INFO    __main__: 6 px adapted: scoring runs\regf-06px-s0\stage3\translated\data.yaml
                   all       1013       8601      0.485      0.386      0.413      0.205
2026-11-15 14:05:48,881 INFO    __main__: 12 px zero_shot: scoring runs\regf-12px-s0\stage3\translated\data.yaml
                   all       1013       8601      0.457      0.254      0.289       0.13
2026-11-15 14:09:01,008 INFO    __main__: 12 px adapted: scoring runs\regf-12px-s0\stage3\translated\data.yaml
                   all       1013       8601      0.461       0.36      0.387      0.177
2026-11-15 14:12:14,305 INFO    __main__: 24 px zero_shot: scoring runs\regf-24px-s0\stage3\translated\data.yaml
                   all       1013       8601       0.62      0.209      0.214     0.0857
2026-11-15 14:15:27,410 INFO    __main__: 24 px adapted: scoring runs\regf-24px-s0\stage3\translated\data.yaml
                   all       1013       8601      0.416      0.291      0.302      0.119
2026-11-15 14:18:37,169 INFO    t2o.metrics.task: captured per-image stats for 1013 images from runs\regf-24px-s0\stage3\translated\data.yaml
2026-11-15 14:52:45,075 INFO    t2o.analysis.resolvability: paired image bootstrap: 1013 of 1013 images x 10000 resamples, delta CI [23.99, 27.77] mAP points
2026-11-15 15:22:47,739 INFO    t2o.analysis.resolvability: paired image bootstrap: 1013 of 1013 images x 10000 resamples, delta CI [16.99, 20.04] mAP points
2026-11-15 15:54:16,056 INFO    t2o.analysis.resolvability: paired image bootstrap: 1013 of 1013 images x 10000 resamples, delta CI [20.17, 23.90] mAP points
2026-11-15 16:32:53,423 INFO    t2o.analysis.resolvability: paired image bootstrap: 1013 of 1013 images x 10000 resamples, delta CI [10.77, 13.68] mAP points
2026-11-15 17:21:55,473 INFO    t2o.analysis.resolvability: paired image bootstrap: 1013 of 1013 images x 10000 resamples, delta CI [3.70, 6.95] mAP points
2026-11-15 18:00:42,660 INFO    t2o.analysis.resolvability: paired image bootstrap: 1013 of 1013 images x 10000 resamples, delta CI [4.08, 7.23] mAP points
2026-11-15 18:48:57,251 INFO    t2o.analysis.resolvability: paired image bootstrap: 1013 of 1013 images x 10000 resamples, delta CI [-0.10, 3.53] mAP points
2026-11-15 19:52:18,040 INFO    t2o.analysis.resolvability: paired image bootstrap: 1013 of 1013 images x 10000 resamples, delta CI [-11.10, -7.46] mAP points
2026-11-15 19:52:18,061 INFO    __main__: === flir: mAP50 over bicycle car person against misalignment px ===
2026-11-15 19:52:18,061 INFO    __main__: label_transfer     0 px  0.7764  seeds identical (n=3)  delta +0.0000
2026-11-15 19:52:18,061 INFO    __main__: label_transfer     3 px  0.7764  seeds identical (n=3)  delta +0.0001
2026-11-15 19:52:18,061 INFO    __main__: label_transfer     6 px  0.7631  seeds identical (n=3)  delta -0.0132
2026-11-15 19:52:18,061 INFO    __main__: label_transfer    12 px  0.7185  seeds identical (n=3)  delta -0.0579
2026-11-15 19:52:18,061 INFO    __main__: label_transfer    24 px  0.5547  seeds identical (n=3)  delta -0.2217
2026-11-15 19:52:18,061 INFO    __main__: zero_shot          0 px  0.1616  CI [15.42, 17.07] pts  0 px point  FID 227.62  COLLAPSED (FID > 200)
2026-11-15 19:52:18,061 INFO    __main__: zero_shot          3 px  0.4200  CI [40.15, 44.09] pts  delta +0.2584 [+23.99, +27.77] pts  FID 94.76
2026-11-15 19:52:18,061 INFO    __main__: zero_shot          6 px  0.3461  CI [33.12, 36.39] pts  delta +0.1844 [+16.99, +20.04] pts  FID 163.96
2026-11-15 19:52:18,061 INFO    __main__: zero_shot         12 px  0.3817  CI [36.31, 40.20] pts  delta +0.2201 [+20.17, +23.90] pts  FID 148.46
2026-11-15 19:52:18,061 INFO    __main__: zero_shot         24 px  0.2839  CI [26.99, 29.91] pts  delta +0.1223 [+10.77, +13.68] pts  FID 162.91
2026-11-15 19:52:18,061 INFO    __main__: adapted            0 px  0.4950  CI [47.52, 51.52] pts  0 px point  FID 227.62  COLLAPSED (FID > 200)
2026-11-15 19:52:18,061 INFO    __main__: adapted            3 px  0.5474  CI [52.76, 56.90] pts  delta +0.0524 [+3.70, +6.95] pts  FID 94.76
2026-11-15 19:52:18,061 INFO    __main__: adapted            6 px  0.5504  CI [53.19, 57.15] pts  delta +0.0555 [+4.08, +7.23] pts  FID 163.96
2026-11-15 19:52:18,061 INFO    __main__: adapted           12 px  0.5122  CI [49.32, 53.17] pts  delta +0.0172 [-0.10, +3.53] pts  FID 148.46
2026-11-15 19:52:18,061 INFO    __main__: adapted           24 px  0.4010  CI [38.56, 41.97] pts  delta -0.0940 [-11.10, -7.46] pts  FID 162.91
2026-11-15 19:52:18,061 INFO    __main__: label transfer: the seeds gave identical results at 5 of 5 levels, so it is n = 1 in effect there, with no seed band (F227)
2026-11-15 19:52:18,061 INFO    __main__: translation: the translator collapsed at 0 px (stage FID > 200, F231). One seed with collapses draws points, not a curve; a delta to a collapsed 0 px is not a drop
2026-11-15 19:52:18,061 INFO    __main__: translation CIs are over val images at one training seed; they carry no seed noise
2026-11-15 19:52:18,061 INFO    __main__: Neither 0 px is truly zero. The custom set's thermal boxes were moved from visible by the sister
2026-11-15 19:52:18,061 INFO    __main__: repo's registration, so every level sits on that error. FLIR keeps about 2.25 px of median
2026-11-15 19:52:18,061 INFO    __main__: per-pair scatter after calibration (record 012).
2026-11-15 19:52:18,061 INFO    __main__: wrote D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-report\flir-registration-curve.csv
=== FINISHED 2026-11-15 19:52:19 +03:00 ===
```

Between the scoring lines, each log holds ultralytics' progress bars and load lines, cut here and
kept in full in the logs. The CSVs carry the same rows at full precision.

Re-render without re-running: none. The script re-scores every export and refuses an existing CSV.
The two CSVs are the results files.

## Findings

### F240 — On custom, image CIs resolve zero-shot's drop from the aligned seed at 3, 12 and 24 px but not at 6 px; the 3 px point that broke monotonicity in F229 is clear of image noise

Against `e3b-loop-s0` (aligned, same judge, paired on the same 153 val images), stage-3 zero-shot
changes by −0.0866 [−13.12, −3.74], −0.0292 [−6.76, +0.36], −0.0581 [−10.68, −1.49] and
−0.1165 [−15.20, −7.68] mAP points at 3 / 6 / 12 / 24 px. Three of four CIs exclude zero. The 6 px
CI reaches +0.36. The point estimates reproduce record 037's exactly, and the script checks that.

The CIs carry no seed noise. The aligned seed set's sd is 0.0140 over 6 seeds, the same as record
007's. Against it, the drops are 6.2, 2.1, 4.1 and 8.3 sd. So 3, 12 and 24 px are not image noise at
seed 0, and the 3 px point still reads below 6 and 12 px. Whether a second seed would keep that
order was not tested.

### F241 — On custom, the adapted arm is flat within image noise to 12 px and drops only at 24 px, −0.0768 [−10.30, −4.35], the same shape and size as label transfer's −0.0729

The 0 px anchor F230 lacked is now measured: `e3b-loop-s0`'s adapted arm reads 0.9145, with the
aligned seed set's sd at 0.0050 over 6 seeds. The paired deltas to it are +0.0097 [−0.80, +3.60],
−0.0092 [−3.13, +1.38], −0.0301 [−5.83, +0.09] and −0.0768 [−10.30, −4.35] at 3 / 6 / 12 / 24 px.
Only 24 px excludes zero. The 3 px point reads above 0 px. So F230's "declines steadily" from 3 px is
not resolved until 24 px.

Label transfer has the same shape: flat to 6 px (−0.0010), −0.0153 at 12 px, −0.0729 at 24 px
(record 036). The adapted arm sits 0.0166, 0.0061, 0.0249, 0.0314 and 0.0206 below the label-transfer
seed means at 0 / 3 / 6 / 12 / 24 px. Zero-shot, by contrast, sits 0.0763–0.1621 below them.

### F242 — On FLIR's 3-class primary, the collapsed 0 px run sits below every other level on zero-shot by +10.77 to +27.77 points, so a 0 px-anchored FLIR curve reads as a gain from misalignment

The 0 px stage 3 collapsed (FID 227.62) and reads 0.1616 on bicycle, car and person. Every other
level reads above it on paired CIs: +0.2584 [+23.99, +27.77], +0.1844 [+16.99, +20.04],
+0.2201 [+20.17, +23.90] and +0.1223 [+10.77, +13.68] at 3 / 6 / 12 / 24 px. The adapted arm is
above 0 px at 3 and 6 px (+0.0524 [+3.70, +6.95], +0.0555 [+4.08, +7.23]), unresolved at 12 px
(+0.0172 [−0.10, +3.53]), and below it at 24 px (−0.0940 [−11.10, −7.46]).

Without the collapsed point, zero-shot reads 0.4200 / 0.3461 / 0.3817 / 0.2839 at 3 / 6 / 12 / 24 px.
That is not monotone. Its order agrees with FID's (94.76 / 163.96 / 148.46 / 162.91) on five of six
pairs, and with the level's on five of six too.
This is F231's reading, now on the 3-class primary and with image CIs: one FLIR seed draws points,
not a curve.

### F243 — On the matched 3-class primary, FLIR label transfer leads translation by more than it did all-class: zero-shot by 0.2708–0.4171 and adapted by 0.1538–0.2290 at 3–24 px

Level by level (3 / 6 / 12 / 24 px), label transfer's 3-class mAP50 (record 036) leads stage-3
zero-shot by 0.3564, 0.4171, 0.3368 and 0.2708. It leads the adapted arm by 0.2290, 0.2127, 0.2063
and 0.1538. At the collapsed 0 px the gaps are 0.6147 and 0.2814. Record 037's all-class gaps at
3–24 px were 0.2649 / 0.3327 / 0.2694 / 0.2016 and 0.1762 / 0.1792 / 0.1713 / 0.1150. Dropping dog
lifts translation zero-shot by 0.04–0.09 (0.1616 against 0.1212 at 0 px, 0.4200 against 0.3416 at
3 px), and lifts label transfer more.

The gaps are differences of point estimates. No CI covers them: label transfer is n = 1 in effect
(F227, F233), and the two halves share neither corpus nor detector. F232's confound holds unchanged.
Label transfer is a `yolo11s` trained for 100 epochs on 4,129 images. The adapted arm is a `yolo11n`
fine-tuned for 50 epochs on 600.

### F244 — The FLIR report costs 6 h 06 min against custom's 49 min, and 5 h 34 min of it comes after the last val pass, in eight 1,013-image paired bootstraps logged 30–63 min apart

Both datasets spend about 31 min on their ten val passes, about 3 min each (custom 30 min 32 s,
FLIR 31 min 35 s). After the last pass, custom's eight paired bootstraps are logged 1 min 47 s to
3 min 01 s apart, 17 min 31 s in all. FLIR's are 34:08, 30:02, 31:29, 38:37, 49:02, 38:47, 48:15 and
63:21 apart, 5 h 33 min 41 s in all. The gaps also hold each point's own-CI bootstrap, which logs no
line. The script is not resumable and writes its CSV only at the end, so a FLIR crash in that
stretch loses the whole run. Why the later gaps grow was not established. Custom finished at
14:34, so the two jobs overlapped only in FLIR's first gap.

## Provenance caveats

- **Translation CIs are over val images at one training seed.** They carry no seed noise. Custom
  has a seed yardstick: the six aligned `e3b-loop-s*` seeds, sd 0.0140 zero-shot and 0.0050
  adapted. Its misaligned levels are single seeds. FLIR has no yardstick.
- **The adapted re-score is not checked against a stored value.** The script checks only zero-shot
  against `metrics.json`. The adapted arm's training-end value comes from ultralytics. Against record
  037's stage-3 adapted values, custom's re-score differs by +0.0001, +0.0001, −0.0009 and −0.0020 at
  3 / 6 / 12 / 24 px. FLIR's all-class `all` rows match record 037 to three decimals at 0–12 px, and
  read 0.302 against 0.3011 at 24 px.
- **The custom 0 px is a different run**: `e3b-loop-s0`, trained in August at an earlier commit
  (record 037). The two share a config, not an invocation.
- **FLIR's label-transfer seeds are identical at every level** (F227, F233), so that half is n = 1.
- **FLIR's 0 px translation run collapsed** and was resumed after a crash (record 037). Every FLIR
  translation delta is measured against it.
- **Neither 0 px is truly zero.** The custom set's val boxes came through the sister repo's
  registration. FLIR keeps about 2.25 px of median per-pair scatter after calibration (record 012).
- **The CSVs under `logs/` are copies** of `runs/registration-report/*-registration-curve.csv`,
  carried back by hand. Their rows match the console blocks at printed precision. Byte identity with
  the server files was not checked.
- **No out-of-family judge or faithfulness pass** was run on these exports, as in record 037.

## Next

- **The axis is read.** Custom carries the curve. Its adapted arm is the clean result: flat to
  12 px, a drop at 24 px matching label transfer (F241). FLIR's translation half is reported as
  unstable at n = 1 (F242), as decided on 2026-11-15.
- **The custom zero-shot ordering at 3 px is untested across seeds** (F240). Prediction, if a second
  3 px seed is ever run: its stage-3 zero-shot lands above 0.7682, the first seed's reading. Nothing
  is planned to test it.
- **A rerun of the FLIR report costs about 6 h** and cannot resume (F244). Budget for that before
  adding exports to it.
