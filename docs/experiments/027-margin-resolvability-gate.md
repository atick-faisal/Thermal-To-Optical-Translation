# 027 — Margin resolvability gate: ~100 test images cannot resolve +2 mAP@50 on the custom set

**Date:** 2026-10-20 · **Task:** MARGIN-GATE-04 · **Machine:** VSIBMEOT05 (e3b on `cuda:0`, e3t on `cuda:1`, concurrently) · **Wall clock:** e3b 54 min 03 s (15:30:59 → 16:25:02 +03:00); e3t 29 min 23 s (15:29:57 → 15:59:20 +03:00)
**git_sha:** `pre-spec-migration-145-gbb9df77` (`git describe --always --dirty`, so a clean tree at `bb9df77`) · **W&B:** none (the script does not log to W&B; neither log mentions it) · **Log:** `logs/2026-10-20-margin-gate-04-e3b-resolvability.txt`, `logs/2026-10-20-margin-gate-04-e3t-resolvability.txt`, and the two result CSVs copied beside them, `logs/2026-10-20-margin-gate-04-e3{b,t}-resolvability.csv`

## Command

From each log's provenance block, as run:

```
uv run python scripts/margin_resolvability.py --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --out runs/margin-resolvability-e3b --loop-template runs/e3b-loop-s{seed}/stage3/translated --control-template runs/e3b-control-s{seed}/stage3/translated --seeds 0 1 2 3 4 5 --reference-weights runs/reference-yolo11s/weights/best.pt --device cuda:0
uv run python scripts/margin_resolvability.py --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --out runs/margin-resolvability-e3t --loop-template runs/e3t-loop-s{seed}/stage3/translated --control-template runs/e3t-control-s{seed}/stage3/translated --seeds 0 1 3 --reference-weights runs/reference-yolo11s/weights/best.pt --device cuda:1
```

## Environment

e3b's block, verbatim:

```
=== PROVENANCE ===
captured: 2026-10-20 15:30:59 +03:00
hostname: VSIBMEOT05
os: Microsoft Windows Server 2022 Standard 10.0.20348 (build 20348)
cpu: AMD EPYC 7763 64-Core Processor
cpu_logical_cores: 64
ram_total: 128.0 GB
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16; NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16
cuda_driver: 13.0
git_sha: pre-spec-migration-145-gbb9df77
working_dir: D:\Atick\GitHub\Thermal-To-Optical-Translation
command: uv run python scripts/margin_resolvability.py --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --out runs/margin-resolvability-e3b --loop-template runs/e3b-loop-s{seed}/stage3/translated --control-template runs/e3b-control-s{seed}/stage3/translated --seeds 0 1 2 3 4 5 --reference-weights runs/reference-yolo11s/weights/best.pt --device cuda:0
error: Extra `matchers` is not defined in the project's `optional-dependencies` table
=== END PROVENANCE ===
```

e3t's block is the same apart from `captured: 2026-10-20 15:29:57 +03:00` and its command line.
The `error:` line stands where the Python and torch probe should have printed. Every val run's
ultralytics banner fills part of that gap: `Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130`.

## Configuration

The margin-gate plan (`docs/features/margin-gate/plan.md`, Q2–Q4) and its ADR-01.

| Axis | Value | Where it is evidenced |
| --- | --- | --- |
| Scorer | the zero-shot reference `yolo11s`, `runs\reference-yolo11s\weights\best.pt` (record 004), for every arm | the CSVs' `weights`; ADR-01 |
| Campaigns | e3b (pix2pix, record 007), seeds 0–5; e3t (turbo, record 010), seeds 0, 1, 3 | the commands |
| Arms per seed | loop (λ_det ramp) and control (λ = 0) stage-3 exports; raw thermal once per invocation | the CSVs' `loop_data` / `baseline_data` |
| Raw thermal | `write_budget_manifest(..., 1.0, infrared=True)` over `--data`: val from `val\infrared`, scored against the thermal-side labels | the `budget manifest ... infrared` line; the Scanning line |
| Pool | the 153 custom val images, 423 instances, 4 backgrounds, in every arm | every `all` row; the CSVs' `pool` |
| Bootstrap | paired, images drawn with replacement, `n = 100` per draw (the test split's size), 10,000 resamples, `bootstrap_seed = 0`; mAP@50 recomputed per draw through `ap_per_class` | the `paired image bootstrap:` lines; the CSVs |
| Units | `loop_map50` / `baseline_map50` are fractions over the whole pool; `low` / `high` / `half_width` are mAP points | `scripts/margin_resolvability.py:78-83` |
| Gate | resolves when the widest half-width over every row in the CSV is ≤ 2 mAP points (plan Q3, Q4) | `scripts/margin_resolvability.py:155-176` |
| imgsz / batch | 640 / 16 (the script's defaults; neither command overrides them) | `scripts/margin_resolvability.py:64-65` |

## Results

Each log from the end of its provenance block to its finish stamp. ANSI colour escapes, blank lines
and the per-batch progress-bar lines (`Class Images Instances ... 10/10`) are stripped. Each val
run's `all` row, which is the only number those lines lead to, is kept. Everything else is as the
logs hold it, including the mangled characters in e3b's Scanning lines.

**e3b:**

```
2026-10-20 15:31:20,402 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\translated
2026-10-20 15:31:20,424 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s1\stage3\translated
2026-10-20 15:31:20,463 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s2\stage3\translated
2026-10-20 15:31:20,482 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s3\stage3\translated
2026-10-20 15:31:20,505 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s4\stage3\translated
2026-10-20 15:31:20,524 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s5\stage3\translated
2026-10-20 15:31:20,549 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s0\stage3\translated
2026-10-20 15:31:20,576 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s1\stage3\translated
2026-10-20 15:31:20,596 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s2\stage3\translated
2026-10-20 15:31:20,614 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s3\stage3\translated
2026-10-20 15:31:20,630 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s4\stage3\translated
2026-10-20 15:31:20,648 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s5\stage3\translated
2026-10-20 15:31:20,655 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-10-20 15:31:20,899 INFO    t2o.data.budget: budget manifest D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\margin-resolvability-e3b\manifests\thermal\data.yaml: 600/600 train images (fraction 1.0000, seed 0, infrared)
2026-10-20 15:31:20,901 INFO    __main__: 12 row(s): 0 already in D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\margin-resolvability-e3b\margin-resolvability.csv, 12 to run
2026-10-20 15:31:20,901 INFO    __main__: seed 0: loop-vs-control, loop-vs-thermal
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 42.02.1 MB/s, size: 582.3 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\translated\val\labels.cache... 153 images, 4 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 153/153  0.0s
                   all        153        423      0.855      0.783      0.855      0.568
Speed: 1.3ms preprocess, 2.9ms inference, 0.0ms loss, 1.2ms postprocess per image
2026-10-20 15:34:09,742 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\translated\data.yaml
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.1 ms, read: 23.98.4 MB/s, size: 551.0 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s0\stage3\translated\val\labels.cache... 153 images, 4 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 153/153  0.0s
                   all        153        423      0.865      0.737      0.813      0.546
Speed: 1.4ms preprocess, 1.3ms inference, 0.0ms loss, 1.2ms postprocess per image
2026-10-20 15:36:50,739 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s0\stage3\translated\data.yaml
2026-10-20 15:38:04,409 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [-0.01, 8.07] mAP points
2026-10-20 15:38:04,419 INFO    __main__: seed 0 loop-vs-control: delta CI [-0.01, 8.07], half-width 4.04 mAP points
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 5.82.1 MB/s, size: 96.9 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\val\infrared\labels.cache... 153 images, 4 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 153/153  0.0s
                   all        153        423      0.224      0.172      0.155     0.0675
Speed: 1.4ms preprocess, 1.3ms inference, 0.0ms loss, 1.3ms postprocess per image
2026-10-20 15:40:48,774 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\margin-resolvability-e3b\manifests\thermal\data.yaml
2026-10-20 15:42:27,907 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [59.63, 77.30] mAP points
2026-10-20 15:42:27,917 INFO    __main__: seed 0 loop-vs-thermal: delta CI [59.63, 77.30], half-width 8.84 mAP points
2026-10-20 15:42:27,917 INFO    __main__: seed 1: loop-vs-control, loop-vs-thermal
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 45.85.6 MB/s, size: 571.2 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s1\stage3\translated\val\labels.cache... 153 images, 4 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 153/153  0.0s
                   all        153        423      0.878      0.774      0.848      0.573
Speed: 1.3ms preprocess, 1.2ms inference, 0.0ms loss, 1.0ms postprocess per image
2026-10-20 15:45:14,604 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s1\stage3\translated\data.yaml
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 26.11.8 MB/s, size: 562.5 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s1\stage3\translated\val\labels.cache... 153 images, 4 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 153/153  0.0s
                   all        153        423      0.869      0.795      0.844      0.566
Speed: 1.0ms preprocess, 1.2ms inference, 0.0ms loss, 1.0ms postprocess per image
2026-10-20 15:48:01,645 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s1\stage3\translated\data.yaml
2026-10-20 15:49:15,543 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [-3.96, 4.29] mAP points
2026-10-20 15:49:15,559 INFO    __main__: seed 1 loop-vs-control: delta CI [-3.96, 4.29], half-width 4.13 mAP points
2026-10-20 15:50:57,008 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [58.23, 77.04] mAP points
2026-10-20 15:50:57,031 INFO    __main__: seed 1 loop-vs-thermal: delta CI [58.23, 77.04], half-width 9.41 mAP points
2026-10-20 15:50:57,031 INFO    __main__: seed 2: loop-vs-control, loop-vs-thermal
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 31.37.3 MB/s, size: 596.4 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s2\stage3\translated\val\labels.cache... 153 images, 4 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 153/153  0.0s
                   all        153        423      0.896      0.765      0.869      0.566
Speed: 1.0ms preprocess, 1.1ms inference, 0.0ms loss, 1.1ms postprocess per image
2026-10-20 15:53:45,436 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s2\stage3\translated\data.yaml
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 39.412.1 MB/s, size: 619.1 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s2\stage3\translated\val\labels.cache... 153 images, 4 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 153/153  0.0s
                   all        153        423      0.777      0.713      0.758      0.491
Speed: 1.6ms preprocess, 1.4ms inference, 0.0ms loss, 1.3ms postprocess per image
2026-10-20 15:56:35,897 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s2\stage3\translated\data.yaml
2026-10-20 15:57:50,275 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [5.94, 16.70] mAP points
2026-10-20 15:57:50,286 INFO    __main__: seed 2 loop-vs-control: delta CI [5.94, 16.70], half-width 5.38 mAP points
2026-10-20 15:59:32,341 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [60.88, 78.60] mAP points
2026-10-20 15:59:32,370 INFO    __main__: seed 2 loop-vs-thermal: delta CI [60.88, 78.60], half-width 8.86 mAP points
2026-10-20 15:59:32,370 INFO    __main__: seed 3: loop-vs-control, loop-vs-thermal
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 34.69.8 MB/s, size: 560.9 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s3\stage3\translated\val\labels.cache... 153 images, 4 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 153/153  0.0s
                   all        153        423      0.839      0.802      0.847       0.57
Speed: 1.4ms preprocess, 1.2ms inference, 0.0ms loss, 1.0ms postprocess per image
2026-10-20 16:02:19,148 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s3\stage3\translated\data.yaml
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 44.63.8 MB/s, size: 575.2 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s3\stage3\translated\val\labels.cache... 153 images, 4 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 153/153  0.0s
                   all        153        423      0.796      0.717      0.768      0.498
Speed: 1.1ms preprocess, 1.2ms inference, 0.0ms loss, 1.0ms postprocess per image
2026-10-20 16:05:05,570 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s3\stage3\translated\data.yaml
2026-10-20 16:06:20,265 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [1.73, 14.59] mAP points
2026-10-20 16:06:20,274 INFO    __main__: seed 3 loop-vs-control: delta CI [1.73, 14.59], half-width 6.43 mAP points
2026-10-20 16:08:02,658 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [58.81, 77.25] mAP points
2026-10-20 16:08:02,680 INFO    __main__: seed 3 loop-vs-thermal: delta CI [58.81, 77.25], half-width 9.22 mAP points
2026-10-20 16:08:02,680 INFO    __main__: seed 4: loop-vs-control, loop-vs-thermal
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 38.612.1 MB/s, size: 549.9 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s4\stage3\translated\val\labels.cache... 153 images, 4 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 153/153  0.0s
                   all        153        423      0.832      0.789      0.826      0.542
Speed: 1.0ms preprocess, 1.2ms inference, 0.0ms loss, 1.0ms postprocess per image
2026-10-20 16:10:50,491 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s4\stage3\translated\data.yaml
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 48.93.6 MB/s, size: 590.9 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s4\stage3\translated\val\labels.cache... 153 images, 4 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 153/153  0.0s
                   all        153        423      0.845      0.717      0.784      0.522
Speed: 1.5ms preprocess, 1.2ms inference, 0.0ms loss, 1.0ms postprocess per image
2026-10-20 16:13:36,187 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s4\stage3\translated\data.yaml
2026-10-20 16:14:50,571 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [-0.12, 8.40] mAP points
2026-10-20 16:14:50,575 INFO    __main__: seed 4 loop-vs-control: delta CI [-0.12, 8.40], half-width 4.26 mAP points
2026-10-20 16:16:31,113 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [56.81, 75.08] mAP points
2026-10-20 16:16:31,125 INFO    __main__: seed 4 loop-vs-thermal: delta CI [56.81, 75.08], half-width 9.14 mAP points
2026-10-20 16:16:31,125 INFO    __main__: seed 5: loop-vs-control, loop-vs-thermal
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 36.36.3 MB/s, size: 534.3 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s5\stage3\translated\val\labels.cache... 153 images, 4 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 153/153  0.0s
                   all        153        423      0.882      0.768      0.847      0.569
Speed: 1.2ms preprocess, 1.4ms inference, 0.0ms loss, 1.3ms postprocess per image
2026-10-20 16:19:19,063 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s5\stage3\translated\data.yaml
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 28.25.2 MB/s, size: 529.8 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s5\stage3\translated\val\labels.cache... 153 images, 4 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 153/153  0.0s
                   all        153        423      0.819      0.772      0.818      0.551
Speed: 1.4ms preprocess, 1.2ms inference, 0.0ms loss, 1.0ms postprocess per image
2026-10-20 16:22:06,688 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s5\stage3\translated\data.yaml
2026-10-20 16:23:19,172 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [-1.93, 8.45] mAP points
2026-10-20 16:23:19,182 INFO    __main__: seed 5 loop-vs-control: delta CI [-1.93, 8.45], half-width 5.19 mAP points
2026-10-20 16:25:00,145 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [58.09, 76.88] mAP points
2026-10-20 16:25:00,156 INFO    __main__: seed 5 loop-vs-thermal: delta CI [58.09, 76.88], half-width 9.40 mAP points
2026-10-20 16:25:00,179 INFO    __main__: verdict over 12 row(s): widest half-width 9.41 mAP points (seed 1, loop-vs-thermal) at n=100
2026-10-20 16:25:00,179 INFO    __main__: verdict: DOES NOT RESOLVE -- +2 mAP@50 cannot be told from zero at n=100; Margin becomes non-inferiority to ModTr (goal.md)
exit code: 0
=== FINISHED 2026-10-20 16:25:02 +03:00 ===
```

**e3t:**

```
2026-10-20 15:30:17,617 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s0\stage3\translated
2026-10-20 15:30:17,627 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s1\stage3\translated
2026-10-20 15:30:17,627 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s3\stage3\translated
2026-10-20 15:30:17,627 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-control-s0\stage3\translated
2026-10-20 15:30:17,633 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-control-s1\stage3\translated
2026-10-20 15:30:17,633 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-control-s3\stage3\translated
2026-10-20 15:30:17,637 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-10-20 15:30:17,794 INFO    t2o.data.budget: budget manifest D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\margin-resolvability-e3t\manifests\thermal\data.yaml: 600/600 train images (fraction 1.0000, seed 0, infrared)
2026-10-20 15:30:17,794 INFO    __main__: 6 row(s): 0 already in D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\margin-resolvability-e3t\margin-resolvability.csv, 6 to run
2026-10-20 15:30:17,794 INFO    __main__: seed 0: loop-vs-control, loop-vs-thermal
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 1644.2578.6 MB/s, size: 393.1 KB)
                   all        153        423       0.88      0.829      0.852      0.567
Speed: 1.9ms preprocess, 3.0ms inference, 0.0ms loss, 1.4ms postprocess per image
2026-10-20 15:33:06,206 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s0\stage3\translated\data.yaml
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 1538.1163.4 MB/s, size: 390.8 KB)
                   all        153        423      0.837      0.771      0.818      0.546
Speed: 1.4ms preprocess, 1.8ms inference, 0.0ms loss, 1.6ms postprocess per image
2026-10-20 15:35:51,799 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-control-s0\stage3\translated\data.yaml
2026-10-20 15:37:08,888 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [-2.05, 9.09] mAP points
2026-10-20 15:37:08,909 INFO    __main__: seed 0 loop-vs-control: delta CI [-2.05, 9.09], half-width 5.57 mAP points
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 7.71.0 MB/s, size: 113.0 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
                   all        153        423      0.224      0.172      0.155     0.0675
Speed: 1.2ms preprocess, 1.2ms inference, 0.0ms loss, 1.1ms postprocess per image
2026-10-20 15:39:54,297 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\margin-resolvability-e3t\manifests\thermal\data.yaml
2026-10-20 15:41:39,858 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [58.87, 77.67] mAP points
2026-10-20 15:41:39,869 INFO    __main__: seed 0 loop-vs-thermal: delta CI [58.87, 77.67], half-width 9.40 mAP points
2026-10-20 15:41:39,869 INFO    __main__: seed 1: loop-vs-control, loop-vs-thermal
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 1401.1426.4 MB/s, size: 395.2 KB)
                   all        153        423      0.835      0.843      0.862      0.584
Speed: 1.5ms preprocess, 1.4ms inference, 0.0ms loss, 1.3ms postprocess per image
2026-10-20 15:44:24,584 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s1\stage3\translated\data.yaml
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 1612.5522.8 MB/s, size: 419.0 KB)
                   all        153        423      0.915      0.778       0.85      0.563
Speed: 1.5ms preprocess, 1.5ms inference, 0.0ms loss, 1.4ms postprocess per image
2026-10-20 15:47:13,694 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-control-s1\stage3\translated\data.yaml
2026-10-20 15:48:31,421 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [-2.22, 3.83] mAP points
2026-10-20 15:48:31,431 INFO    __main__: seed 1 loop-vs-control: delta CI [-2.22, 3.83], half-width 3.03 mAP points
2026-10-20 15:50:14,685 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [59.50, 78.20] mAP points
2026-10-20 15:50:14,701 INFO    __main__: seed 1 loop-vs-thermal: delta CI [59.50, 78.20], half-width 9.35 mAP points
2026-10-20 15:50:14,701 INFO    __main__: seed 3: loop-vs-control, loop-vs-thermal
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 1747.1221.3 MB/s, size: 408.9 KB)
                   all        153        423      0.894      0.823      0.875      0.577
Speed: 1.6ms preprocess, 1.2ms inference, 0.0ms loss, 1.1ms postprocess per image
2026-10-20 15:53:00,124 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s3\stage3\translated\data.yaml
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 1443.8371.2 MB/s, size: 369.9 KB)
                   all        153        423      0.796      0.805      0.829      0.562
Speed: 1.6ms preprocess, 1.3ms inference, 0.0ms loss, 1.3ms postprocess per image
2026-10-20 15:55:51,421 INFO    t2o.metrics.task: captured per-image stats for 153 images from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-control-s3\stage3\translated\data.yaml
2026-10-20 15:57:17,851 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [0.52, 8.55] mAP points
2026-10-20 15:57:17,861 INFO    __main__: seed 3 loop-vs-control: delta CI [0.52, 8.55], half-width 4.02 mAP points
2026-10-20 15:59:18,169 INFO    t2o.analysis.resolvability: paired image bootstrap: 100 of 153 images x 10000 resamples, delta CI [62.17, 78.57] mAP points
2026-10-20 15:59:18,180 INFO    __main__: seed 3 loop-vs-thermal: delta CI [62.17, 78.57], half-width 8.20 mAP points
2026-10-20 15:59:18,200 INFO    __main__: verdict over 6 row(s): widest half-width 9.40 mAP points (seed 0, loop-vs-thermal) at n=100
2026-10-20 15:59:18,200 INFO    __main__: verdict: DOES NOT RESOLVE -- +2 mAP@50 cannot be told from zero at n=100; Margin becomes non-inferiority to ModTr (goal.md)
exit code: 0
=== FINISHED 2026-10-20 15:59:20 +03:00 ===
```

**The result CSVs**, verbatim. e3b:

```
seed,pair,n,pool,resamples,bootstrap_seed,loop_map50,baseline_map50,low,high,half_width,resolves,loop_data,baseline_data,weights
0,loop-vs-control,100,153,10000,0,0.8547938638426396,0.8128402906700233,-0.008041282296213285,8.069081440326828,4.038561361311521,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s0\stage3\translated\data.yaml,runs\reference-yolo11s\weights\best.pt
0,loop-vs-thermal,100,153,10000,0,0.8547938638426396,0.1552287405370524,59.62705806597286,77.30027965160392,8.836610792815527,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\margin-resolvability-e3b\manifests\thermal\data.yaml,runs\reference-yolo11s\weights\best.pt
1,loop-vs-control,100,153,10000,0,0.8475402463498627,0.8441498794294119,-3.963719104456852,4.290257647249256,4.126988375853054,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s1\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s1\stage3\translated\data.yaml,runs\reference-yolo11s\weights\best.pt
1,loop-vs-thermal,100,153,10000,0,0.8475402463498627,0.1552287405370524,58.22916503115281,77.03975584357337,9.405295406210282,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s1\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\margin-resolvability-e3b\manifests\thermal\data.yaml,runs\reference-yolo11s\weights\best.pt
2,loop-vs-control,100,153,10000,0,0.8692232659519413,0.7576454698987364,5.9383073205974855,16.702295739281205,5.38199420934186,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s2\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s2\stage3\translated\data.yaml,runs\reference-yolo11s\weights\best.pt
2,loop-vs-thermal,100,153,10000,0,0.8692232659519413,0.1552287405370524,60.875759249462035,78.5976925193614,8.860966634949683,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s2\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\margin-resolvability-e3b\manifests\thermal\data.yaml,runs\reference-yolo11s\weights\best.pt
3,loop-vs-control,100,153,10000,0,0.8473979197736736,0.7680947851238659,1.7301821682296814,14.5864201400662,6.4281189859182595,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s3\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s3\stage3\translated\data.yaml,runs\reference-yolo11s\weights\best.pt
3,loop-vs-thermal,100,153,10000,0,0.8473979197736736,0.1552287405370524,58.81349548840929,77.25490096846072,9.220702740025715,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s3\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\margin-resolvability-e3b\manifests\thermal\data.yaml,runs\reference-yolo11s\weights\best.pt
4,loop-vs-control,100,153,10000,0,0.8260504687506377,0.7843855761288185,-0.12304786667992865,8.399975672672362,4.2615117696761455,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s4\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s4\stage3\translated\data.yaml,runs\reference-yolo11s\weights\best.pt
4,loop-vs-thermal,100,153,10000,0,0.8260504687506377,0.1552287405370524,56.811601792213075,75.08314692449474,9.135772566140833,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s4\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\margin-resolvability-e3b\manifests\thermal\data.yaml,runs\reference-yolo11s\weights\best.pt
5,loop-vs-control,100,153,10000,0,0.8473956210400941,0.8179749368503605,-1.9329947383285369,8.452384253974959,5.192689496151748,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s5\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s5\stage3\translated\data.yaml,runs\reference-yolo11s\weights\best.pt
5,loop-vs-thermal,100,153,10000,0,0.8473956210400941,0.1552287405370524,58.09075411481235,76.88360651151876,9.396426198353208,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s5\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\margin-resolvability-e3b\manifests\thermal\data.yaml,runs\reference-yolo11s\weights\best.pt
```

e3t:

```
seed,pair,n,pool,resamples,bootstrap_seed,loop_map50,baseline_map50,low,high,half_width,resolves,loop_data,baseline_data,weights
0,loop-vs-control,100,153,10000,0,0.8522506023318381,0.8184094244827931,-2.0451170185183725,9.088289376525498,5.566703197521935,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s0\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-control-s0\stage3\translated\data.yaml,runs\reference-yolo11s\weights\best.pt
0,loop-vs-thermal,100,153,10000,0,0.8522506023318381,0.1552287405370524,58.87445828855304,77.67382038367731,9.399681047562133,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s0\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\margin-resolvability-e3t\manifests\thermal\data.yaml,runs\reference-yolo11s\weights\best.pt
1,loop-vs-control,100,153,10000,0,0.8622155266083098,0.8501035623672061,-2.217888431965605,3.833343703341024,3.0256160676533144,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s1\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-control-s1\stage3\translated\data.yaml,runs\reference-yolo11s\weights\best.pt
1,loop-vs-thermal,100,153,10000,0,0.8622155266083098,0.1552287405370524,59.49940109940265,78.19579187931477,9.34819538995606,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s1\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\margin-resolvability-e3t\manifests\thermal\data.yaml,runs\reference-yolo11s\weights\best.pt
3,loop-vs-control,100,153,10000,0,0.8752003112222961,0.8292772816109224,0.5207167749541393,8.5537366540212,4.016509939533531,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s3\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-control-s3\stage3\translated\data.yaml,runs\reference-yolo11s\weights\best.pt
3,loop-vs-thermal,100,153,10000,0,0.8752003112222961,0.1552287405370524,62.172630002465745,78.56503388803768,8.196201942785965,False,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s3\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\margin-resolvability-e3t\manifests\thermal\data.yaml,runs\reference-yolo11s\weights\best.pt
```

The verdict re-renders from either CSV without a GPU: re-running the same command against the
same `--out` finds every row already written, skips them all, and logs the verdict over the whole
CSV again (`scripts/margin_resolvability.py:234-271`). The command still loads every manifest
before it gets there.

Derived from the CSVs above. These are the mean and sample sd of the per-seed whole-pool deltas, in
mAP points, and the per-seed ratio of the two pairs' half-widths:

| Campaign | loop − control, per seed | mean / sd | loop-vs-control half-width | loop-vs-thermal half-width | thermal / control ratio |
| --- | --- | --- | --- | --- | --- |
| e3b | 4.20, 0.34, 11.16, 7.93, 4.17, 2.94 | 5.12 / 3.84 | 4.04, 4.13, 5.38, 6.43, 4.26, 5.19 (mean 4.90) | 8.84, 9.41, 8.86, 9.22, 9.14, 9.40 | 2.19, 2.28, 1.65, 1.43, 2.14, 1.81 |
| e3t | 3.38, 1.21, 4.59 | 3.06 / 1.71 | 5.57, 3.03, 4.02 (mean 4.20) | 9.40, 9.35, 8.20 | 1.69, 3.09, 2.04 |

## Findings

### F189 — At the test split's size the Margin CI cannot resolve +2 mAP@50: the widest 95% image-level half-width is 9.41 mAP points, 4.7× the 2-point bar

Both campaigns return `verdict: DOES NOT RESOLVE`. e3b's widest row is seed 1 loop-vs-thermal,
9.41 points, over 12 rows. e3t's is seed 0 loop-vs-thermal, 9.40, over 6 rows. Not one of the 18
rows resolves (`resolves` is `False` on every row); the narrowest half-width anywhere is 3.03. The
comparison is the 2-point bar that plan Q3 set. By `goal.md`'s own clause, Margin therefore
becomes non-inferiority to ModTr, and Transfer and Faithfulness carry the claim. The result does
not depend on how the loop performs: the gate reads only interval widths.

### F190 — The verdict does not hinge on the pair Q4 chose: the closest proxy, loop vs its own no-loop control, is 3.03–6.43 points on all nine seeds, at least 1.5× the bar

Q4 gated on the wider of two pairs as the conservative choice. The narrower pair fails on its own.
Its half-widths run 4.04–6.43 on e3b and 3.03–5.57 on e3t, and its narrowest (e3t seed 1, 3.03)
is still 1.5× the 2-point bar. As the plan predicted, the raw-thermal pair is wider on every seed:
8.20–9.41 points, 1.43–3.09× the control pair at the same seed. The comparison is the 2-point bar,
and each seed's other pair. A loop-vs-ModTr interval could still be narrower than loop vs
control if ModTr's per-image errors track the loop's more closely than its own control's do. This
run cannot rule that out.

### F191 — The per-image capture path reproduces the one evaluation path on server data: all 18 whole-pool mAP50s equal the stored stage-3 zero-shot values to every printed digit

Each `loop_map50` and control `baseline_map50` here equals that run's stage-3
`zero_shot.map50` in `logs/2026-10-19-runs-e3{b,t}-judge-tidy.csv` to all 16 printed digits:
e3b-loop-s0 `0.8547938638426396` in both, e3t-control-s3 `0.8292772816109224` in both, and so on
for all 18 exports. The tidy files were written by `t2o aggregate` from `evaluate_detector`'s
write-backs. Raw thermal's `0.1552287405370524`, identical in both invocations, matches F10's
0.1552 clean-judge floor (and F86's A0). The per-seed means re-derive F35's +0.0512 (e3b, 5.12
points) and F72's +0.0306 (e3t, 3.06). The comparison is `evaluate_detector` on the same weights
and the same exports. MARGIN-GATE-01's tests could only show this equality on hand-built records,
because the fixture detector scores 0.0 (margin-gate pitfalls). This is the first run of it with a
detector that finds objects. It reproduces those numbers through the same AP code, so it is
not second evidence for any of them.

### F192 — Seed-to-seed spread of the paired val delta is as large as one seed's image-level noise: sd 3.84 points across six e3b seeds against a mean half-width of 4.90

The per-seed whole-pool loop − control deltas are scored on the same 153 images by the same
judge, so their spread is training-seed variance alone. On e3b they run +0.34 to +11.16 points,
sd 3.84. Read as roughly normal, a 4.90-point mean half-width is an image-level sd of about 2.5.
The seed spread is the larger of the two. e3t's three seeds give sd 1.71 against a mean
half-width of 4.20 (about 2.1), the same order. The same spread shows in which per-seed intervals
exclude zero: 3 of 9 (e3b seeds 2 and 3, e3t seed 3), with every point delta positive. The
comparison is the image-level half-width within a seed. This quantifies ADR-01's consequence that
a one-seed image CI ignores seed variance. A Margin number on one seed's interval would understate
its uncertainty by about as much again, so Stability's seeds have to be reported beside it.

## Provenance caveats

- **The provenance block's Python probe failed.** Its `error:` line (quoted in Environment)
  stands in for the `python`, `torch`, `cuda_torch_build`, `cudnn`, `cuda_available` and
  `gpu0_torch` lines. t2o's `pyproject.toml` defines `cpu` and `gpu` extras but no
  `matchers`, so the server's `Write-Provenance` is most likely the sister project's version,
  whose probe runs `uv run --extra gpu --extra matchers`. That is inferred from the error, not
  checked on the server. The ultralytics banners give Python 3.12.7
  and torch 2.12.1+cu130. cuDNN and torch's own CUDA build are not recorded.
- **The pool is val, not test.** Every interval is an m-out-of-n draw (100 of 153) from the
  validation images. It estimates what a CI over ~100 different test images would look like, and
  is not one.
- **One bootstrap seed.** Every row uses `bootstrap_seed = 0` at 10,000 resamples. Monte Carlo
  error on the half-widths was not measured. At the 2-point bar against a narrowest 3.03, it
  would have to be over a point to change the verdict.
- **The loop-vs-thermal pair is scored against two label sets.** Exports carry the visible labels
  they were mirrored from, while raw thermal carries the transferred thermal-side labels. The
  pairing check passed on all nine pairs: each stem had the same target classes in both, and both
  sides count 423 instances. Box geometry was not compared, so part of that pair's width may be
  label transfer rather than imagery.
- **Raw thermal was collected twice**, once per invocation, into two `manifests\thermal` trees.
  The two collections agree to every digit.
- **e3b read its images slowly** ("Slow image access detected", 5.8–48.9 MB/s, against e3t's
  1,401–1,747 MB/s). That cost time, not numbers.

## Next

Margin is now non-inferiority to ModTr. The later baselines feature builds ModTr first, trained
against the same in-loop `yolo11n` (ADR-01), and HalluciDet and Grounding DINO drop out of the
margin claim. The test split still has to be pinned before its first read, which is
MARGIN-GATE-05. A falsifiable prediction for that feature: on val, loop vs ModTr under this
scorer will also have a half-width above 2 points on every seed (F190's narrowest pair is 3.03).
`goal.md` does not name the non-inferiority margin, and it has to be fixed before the test split
is read.
