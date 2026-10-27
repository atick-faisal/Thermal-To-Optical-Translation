# 031 — ModTr FLIR campaign: the loop is not non-inferior, and ModTr is 5.80 points ahead

**Date:** 2026-10-26 (scored 2026-10-27) · **Task:** MODTR-BASELINE-06 · **Machine:** VSIBMEOT05 (`cuda:1`) · **Wall clock:** 10 h 51 min 43 s for the three seeds back to back (2026-10-26 13:24:31 → 2026-10-27 00:16:14 +03:00): s0 3 h 34 min 49 s, s1 3 h 35 min 54 s, s3 3 h 40 min 54 s. Then 32 min 9 s for the RT-DETR judge and 1 h 37 min 47 s for the non-inferiority run
**git_sha:** `pre-spec-migration-189-g2413103` (`git describe --always --dirty`, so a clean tree at `2413103`) · **W&B:** none (`runtime.wandb: false` in `experiments/modtr_flir.yaml:88`; no W&B line in any log) · **Log:** `logs/2026-10-26-modtr-baseline-06-flir-campaign-s0.txt`, `logs/2026-10-26-modtr-baseline-06-flir-campaign-s1.txt`, `logs/2026-10-26-modtr-baseline-06-flir-campaign-s3.txt`, `logs/2026-10-27-modtr-baseline-06-rtdetr-flir.txt`, `logs/2026-10-27-modtr-baseline-06-non-inferiority-flir.txt`, with `logs/2026-10-27-modtr-non-inferiority-flir.csv`

## Command

From each log's provenance block, as run, in order:

```
uv run t2o loop --config experiments/modtr_flir.yaml --seed 0 --name modtr-flir-s0 --device cuda:1
uv run t2o loop --config experiments/modtr_flir.yaml --seed 1 --name modtr-flir-s1 --device cuda:1
uv run t2o loop --config experiments/modtr_flir.yaml --seed 3 --name modtr-flir-s3 --device cuda:1
uv run t2o evaluate --weights runs/reference-flir-rtdetr-l/weights/best.pt --data runs/modtr-flir-s0/stage0/translated/data.yaml --write-back --judge rtdetr --device 1
uv run t2o evaluate --weights runs/reference-flir-rtdetr-l/weights/best.pt --data runs/modtr-flir-s1/stage0/translated/data.yaml --write-back --judge rtdetr --device 1
uv run t2o evaluate --weights runs/reference-flir-rtdetr-l/weights/best.pt --data runs/modtr-flir-s3/stage0/translated/data.yaml --write-back --judge rtdetr --device 1
uv run python scripts/modtr_non_inferiority.py --loop-template runs/e3f-loop-s{seed}/stage3/translated --modtr-template runs/modtr-flir-s{seed}/stage0/translated --reference-weights runs/reference-flir-yolo11s/weights/best.pt --out runs/modtr-non-inferiority-flir --device cuda:1
```

The non-inferiority script's default seeds, 0, 1 and 3, fill what the command leaves out. The
log's `data.yaml` lines confirm them.

## Environment

Seed 0's block, verbatim:

```
=== PROVENANCE ===
captured: 2026-10-26 13:24:31 +03:00
hostname: VSIBMEOT05
os: Microsoft Windows Server 2022 Standard 10.0.20348 (build 20348)
cpu: AMD EPYC 7763 64-Core Processor
cpu_logical_cores: 64
ram_total: 128.0 GB
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16; NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16
cuda_driver: 13.0
git_sha: pre-spec-migration-189-g2413103
working_dir: D:\Atick\GitHub\Thermal-To-Optical-Translation
command: uv run t2o loop --config experiments/modtr_flir.yaml --seed 0 --name modtr-flir-s0 --device cuda:1
python: 3.12.7
torch: 2.12.1+cu130
cuda_torch_build: 13.0
cudnn: 92000
cuda_available: True
gpu0_torch: NVIDIA A100-SXM4-40GB, 39.7 GiB, sm 8.0
=== END PROVENANCE ===
```

The other six blocks are the same apart from `captured:` and their command lines. Seeds 1 and
3: 2026-10-26 16:59:22 and 20:35:20. The RT-DETR judge: 2026-10-27 09:10:07, 09:11:38 and
09:40:44. Non-inferiority: 09:55:03. All seven carry the same `git_sha`. The Python probe ran:
the `python`, `torch` and `cudnn` lines are present and no `error:` line is.

## Configuration

`experiments/modtr_flir.yaml` at `2413103`, with the budget set by record 029.

| Axis | Value | Where it is evidenced |
| --- | --- | --- |
| Translator | ModTr (ResNet34 U-Net, fusion `clamp(x·h(x), 0, 1)`), detection-only objective | modtr-baseline ADR-01; `experiments/modtr_flir.yaml` |
| Budget | 1 stage × 339 epochs, `task_weight` 1.0, `grad_scale` 1.0 | `stage 0 done: task_weight=1.0 epochs=339`; `modtr_flir.yaml:33, :51-52`; F197 |
| Seeds | 0, 1, 3, so each pairs with an e3f seed | the commands |
| Training data | FLIR's 600-pair campaign subset, `subset_seed: 0` | `600 paired samples (capped from 4129)`; `modtr_flir.yaml:23, :28` |
| Val-loss cut | 153 of the 1,013 val pairs, used only for the training-time val loss | `153 paired samples (capped from 1013)`; `modtr_flir.yaml:26` |
| Scored split | all 1,013 FLIR val pairs, as in record 016 (F123) | `exported 1013 translated images`; `1013 images` on every eval |
| In-loop detector | `runs/inloop-flir-yolo11n/weights/best.pt`, the one the e3f loop trained against | `modtr_flir.yaml:59`; `model=runs\inloop-flir-yolo11n\weights\best.pt` in the fine-tune args |
| Zero-shot scorer | FLIR's reference `yolo11s`, `runs/reference-flir-yolo11s/weights/best.pt` | `modtr_flir.yaml:68`; the non-inferiority command |
| Out-of-family judge | FLIR's RT-DETR-l, `runs/reference-flir-rtdetr-l/weights/best.pt` (record 026) | the RT-DETR commands |
| Fine-tuned arm | in-loop `yolo11n` fine-tuned 50 epochs on each seed's translated train split | `modtr_flir.yaml:62-63`; `epochs=50` in the ultralytics args line |
| Monitor | off | no `--monitor-map`; no per-epoch `exported` or `eval` lines |
| Non-inferiority test | ADR-02: paired 95% bootstrap over the val images on seed-averaged mAP@50, 10,000 resamples, bootstrap seed 0; pass when the lower bound > −5 points | `seed-averaged bootstrap: 3 seed(s) x 1013 images x 10000 resamples`; the CSV's `margin`, `resamples` and `bootstrap_seed` columns |
| Endpoint | **all-class (4-class) mAP@50**, not FLIR's 3-class primary (see Provenance caveats) | `scripts/modtr_non_inferiority.py:111-112` (`map50_from_stats` over every class) |
| Comparison arm | e3f loop (pix2pix, record 016), stage-3 exports, seeds 0, 1, 3 | the log's `data.yaml` lines |
| `config_hash` | not in the logs | — |

## Results

### Campaign

Each campaign log is about 4,700 lines. Per seed, the opening through epoch 0, the end of training
through the zero-shot and fidelity evaluation, and the fine-tuned detector's close follow. ANSI
colour escapes, the per-batch `Class Images Instances Box(P ...` progress lines and the
in-progress `Scanning` lines are stripped; the final `Scanning ... 100%` line is kept. `[...]`
marks epochs 1–337, and then the fine-tuned detector's training, the ultralytics args line, the
model summary and 50 epochs of progress. All of it is in the logs.

```
2026-10-26 13:24:53,887 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir
2026-10-26 13:24:54,474 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\train\visible\images: 600 paired samples (capped from 4129), 600 annotated
2026-10-26 13:24:55,036 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir
2026-10-26 13:25:03,223 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\train\visible\images: 600 paired samples (capped from 4129), 600 annotated
2026-10-26 13:25:05,528 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\val\visible\images: 153 paired samples (capped from 1013), 153 annotated
2026-10-26 13:25:39,443 INFO    t2o.engine.trainer: epoch 0 took 33.9 s
2026-10-26 13:25:39,444 INFO    t2o.engine.trainer: epoch 0 peak memory: 5.56 GiB allocated, 6.10 GiB reserved of 39.70 GiB
[...]
2026-10-26 16:38:54,064 INFO    t2o.engine.trainer: epoch 338 took 34.9 s
2026-10-26 16:38:54,064 INFO    t2o.engine.trainer: epoch 338 peak memory: 5.56 GiB allocated, 6.10 GiB reserved of 39.70 GiB
2026-10-26 16:38:55,432 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\train\visible\images: 600 paired samples (capped from 4129), 600 annotated
2026-10-26 16:39:46,292 INFO    t2o.engine.export: exported 600 translated images to runs\modtr-flir-s0\stage0\translated\train (normalize=clamp)
2026-10-26 16:39:58,710 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\val\visible\images: 1013 paired samples, 1013 annotated
2026-10-26 16:41:17,110 INFO    t2o.engine.export: exported 1013 translated images to runs\modtr-flir-s0\stage0\translated\val (normalize=clamp)
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,348 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 23.34.0 MB/s, size: 550.5 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\translated\val\labels... 1013 images, 0 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 1013/1013 297.8it/s 3.4s
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\translated\val\images\FLIR_09042.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\translated\val\images\FLIR_09055.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\translated\val\images\FLIR_09653.png: 1 duplicate labels removed
val: New cache created: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\translated\val\labels.cache
                   all       1013       8601      0.512      0.372      0.413       0.19
Speed: 1.4ms preprocess, 1.4ms inference, 0.0ms loss, 1.2ms postprocess per image
2026-10-26 16:42:10,099 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5116  R 0.3722  mAP50 0.4134  mAP50-95 0.1903
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torchmetrics\utilities\prints.py:43: UserWarning: Metric `Kernel Inception Distance` will save all extracted features in buffer. For large datasets this may lead to large memory footprint.
  warnings.warn(*args, **kwargs)
2026-10-26 16:42:53,029 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 6.13  SSIM 0.2430  LPIPS 0.7784  FID 315.70  KID 0.3917+-0.0241
[...]
50 epochs completed in 0.201 hours.
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\detector\stage0\weights\last.pt, 5.5MB
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\detector\stage0\weights\best.pt, 5.5MB
Validating D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\detector\stage0\weights\best.pt...
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11n summary (fused): 101 layers, 2,582,932 parameters, 0 gradients, 6.4 GFLOPs
                   all       1013       8601      0.501       0.43      0.463      0.232
Speed: 0.2ms preprocess, 0.7ms inference, 0.0ms loss, 1.3ms postprocess per image
2026-10-26 16:59:15,213 INFO    t2o.engine.detector_stage: detector stage0: P 0.5014  R 0.4301  mAP50 0.4625  mAP50-95 0.2324
2026-10-26 16:59:15,224 INFO    t2o.engine.loop: stage 0 done: task_weight=1.0 epochs=339
2026-10-26 16:59:15,234 INFO    t2o.cli: stage 0 (task_weight=1.0): 339 epochs | zero-shot mAP50 0.4134 | LPIPS 0.7784 | fine-tuned mAP50 0.4625
=== FINISHED 2026-10-26 16:59:20 +03:00 ===

2026-10-26 16:59:47,793 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir
2026-10-26 16:59:48,222 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\train\visible\images: 600 paired samples (capped from 4129), 600 annotated
2026-10-26 16:59:48,766 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir
2026-10-26 16:59:50,017 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\train\visible\images: 600 paired samples (capped from 4129), 600 annotated
2026-10-26 16:59:50,184 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\val\visible\images: 153 paired samples (capped from 1013), 153 annotated
2026-10-26 17:00:24,643 INFO    t2o.engine.trainer: epoch 0 took 34.5 s
2026-10-26 17:00:24,643 INFO    t2o.engine.trainer: epoch 0 peak memory: 5.56 GiB allocated, 6.10 GiB reserved of 39.70 GiB
[...]
2026-10-26 20:15:22,331 INFO    t2o.engine.trainer: epoch 338 took 34.2 s
2026-10-26 20:15:22,331 INFO    t2o.engine.trainer: epoch 338 peak memory: 5.55 GiB allocated, 6.10 GiB reserved of 39.70 GiB
2026-10-26 20:15:23,942 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\train\visible\images: 600 paired samples (capped from 4129), 600 annotated
2026-10-26 20:16:13,079 INFO    t2o.engine.export: exported 600 translated images to runs\modtr-flir-s1\stage0\translated\train (normalize=clamp)
2026-10-26 20:16:26,051 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\val\visible\images: 1013 paired samples, 1013 annotated
2026-10-26 20:17:40,746 INFO    t2o.engine.export: exported 1013 translated images to runs\modtr-flir-s1\stage0\translated\val (normalize=clamp)
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,348 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 28.88.0 MB/s, size: 547.0 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\translated\val\labels... 1013 images, 0 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 1013/1013 333.3it/s 3.0s
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\translated\val\images\FLIR_09042.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\translated\val\images\FLIR_09055.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\translated\val\images\FLIR_09653.png: 1 duplicate labels removed
val: New cache created: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\translated\val\labels.cache
                   all       1013       8601      0.758      0.365       0.41      0.197
Speed: 1.4ms preprocess, 1.4ms inference, 0.0ms loss, 1.2ms postprocess per image
2026-10-26 20:18:32,926 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7578  R 0.3652  mAP50 0.4098  mAP50-95 0.1968
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torchmetrics\utilities\prints.py:43: UserWarning: Metric `Kernel Inception Distance` will save all extracted features in buffer. For large datasets this may lead to large memory footprint.
  warnings.warn(*args, **kwargs)
2026-10-26 20:19:14,884 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 6.42  SSIM 0.2516  LPIPS 0.7416  FID 334.58  KID 0.4148+-0.0250
[...]
50 epochs completed in 0.193 hours.
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\detector\stage0\weights\last.pt, 5.5MB
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\detector\stage0\weights\best.pt, 5.5MB
Validating D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\detector\stage0\weights\best.pt...
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11n summary (fused): 101 layers, 2,582,932 parameters, 0 gradients, 6.4 GFLOPs
                   all       1013       8601      0.704      0.428      0.474      0.234
Speed: 0.1ms preprocess, 0.7ms inference, 0.0ms loss, 1.6ms postprocess per image
2026-10-26 20:35:12,908 INFO    t2o.engine.detector_stage: detector stage0: P 0.7040  R 0.4280  mAP50 0.4738  mAP50-95 0.2338
2026-10-26 20:35:12,917 INFO    t2o.engine.loop: stage 0 done: task_weight=1.0 epochs=339
2026-10-26 20:35:12,924 INFO    t2o.cli: stage 0 (task_weight=1.0): 339 epochs | zero-shot mAP50 0.4098 | LPIPS 0.7416 | fine-tuned mAP50 0.4738
=== FINISHED 2026-10-26 20:35:16 +03:00 ===

2026-10-26 20:36:32,430 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir
2026-10-26 20:36:32,985 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\train\visible\images: 600 paired samples (capped from 4129), 600 annotated
2026-10-26 20:36:33,556 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir
2026-10-26 20:36:34,922 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\train\visible\images: 600 paired samples (capped from 4129), 600 annotated
2026-10-26 20:36:35,173 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\val\visible\images: 153 paired samples (capped from 1013), 153 annotated
2026-10-26 20:37:12,140 INFO    t2o.engine.trainer: epoch 0 took 37.0 s
2026-10-26 20:37:12,140 INFO    t2o.engine.trainer: epoch 0 peak memory: 5.56 GiB allocated, 6.10 GiB reserved of 39.70 GiB
[...]
2026-10-26 23:55:06,803 INFO    t2o.engine.trainer: epoch 338 took 35.1 s
2026-10-26 23:55:06,803 INFO    t2o.engine.trainer: epoch 338 peak memory: 5.56 GiB allocated, 6.10 GiB reserved of 39.70 GiB
2026-10-26 23:55:08,017 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\train\visible\images: 600 paired samples (capped from 4129), 600 annotated
2026-10-26 23:55:58,837 INFO    t2o.engine.export: exported 600 translated images to runs\modtr-flir-s3\stage0\translated\train (normalize=clamp)
2026-10-26 23:55:59,435 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\val\visible\images: 1013 paired samples, 1013 annotated
2026-10-26 23:57:17,363 INFO    t2o.engine.export: exported 1013 translated images to runs\modtr-flir-s3\stage0\translated\val (normalize=clamp)
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,348 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 46.18.8 MB/s, size: 585.0 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\translated\val\labels... 1013 images, 0 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 1013/1013 210.6it/s 4.8s
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\translated\val\images\FLIR_09042.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\translated\val\images\FLIR_09055.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\translated\val\images\FLIR_09653.png: 1 duplicate labels removed
val: New cache created: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\translated\val\labels.cache
                   all       1013       8601      0.749      0.366        0.4      0.188
Speed: 1.4ms preprocess, 1.7ms inference, 0.0ms loss, 1.3ms postprocess per image
2026-10-26 23:58:04,710 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7486  R 0.3663  mAP50 0.4000  mAP50-95 0.1880
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torchmetrics\utilities\prints.py:43: UserWarning: Metric `Kernel Inception Distance` will save all extracted features in buffer. For large datasets this may lead to large memory footprint.
  warnings.warn(*args, **kwargs)
2026-10-26 23:58:51,035 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 6.98  SSIM 0.2455  LPIPS 0.7902  FID 329.62  KID 0.3977+-0.0234
[...]
50 epochs completed in 0.217 hours.
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\detector\stage0\weights\last.pt, 5.5MB
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\detector\stage0\weights\best.pt, 5.5MB
Validating D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\detector\stage0\weights\best.pt...
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11n summary (fused): 101 layers, 2,582,932 parameters, 0 gradients, 6.4 GFLOPs
                   all       1013       8601      0.695      0.415      0.486       0.23
Speed: 0.1ms preprocess, 0.8ms inference, 0.0ms loss, 1.7ms postprocess per image
2026-10-27 00:16:09,508 INFO    t2o.engine.detector_stage: detector stage0: P 0.6949  R 0.4153  mAP50 0.4864  mAP50-95 0.2304
2026-10-27 00:16:09,530 INFO    t2o.engine.loop: stage 0 done: task_weight=1.0 epochs=339
2026-10-27 00:16:09,535 INFO    t2o.cli: stage 0 (task_weight=1.0): 339 epochs | zero-shot mAP50 0.4000 | LPIPS 0.7902 | fine-tuned mAP50 0.4864
=== FINISHED 2026-10-27 00:16:14 +03:00 ===
```

Derived from all 339 `epoch N took X s` lines and the memory lines of each campaign log:

| seed | median epoch | range | sum of epochs | peak allocated | peak reserved |
| --- | --- | --- | --- | --- | --- |
| 0 | 33.8 s | 30.4–36.6 s | 3.17 h | 5.57 GiB | 6.23 GiB |
| 1 | 34.2 s | 31.1–50.6 s | 3.20 h | 5.57 GiB | 6.23 GiB |
| 3 | 34.8 s | 30.2–38.1 s | 3.25 h | 5.57 GiB | 6.23 GiB |

The three stage summaries, side by side (`t2o.cli`, `t2o.metrics.fidelity` and `t2o.metrics.task`
lines above). Every mAP here is all four classes, on all 1,013 val images:

| seed | zero-shot mAP50 (`yolo11s`) | mAP50-95 | LPIPS | FID | PSNR | SSIM | fine-tuned mAP50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.4134 | 0.1903 | 0.7784 | 315.70 | 6.13 | 0.2430 | 0.4625 |
| 1 | 0.4098 | 0.1968 | 0.7416 | 334.58 | 6.42 | 0.2516 | 0.4738 |
| 3 | 0.4000 | 0.1880 | 0.7902 | 329.62 | 6.98 | 0.2455 | 0.4864 |
| mean | 0.4077 | 0.1917 | 0.7701 | 326.63 | 6.51 | 0.2467 | 0.4742 |

### RT-DETR judge

From the end of the first provenance block to the finish stamp, the provenance blocks removed,
ANSI escapes, per-batch progress lines and in-progress `Scanning` lines stripped:

```
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
rt-detr-l summary: 315 layers, 31,991,960 parameters, 0 gradients, 105.3 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 45.61.7 MB/s, size: 546.5 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/

val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\translated\val\labels.cache... 1013 images, 0 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 1013/1013  0.0s
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\translated\val\images\FLIR_09042.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\translated\val\images\FLIR_09055.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\translated\val\images\FLIR_09653.png: 1 duplicate labels removed

                   all       1013       8601      0.627      0.507      0.572       0.28
Speed: 1.6ms preprocess, 4.8ms inference, 0.0ms loss, 0.2ms postprocess per image
2026-10-27 09:11:34,000 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.6270  R 0.5072  mAP50 0.5720  mAP50-95 0.2799
2026-10-27 09:11:34,000 INFO    t2o.cli:   bicycle              AP50=0.4960
2026-10-27 09:11:34,000 INFO    t2o.cli:   car                  AP50=0.7897
2026-10-27 09:11:34,000 INFO    t2o.cli:   dog                  AP50=0.3031
2026-10-27 09:11:34,000 INFO    t2o.cli:   person               AP50=0.6992
2026-10-27 09:11:34,041 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\metrics.json stage 0
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
rt-detr-l summary: 315 layers, 31,991,960 parameters, 0 gradients, 105.3 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 861.6366.7 MB/s, size: 534.3 KB)

val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\translated\val\labels.cache... 1013 images, 0 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 1013/1013  0.0s
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\translated\val\images\FLIR_09042.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\translated\val\images\FLIR_09055.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\translated\val\images\FLIR_09653.png: 1 duplicate labels removed

                   all       1013       8601      0.593      0.435      0.512      0.265
Speed: 1.7ms preprocess, 4.8ms inference, 0.0ms loss, 0.3ms postprocess per image
2026-10-27 09:13:06,775 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5928  R 0.4351  mAP50 0.5121  mAP50-95 0.2649
2026-10-27 09:13:06,776 INFO    t2o.cli:   bicycle              AP50=0.4717
2026-10-27 09:13:06,776 INFO    t2o.cli:   car                  AP50=0.8086
2026-10-27 09:13:06,776 INFO    t2o.cli:   dog                  AP50=0.0785
2026-10-27 09:13:06,777 INFO    t2o.cli:   person               AP50=0.6897
2026-10-27 09:13:06,808 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\metrics.json stage 0
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
rt-detr-l summary: 315 layers, 31,991,960 parameters, 0 gradients, 105.3 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 1438.4278.5 MB/s, size: 579.7 KB)

val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\translated\val\labels.cache... 1013 images, 0 backgrounds, 0 corrupt: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 1013/1013  0.0s
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\translated\val\images\FLIR_09042.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\translated\val\images\FLIR_09055.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\translated\val\images\FLIR_09653.png: 1 duplicate labels removed

                   all       1013       8601      0.621      0.412      0.479      0.241
Speed: 1.5ms preprocess, 4.8ms inference, 0.0ms loss, 0.2ms postprocess per image
2026-10-27 09:42:13,518 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.6209  R 0.4121  mAP50 0.4791  mAP50-95 0.2413
2026-10-27 09:42:13,519 INFO    t2o.cli:   bicycle              AP50=0.4552
2026-10-27 09:42:13,519 INFO    t2o.cli:   car                  AP50=0.7700
2026-10-27 09:42:13,519 INFO    t2o.cli:   dog                  AP50=0.0363
2026-10-27 09:42:13,519 INFO    t2o.cli:   person               AP50=0.6548
2026-10-27 09:42:13,569 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\metrics.json stage 0
=== FINISHED 2026-10-27 09:42:16 +03:00 ===
```

### Non-inferiority

The bootstrap, verdict and finish lines, verbatim. The rest of the log is six `data.yaml` lines,
a `seed N: scoring the loop and ModTr` line and two ultralytics evaluations per seed, each closing
with a `captured per-image stats for 1013 images` line:

```
2026-10-27 11:32:48,433 INFO    t2o.analysis.resolvability: seed-averaged bootstrap: 3 seed(s) x 1013 images x 10000 resamples, delta CI [-7.00, -3.65] mAP points
2026-10-27 11:32:48,890 INFO    __main__: wrote D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-non-inferiority-flir\modtr-non-inferiority.csv
2026-10-27 11:32:48,890 INFO    __main__: per-seed deltas (loop - ModTr): s0 -3.87, s1 -5.42, s3 -8.10; sd 2.14 mAP points
2026-10-27 11:32:48,890 INFO    __main__: verdict: NOT NON-INFERIOR -- lower bound -7.00 <= -5.0 mAP points (delta -5.80)
=== FINISHED 2026-10-27 11:32:50 +03:00 ===
```

The six ultralytics `all` rows, in log order (images, instances, P, R, mAP50, mAP50-95):

```
e3f-loop-s0     all       1013       8601      0.531      0.342      0.375      0.179
modtr-flir-s0   all       1013       8601      0.512      0.372      0.413       0.19
e3f-loop-s1     all       1013       8601      0.483      0.314      0.356      0.169
modtr-flir-s1   all       1013       8601      0.758      0.365       0.41      0.197
e3f-loop-s3     all       1013       8601      0.471      0.285      0.319      0.152
modtr-flir-s3   all       1013       8601      0.749      0.366        0.4      0.188
```

The run label at the start of each row is added here, from the `captured per-image stats ... from
<path>` line that follows it in the log. The rest of each row is verbatim.

`logs/2026-10-27-modtr-non-inferiority-flir.csv`, verbatim:

```
seed,pool,loop_map50,modtr_map50,delta,low,high,half_width,seed_sd,margin,non_inferior,resamples,bootstrap_seed,loop_data,modtr_data,weights
0,1013,0.37462111901913786,0.4133647976309352,-3.874367861179734,,,,,5.0,,10000,0,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s0\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\stage0\translated\data.yaml,runs\reference-flir-yolo11s\weights\best.pt
1,1013,0.35567280630879894,0.4098347957752315,-5.416198946643258,,,,,5.0,,10000,0,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s1\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\stage0\translated\data.yaml,runs\reference-flir-yolo11s\weights\best.pt
3,1013,0.3190179458341064,0.40004402366654973,-8.10260778324433,,,,,5.0,,10000,0,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-loop-s3\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\stage0\translated\data.yaml,runs\reference-flir-yolo11s\weights\best.pt
mean,1013,0.3497706237206811,0.4077478723575722,-5.797724863689107,-7.001401707604438,-3.6538294314848763,1.6737861380597807,2.1397838041887463,5.0,False,10000,0,runs/e3f-loop-s{seed}/stage3/translated,runs/modtr-flir-s{seed}/stage0/translated,runs\reference-flir-yolo11s\weights\best.pt
```

Re-render the verdict without re-training, on the server: the
`scripts/modtr_non_inferiority.py` command above with a new `--out`, since the script refuses to
overwrite its CSV. Per-seed `metrics.json`, including the RT-DETR result written back, is at
`runs/modtr-flir-s{0,1,3}/metrics.json` there.

## Findings

### F205 — Margin's non-inferiority test fails on FLIR: loop − ModTr −5.80 mAP points, 95% CI [−7.00, −3.65]

ADR-02's pass rule is that the lower bound of the seed-averaged paired 95% CI on loop − ModTr is
above −5 mAP@50 points. On FLIR, scored as ADR-02's script scores it (all four classes, the
reference `yolo11s`, all 1,013 val images), it is −7.00.

| loop | seed-averaged delta | 95% CI | per-seed deltas (s0, s1, s3) | seed sd |
| --- | --- | --- | --- | --- |
| e3f (pix2pix) | −5.80 | [−7.00, −3.65] | −3.87, −5.42, −8.10 | 2.14 |

- **Every per-seed delta is negative.** The loop's 4-class mAP@50 is 0.3746, 0.3557 and 0.3190;
  ModTr's is 0.4134, 0.4098 and 0.4000.
- **The CI excludes zero from below.** So at 95%, ModTr is ahead of the loop on this endpoint.
- **The CI does not show the loop worse than ModTr by more than δ either.** Its upper end, −3.65,
  is above −5. The verdict ADR-02 defines is "not shown non-inferior", and the point estimate,
  −5.80, lies past −δ.
- **mAP@50:95 points the same way.** Loop 0.1665 against ModTr 0.1917 (seed means), −2.52 points.
  `goal.md`'s main Margin bar also asks for a non-negative mAP@50:95 delta.
- **This is the opposite of the custom set.** There, record 030's F200 put the loop 13.78 (e3b)
  and 15.11 (e3t) points ahead, with lower bounds about 15 points clear of −5.
- **The CI is half as wide as the custom set's.** Half-width 1.67 on 1,013 images, against 3.40
  (e3b) on 153. ADR-02 expected a half-width of about 4 at n = 153.
- **The loop arm reproduces record 016.** The loop's three 4-class values equal record 026's tidy
  CSV `zero_shot.map50` for the same seeds (0.374621, 0.355673, 0.319018) to every printed digit.
  That is the same exports read again, not second evidence.

### F206 — Under RT-DETR, ModTr's lead on FLIR grows to 10.41 points on 4-class and 10.83 on the 3-class primary

RT-DETR-l zero-shot mAP@50 for ModTr comes from this run's log. The loop's comes from record 026's
`judges.rtdetr` columns for the same seeds. The 3-class value is the mean of bicycle, car and
person AP, as `metrics/task.py::primary_mean` takes it:

| | ModTr s0 / s1 / s3 | ModTr mean | loop mean | loop − ModTr per seed | loop − ModTr mean |
| --- | --- | --- | --- | --- | --- |
| 4-class | 0.5720 / 0.5121 / 0.4791 | 0.5211 | 0.4169 | −15.36, −10.92, −4.96 | **−10.41** |
| 3-class (bicycle, car, person) | 0.6616 / 0.6567 / 0.6267 | 0.6483 | 0.5400 | −10.98, −12.19, −9.33 | **−10.83** |

- **The dog class does not carry the gap under RT-DETR.** Dropping it changes the mean gap by
  0.42 points, and all six per-seed gaps stay negative. Dog's AP for ModTr swings across seeds
  (0.3031, 0.0785, 0.0363), as F94 warned a 13-instance class would.
- **Under `yolo11s`, ModTr's 3-class value is not in any log.** The campaign log and the
  non-inferiority log print only the `all` row. What can be said is arithmetic: for the 3-class
  loop − ModTr gap to reach zero, ModTr's `yolo11s` dog AP would need to average ≥ 0.26. That is
  above the 0.2396 the same judge scores on real visible images (record 014), and the loop's is
  0.0298.
- **The judge moves the gap toward ModTr, as on the custom set**, but here both judges agree on
  its sign. On FLIR, RT-DETR lifts ModTr by 11.34 points over `yolo11s`, the loop by 6.71 and the
  control by 9.66. On the custom set (F203), it lifted ModTr by 9.52 and the loop by 1.52.
- **These are point estimates, without a CI.** ADR-02 pre-registered only the `yolo11s` verdict.

### F207 — On FLIR, ModTr is the best zero-shot translator measured: 6.43 points above de-rolled raw thermal, where the loop is 0.63 above it

All-class (4-class) `yolo11s` mAP@50 on the same 1,013 val images. The loop and control values are
record 026's tidy CSV rows for the same seeds. The floor and ceiling are record 014's de-rolled
gate, scored with the same judge:

| arm | s0 | s1 | s3 | mean | mAP50-95 mean | fine-tuned mAP50 mean |
| --- | --- | --- | --- | --- | --- | --- |
| ModTr | 0.4134 | 0.4098 | 0.4000 | **0.4077** | 0.1917 | 0.4742 |
| e3f loop (stage 3) | 0.3746 | 0.3557 | 0.3190 | 0.3498 | 0.1665 | 0.4307 |
| e3f control (stage 3) | 0.1872 | 0.0920 | 0.0002 | 0.0931 | 0.0436 | 0.3326 |
| raw thermal, de-rolled (floor) | | | | 0.3435 | 0.1514 | |
| visible (ceiling) | | | | 0.5523 | 0.2767 | |

- **ModTr closes 31% of the 4-class headroom**: (0.4077 − 0.3435) / (0.5523 − 0.3435). The
  loop closes 3%.
- **ModTr is ahead of the control at every seed**, by +22.62, +31.78 and +39.98 points.
- **The fine-tuned arm agrees.** ModTr's fine-tuned `yolo11n` reaches 0.4625, 0.4738 and 0.4864,
  above the loop's 0.4289, 0.4255 and 0.4378 at every seed (+4.35 mean).
- **This reverses the custom set's order.** There, F201 found ModTr below both no-loop controls on
  every seed. On FLIR it is above the loop, and far above the control.

### F208 — ModTr's FLIR exports are farther from visible than the custom set's: val LPIPS 0.7701, 1.8× the e3f loop's 0.4197

Seed means on the 1,013 val images. The loop and control values are record 026's tidy CSV rows
for the same seeds:

| arm | LPIPS | FID | PSNR | SSIM |
| --- | --- | --- | --- | --- |
| ModTr | 0.7701 (0.7416–0.7902) | 326.63 (315.70–334.58) | 6.51 | 0.2467 |
| e3f loop (stage 3) | 0.4197 | 94.18 | 12.54 | 0.4760 |
| e3f control (stage 3) | 0.4658 | 141.49 | 11.10 | 0.3077 |

- **It is the same pattern as the custom set, stronger.** F202 measured ModTr's custom-set LPIPS
  at 0.6387, 2.1× the e3b loop's.
- **On FLIR, the arm farthest from visible on all four fidelity metrics is the one that detects
  best** (F207). That matters wherever ModTr is compared on anything but mAP, Faithfulness included.

### F209 — A 339-epoch FLIR ModTr seed costs 3 h 35 min – 3 h 41 min on one A100 without the monitor, inside F199's ≤ 6.9 h prediction

Provenance to finish stamp is 3 h 34 min 49 s, 3 h 35 min 54 s and 3 h 40 min 54 s. That covers
training, the export, zero-shot and fidelity evaluation, and the 50-epoch fine-tuned detector.

- **F199 predicted at most about 6.9 h** a seed, from a monitor-free epoch estimate of 58.3 s. The
  observed median epoch is 33.8–34.8 s, so that estimate was 1.7× too high. The prediction holds
  with about 3.2 h to spare.
- **A FLIR epoch costs a little less than a custom-set one**: 33.8–34.8 s median here against
  37.2–39.2 s (F204), both on 600 training images.
- **Peak memory is 6.23 GiB reserved** on every seed, against 5.82 on the custom set (F204).
- **The non-inferiority bootstrap is the slow part of scoring.** 3 seeds × 1,013 images × 10,000
  resamples took 80 min 30 s (10:12:18 → 11:32:48). Scoring the six exports before it took
  16 min 48 s.

## Provenance caveats

- **The endpoint is 4-class, not FLIR's 3-class primary.** Since record 012's F94, FLIR's primary
  is 3-class mAP@50 (bicycle, car, person), because dog has 13 val instances carrying a quarter of
  a 4-class mean. Record 016 fixed that endpoint before reading its numbers. ADR-02 names
  "mAP@50" with no class subset, and was written with the 5-class custom set in mind.
  `scripts/modtr_non_inferiority.py:111-112` takes `map50_from_stats` over every class. So F205 is
  a verdict on the endpoint ADR-02's code pre-registered, which is the one record 016 demoted for
  FLIR. F206 bears on whether that matters (under RT-DETR it does not), but no `yolo11s` 3-class
  value for ModTr exists in these logs.
- **The CSV carries no task segment in its name** (`2026-10-27-modtr-non-inferiority-flir.csv`),
  like the custom set's two. It is the server's
  `runs/modtr-non-inferiority-flir/modtr-non-inferiority.csv`, copied back and renamed.
- **The non-inferiority log is named `-06-`.** The runbook wrote `-05-` (MODTR-BASELINE-05 owns
  the script), as in record 030. It is cited here as saved.
- **The loop and control arms are borrowed.** Their RT-DETR, fidelity, fine-tuned and control
  values are record 026's tidy CSV, from record 016's campaign at an earlier SHA. Only the loop's
  4-class `yolo11s` mAP was re-scored here, and it matched.
- **The arms are not compute-matched.** The e3f loop is pix2pix, 4 stages × 100 epochs, at
  `grad_scale` 0.15, with a 21.1% detection share at stage 3 (record 016). ModTr is 339
  detection-only epochs, the budget record 029 set.
- **FLIR's pix2pix arms carry record 016's roll confound (F119).** A 5.90 px roll sits between
  pix2pix's input and its `l2` + `lpips` target, and the generator can only meet it by blurring.
  ModTr has no reconstruction target, so the confound cannot reach it. This may favour ModTr on
  FLIR alone. It is a reading of the two objectives, not a measurement.
- **The in-loop detector was most likely trained on FLIR's visible training images**, as ADR-05's
  consequences note. ModTr and the loop train against it the same way, so the comparison is
  matched. But the fine-tuned arm starts from a detector that may already know the training scenes.
- **Three val images lose one duplicate label each** (`FLIR_09042`, `FLIR_09055`, `FLIR_09653`) in
  every evaluation. The non-inferiority log shows the same three in both arms, so the pairing holds.
- **The RT-DETR log has a 27 min 38 s gap** between seed 1's last line (09:13:06) and seed 3's
  provenance block (09:40:44). Each evaluation took about 1.5 min. The log does not say why. It
  affects no number.
- **One RT-DETR judge, no CI.** F206 inherits record 026's caveats about that judge, F188's
  non-reproducible training among them.
- **The runs' `config_hash` is not in the logs.** It is in `runs/modtr-flir-s{0,1,3}/config.yaml`
  on the server.
- **Slow image access.** ultralytics warns "Slow image access detected" on most first-pass
  evaluations, as in records 027 and 030. That cost time, not numbers.

## Next

Margin now splits by dataset. On the custom set the loop is non-inferior to ModTr and well ahead
(F200). On FLIR it is not shown non-inferior, and ModTr is ahead under both judges (F205, F206).

- **Which endpoint FLIR's Margin is read on is a decision, not a re-run.** ADR-02's code scored
  4-class, and that is the verdict as pre-registered. A 3-class `yolo11s` re-read is cheap: the
  per-class zero-shot AP should already be in `runs/modtr-flir-s{0,1,3}/metrics.json` on the
  server, as it is for every e3f run in record 026's CSV. That was not checked. Any 3-class
  verdict is a secondary analysis, labelled as one.
- **Prediction, posted now.** On 3-class `yolo11s` mAP@50, ModTr's seed mean will be above the
  loop's 0.4564, so the loop trails ModTr on FLIR's own primary endpoint too. It would fail only if
  ModTr's `yolo11s` dog AP averages ≥ 0.26 (F206).
- **The paper has to say why the two datasets disagree.** The roll confound (F119) is the leading
  reading, because it reaches pix2pix's objective and not ModTr's. It is a hypothesis this run
  does not test.
