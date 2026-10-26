# 030 — ModTr custom-set campaign: the loop is non-inferior, and 14–15 points ahead

**Date:** 2026-10-25 (scored 2026-10-26) · **Task:** MODTR-BASELINE-06 · **Machine:** VSIBMEOT05 (`cuda:0`) · **Wall clock:** 13 h 53 min 36 s for the three seeds back to back (2026-10-25 12:49:35 → 2026-10-26 02:43:11 +03:00): s0 4 h 43 min 33 s, s1 4 h 30 min 36 s, s3 4 h 39 min 23 s. Then 4 min 5 s for the RT-DETR judge and 37 min 45 s for both non-inferiority runs
**git_sha:** `pre-spec-migration-184-gddeeafe` (`git describe --always --dirty`, so a clean tree at `ddeeafe`) · **W&B:** none (`runtime.wandb: false` in `experiments/modtr_custom.yaml:106`; no W&B line in any log) · **Log:** `logs/2026-10-25-modtr-baseline-06-campaign-s0.txt`, `logs/2026-10-25-modtr-baseline-06-campaign-s1.txt`, `logs/2026-10-25-modtr-baseline-06-campaign-s3.txt`, `logs/2026-10-26-modtr-baseline-06-rtdetr-custom.txt`, `logs/2026-10-26-modtr-baseline-06-non-inferiority-custom.txt`, with `logs/2026-10-26-modtr-non-inferiority-e3b.csv` and `logs/2026-10-26-modtr-non-inferiority-e3t.csv`

## Command

From each log's provenance block, as run, in order:

```
uv run t2o loop --config experiments/modtr_custom.yaml --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --in-loop-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --eval-init-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --device cuda:0 --seed 0 --name modtr-s0
uv run t2o loop --config experiments/modtr_custom.yaml --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --in-loop-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --eval-init-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --device cuda:0 --seed 1 --name modtr-s1
uv run t2o loop --config experiments/modtr_custom.yaml --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --in-loop-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --eval-init-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --device cuda:0 --seed 3 --name modtr-s3
uv run t2o evaluate --weights runs/reference-rtdetr-l/weights/best.pt --data runs/modtr-s0/stage0/translated/data.yaml --write-back --judge rtdetr --device 0
uv run t2o evaluate --weights runs/reference-rtdetr-l/weights/best.pt --data runs/modtr-s1/stage0/translated/data.yaml --write-back --judge rtdetr --device 0
uv run t2o evaluate --weights runs/reference-rtdetr-l/weights/best.pt --data runs/modtr-s3/stage0/translated/data.yaml --write-back --judge rtdetr --device 0
uv run python scripts/modtr_non_inferiority.py --modtr-template runs/modtr-s{seed}/stage0/translated --out runs/modtr-non-inferiority-e3b --device cuda:0
uv run python scripts/modtr_non_inferiority.py --loop-template runs/e3t-loop-s{seed}/stage3/translated --modtr-template runs/modtr-s{seed}/stage0/translated --out runs/modtr-non-inferiority-e3t --device cuda:0
```

The non-inferiority script's defaults fill what the first of its two commands leaves out: the
loop template `runs/e3b-loop-s{seed}/stage3/translated`, seeds 0, 1 and 3, and the reference
`yolo11s` weights. The log's `data.yaml` lines and the CSV's `loop_data` and `weights` columns
confirm all three.

## Environment

Seed 0's block, verbatim:

```
=== PROVENANCE ===
captured: 2026-10-25 12:49:35 +03:00
hostname: VSIBMEOT05
os: Microsoft Windows Server 2022 Standard 10.0.20348 (build 20348)
cpu: AMD EPYC 7763 64-Core Processor
cpu_logical_cores: 64
ram_total: 128.0 GB
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16; NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16
cuda_driver: 13.0
git_sha: pre-spec-migration-184-gddeeafe
working_dir: D:\Atick\GitHub\Thermal-To-Optical-Translation
command: uv run t2o loop --config experiments/modtr_custom.yaml --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --in-loop-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --eval-init-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --device cuda:0 --seed 0 --name modtr-s0
python: 3.12.7
torch: 2.12.1+cu130
cuda_torch_build: 13.0
cudnn: 92000
cuda_available: True
gpu0_torch: NVIDIA A100-SXM4-40GB, 39.7 GiB, sm 8.0
=== END PROVENANCE ===
```

The other seven blocks are the same apart from `captured:` and their command lines. Seeds 1 and
3: 2026-10-25 17:33:10 and 22:03:48. The RT-DETR judge: 2026-10-26 10:54:22, 10:55:41 and
10:57:05. Non-inferiority: 11:23:07 (e3b) and 11:42:21 (e3t). All eight carry the same
`git_sha`. As in record 029, the Python probe ran: the `python`, `torch` and `cudnn` lines are
present and no `error:` line is.

## Configuration

`experiments/modtr_custom.yaml` at `ddeeafe`, with the budget set by record 028.

| Axis | Value | Where it is evidenced |
| --- | --- | --- |
| Translator | ModTr (ResNet34 U-Net, fusion `clamp(x·h(x), 0, 1)`), detection-only objective | modtr-baseline ADR-01; `experiments/modtr_custom.yaml` |
| Budget | 1 stage × 400 epochs, `task_weight` 1.0, `grad_scale` 1.0 | `stage 0 done: task_weight=1.0 epochs=400`; `modtr_custom.yaml:33, :65`; F193 |
| Seeds | 0, 1, 3, so each pairs with an e3b and an e3t seed | the commands |
| Training data | all 600 training pairs of the custom set | `600 paired samples, 600 annotated` |
| In-loop detector | `optical_best_n_1.pt` (`yolo11n`), the same one the e3b and e3t loops trained against (margin-gate ADR-01) | the command's `--in-loop-weights` |
| Zero-shot scorer | reference `yolo11s`, `runs/reference-yolo11s/weights/best.pt` | `YOLO11s summary (fused)`; the CSVs' `weights` column |
| Out-of-family judge | RT-DETR-l, `runs/reference-rtdetr-l/weights/best.pt` (record 026) | the RT-DETR commands |
| Fine-tuned arm | `optical_best_n_1.pt` fine-tuned 50 epochs on each seed's translated train split | `epochs=50`, `model=...optical_best_n_1.pt` in the ultralytics args line |
| Monitor | off | no `--monitor-map`; no per-epoch `exported` or `eval` lines |
| Non-inferiority test | ADR-02: paired 95% bootstrap over the 153 val images on seed-averaged mAP@50, 10,000 resamples, bootstrap seed 0; pass when the lower bound > −5 points | `seed-averaged bootstrap: 3 seed(s) x 153 images x 10000 resamples`; the CSVs' `margin`, `resamples` and `bootstrap_seed` columns |
| Comparison arms | e3b loop (pix2pix, record 007) and e3t loop (turbo, record 010), stage-3 exports, seeds 0, 1, 3 | the log's `data.yaml` lines |
| `config_hash` | not in the logs | — |

## Results

### Campaign

Each campaign log is 1,153 lines. Per seed, the opening through epoch 0, the end of training
through the zero-shot and fidelity evaluation, and the fine-tuned detector's close follow.
ANSI colour escapes, the per-batch `Class Images Instances Box(P ...` progress lines and
ultralytics' `New https://pypi.org/project/ultralytics/8.4.189 available` notice are stripped.
`[...]` marks epochs 1–398, and then the fine-tuned detector's training, the ultralytics args
line, the model summary and 50 epochs of progress. All of it is in the logs.

```
2026-10-25 12:50:01,511 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-10-25 12:50:01,571 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\train\visible\images: 600 paired samples, 600 annotated
2026-10-25 12:50:02,104 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-10-25 12:50:12,860 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\train\visible\images: 600 paired samples, 600 annotated
2026-10-25 12:50:15,308 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\val\visible\images: 153 paired samples, 153 annotated
2026-10-25 12:50:53,354 INFO    t2o.engine.trainer: epoch 0 took 38.0 s
2026-10-25 12:50:53,354 INFO    t2o.engine.trainer: epoch 0 peak memory: 5.20 GiB allocated, 5.74 GiB reserved of 39.70 GiB
[...]
2026-10-25 17:18:26,090 INFO    t2o.engine.trainer: epoch 399 took 44.0 s
2026-10-25 17:18:26,090 INFO    t2o.engine.trainer: epoch 399 peak memory: 5.20 GiB allocated, 5.82 GiB reserved of 39.70 GiB
2026-10-25 17:18:27,258 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\train\visible\images: 600 paired samples, 600 annotated
2026-10-25 17:19:32,908 INFO    t2o.engine.export: exported 600 translated images to runs\modtr-s0\stage0\translated\train (normalize=clamp)
2026-10-25 17:19:33,013 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\val\visible\images: 153 paired samples, 153 annotated
2026-10-25 17:20:03,294 INFO    t2o.engine.export: exported 153 translated images to runs\modtr-s0\stage0\translated\val (normalize=clamp)
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 16.15.2 MB/s, size: 338.3 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s0\stage0\translated\val\labels... 153 images, 4 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 153/153 183.4it/s 0.8s
val: New cache created: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s0\stage0\translated\val\labels.cache
                   all        153        423      0.841      0.644      0.696      0.466
Speed: 1.8ms preprocess, 2.8ms inference, 0.0ms loss, 1.3ms postprocess per image
2026-10-25 17:20:54,603 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8415  R 0.6442  mAP50 0.6960  mAP50-95 0.4661
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torchmetrics\utilities\prints.py:43: UserWarning: Metric `Kernel Inception Distance` will save all extracted features in buffer. For large datasets this may lead to large memory footprint.
  warnings.warn(*args, **kwargs)
2026-10-25 17:21:04,543 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 6.91  SSIM 0.1338  LPIPS 0.6590  FID 191.16  KID 0.1214+-0.0101
[...]
50 epochs completed in 0.159 hours.
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s0\stage0\detector\stage0\weights\last.pt, 5.5MB
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s0\stage0\detector\stage0\weights\best.pt, 5.5MB

Validating D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s0\stage0\detector\stage0\weights\best.pt...
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11n summary (fused): 101 layers, 2,583,127 parameters, 0 gradients, 6.4 GFLOPs
                   all        153        423      0.919      0.854      0.917      0.653
Speed: 0.1ms preprocess, 1.0ms inference, 0.0ms loss, 2.0ms postprocess per image
2026-10-25 17:33:05,678 INFO    t2o.engine.detector_stage: detector stage0: P 0.9190  R 0.8537  mAP50 0.9172  mAP50-95 0.6527
2026-10-25 17:33:05,699 INFO    t2o.engine.loop: stage 0 done: task_weight=1.0 epochs=400
2026-10-25 17:33:05,704 INFO    t2o.cli: stage 0 (task_weight=1.0): 400 epochs | zero-shot mAP50 0.6960 | LPIPS 0.6590 | fine-tuned mAP50 0.9172
=== FINISHED 2026-10-25 17:33:08 +03:00 ===

2026-10-25 17:33:43,513 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-10-25 17:33:43,572 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\train\visible\images: 600 paired samples, 600 annotated
2026-10-25 17:33:44,113 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-10-25 17:33:45,106 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\train\visible\images: 600 paired samples, 600 annotated
2026-10-25 17:33:45,212 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\val\visible\images: 153 paired samples, 153 annotated
2026-10-25 17:34:20,526 INFO    t2o.engine.trainer: epoch 0 took 35.3 s
2026-10-25 17:34:20,527 INFO    t2o.engine.trainer: epoch 0 peak memory: 5.20 GiB allocated, 5.74 GiB reserved of 39.70 GiB
[...]
2026-10-25 21:51:35,391 INFO    t2o.engine.trainer: epoch 399 took 31.8 s
2026-10-25 21:51:35,391 INFO    t2o.engine.trainer: epoch 399 peak memory: 5.20 GiB allocated, 5.82 GiB reserved of 39.70 GiB
2026-10-25 21:51:36,327 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\train\visible\images: 600 paired samples, 600 annotated
2026-10-25 21:52:29,896 INFO    t2o.engine.export: exported 600 translated images to runs\modtr-s1\stage0\translated\train (normalize=clamp)
2026-10-25 21:52:30,012 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\val\visible\images: 153 paired samples, 153 annotated
2026-10-25 21:52:49,886 INFO    t2o.engine.export: exported 153 translated images to runs\modtr-s1\stage0\translated\val (normalize=clamp)
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 35.46.9 MB/s, size: 452.0 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s1\stage0\translated\val\labels... 153 images, 4 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 153/153 274.3it/s 0.6s
val: New cache created: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s1\stage0\translated\val\labels.cache
                   all        153        423      0.782      0.656      0.708      0.485
Speed: 1.3ms preprocess, 2.5ms inference, 0.0ms loss, 1.2ms postprocess per image
2026-10-25 21:53:34,098 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7819  R 0.6562  mAP50 0.7075  mAP50-95 0.4849
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torchmetrics\utilities\prints.py:43: UserWarning: Metric `Kernel Inception Distance` will save all extracted features in buffer. For large datasets this may lead to large memory footprint.
  warnings.warn(*args, **kwargs)
2026-10-25 21:53:44,834 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 6.97  SSIM 0.1156  LPIPS 0.6239  FID 193.70  KID 0.1294+-0.0078
[...]
50 epochs completed in 0.130 hours.
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s1\stage0\detector\stage0\weights\last.pt, 5.5MB
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s1\stage0\detector\stage0\weights\best.pt, 5.5MB

Validating D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s1\stage0\detector\stage0\weights\best.pt...
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11n summary (fused): 101 layers, 2,583,127 parameters, 0 gradients, 6.4 GFLOPs
                   all        153        423      0.915      0.821      0.892      0.625
Speed: 0.1ms preprocess, 0.9ms inference, 0.0ms loss, 1.7ms postprocess per image
2026-10-25 22:03:43,505 INFO    t2o.engine.detector_stage: detector stage0: P 0.9150  R 0.8207  mAP50 0.8915  mAP50-95 0.6250
2026-10-25 22:03:43,526 INFO    t2o.engine.loop: stage 0 done: task_weight=1.0 epochs=400
2026-10-25 22:03:43,526 INFO    t2o.cli: stage 0 (task_weight=1.0): 400 epochs | zero-shot mAP50 0.7075 | LPIPS 0.6239 | fine-tuned mAP50 0.8915
=== FINISHED 2026-10-25 22:03:46 +03:00 ===

2026-10-25 22:05:02,173 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-10-25 22:05:02,300 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\train\visible\images: 600 paired samples, 600 annotated
2026-10-25 22:05:02,993 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-10-25 22:05:03,858 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\train\visible\images: 600 paired samples, 600 annotated
2026-10-25 22:05:03,947 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\val\visible\images: 153 paired samples, 153 annotated
2026-10-25 22:05:42,606 INFO    t2o.engine.trainer: epoch 0 took 38.7 s
2026-10-25 22:05:42,606 INFO    t2o.engine.trainer: epoch 0 peak memory: 5.19 GiB allocated, 5.74 GiB reserved of 39.70 GiB
[...]
2026-10-26 02:30:30,151 INFO    t2o.engine.trainer: epoch 399 took 35.8 s
2026-10-26 02:30:30,151 INFO    t2o.engine.trainer: epoch 399 peak memory: 5.20 GiB allocated, 5.82 GiB reserved of 39.70 GiB
2026-10-26 02:30:31,324 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\train\visible\images: 600 paired samples, 600 annotated
2026-10-26 02:31:32,536 INFO    t2o.engine.export: exported 600 translated images to runs\modtr-s3\stage0\translated\train (normalize=clamp)
2026-10-26 02:31:32,620 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\val\visible\images: 153 paired samples, 153 annotated
2026-10-26 02:31:55,449 INFO    t2o.engine.export: exported 153 translated images to runs\modtr-s3\stage0\translated\val (normalize=clamp)
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 25.23.9 MB/s, size: 454.9 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s3\stage0\translated\val\labels... 153 images, 4 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 153/153 203.8it/s 0.8s
val: New cache created: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s3\stage0\translated\val\labels.cache
                   all        153        423      0.764      0.689      0.733       0.49
Speed: 2.0ms preprocess, 3.3ms inference, 0.0ms loss, 1.4ms postprocess per image
2026-10-26 02:32:38,981 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7640  R 0.6887  mAP50 0.7328  mAP50-95 0.4895
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torchmetrics\utilities\prints.py:43: UserWarning: Metric `Kernel Inception Distance` will save all extracted features in buffer. For large datasets this may lead to large memory footprint.
  warnings.warn(*args, **kwargs)
2026-10-26 02:32:48,724 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 6.99  SSIM 0.1177  LPIPS 0.6333  FID 197.02  KID 0.1319+-0.0090
[...]
50 epochs completed in 0.138 hours.
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s3\stage0\detector\stage0\weights\last.pt, 5.5MB
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s3\stage0\detector\stage0\weights\best.pt, 5.5MB

Validating D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s3\stage0\detector\stage0\weights\best.pt...
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11n summary (fused): 101 layers, 2,583,127 parameters, 0 gradients, 6.4 GFLOPs
                   all        153        423      0.928      0.814      0.891      0.638
Speed: 0.1ms preprocess, 0.7ms inference, 0.0ms loss, 1.7ms postprocess per image
2026-10-26 02:43:08,029 INFO    t2o.engine.detector_stage: detector stage0: P 0.9284  R 0.8137  mAP50 0.8914  mAP50-95 0.6380
2026-10-26 02:43:08,061 INFO    t2o.engine.loop: stage 0 done: task_weight=1.0 epochs=400
2026-10-26 02:43:08,069 INFO    t2o.cli: stage 0 (task_weight=1.0): 400 epochs | zero-shot mAP50 0.7328 | LPIPS 0.6333 | fine-tuned mAP50 0.8914
=== FINISHED 2026-10-26 02:43:11 +03:00 ===
```

Derived from all 400 `epoch N took X s` lines and the memory lines of each campaign log:

| seed | median epoch | range | sum of epochs | peak reserved |
| --- | --- | --- | --- | --- |
| 0 | 39.2 s | 31.1–49.8 s | 4.39 h | 5.82 GiB |
| 1 | 37.2 s | 30.7–55.4 s | 4.22 h | 5.82 GiB |
| 3 | 38.4 s | 29.8–50.0 s | 4.34 h | 5.82 GiB |

The three stage summaries, side by side (`t2o.cli`, `t2o.metrics.fidelity` and `t2o.metrics.task`
lines above):

| seed | zero-shot mAP50 (`yolo11s`) | mAP50-95 | LPIPS | FID | PSNR | SSIM | fine-tuned mAP50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.6960 | 0.4661 | 0.6590 | 191.16 | 6.91 | 0.1338 | 0.9172 |
| 1 | 0.7075 | 0.4849 | 0.6239 | 193.70 | 6.97 | 0.1156 | 0.8915 |
| 3 | 0.7328 | 0.4895 | 0.6333 | 197.02 | 6.99 | 0.1177 | 0.8914 |
| mean | 0.7121 | | 0.6387 | 193.96 | 6.96 | 0.1224 | 0.9000 |

### RT-DETR judge

From the end of the first provenance block to the finish stamp, the provenance blocks removed,
ANSI escapes and per-batch progress lines stripped:

```
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
rt-detr-l summary: 315 layers, 31,994,015 parameters, 0 gradients, 105.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 18.35.0 MB/s, size: 408.9 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/

                   all        153        423      0.888      0.714      0.801       0.52
Speed: 1.3ms preprocess, 5.5ms inference, 0.0ms loss, 0.2ms postprocess per image
2026-10-26 10:55:36,896 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8882  R 0.7140  mAP50 0.8007  mAP50-95 0.5197
2026-10-26 10:55:36,896 INFO    t2o.cli:   Fuse                 AP50=0.8360
2026-10-26 10:55:36,896 INFO    t2o.cli:   Pole                 AP50=0.9231
2026-10-26 10:55:36,896 INFO    t2o.cli:   Switch               AP50=0.5347
2026-10-26 10:55:36,896 INFO    t2o.cli:   Transformer          AP50=0.9090
2026-10-26 10:55:36,938 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s0\metrics.json stage 0
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
rt-detr-l summary: 315 layers, 31,994,015 parameters, 0 gradients, 105.4 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 1469.9507.4 MB/s, size: 446.6 KB)

                   all        153        423      0.832      0.727      0.797      0.525
Speed: 1.9ms preprocess, 5.4ms inference, 0.0ms loss, 0.2ms postprocess per image
2026-10-26 10:57:01,065 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8324  R 0.7275  mAP50 0.7970  mAP50-95 0.5250
2026-10-26 10:57:01,066 INFO    t2o.cli:   Fuse                 AP50=0.8166
2026-10-26 10:57:01,066 INFO    t2o.cli:   Pole                 AP50=0.9223
2026-10-26 10:57:01,066 INFO    t2o.cli:   Switch               AP50=0.5886
2026-10-26 10:57:01,066 INFO    t2o.cli:   Transformer          AP50=0.8603
2026-10-26 10:57:01,116 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s1\metrics.json stage 0
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
rt-detr-l summary: 315 layers, 31,994,015 parameters, 0 gradients, 105.4 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 1162.8293.4 MB/s, size: 358.5 KB)

                   all        153        423      0.882      0.764      0.824      0.533
Speed: 1.4ms preprocess, 6.0ms inference, 0.0ms loss, 0.2ms postprocess per image
2026-10-26 10:58:24,974 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8823  R 0.7638  mAP50 0.8241  mAP50-95 0.5331
2026-10-26 10:58:24,974 INFO    t2o.cli:   Fuse                 AP50=0.8540
2026-10-26 10:58:24,974 INFO    t2o.cli:   Pole                 AP50=0.9269
2026-10-26 10:58:24,974 INFO    t2o.cli:   Switch               AP50=0.5905
2026-10-26 10:58:24,974 INFO    t2o.cli:   Transformer          AP50=0.9248
2026-10-26 10:58:25,033 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s3\metrics.json stage 0
=== FINISHED 2026-10-26 10:58:27 +03:00 ===
```

### Non-inferiority

The bootstrap, verdict and finish lines, verbatim. The rest of the log is six `data.yaml` lines,
three `captured per-image stats` lines per seed and ultralytics banners, per run:

```
2026-10-26 11:42:16,971 INFO    t2o.analysis.resolvability: seed-averaged bootstrap: 3 seed(s) x 153 images x 10000 resamples, delta CI [10.20, 17.01] mAP points
2026-10-26 11:42:17,006 INFO    __main__: wrote D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-non-inferiority-e3b\modtr-non-inferiority.csv
2026-10-26 11:42:17,006 INFO    __main__: per-seed deltas (loop - ModTr): s0 +15.88, s1 +14.00, s3 +11.46; sd 2.22 mAP points
2026-10-26 11:42:17,007 INFO    __main__: verdict: NON-INFERIOR -- lower bound 10.20 > -5.0 mAP points (delta +13.78)
2026-10-26 12:00:50,429 INFO    t2o.analysis.resolvability: seed-averaged bootstrap: 3 seed(s) x 153 images x 10000 resamples, delta CI [10.87, 18.97] mAP points
2026-10-26 12:00:50,468 INFO    __main__: wrote D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-non-inferiority-e3t\modtr-non-inferiority.csv
2026-10-26 12:00:50,468 INFO    __main__: per-seed deltas (loop - ModTr): s0 +15.63, s1 +15.47, s3 +14.24; sd 0.76 mAP points
2026-10-26 12:00:50,468 INFO    __main__: verdict: NON-INFERIOR -- lower bound 10.87 > -5.0 mAP points (delta +15.11)
=== FINISHED 2026-10-26 12:00:52 +03:00 ===
```

`logs/2026-10-26-modtr-non-inferiority-e3b.csv`, verbatim:

```
seed,pool,loop_map50,modtr_map50,delta,low,high,half_width,seed_sd,margin,non_inferior,resamples,bootstrap_seed,loop_data,modtr_data,weights
0,153,0.8547938638426396,0.6959755964291388,15.88182674135008,,,,,5.0,,10000,0,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s0\stage0\translated\data.yaml,runs\reference-yolo11s\weights\best.pt
1,153,0.8475402463498627,0.7075368995637781,14.00033467860846,,,,,5.0,,10000,0,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s1\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s1\stage0\translated\data.yaml,runs\reference-yolo11s\weights\best.pt
3,153,0.8473979197736736,0.7327569972277532,11.464092254592039,,,,,5.0,,10000,0,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s3\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s3\stage0\translated\data.yaml,runs\reference-yolo11s\weights\best.pt
mean,153,0.849910676655392,0.7120898310735567,13.782084558183527,10.196399669974781,17.006148264750614,3.4048742973879165,2.2169391816583417,5.0,True,10000,0,runs/e3b-loop-s{seed}/stage3/translated,runs/modtr-s{seed}/stage0/translated,runs\reference-yolo11s\weights\best.pt
```

`logs/2026-10-26-modtr-non-inferiority-e3t.csv`, verbatim:

```
seed,pool,loop_map50,modtr_map50,delta,low,high,half_width,seed_sd,margin,non_inferior,resamples,bootstrap_seed,loop_data,modtr_data,weights
0,153,0.8522506023318381,0.6959755964291388,15.627500590269939,,,,,5.0,,10000,0,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s0\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s0\stage0\translated\data.yaml,runs\reference-yolo11s\weights\best.pt
1,153,0.8622155266083098,0.7075368995637781,15.467862704453172,,,,,5.0,,10000,0,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s1\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s1\stage0\translated\data.yaml,runs\reference-yolo11s\weights\best.pt
3,153,0.8752003112222961,0.7327569972277532,14.24433139945429,,,,,5.0,,10000,0,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s3\stage3\translated\data.yaml,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s3\stage0\translated\data.yaml,runs\reference-yolo11s\weights\best.pt
mean,153,0.8632221467208147,0.7120898310735567,15.113231564725801,10.868266433485715,18.970766222186043,4.051249894350164,0.7567110984739516,5.0,True,10000,0,runs/e3t-loop-s{seed}/stage3/translated,runs/modtr-s{seed}/stage0/translated,runs\reference-yolo11s\weights\best.pt
```

Re-render the non-inferiority verdict without re-training, on the server: the two
`scripts/modtr_non_inferiority.py` commands above, each with a new `--out`, since the script
refuses to overwrite its CSV. Per-seed `metrics.json`, including the RT-DETR result written
back, is at `runs/modtr-s{0,1,3}/metrics.json` there.

## Findings

### F200 — Margin's non-inferiority test passes on both custom campaigns, with the loop 13.78 and 15.11 points ahead of ModTr

ADR-02's pass rule is that the lower bound of the seed-averaged paired 95% CI on loop − ModTr is
above −5 mAP@50 points. Both runs clear it:

| loop | seed-averaged delta | 95% CI | per-seed deltas (s0, s1, s3) | seed sd | margin of the lower bound over −5 |
| --- | --- | --- | --- | --- | --- |
| e3b (pix2pix) | +13.78 | [10.20, 17.01] | +15.88, +14.00, +11.46 | 2.22 | 15.20 |
| e3t (turbo) | +15.11 | [10.87, 18.97] | +15.63, +15.47, +14.24 | 0.76 | 15.87 |

- **Every one of the six per-seed deltas is positive**, the smallest +11.46.
- **The CIs also exclude zero and clear +2.** Against ModTr alone, then, the loop meets
  `goal.md`'s main Margin bar, not just the fallback. It does not make Margin pass on the main
  bar: that bar is set by the strongest baseline, and ModTr is not it (F201). The loop's own
  no-loop control is, and record 027 showed that comparison does not resolve +2 (F189, F190).
- **The loop arms reproduce record 027.** Every `loop_map50` here equals record 027's for the
  same seed to every printed digit (e3b-loop-s0 `0.8547938638426396` in both). That is the
  per-image path reading the same exports again, not second evidence.

### F201 — This ModTr is weaker than the loop's own no-loop translator: 0.7121 seed-averaged against controls of 0.8084 and 0.8326

Zero-shot `yolo11s` mAP@50 at seeds 0, 1 and 3, all on the same 153 val images. The control
values are record 027's CSV rows for the same seeds:

| arm | s0 | s1 | s3 | mean |
| --- | --- | --- | --- | --- |
| ModTr | 0.6960 | 0.7075 | 0.7328 | 0.7121 |
| e3b control (pix2pix, no loop) | 0.8128 | 0.8441 | 0.7681 | 0.8084 |
| e3t control (turbo, no loop) | 0.8184 | 0.8501 | 0.8293 | 0.8326 |
| raw thermal (F10, F191) | | | | 0.1552 |

- **ModTr is below both controls on every seed.** Per seed, ModTr − e3b control is −11.69,
  −13.66 and −3.53 points (mean −9.63). ModTr − e3t control is −12.24, −14.26 and −9.65
  (mean −12.05). Even the best ModTr seed, 0.7328, is below the worst control seed, 0.7681.
- **It still transforms thermal a long way.** It sits 55.69 points above raw thermal. That is
  80% of the e3b loop's 69.47-point lift over the same floor.
- **ADR-02's δ never came close to binding.** ADR-02 sized δ = 5 so that a loop roughly tied
  with ModTr could pass on 153 images. The observed gap is about three times δ in the loop's favour. So the test
  says nothing about whether δ = 5 is too generous.
- **The fine-tuned arm hides the gap.** Fine-tuned mAP50 is 0.8914–0.9172. That lies in or next
  to the 0.8984–0.9199 band F05 found saturated for M1's arms. So a translator 14 points behind
  on zero-shot reaches the same adapted score.

### F202 — ModTr's exports are far from visible: val LPIPS 0.6387, 2.1× the e3b loop's stage-3 0.3002

Seed means: LPIPS 0.6387 (0.6239–0.6590), FID 193.96 (191.16–197.02), PSNR 6.96, SSIM 0.1224.
The e3b loop's stage-3 val LPIPS is 0.3002 ± .0106 over six seeds (record 007). M1's loop
reached FID ~90 and PSNR ~15.5 (F08). This is what a detection-only objective with no
reconstruction term produces (plan, "ModTr has no reconstruction term"). It matters wherever
ModTr's exports are compared on anything but mAP, Faithfulness included.

### F203 — The loop's lead over ModTr depends on the judge: under RT-DETR it shrinks to +5.78 (e3b) and +6.87 (e3t), 42% and 45% of the `yolo11s` lead

RT-DETR-l zero-shot mAP@50 for ModTr is 0.8007, 0.7970 and 0.8241 (mean 0.8073). For the loop
and control arms at the same seeds, the values come from record 026's
`logs/2026-10-19-runs-e3{b,t}-judge-tidy.csv`:

| arm | `yolo11s` mean | RT-DETR mean | RT-DETR − `yolo11s` |
| --- | --- | --- | --- |
| ModTr | 0.7121 | 0.8073 | +9.52 |
| e3b loop | 0.8499 | 0.8651 | +1.52 |
| e3b control | 0.8084 | 0.8235 | +1.51 |
| e3t loop | 0.8632 | 0.8760 | +1.27 |
| e3t control | 0.8326 | 0.8456 | +1.30 |

- **RT-DETR lifts ModTr about six times as much as it lifts any E3 arm.** Per seed, loop − ModTr
  under RT-DETR is +7.50, +5.04, +4.82 (e3b) and +8.02, +6.05, +6.54 (e3t). Every one is still
  positive.
- **Under RT-DETR, ModTr is close to the no-loop controls**: 1.62 points below e3b's and 3.83
  below e3t's, against 9.63 and 12.05 under `yolo11s`.
- **These are point estimates, without a CI.** ADR-02 pre-registered only the `yolo11s` verdict.
  Against record 026's F182, where RT-DETR kept 85–99% of the loop − control gain, the
  loop − ModTr gap keeps less than half. The logs do not say why RT-DETR reads ModTr's images
  so differently from `yolo11s`.

### F204 — A campaign seed costs 4 h 31 min to 4 h 44 min on one A100 without the monitor, inside F195's 6.6 h prediction

Provenance to finish stamp is 4 h 43 min 33 s, 4 h 30 min 36 s and 4 h 39 min 23 s. That covers
training, the export, zero-shot and fidelity evaluation, and the 50-epoch fine-tuned detector.
Record 028's Next predicted at most about 6.6 h per seed, because the campaign does not pay for
the monitor. Median epochs are 37.2–39.2 s on 600 images, against the calibration's 59.2 s on
480 images with the monitor on (F195): 34–37% less per epoch on 25% more images. Peak memory is
5.82 GiB reserved on every seed, against 5.90 in the calibration.

## Provenance caveats

- **The non-inferiority log is named `-06-`.** The runbook wrote `-05-` (MODTR-BASELINE-05 owns
  the script), and the request named it that way too. It is cited here as saved. The two CSVs
  carry no task segment in their names (`2026-10-26-modtr-non-inferiority-e3{b,t}.csv`). They
  are the server's `runs/modtr-non-inferiority-e3{b,t}/modtr-non-inferiority.csv`, copied back
  and renamed.
- **The in-loop detector has seen every training image**, as record 028 recorded.
  `optical_best_n_1.pt` was trained on all 600 training images, as visible images. ModTr and
  both loops train against it the same way, so the comparison is matched. But the fine-tuned
  arm starts from a detector that already knows the training scenes.
- **The val split is the reporting split**, decided 2026-10-22 because the custom set has no test
  split. Every number here is on the 153 val images, which the loop campaigns were also scored on.
- **RT-DETR's ModTr values have four decimals.** They come from the log's `t2o.metrics.task`
  lines. The E3 values in F203 come from the tidy CSVs at full precision, so F203's differences
  can be off in the second decimal.
- **The ModTr mAP50-95 values are logged, the loop's are not.** `goal.md`'s main Margin bar also
  asks for a non-negative mAP@50:95 delta. The non-inferiority CSV carries mAP@50 only, so that
  delta against ModTr is not computed here.
- **One RT-DETR judge, no CI.** F203 inherits record 026's caveats about that judge, F188's
  non-reproducible training among them.
- **The runs' `config_hash` is not in the logs.** It is in `runs/modtr-s{0,1,3}/config.yaml` on
  the server.
- **Slow image access.** ultralytics warns "Slow image access detected" on most evaluation passes,
  as in records 027 and 028. That cost time, not numbers.

## Next

Margin is settled on the custom set as ADR-02 framed it: non-inferior to ModTr, with Transfer and
Faithfulness carrying the claim. Two things a reviewer will ask follow from F201 and F203.

- **Is this ModTr under-trained?** It loses to a pix2pix with no detection loss at all. Record
  028's calibration curve was within 2 points of its peak from epoch 64, which suggests more
  epochs alone would not close the gap. That curve was scored by the in-loop `yolo11n` on
  inner-val, though, not by `yolo11s` on val, so this is a reading, not a measurement. The paper has to say this is a reimplementation trained against the
  loop's `yolo11n`, as the plan already requires.
- **The judge dependence.** F203's gap can get a CI from the same seed-averaged bootstrap,
  scored by RT-DETR instead of `yolo11s`. That would be a secondary analysis, labelled as one
  and outside ADR-02.

The FLIR campaign (Track C) waits on MODTR-BASELINE-09's 339-epoch budget. A falsifiable
prediction from F204 and F199: a FLIR seed at 339 epochs trains in at most about 6.9 h on one
card, since the custom set's per-epoch cost without the monitor came in under F199's
monitor-free estimate.
