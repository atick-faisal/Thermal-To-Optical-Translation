# 037 — Registration-error axis, translation half: custom zero-shot sits below its aligned seed at every level while its adapted arm tracks label transfer; FLIR's single-seed curve is set by translator collapses and reads lowest at 0 px

**Date:** 2026-11-11 · **Task:** REGISTRATION-AXIS-06 · **Machine:** VSIBMEOT05, 2× A100 (custom 3–24 px, then FLIR 24 px, on `cuda:0`; FLIR 0 / 3 / 6 / 12 px on `cuda:1`, with record 036's FLIR sweep between 0 and 3 px) · **Wall clock:** custom 3 / 6 / 12 / 24 px 9 h 25 min / 10 h 37 min / 9 h 37 min / 9 h 20 min; FLIR 0 / 3 / 6 / 12 / 24 px 9 h 56 min (resumed; plus 18 min before the crash) / 10 h 21 min / 10 h 06 min / 9 h 42 min / 10 h 06 min. From 2026-11-11 09:34:46 to 2026-11-14 01:48:01 +03:00
**git_sha:** `pre-spec-migration-256-gb541e76` (custom 3 and 6 px, FLIR 0 px and its crashed attempt), `pre-spec-migration-257-g8af7e17` (custom 12 px), `pre-spec-migration-258-g55e697b` (custom 24 px, FLIR 3–24 px). All clean, with no `-dirty` suffix. `git diff --stat b541e76 55e697b` touches only `docs/explainers/t2o-field-guide.html` and `docs/progress/2026-11-12-update-board.html`, so all nine runs ran the same code · **W&B:** none (`runtime.wandb: false`; no W&B line in any log) · **Log:** `logs/2026-11-11-REGISTRATION-AXIS-06-regc-03px.txt`, `logs/2026-11-11-REGISTRATION-AXIS-06-regc-06px.txt`, `logs/2026-11-11-REGISTRATION-AXIS-06-regc-12px.txt`, `logs/2026-11-11-REGISTRATION-AXIS-06-regc-24px.txt`, `logs/2026-11-11-REGISTRATION-AXIS-06-regf-00px.txt`, `logs/2026-11-10-REGISTRATION-AXIS-06-regf-03px.txt`, `logs/2026-11-10-REGISTRATION-AXIS-06-regf-06px.txt`, `logs/2026-11-10-REGISTRATION-AXIS-06-regf-12px.txt`, `logs/2026-11-10-REGISTRATION-AXIS-06-regf-24px.txt`, `logs/2026-11-10-REGISTRATION-AXIS-06-regf-00px.txt` (the crashed attempt)

This is the translation half of goal C2's registration-error axis (legacy C4). Record 036 is the
label-transfer half. Each run is the pix2pix loop arm, λ_det [0, 1, 2, 3] at `grad_scale` 0.15,
seed 0, with every **train** pair's visible target warped by a known misalignment. The train boxes
move with the target (ADR-01), and val never moves. The custom set runs 3 / 6 / 12 / 24 px, and its
0 px point is the existing aligned `e3b-loop-s0` (plan Q4). FLIR runs 0 / 3 / 6 / 12 / 24 px on
de-rolled pairs scored against de-rolled thermal-side boxes. Its 0 px point is this campaign's own
`regf-00px-s0` (ADR-02).

## Command

From the provenance blocks, as run:

```
uv run t2o loop --config experiments/registration/flir_00px.yaml --workers 8 --resume --device cuda:1
uv run t2o loop --config experiments/registration/custom_03px.yaml --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --in-loop-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --eval-init-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --workers 4 --resume --device cuda:0
uv run t2o loop --config experiments/registration/custom_06px.yaml --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --in-loop-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --eval-init-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --workers 4 --resume --device cuda:0
uv run t2o loop --config experiments/registration/custom_12px.yaml --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --in-loop-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --eval-init-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --workers 4 --resume --device cuda:0
uv run t2o loop --config experiments/registration/custom_24px.yaml --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --in-loop-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --eval-init-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --workers 4 --resume --device cuda:0
uv run t2o loop --config experiments/registration/flir_00px.yaml --workers 4 --resume --device cuda:1
uv run t2o loop --config experiments/registration/flir_03px.yaml --workers 4 --resume --device cuda:1
uv run t2o loop --config experiments/registration/flir_06px.yaml --workers 4 --resume --device cuda:1
uv run t2o loop --config experiments/registration/flir_12px.yaml --workers 4 --resume --device cuda:1
uv run t2o loop --config experiments/registration/flir_24px.yaml --workers 4 --resume --device cuda:0
```

The first line is the 2026-11-10 attempt, which hung at epoch 20 (record 036, F228). The fifth
line is its restart. With `--resume`, it picked up at epoch 21.

## Environment

The block from the custom 3 px log. All ten logs carry the same block, apart from the `captured`,
`command` and `git_sha` lines (checked by hashing each block with those three lines removed).

```
=== PROVENANCE ===
captured: 2026-11-11 13:13:59 +03:00
hostname: VSIBMEOT05
os: Microsoft Windows Server 2022 Standard 10.0.20348 (build 20348)
cpu: AMD EPYC 7763 64-Core Processor
cpu_logical_cores: 64
ram_total: 128.0 GB
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16; NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16
cuda_driver: 13.0
git_sha: pre-spec-migration-256-gb541e76
working_dir: D:\Atick\GitHub\Thermal-To-Optical-Translation
command: uv run t2o loop --config experiments/registration/custom_03px.yaml --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --in-loop-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --eval-init-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --workers 4 --resume --device cuda:0
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
| Base config | `experiments/e3_pix2pix_loop.yaml` (e3b) with `data.misalignment_px` and the run name changed, nothing else | the e3f cell (record 016) with `data.calibration: calibration/flir.json`, `data.misalignment_px` and the run name changed |
| Configs | `experiments/registration/custom_{03,06,12,24}px.yaml` | `experiments/registration/flir_{00,03,06,12,24}px.yaml` |
| Translator | pix2pix `resnet_9blocks` / `basic`, 4 stages × 100 epochs, warm-started, batch 8, bf16 | same |
| Loop | λ_det [0, 1, 2, 3], `grad_scale` 0.15, `train.seed` 0, loop arm only | same |
| Train corpus | all 600 pairs, visible target and boxes moved by the level | 600 of 4,129 pairs (`subset_seed` 0); target de-rolled, then moved; boxes are the de-rolled thermal-side labels moved by the level |
| Val | 153 images, untouched | all 1,013 images scored (153 subsampled for the val loss only); untouched de-rolled thermal-side labels |
| In-loop and adapted-arm init | `optical_best_n_1.pt` (`--in-loop-weights`, `--eval-init-weights`) | `runs/inloop-flir-yolo11n/weights/best.pt` |
| Zero-shot judge | `runs/reference-yolo11s/weights/best.pt` | `runs/reference-flir-yolo11s/weights/best.pt` |
| Calibration digest | none | `210142740cdba163`, checked against both splits' thermal labels at startup in every run |
| Zero point | `e3b-loop-s0` stage 3, 0.8548 (record 027's `0.8547938638426396`) | `regf-00px-s0` (this record) |
| Config hash | not printed in the logs | not printed in the logs |

**What the two mAP columns are.** `zero-shot mAP50` is the fixed reference `yolo11s` scoring the
stage's translated val export (`src/t2o/engine/loop.py:229-233`). It is the same judge, and the same
metric, as record 007's e3b table. `fine-tuned mAP50` is the adapted arm. Its evaluation detector
is fine-tuned for 50 epochs on that stage's translated train export, whose boxes moved with the
level (ADR-01), and then validated on translated val. FLIR's zero-shot is all-class over four
classes, dog included. The logs print no per-class AP, so the 3-class primary is not available
here.

## Results

Per run, every stage's zero-shot, fidelity, adapted-arm and summary line, verbatim and in log
order.

**custom 3 px** (`regc-03px-s0`)

```
2026-11-11 14:40:56,047 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8881  R 0.7278  mAP50 0.7982  mAP50-95 0.5333
2026-11-11 14:41:05,521 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 14.56  SSIM 0.5038  LPIPS 0.3233  FID 99.42  KID 0.0355+-0.0077
2026-11-11 14:48:34,553 INFO    t2o.engine.detector_stage: detector stage0: P 0.9002  R 0.8761  mAP50 0.9116  mAP50-95 0.6387
2026-11-11 14:48:34,563 INFO    t2o.engine.loop: stage 0 done: task_weight=0.0 epochs=100
2026-11-11 16:51:59,887 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8230  R 0.7629  mAP50 0.8346  mAP50-95 0.5549
2026-11-11 16:52:09,120 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 15.15  SSIM 0.4965  LPIPS 0.2900  FID 89.35  KID 0.0247+-0.0055
2026-11-11 16:59:13,056 INFO    t2o.engine.detector_stage: detector stage1: P 0.9405  R 0.8813  mAP50 0.9348  mAP50-95 0.6500
2026-11-11 16:59:13,072 INFO    t2o.engine.loop: stage 1 done: task_weight=1.0 epochs=100
2026-11-11 19:28:47,434 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8237  R 0.6987  mAP50 0.7919  mAP50-95 0.5041
2026-11-11 19:28:56,161 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 15.63  SSIM 0.4972  LPIPS 0.2882  FID 90.58  KID 0.0231+-0.0057
2026-11-11 19:35:59,720 INFO    t2o.engine.detector_stage: detector stage2: P 0.9340  R 0.8496  mAP50 0.9100  mAP50-95 0.6402
2026-11-11 19:35:59,735 INFO    t2o.engine.loop: stage 2 done: task_weight=2.0 epochs=100
2026-11-11 22:31:40,897 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8131  R 0.7106  mAP50 0.7682  mAP50-95 0.4739
2026-11-11 22:31:49,318 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 14.98  SSIM 0.4881  LPIPS 0.3000  FID 97.71  KID 0.0249+-0.0061
2026-11-11 22:38:53,739 INFO    t2o.engine.detector_stage: detector stage3: P 0.9039  R 0.8705  mAP50 0.9241  mAP50-95 0.6410
2026-11-11 22:38:53,754 INFO    t2o.engine.loop: stage 3 done: task_weight=3.0 epochs=100
2026-11-11 22:38:53,760 INFO    t2o.cli: stage 0 (task_weight=0.0): 100 epochs | zero-shot mAP50 0.7982 | LPIPS 0.3233 | fine-tuned mAP50 0.9116
2026-11-11 22:38:53,760 INFO    t2o.cli: stage 1 (task_weight=1.0): 100 epochs | zero-shot mAP50 0.8346 | LPIPS 0.2900 | fine-tuned mAP50 0.9348
2026-11-11 22:38:53,760 INFO    t2o.cli: stage 2 (task_weight=2.0): 100 epochs | zero-shot mAP50 0.7919 | LPIPS 0.2882 | fine-tuned mAP50 0.9100
2026-11-11 22:38:53,760 INFO    t2o.cli: stage 3 (task_weight=3.0): 100 epochs | zero-shot mAP50 0.7682 | LPIPS 0.3000 | fine-tuned mAP50 0.9241
```

**custom 6 px** (`regc-06px-s0`)

```
2026-11-12 00:08:08,402 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8079  R 0.6786  mAP50 0.7415  mAP50-95 0.4792
2026-11-12 00:08:20,313 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 14.96  SSIM 0.4671  LPIPS 0.3390  FID 115.25  KID 0.0458+-0.0067
2026-11-12 00:17:57,866 INFO    t2o.engine.detector_stage: detector stage0: P 0.9073  R 0.8512  mAP50 0.9219  mAP50-95 0.6238
2026-11-12 00:17:57,866 INFO    t2o.engine.loop: stage 0 done: task_weight=0.0 epochs=100
2026-11-12 02:35:28,162 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7704  R 0.7218  mAP50 0.7634  mAP50-95 0.5021
2026-11-12 02:35:38,733 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 14.74  SSIM 0.4231  LPIPS 0.3510  FID 100.27  KID 0.0301+-0.0069
2026-11-12 02:44:59,714 INFO    t2o.engine.detector_stage: detector stage1: P 0.9371  R 0.8440  mAP50 0.9138  mAP50-95 0.6258
2026-11-12 02:44:59,724 INFO    t2o.engine.loop: stage 1 done: task_weight=1.0 epochs=100
2026-11-12 05:35:21,805 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8501  R 0.7352  mAP50 0.8110  mAP50-95 0.5279
2026-11-12 05:35:32,181 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 13.99  SSIM 0.4079  LPIPS 0.3246  FID 100.79  KID 0.0287+-0.0057
2026-11-12 05:44:34,584 INFO    t2o.engine.detector_stage: detector stage2: P 0.9121  R 0.8464  mAP50 0.9049  mAP50-95 0.6188
2026-11-12 05:44:34,605 INFO    t2o.engine.loop: stage 2 done: task_weight=2.0 epochs=100
2026-11-12 09:06:45,971 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8459  R 0.7655  mAP50 0.8256  mAP50-95 0.5546
2026-11-12 09:06:56,170 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 15.41  SSIM 0.4197  LPIPS 0.3023  FID 97.13  KID 0.0299+-0.0068
2026-11-12 09:15:47,174 INFO    t2o.engine.detector_stage: detector stage3: P 0.9139  R 0.8371  mAP50 0.9052  mAP50-95 0.6294
2026-11-12 09:15:47,206 INFO    t2o.engine.loop: stage 3 done: task_weight=3.0 epochs=100
2026-11-12 09:15:47,217 INFO    t2o.cli: stage 0 (task_weight=0.0): 100 epochs | zero-shot mAP50 0.7415 | LPIPS 0.3390 | fine-tuned mAP50 0.9219
2026-11-12 09:15:47,217 INFO    t2o.cli: stage 1 (task_weight=1.0): 100 epochs | zero-shot mAP50 0.7634 | LPIPS 0.3510 | fine-tuned mAP50 0.9138
2026-11-12 09:15:47,217 INFO    t2o.cli: stage 2 (task_weight=2.0): 100 epochs | zero-shot mAP50 0.8110 | LPIPS 0.3246 | fine-tuned mAP50 0.9049
2026-11-12 09:15:47,217 INFO    t2o.cli: stage 3 (task_weight=3.0): 100 epochs | zero-shot mAP50 0.8256 | LPIPS 0.3023 | fine-tuned mAP50 0.9052
```

**custom 12 px** (`regc-12px-s0`)

```
2026-11-12 10:43:57,033 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7257  R 0.6653  mAP50 0.6720  mAP50-95 0.4334
2026-11-12 10:44:06,540 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 13.47  SSIM 0.4019  LPIPS 0.3698  FID 148.96  KID 0.0899+-0.0129
2026-11-12 10:52:00,878 INFO    t2o.engine.detector_stage: detector stage0: P 0.9281  R 0.7902  mAP50 0.8855  mAP50-95 0.5661
2026-11-12 10:52:00,888 INFO    t2o.engine.loop: stage 0 done: task_weight=0.0 epochs=100
2026-11-12 12:57:31,140 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8058  R 0.7398  mAP50 0.8056  mAP50-95 0.5259
2026-11-12 12:57:39,794 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 13.94  SSIM 0.4048  LPIPS 0.3963  FID 134.76  KID 0.0628+-0.0088
2026-11-12 13:04:56,522 INFO    t2o.engine.detector_stage: detector stage1: P 0.9134  R 0.8239  mAP50 0.8877  mAP50-95 0.5925
2026-11-12 13:04:56,527 INFO    t2o.engine.loop: stage 1 done: task_weight=1.0 epochs=100
2026-11-12 15:36:30,197 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8181  R 0.7224  mAP50 0.7811  mAP50-95 0.5042
2026-11-12 15:36:38,420 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 14.30  SSIM 0.3521  LPIPS 0.3546  FID 144.26  KID 0.0732+-0.0101
2026-11-12 15:43:52,000 INFO    t2o.engine.detector_stage: detector stage2: P 0.9287  R 0.8361  mAP50 0.8907  mAP50-95 0.5705
2026-11-12 15:43:52,016 INFO    t2o.engine.loop: stage 2 done: task_weight=2.0 epochs=100
2026-11-12 18:44:48,753 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8567  R 0.7183  mAP50 0.7967  mAP50-95 0.5108
2026-11-12 18:44:56,729 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 13.93  SSIM 0.3504  LPIPS 0.3518  FID 133.52  KID 0.0584+-0.0094
2026-11-12 18:52:48,348 INFO    t2o.engine.detector_stage: detector stage3: P 0.9147  R 0.8238  mAP50 0.8853  mAP50-95 0.5848
2026-11-12 18:52:48,359 INFO    t2o.engine.loop: stage 3 done: task_weight=3.0 epochs=100
2026-11-12 18:52:48,370 INFO    t2o.cli: stage 0 (task_weight=0.0): 100 epochs | zero-shot mAP50 0.6720 | LPIPS 0.3698 | fine-tuned mAP50 0.8855
2026-11-12 18:52:48,370 INFO    t2o.cli: stage 1 (task_weight=1.0): 100 epochs | zero-shot mAP50 0.8056 | LPIPS 0.3963 | fine-tuned mAP50 0.8877
2026-11-12 18:52:48,370 INFO    t2o.cli: stage 2 (task_weight=2.0): 100 epochs | zero-shot mAP50 0.7811 | LPIPS 0.3546 | fine-tuned mAP50 0.8907
2026-11-12 18:52:48,370 INFO    t2o.cli: stage 3 (task_weight=3.0): 100 epochs | zero-shot mAP50 0.7967 | LPIPS 0.3518 | fine-tuned mAP50 0.8853
```

**custom 24 px** (`regc-24px-s0`)

```
2026-11-12 20:19:06,416 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.6691  R 0.4901  mAP50 0.5268  mAP50-95 0.2864
2026-11-12 20:19:15,408 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 14.70  SSIM 0.3931  LPIPS 0.3911  FID 138.79  KID 0.0746+-0.0097
2026-11-12 20:27:10,624 INFO    t2o.engine.detector_stage: detector stage0: P 0.8040  R 0.7384  mAP50 0.7668  mAP50-95 0.4155
2026-11-12 20:27:10,630 INFO    t2o.engine.loop: stage 0 done: task_weight=0.0 epochs=100
2026-11-12 22:27:36,494 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7257  R 0.6865  mAP50 0.7278  mAP50-95 0.4188
2026-11-12 22:27:44,857 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 14.76  SSIM 0.3544  LPIPS 0.3736  FID 133.74  KID 0.0652+-0.0100
2026-11-12 22:34:34,727 INFO    t2o.engine.detector_stage: detector stage1: P 0.8961  R 0.7684  mAP50 0.8427  mAP50-95 0.4509
2026-11-12 22:34:34,738 INFO    t2o.engine.loop: stage 1 done: task_weight=1.0 epochs=100
2026-11-13 01:04:34,262 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8151  R 0.6843  mAP50 0.7453  mAP50-95 0.4308
2026-11-13 01:04:42,912 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 13.96  SSIM 0.3729  LPIPS 0.3593  FID 131.68  KID 0.0544+-0.0098
2026-11-13 01:11:39,392 INFO    t2o.engine.detector_stage: detector stage2: P 0.8582  R 0.8191  mAP50 0.8665  mAP50-95 0.4749
2026-11-13 01:11:39,412 INFO    t2o.engine.loop: stage 2 done: task_weight=2.0 epochs=100
2026-11-13 04:05:50,890 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7852  R 0.6971  mAP50 0.7383  mAP50-95 0.4249
2026-11-13 04:05:59,235 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 13.88  SSIM 0.3958  LPIPS 0.3737  FID 140.79  KID 0.0674+-0.0103
2026-11-13 04:13:12,049 INFO    t2o.engine.detector_stage: detector stage3: P 0.9182  R 0.7704  mAP50 0.8397  mAP50-95 0.4637
2026-11-13 04:13:12,102 INFO    t2o.engine.loop: stage 3 done: task_weight=3.0 epochs=100
2026-11-13 04:13:12,112 INFO    t2o.cli: stage 0 (task_weight=0.0): 100 epochs | zero-shot mAP50 0.5268 | LPIPS 0.3911 | fine-tuned mAP50 0.7668
2026-11-13 04:13:12,112 INFO    t2o.cli: stage 1 (task_weight=1.0): 100 epochs | zero-shot mAP50 0.7278 | LPIPS 0.3736 | fine-tuned mAP50 0.8427
2026-11-13 04:13:12,112 INFO    t2o.cli: stage 2 (task_weight=2.0): 100 epochs | zero-shot mAP50 0.7453 | LPIPS 0.3593 | fine-tuned mAP50 0.8665
2026-11-13 04:13:12,112 INFO    t2o.cli: stage 3 (task_weight=3.0): 100 epochs | zero-shot mAP50 0.7383 | LPIPS 0.3737 | fine-tuned mAP50 0.8397
```

**FLIR 0 px** (`regf-00px-s0`, resumed)

```
2026-11-11 10:48:58,189 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.1854  R 0.0546  mAP50 0.0453  mAP50-95 0.0173
2026-11-11 10:49:41,454 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 12.18  SSIM 0.2457  LPIPS 0.4715  FID 211.81  KID 0.2027+-0.0136
2026-11-11 11:04:56,190 INFO    t2o.engine.detector_stage: detector stage0: P 0.4699  R 0.3070  mAP50 0.3270  mAP50-95 0.1580
2026-11-11 11:04:56,201 INFO    t2o.engine.loop: stage 0 done: task_weight=0.0 epochs=79
2026-11-11 13:13:53,227 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.6774  R 0.2743  mAP50 0.3003  mAP50-95 0.1394
2026-11-11 13:14:30,556 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 12.89  SSIM 0.3946  LPIPS 0.4591  FID 131.41  KID 0.0930+-0.0112
2026-11-11 13:28:29,473 INFO    t2o.engine.detector_stage: detector stage1: P 0.6947  R 0.3978  mAP50 0.4532  mAP50-95 0.2206
2026-11-11 13:28:29,494 INFO    t2o.engine.loop: stage 1 done: task_weight=1.0 epochs=100
2026-11-11 16:02:45,136 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.4392  R 0.2811  mAP50 0.3168  mAP50-95 0.1569
2026-11-11 16:03:27,389 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 12.36  SSIM 0.4118  LPIPS 0.4465  FID 122.37  KID 0.0868+-0.0118
2026-11-11 16:17:40,988 INFO    t2o.engine.detector_stage: detector stage2: P 0.6185  R 0.4677  mAP50 0.4677  mAP50-95 0.2341
2026-11-11 16:17:41,013 INFO    t2o.engine.loop: stage 2 done: task_weight=2.0 epochs=100
2026-11-11 19:18:03,681 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5775  R 0.1186  mAP50 0.1212  mAP50-95 0.0581
2026-11-11 19:18:37,012 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 9.98  SSIM 0.2509  LPIPS 0.5075  FID 227.62  KID 0.2376+-0.0169
2026-11-11 19:30:49,333 INFO    t2o.engine.detector_stage: detector stage3: P 0.4699  R 0.3409  mAP50 0.3729  mAP50-95 0.1863
2026-11-11 19:30:49,354 INFO    t2o.engine.loop: stage 3 done: task_weight=3.0 epochs=100
2026-11-11 19:30:49,360 INFO    t2o.cli: stage 0 (task_weight=0.0): 79 epochs | zero-shot mAP50 0.0453 | LPIPS 0.4715 | fine-tuned mAP50 0.3270
2026-11-11 19:30:49,360 INFO    t2o.cli: stage 1 (task_weight=1.0): 100 epochs | zero-shot mAP50 0.3003 | LPIPS 0.4591 | fine-tuned mAP50 0.4532
2026-11-11 19:30:49,360 INFO    t2o.cli: stage 2 (task_weight=2.0): 100 epochs | zero-shot mAP50 0.3168 | LPIPS 0.4465 | fine-tuned mAP50 0.4677
2026-11-11 19:30:49,360 INFO    t2o.cli: stage 3 (task_weight=3.0): 100 epochs | zero-shot mAP50 0.1212 | LPIPS 0.5075 | fine-tuned mAP50 0.3729
```

**FLIR 3 px** (`regf-03px-s0`)

```
2026-11-12 21:09:41,156 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.2537  R 0.1115  mAP50 0.0963  mAP50-95 0.0441
2026-11-12 21:10:17,374 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 12.54  SSIM 0.4242  LPIPS 0.4213  FID 112.99  KID 0.0744+-0.0088
2026-11-12 21:22:34,731 INFO    t2o.engine.detector_stage: detector stage0: P 0.4304  R 0.3247  mAP50 0.3518  mAP50-95 0.1761
2026-11-12 21:22:34,752 INFO    t2o.engine.loop: stage 0 done: task_weight=0.0 epochs=100
2026-11-12 23:23:02,269 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.4778  R 0.2879  mAP50 0.3277  mAP50-95 0.1521
2026-11-12 23:23:43,189 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 13.04  SSIM 0.4145  LPIPS 0.4517  FID 127.29  KID 0.0947+-0.0118
2026-11-12 23:37:25,005 INFO    t2o.engine.detector_stage: detector stage1: P 0.7239  R 0.3793  mAP50 0.4163  mAP50-95 0.2080
2026-11-12 23:37:25,017 INFO    t2o.engine.loop: stage 1 done: task_weight=1.0 epochs=100
2026-11-13 02:20:20,273 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.4726  R 0.2964  mAP50 0.3299  mAP50-95 0.1598
2026-11-13 02:21:04,148 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 12.95  SSIM 0.4584  LPIPS 0.4216  FID 93.13  KID 0.0479+-0.0095
2026-11-13 02:35:10,731 INFO    t2o.engine.detector_stage: detector stage2: P 0.5152  R 0.3757  mAP50 0.4370  mAP50-95 0.2161
2026-11-13 02:35:10,767 INFO    t2o.engine.loop: stage 2 done: task_weight=2.0 epochs=100
2026-11-13 05:45:03,638 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5989  R 0.3085  mAP50 0.3416  mAP50-95 0.1656
2026-11-13 05:45:42,723 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 12.23  SSIM 0.4391  LPIPS 0.4416  FID 94.76  KID 0.0543+-0.0105
2026-11-13 06:00:01,975 INFO    t2o.engine.detector_stage: detector stage3: P 0.5056  R 0.4183  mAP50 0.4303  mAP50-95 0.2150
2026-11-13 06:00:01,997 INFO    t2o.engine.loop: stage 3 done: task_weight=3.0 epochs=100
2026-11-13 06:00:02,007 INFO    t2o.cli: stage 0 (task_weight=0.0): 100 epochs | zero-shot mAP50 0.0963 | LPIPS 0.4213 | fine-tuned mAP50 0.3518
2026-11-13 06:00:02,007 INFO    t2o.cli: stage 1 (task_weight=1.0): 100 epochs | zero-shot mAP50 0.3277 | LPIPS 0.4517 | fine-tuned mAP50 0.4163
2026-11-13 06:00:02,007 INFO    t2o.cli: stage 2 (task_weight=2.0): 100 epochs | zero-shot mAP50 0.3299 | LPIPS 0.4216 | fine-tuned mAP50 0.4370
2026-11-13 06:00:02,007 INFO    t2o.cli: stage 3 (task_weight=3.0): 100 epochs | zero-shot mAP50 0.3416 | LPIPS 0.4416 | fine-tuned mAP50 0.4303
```

**FLIR 6 px** (`regf-06px-s0`)

```
2026-11-13 07:30:18,949 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5673  R 0.0667  mAP50 0.0668  mAP50-95 0.0315
2026-11-13 07:30:54,167 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 11.86  SSIM 0.3213  LPIPS 0.4941  FID 243.31  KID 0.2574+-0.0162
2026-11-13 07:43:28,149 INFO    t2o.engine.detector_stage: detector stage0: P 0.5593  R 0.3212  mAP50 0.3638  mAP50-95 0.1694
2026-11-13 07:43:28,163 INFO    t2o.engine.loop: stage 0 done: task_weight=0.0 epochs=100
2026-11-13 09:52:52,524 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.3640  R 0.0577  mAP50 0.0278  mAP50-95 0.0115
2026-11-13 09:53:25,258 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 8.58  SSIM 0.1458  LPIPS 0.5880  FID 289.90  KID 0.3237+-0.0183
2026-11-13 10:05:33,953 INFO    t2o.engine.detector_stage: detector stage1: P 0.7399  R 0.3242  mAP50 0.3670  mAP50-95 0.1796
2026-11-13 10:05:33,963 INFO    t2o.engine.loop: stage 1 done: task_weight=1.0 epochs=100
2026-11-13 12:42:53,323 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5359  R 0.1286  mAP50 0.1204  mAP50-95 0.0529
2026-11-13 12:43:26,675 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 7.76  SSIM 0.1675  LPIPS 0.6059  FID 299.77  KID 0.3463+-0.0190
2026-11-13 12:55:42,648 INFO    t2o.engine.detector_stage: detector stage2: P 0.4114  R 0.3218  mAP50 0.3345  mAP50-95 0.1620
2026-11-13 12:55:42,668 INFO    t2o.engine.loop: stage 2 done: task_weight=2.0 epochs=100
2026-11-13 15:54:42,187 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.4185  R 0.2512  mAP50 0.2596  mAP50-95 0.1250
2026-11-13 15:55:14,480 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 11.92  SSIM 0.3228  LPIPS 0.5157  FID 163.96  KID 0.1461+-0.0132
2026-11-13 16:06:25,526 INFO    t2o.engine.detector_stage: detector stage3: P 0.4833  R 0.3860  mAP50 0.4131  mAP50-95 0.2052
2026-11-13 16:06:25,544 INFO    t2o.engine.loop: stage 3 done: task_weight=3.0 epochs=100
2026-11-13 16:06:25,554 INFO    t2o.cli: stage 0 (task_weight=0.0): 100 epochs | zero-shot mAP50 0.0668 | LPIPS 0.4941 | fine-tuned mAP50 0.3638
2026-11-13 16:06:25,554 INFO    t2o.cli: stage 1 (task_weight=1.0): 100 epochs | zero-shot mAP50 0.0278 | LPIPS 0.5880 | fine-tuned mAP50 0.3670
2026-11-13 16:06:25,554 INFO    t2o.cli: stage 2 (task_weight=2.0): 100 epochs | zero-shot mAP50 0.1204 | LPIPS 0.6059 | fine-tuned mAP50 0.3345
2026-11-13 16:06:25,554 INFO    t2o.cli: stage 3 (task_weight=3.0): 100 epochs | zero-shot mAP50 0.2596 | LPIPS 0.5157 | fine-tuned mAP50 0.4131
```

**FLIR 12 px** (`regf-12px-s0`)

```
2026-11-13 17:34:19,738 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.3350  R 0.0190  mAP50 0.0157  mAP50-95 0.0056
2026-11-13 17:34:52,460 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 11.53  SSIM 0.2653  LPIPS 0.5000  FID 216.97  KID 0.1966+-0.0164
2026-11-13 17:46:13,304 INFO    t2o.engine.detector_stage: detector stage0: P 0.4255  R 0.3412  mAP50 0.3236  mAP50-95 0.1432
2026-11-13 17:46:13,310 INFO    t2o.engine.loop: stage 0 done: task_weight=0.0 epochs=100
2026-11-13 19:51:13,298 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.6722  R 0.2344  mAP50 0.2550  mAP50-95 0.1236
2026-11-13 19:51:47,136 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 13.31  SSIM 0.4349  LPIPS 0.4384  FID 114.99  KID 0.0665+-0.0087
2026-11-13 20:02:46,291 INFO    t2o.engine.detector_stage: detector stage1: P 0.7340  R 0.3376  mAP50 0.3802  mAP50-95 0.1707
2026-11-13 20:02:46,300 INFO    t2o.engine.loop: stage 1 done: task_weight=1.0 epochs=100
2026-11-13 22:30:38,540 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.4493  R 0.2404  mAP50 0.2690  mAP50-95 0.1245
2026-11-13 22:31:13,128 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 13.21  SSIM 0.4381  LPIPS 0.4589  FID 132.78  KID 0.0904+-0.0110
2026-11-13 22:42:10,326 INFO    t2o.engine.detector_stage: detector stage2: P 0.4769  R 0.3398  mAP50 0.3711  mAP50-95 0.1706
2026-11-13 22:42:10,347 INFO    t2o.engine.loop: stage 2 done: task_weight=2.0 epochs=100
2026-11-14 01:36:47,843 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.4571  R 0.2536  mAP50 0.2888  mAP50-95 0.1299
2026-11-14 01:37:19,433 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 13.16  SSIM 0.4017  LPIPS 0.4847  FID 148.46  KID 0.1143+-0.0142
2026-11-14 01:47:58,066 INFO    t2o.engine.detector_stage: detector stage3: P 0.4165  R 0.3855  mAP50 0.3869  mAP50-95 0.1772
2026-11-14 01:47:58,083 INFO    t2o.engine.loop: stage 3 done: task_weight=3.0 epochs=100
2026-11-14 01:47:58,087 INFO    t2o.cli: stage 0 (task_weight=0.0): 100 epochs | zero-shot mAP50 0.0157 | LPIPS 0.5000 | fine-tuned mAP50 0.3236
2026-11-14 01:47:58,087 INFO    t2o.cli: stage 1 (task_weight=1.0): 100 epochs | zero-shot mAP50 0.2550 | LPIPS 0.4384 | fine-tuned mAP50 0.3802
2026-11-14 01:47:58,087 INFO    t2o.cli: stage 2 (task_weight=2.0): 100 epochs | zero-shot mAP50 0.2690 | LPIPS 0.4589 | fine-tuned mAP50 0.3711
2026-11-14 01:47:58,087 INFO    t2o.cli: stage 3 (task_weight=3.0): 100 epochs | zero-shot mAP50 0.2888 | LPIPS 0.4847 | fine-tuned mAP50 0.3869
```

**FLIR 24 px** (`regf-24px-s0`)

```
2026-11-13 05:42:07,851 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.4381  R 0.0530  mAP50 0.0451  mAP50-95 0.0160
2026-11-13 05:42:40,356 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 10.00  SSIM 0.1901  LPIPS 0.5408  FID 350.88  KID 0.4127+-0.0195
2026-11-13 05:55:10,437 INFO    t2o.engine.detector_stage: detector stage0: P 0.3608  R 0.2204  mAP50 0.2302  mAP50-95 0.0865
2026-11-13 05:55:10,437 INFO    t2o.engine.loop: stage 0 done: task_weight=0.0 epochs=100
2026-11-13 08:04:40,577 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.6099  R 0.2059  mAP50 0.2178  mAP50-95 0.0992
2026-11-13 08:05:17,516 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 12.33  SSIM 0.3524  LPIPS 0.4789  FID 163.82  KID 0.1277+-0.0124
2026-11-13 08:17:39,777 INFO    t2o.engine.detector_stage: detector stage1: P 0.6429  R 0.2663  mAP50 0.2881  mAP50-95 0.1127
2026-11-13 08:17:39,788 INFO    t2o.engine.loop: stage 1 done: task_weight=1.0 epochs=100
2026-11-13 10:53:30,656 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.6396  R 0.0657  mAP50 0.0737  mAP50-95 0.0334
2026-11-13 10:54:05,959 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 10.91  SSIM 0.1539  LPIPS 0.5734  FID 347.34  KID 0.4211+-0.0205
2026-11-13 11:06:05,843 INFO    t2o.engine.detector_stage: detector stage2: P 0.4056  R 0.2781  mAP50 0.2890  mAP50-95 0.1085
2026-11-13 11:06:05,863 INFO    t2o.engine.loop: stage 2 done: task_weight=2.0 epochs=100
2026-11-13 14:07:03,665 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.6199  R 0.2095  mAP50 0.2145  mAP50-95 0.0857
2026-11-13 14:07:40,976 INFO    t2o.metrics.fidelity: fidelity val vs visible (1013 images): PSNR 12.75  SSIM 0.4176  LPIPS 0.5124  FID 162.91  KID 0.1313+-0.0154
2026-11-13 14:19:36,331 INFO    t2o.engine.detector_stage: detector stage3: P 0.4154  R 0.2908  mAP50 0.3011  mAP50-95 0.1184
2026-11-13 14:19:36,357 INFO    t2o.engine.loop: stage 3 done: task_weight=3.0 epochs=100
2026-11-13 14:19:36,362 INFO    t2o.cli: stage 0 (task_weight=0.0): 100 epochs | zero-shot mAP50 0.0451 | LPIPS 0.5408 | fine-tuned mAP50 0.2302
2026-11-13 14:19:36,362 INFO    t2o.cli: stage 1 (task_weight=1.0): 100 epochs | zero-shot mAP50 0.2178 | LPIPS 0.4789 | fine-tuned mAP50 0.2881
2026-11-13 14:19:36,362 INFO    t2o.cli: stage 2 (task_weight=2.0): 100 epochs | zero-shot mAP50 0.0737 | LPIPS 0.5734 | fine-tuned mAP50 0.2890
2026-11-13 14:19:36,362 INFO    t2o.cli: stage 3 (task_weight=3.0): 100 epochs | zero-shot mAP50 0.2145 | LPIPS 0.5124 | fine-tuned mAP50 0.3011
```

The same numbers by stage, transcribed from the blocks above (zero-shot / adapted / FID):

| Run | stage 0 | stage 1 | stage 2 | stage 3 |
| --- | --- | --- | --- | --- |
| custom 3 px | 0.7982 / 0.9116 / 99.42 | 0.8346 / 0.9348 / 89.35 | 0.7919 / 0.9100 / 90.58 | 0.7682 / 0.9241 / 97.71 |
| custom 6 px | 0.7415 / 0.9219 / 115.25 | 0.7634 / 0.9138 / 100.27 | 0.8110 / 0.9049 / 100.79 | 0.8256 / 0.9052 / 97.13 |
| custom 12 px | 0.6720 / 0.8855 / 148.96 | 0.8056 / 0.8877 / 134.76 | 0.7811 / 0.8907 / 144.26 | 0.7967 / 0.8853 / 133.52 |
| custom 24 px | 0.5268 / 0.7668 / 138.79 | 0.7278 / 0.8427 / 133.74 | 0.7453 / 0.8665 / 131.68 | 0.7383 / 0.8397 / 140.79 |
| FLIR 0 px | 0.0453 / 0.3270 / **211.81** | 0.3003 / 0.4532 / 131.41 | 0.3168 / 0.4677 / 122.37 | 0.1212 / 0.3729 / **227.62** |
| FLIR 3 px | 0.0963 / 0.3518 / 112.99 | 0.3277 / 0.4163 / 127.29 | 0.3299 / 0.4370 / 93.13 | 0.3416 / 0.4303 / 94.76 |
| FLIR 6 px | 0.0668 / 0.3638 / **243.31** | 0.0278 / 0.3670 / **289.90** | 0.1204 / 0.3345 / **299.77** | 0.2596 / 0.4131 / 163.96 |
| FLIR 12 px | 0.0157 / 0.3236 / **216.97** | 0.2550 / 0.3802 / 114.99 | 0.2690 / 0.3711 / 132.78 | 0.2888 / 0.3869 / 148.46 |
| FLIR 24 px | 0.0451 / 0.2302 / **350.88** | 0.2178 / 0.2881 / 163.82 | 0.0737 / 0.2890 / **347.34** | 0.2145 / 0.3011 / 162.91 |

Bold marks FID above 200.

Each run's results file is `runs/<name>/metrics.json` on the server. No renderer exists for this
axis yet. REGISTRATION-AXIS-07 builds the degradation report.

### The crash and the resume

The FLIR 0 px attempt's last epochs and its error on 2026-11-10, then the restart's first lines on
2026-11-11:

```
2026-11-10 18:21:32,894 INFO    t2o.engine.trainer: epoch 19 took 48.9 s
2026-11-10 18:21:32,894 INFO    t2o.engine.trainer: epoch 19 peak memory: 14.90 GiB allocated, 17.57 GiB reserved of 39.70 GiB
2026-11-10 18:22:23,139 INFO    t2o.engine.trainer: epoch 20 took 49.8 s
2026-11-10 18:22:23,139 INFO    t2o.engine.trainer: epoch 20 peak memory: 14.90 GiB allocated, 17.42 GiB reserved of 39.70 GiB
Traceback (most recent call last):
  File "C:\Users\atick\AppData\Local\Programs\Python\Python312\Lib\multiprocessing\queues.py", line 264, in _feed
    obj = _ForkingPickler.dumps(obj)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\atick\AppData\Local\Programs\Python\Python312\Lib\multiprocessing\reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torch\multiprocessing\reductions.py", line 604, in reduce_storage
    metadata = storage._share_filename_cpu_()
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torch\storage.py", line 449, in wrapper
    return fn(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torch\storage.py", line 531, in _share_filename_cpu_
    return super()._share_filename_cpu_(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Couldn't open shared file mapping: <torch_22800_2855849386_36>, error code: <1455>
=== FINISHED 2026-11-11 09:30:11 +03:00 ===
2026-11-11 09:35:58,793 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir
2026-11-11 09:35:59,752 INFO    t2o.data.misalignment: de-rolled run: calibration\flir.json (digest 210142740cdba163, 9.32 px) matches both splits' thermal labels
2026-11-11 09:36:09,210 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\train\visible\images: 600 paired samples (capped from 4129), 600 annotated
2026-11-11 09:36:09,210 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\train\visible\images: visible side de-rolled by calibration 210142740cdba163; boxes read from the thermal side
2026-11-11 09:36:11,957 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\val\visible\images: 153 paired samples (capped from 1013), 153 annotated
2026-11-11 09:36:54,620 INFO    t2o.engine.trainer: epoch 21 took 42.5 s
2026-11-11 09:36:54,620 INFO    t2o.engine.trainer: epoch 21 peak memory: 14.90 GiB allocated, 17.14 GiB reserved of 39.70 GiB
```

## Findings

### F229 — On custom, stage-3 zero-shot sits below the aligned seed's 0.8548 at every level: 0.7682 / 0.8256 / 0.7967 / 0.7383 at 3 / 6 / 12 / 24 px. The curve is not monotone, while stage 0's is

Against `e3b-loop-s0` (same seed, config and judge, aligned), the stage-3 deficits are −0.0866,
−0.0292, −0.0581 and −0.1165. All four sit below e3b's six-seed loop mean less one sd,
0.8487 − 0.0140 (record 007). The 3 px point reads below 6 and 12 px, which no misalignment
account predicts. At n = 1, the level effect below 24 px cannot be separated from seed noise. At
stage 0 (λ = 0), zero-shot falls with every level: 0.7982 → 0.7415 → 0.6720 → 0.5268. The loop
stages then lift the misaligned runs most: stage 0 → 3 changes by −0.0300, +0.0841, +0.1247 and
+0.2115. Stage 3 also has 400 warm-started epochs against stage 0's 100, and there is no control
arm. So this shows the stage-3 spread narrowing (0.7383–0.8256 against 0.5268–0.7982). It does
not show λ causing that. ADR-01 expected the loop arm might degrade *faster* on misregistered
pairs. This cell cannot test that without a no-loop arm at matched epochs.

Against label transfer on the same levels (record 036, F224), translation zero-shot is lower at
every level: by 0.1621 / 0.1045 / 0.1191 / 0.1199. At 0 px the gap is 0.0763 (0.9311 against
0.8548). From 0 to 24 px, zero-shot loses 0.1165 against label transfer's 0.0729.

### F230 — On custom, the adapted arm declines steadily, 0.9241 / 0.9052 / 0.8853 / 0.8397, and stays within 0.031 of label transfer at every level

The stage-3 fine-tuned mAP50 falls 0.0844 from 3 to 24 px. Label transfer falls 0.0721 over the
same levels (0.9303 → 0.8582). The adapted arm sits 0.0062, 0.0249, 0.0305 and 0.0185 below the
label-transfer seed means at 3, 6, 12 and 24 px. Both arms train a `yolo11n` for 50 epochs on the
same moved boxes (ADR-01). They differ in the pixels, translated against raw thermal, and in the
init, the optical detector against COCO. This anchors no 0 px point: no record read for this one
gives `e3b-loop-s0`'s adapted-arm stage-3 value.

### F231 — FLIR's translator collapses by draw on de-rolled pairs too: 8 of 20 stages have FID 211.81–350.88, every one of them reads zero-shot ≤ 0.1212, and one of them is 0 px's stage 3, so the stage-3 curve reads lowest at its zero point

The other 12 stages have FID 93.13–163.96, and 11 of them read zero-shot 0.2145–0.3416. The
exception is 3 px stage 0, at 0.0963 with FID 112.99. Stage 0 (λ = 0) reads ≤ 0.0963 in all five
runs. The collapses don't follow the level. 6 px collapsed at three stages running and recovered at
stage 3. 24 px collapsed at stages 0 and 2. 0 px was healthy at stages 1–2 (FID 131.41, 122.37;
zero-shot 0.3003, 0.3168) and collapsed at stage 3 (FID 227.62; 0.1212). So the stage-3 curve,
0.1212 / 0.3416 / 0.2596 / 0.2888 / 0.2145 at 0 / 3 / 6 / 12 / 24 px, is ordered by which stage
collapsed, not by misalignment. The custom set shows no collapse (FID 89.35–148.96). This confirms
F140's "a single-seed FLIR pix2pix number is near meaningless", now on ADR-02's de-rolled pairs
and thermal-side boxes, which F140's runs did not use. The numbers themselves are not comparable
with record 016 (ADR-02).

### F232 — On FLIR, translation stays far below label transfer at every level, on both arms: stage-3 zero-shot 0.1212–0.3416 and adapted 0.3011–0.4303, against label transfer's all-class 0.4161–0.6069

Level by level (0 / 3 / 6 / 12 / 24 px), label transfer's all-class mAP50 leads stage-3 zero-shot
by 0.4857, 0.2649, 0.3327, 0.2694 and 0.2016, and leads the adapted arm by 0.2340, 0.1762, 0.1792,
0.1713 and 0.1150. Label transfer at 24 px, 0.4161, beats every zero-shot stage of every run, the best being
0.3416. The adapted arm passes it at five stages, all on the 0 px and 3 px runs (best 0.4677). The comparison is heavily confounded in label transfer's favour. Its
detector is a `yolo11s` trained for 100 epochs on all 4,129 train images. The adapted arm is a
`yolo11n` fine-tuned for 50 epochs on 600, and zero-shot uses a visible-trained judge. So this
bounds what translation would have to close on FLIR. It is not a like-for-like rate comparison.

## Provenance caveats

- **One seed per non-zero level** (plan Q4), and on FLIR at 0 px too. F231 shows a FLIR seed can
  collapse at any stage, so each FLIR point is one draw.
- **The custom zero point is a different run.** `e3b-loop-s0` trained in August at an earlier
  commit (record 007). REGISTRATION-AXIS-05 pins that the default `misalignment_px` leaves its
  config hash unchanged, but the two runs share a config, not an invocation.
- **FLIR 0 px was resumed.** The 2026-11-10 attempt ran epochs 0–20 of stage 0 at `--workers 8`
  and hung. The restart resumed at epoch 21 at `--workers 4`, so its stage-0 summary says
  "79 epochs": the epochs run in that invocation. Stages 1–3 ran wholly after the resume. Whether
  a resumed stage 0 reproduces an uninterrupted one was not checked. The stage-3 collapse sits two
  stages after the resume.
- **Neither 0 px is truly zero.** The custom set's val boxes came through the sister repo's
  registration. FLIR keeps about 2.25 px of median per-pair scatter after calibration (record 012).
- **FLIR's two halves don't share a training corpus.** The translators train on the 600-pair
  subset. Record 036's label-transfer detectors train on all 4,129.
- **FLIR's translation numbers here are not comparable with record 016** (ADR-02): de-rolled
  targets and thermal-side val boxes, against rolled pairs and visible-side boxes.
- **FLIR's zero-shot is all-class only.** The 3-class primary that record 036 reports for FLIR
  needs a re-score of these exports.
- **No out-of-family judge or faithfulness pass was run** on these exports. Faithfulness was
  decided under `rtdetr` (record 033), and none of that applies here yet.
- **Four FLIR logs carry 2026-11-10 in their names** (`regf-03px`, `-06px`, `-12px`, `-24px`), but
  their provenance blocks put them at 2026-11-12 to 2026-11-14. The record cites them as named.
- **The translation logs are raw, not condensed** (1.8–3.0 MB each).

## Next

- **REGISTRATION-AXIS-07**, the degradation report. On custom, the zero-shot curve at n = 1 needs
  the bootstrap CI over val images the plan specifies before any level below 24 px is called a
  drop (F229). The adapted arm (F230) is the cleaner reading.
- **FLIR's translation half cannot be reported as a curve from one seed** (F231). One option: two
  more seeds at each FLIR level, about 21 GPU-hours per level. Another: report FLIR's half as
  "unstable at n = 1" and leave the curve to custom. That choice is the human's. Prediction, if
  seeds are added: a second 0 px seed will not reproduce 0.1212 at stage 3, and FLIR's stage-3
  seed sd will exceed custom's.
- **Re-score FLIR's stage-3 exports on the 3-class primary**, so the two FLIR halves are on one
  metric.
