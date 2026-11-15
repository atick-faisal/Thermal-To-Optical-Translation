# 039 — CycleGAN cascade: a trained unpaired stage 2 on top of frozen pix2pix adds no zero-shot mAP50 over stage 1 alone (control −0.0005, loop −0.0216, seed 0), and the probe's dose projection held at 25.0% against 26%

**Date:** 2026-11-14 · **Task:** CYCLEGAN-CASCADE-04 · **Machine:** VSIBMEOT05, 2× A100 (exports, probe and control on `cuda:0`; loop on `cuda:1`) · **Wall clock:** 26 h 46 min end to end (2026-11-14 09:55:21 → 2026-11-15 12:41:15 +03:00), including a 4 h 51 min stop after the probe for CYCLEGAN-CASCADE-05 and 5 h 7 min idle before the metrics step. Per step: export-control 3 min 36 s, export-loop 3 min 1 s, probe 3 h 7 min 20 s, loss-share 27 s, control 10 h 29 min 29 s, control-faithfulness 44 s, loop 13 h 32 min 5 s (in parallel with control), loop-faithfulness 30 s, metrics 10 s
**git_sha:** `pre-spec-migration-276-gd4b5781` for the exports, probe and loss-share; `pre-spec-migration-278-ge11380b` for the two campaign runs, both faithfulness steps and the metrics dump. Both are clean trees, no `-dirty` suffix. Between them, `git diff d4b5781 e11380b --stat` touches only the two cascade configs (`grad_scale` 0.15 → 0.6, CYCLEGAN-CASCADE-05) and two feature docs · **W&B:** none (`runtime.wandb: false`; no W&B line in any log) · **Log:** `logs/2026-11-14-CYCLEGAN-CASCADE-04-export-control.txt`, `logs/2026-11-14-CYCLEGAN-CASCADE-04-export-loop.txt`, `logs/2026-11-14-CYCLEGAN-CASCADE-04-probe.txt`, `logs/2026-11-14-CYCLEGAN-CASCADE-04-loss-share.txt`, `logs/2026-11-14-CYCLEGAN-CASCADE-04-control.txt`, `logs/2026-11-14-CYCLEGAN-CASCADE-04-control-faithfulness.txt`, `logs/2026-11-14-CYCLEGAN-CASCADE-04-loop.txt`, `logs/2026-11-14-CYCLEGAN-CASCADE-04-loop-faithfulness.txt`, `logs/2026-11-14-CYCLEGAN-CASCADE-04-metrics.txt`

This is the professor's paired→unpaired cascade, framed as paper-2 exploration (plan Q1): no TIM
claim, one seed per arm, no significance test. Stage 1 is record 007's frozen e3b pix2pix at
stage 3, seed 0. Its train and val translations were exported once per arm into a paired tree
(pix2pix's RGB output as the infrared side, the real visible frame beside it). Stage 2 is a plain
CycleGAN trained for 200 epochs on that tree, with unpaired visible targets on train. Two arms:

- **control (`cuda:0`):** `e3b-control-s0` → CycleGAN, no detection term anywhere.
- **loop (`cuda:1`):** `e3b-loop-s0` → CycleGAN → frozen optical `yolo11n`, whose loss trains the
  CycleGAN's forward generator at `grad_scale` 0.6.

Before the campaign, a 50-epoch probe of the loop config at `grad_scale` 0.15 measured the
detection term's share, and CYCLEGAN-CASCADE-05 set 0.6 from it. The comparison of record is
cascade against stage 1 alone, on the same seed, judge and val split.

## Command

From the provenance blocks, as run:

```
# 1a. stage-1 trees (git d4b5781)
uv run t2o export --config experiments/e3_pix2pix_control.yaml --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --checkpoint runs/e3b-control-s0/stage3/translator_last.pt --out runs/e3b-control-s0/stage3/paired --paired --workers 4 --device cuda:0
uv run t2o export --config experiments/e3_pix2pix_loop.yaml --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --checkpoint runs/e3b-loop-s0/stage3/translator_last.pt --out runs/e3b-loop-s0/stage3/paired --paired --workers 4 --device cuda:0

# 1b. dose probe, then its share over the constant-rate first 25 epochs (git d4b5781)
uv run t2o loop --config experiments/cascade_cyclegan_loop.yaml --in-loop-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --eval-init-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --epochs 50 --name cascade-cyclegan-probe --no-detector --workers 4 --resume --device cuda:0
uv run python scripts/loss_share.py --runs runs/cascade-cyclegan-probe --first-epochs 25

# 2A. control, then its faithfulness (git e11380b)
uv run t2o loop --config experiments/cascade_cyclegan_control.yaml --in-loop-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --eval-init-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --workers 4 --resume --device cuda:0
uv run t2o faithfulness --translated runs/cascade-cyclegan-control/stage0/translated --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --weights runs/reference-yolo11s/weights/best.pt --write-back --device cuda:0

# 2B. loop, then its faithfulness (git e11380b)
uv run t2o loop --config experiments/cascade_cyclegan_loop.yaml --in-loop-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --eval-init-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --workers 4 --resume --device cuda:1
uv run t2o faithfulness --translated runs/cascade-cyclegan-loop/stage0/translated --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --weights runs/reference-yolo11s/weights/best.pt --write-back --device cuda:1

# 3. metrics dump (git e11380b)
uv run python -c import pathlib; [print(p, p.read_text(), sep="\n") for p in sorted(pathlib.Path("runs").glob("cascade-cyclegan-*/metrics.json"))]
```

The runbook's last two steps, `loss_share.py` on `runs/cascade-cyclegan-loop` with and without
`--first-epochs 25`, have no log. The share the loop run reached is computed on the Mac from the
metrics dump instead; see Results, step 6.

## Environment

The export-control log's block. The other eight blocks differ from it only in their `captured`,
`command` and `git_sha` lines (checked by hashing each block with those three lines removed: one
hash for all nine).

```
=== PROVENANCE ===
captured: 2026-11-14 09:55:21 +03:00
hostname: VSIBMEOT05
os: Microsoft Windows Server 2022 Standard 10.0.20348 (build 20348)
cpu: AMD EPYC 7763 64-Core Processor
cpu_logical_cores: 64
ram_total: 128.0 GB
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16; NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16
cuda_driver: 13.0
git_sha: pre-spec-migration-276-gd4b5781
working_dir: D:\Atick\GitHub\Thermal-To-Optical-Translation
command: uv run t2o export --config experiments/e3_pix2pix_control.yaml --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --checkpoint runs/e3b-control-s0/stage3/translator_last.pt --out runs/e3b-control-s0/stage3/paired --paired --workers 4 --device cuda:0
python: 3.12.7
torch: 2.12.1+cu130
cuda_torch_build: 13.0
cudnn: 92000
cuda_available: True
gpu0_torch: NVIDIA A100-SXM4-40GB, 39.7 GiB, sm 8.0
=== END PROVENANCE ===
```

## Configuration

| Axis | Value | Where it is evidenced |
| --- | --- | --- |
| Stage 1 | frozen pix2pix, `runs/e3b-{control,loop}-s0/stage3/translator_last.pt` (record 007, `grad_scale` 0.15). Exported with the committed `experiments/e3_pix2pix_{control,loop}.yaml`, not the run's own snapshot (runbook: only the network shape matters to an export). 600 train + 153 val per arm, `normalize=clamp` | the export logs |
| Arms | matched (plan Q5): control ← `e3b-control-s0`, loop ← `e3b-loop-s0`. The configs differ only in `coupling.task_weights` (`[0.0]` / `[1.0]`), `runtime.name` and `data.manifest` | `experiments/cascade_cyclegan_{control,loop}.yaml` at `e11380b`; `tests/test_cascade_cyclegan_experiments.py` |
| Stage 2 | plain CycleGAN (plan Q7): `resnet_9blocks` G, `basic` D, ngf = ndf = 64, `lambda_cycle` 10, `lambda_identity` 0.5 (an effective 5), `loss.gan` 1.0, `l2` = `lpips` = 0, ImagePool 50 | the configs; plan CYCLEGAN-CASCADE-03 |
| Schedule | one stage of 200 epochs (plan Q9), batch 1 (Q11), so 120,000 updates; `lr` 2.0e-4 constant for epochs 0–99, linear to zero over 100–199 (Q10); float32; `hflip` 0.5; full 640×480 frames; `train.seed` 0 | the configs; 200 `epoch N took` lines per arm |
| Data | `data.infrared_channels: 3` (ADR-01), `data.unpaired_visible: true`: train targets drawn by a permutation seeded 0, val stays paired | `visible targets unpaired by a permutation seeded 0` lines |
| Dose | loop `task_weights [1.0]` × `grad_scale` 0.6, set by CYCLEGAN-CASCADE-05 from the probe's 8.2% at 0.15. The probe ran the loop config at `--epochs 50 --no-detector`, so it built no evaluation detector | the loss-share log; `e11380b`'s config comment |
| In-loop detector | `D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt`, the optical `yolo11n` e3b-loop-s0 used (loop arm only) | the commands |
| Zero-shot judge | `detector.reference.weights`, `runs/reference-yolo11s/weights/best.pt` (record 004's `yolo11s`), on real val labels. The same judge scored stage 1's 0.8548 / 0.8128 (record 027) | `YOLO11s summary (fused): ... 9,414,735 parameters` before each `eval best.pt` line |
| Adapted arm | `yolo11n` fine-tuned 50 epochs on each arm's stage-2 train translations, from `--eval-init-weights` | `detector stage0:` lines |
| Faithfulness | `t2o faithfulness` defaults (IoU 0.5, conf 0.25), the same `yolo11s` judge, written back to stage 0 | the faithfulness commands; configs' `metrics` |
| Config hash | not printed by any log | — |

## Results

### Step 1 — stage-1 exports

Everything after the provenance block, verbatim. export-control:

```
2026-11-14 09:56:35,901 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-14 09:56:45,371 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\train\visible\images: 600 paired samples, 600 annotated
2026-11-14 09:58:21,360 INFO    t2o.engine.export: exported 600 translated images to runs\e3b-control-s0\stage3\paired\train (normalize=clamp)
2026-11-14 09:58:23,814 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\val\visible\images: 153 paired samples, 153 annotated
2026-11-14 09:58:54,888 INFO    t2o.engine.export: exported 153 translated images to runs\e3b-control-s0\stage3\paired\val (normalize=clamp)
2026-11-14 09:58:54,903 INFO    t2o.cli: wrote runs\e3b-control-s0\stage3\paired\data.yaml
initialize network with normal
initialize network with normal
=== FINISHED 2026-11-14 09:58:57 +03:00 ===
```

export-loop:

```
2026-11-14 09:59:27,163 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-14 09:59:27,537 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\train\visible\images: 600 paired samples, 600 annotated
2026-11-14 10:01:22,923 INFO    t2o.engine.export: exported 600 translated images to runs\e3b-loop-s0\stage3\paired\train (normalize=clamp)
2026-11-14 10:01:23,050 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\val\visible\images: 153 paired samples, 153 annotated
2026-11-14 10:01:58,893 INFO    t2o.engine.export: exported 153 translated images to runs\e3b-loop-s0\stage3\paired\val (normalize=clamp)
2026-11-14 10:01:58,904 INFO    t2o.cli: wrote runs\e3b-loop-s0\stage3\paired\data.yaml
initialize network with normal
initialize network with normal
=== FINISHED 2026-11-14 10:02:00 +03:00 ===
```

### Step 2 — dose probe

An excerpt: every line but the per-epoch `took` and `peak memory` lines of epochs 1–49. Those 98
lines read 221.0–226.6 s per epoch (all 50 epochs: mean 223.5 s) and 10.54–10.55 GiB allocated,
11.40–11.46 GiB reserved. Rebuilt with
`grep -vE 'epoch ([1-9]|[1-4][0-9]) (took|peak)' logs/2026-11-14-CYCLEGAN-CASCADE-04-probe.txt`.

```
2026-11-14 10:02:31,102 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\paired
2026-11-14 10:02:31,165 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\paired\train\visible\images: 600 paired samples, 600 annotated
2026-11-14 10:02:31,622 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\paired
initialize network with normal
initialize network with normal
initialize network with normal
initialize network with normal
2026-11-14 10:02:35,483 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\paired\train\visible\images: 600 paired samples, 600 annotated
2026-11-14 10:02:35,483 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\paired\train\visible\images: visible targets unpaired by a permutation seeded 0
2026-11-14 10:02:36,206 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\paired\val\visible\images: 153 paired samples, 153 annotated
2026-11-14 10:06:24,827 INFO    t2o.engine.trainer: epoch 0 took 228.6 s
2026-11-14 10:06:24,827 INFO    t2o.engine.trainer: epoch 0 peak memory: 10.54 GiB allocated, 11.48 GiB reserved of 39.70 GiB
2026-11-14 13:09:20,169 INFO    t2o.engine.loop: stage 0 done: task_weight=1.0 epochs=50
2026-11-14 13:09:20,169 INFO    t2o.cli: stage 0 (task_weight=1.0): 50 epochs
=== FINISHED 2026-11-14 13:09:22 +03:00 ===
```

### Step 3 — probe share

Everything after the provenance block, verbatim:

```
2026-11-14 13:09:50,769 INFO    __main__: stage runs  epochs     w lambda_eff     loss_l2  loss_lpips    loss_gan  loss_cycle    loss_idt    loss_det  loss_total     w*det   share
2026-11-14 13:09:50,769 INFO    __main__:     0    1      25   1.0     0.1500          --          --      0.7480      2.0981      0.9616      0.3416      4.1493    0.3416    8.2%
=== FINISHED 2026-11-14 13:09:52 +03:00 ===
```

### Step 4 — the two campaign runs

An excerpt of each: the `t2o` lines, the network-init lines, ultralytics' banner and summary
lines, and the epoch-0 and epoch-199 timing lines. It leaves out the per-epoch lines of epochs
1–198, the ultralytics progress bars, the model table and the trainer argument line. Rebuilt with
`grep -E 't2o\.|=== FINISHED|^initialize|^YOLO11|^Ultralytics 8|Starting training|epochs completed' <log> | grep -vE 'epoch ([1-9]|[1-9][0-9]|1[0-8][0-9]|19[0-8]) (took|peak)'`.

Epoch times over all 200 epochs: control 180.5–201.0 s (mean 185.1 s, 37,017.5 s in total); loop
216.4–262.1 s (mean 240.2 s, 48,030.7 s). Loop epochs that ran while control was still running
averaged 241.3 s (155 epochs), and 236.2 s after it finished (45 epochs). Peak memory: control
10.47–10.48 GiB allocated, loop 10.54–10.55 GiB, of 39.70 GiB.

Control:

```
2026-11-14 18:01:32,572 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s0\stage3\paired
2026-11-14 18:01:32,630 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s0\stage3\paired\train\visible\images: 600 paired samples, 600 annotated
2026-11-14 18:01:33,106 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s0\stage3\paired
2026-11-14 18:01:44,347 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s0\stage3\paired\train\visible\images: 600 paired samples, 600 annotated
2026-11-14 18:01:44,347 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s0\stage3\paired\train\visible\images: visible targets unpaired by a permutation seeded 0
2026-11-14 18:01:46,954 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s0\stage3\paired\val\visible\images: 153 paired samples, 153 annotated
2026-11-14 18:04:55,965 INFO    t2o.engine.trainer: epoch 0 took 189.0 s
2026-11-14 18:04:55,970 INFO    t2o.engine.trainer: epoch 0 peak memory: 10.47 GiB allocated, 11.35 GiB reserved of 39.70 GiB
2026-11-15 04:20:47,256 INFO    t2o.engine.trainer: epoch 199 took 183.7 s
2026-11-15 04:20:47,256 INFO    t2o.engine.trainer: epoch 199 peak memory: 10.47 GiB allocated, 11.29 GiB reserved of 39.70 GiB
2026-11-15 04:20:48,133 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s0\stage3\paired\train\visible\images: 600 paired samples, 600 annotated
2026-11-15 04:21:46,133 INFO    t2o.engine.export: exported 600 translated images to runs\cascade-cyclegan-control\stage0\translated\train (normalize=clamp)
2026-11-15 04:21:46,226 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-control-s0\stage3\paired\val\visible\images: 153 paired samples, 153 annotated
2026-11-15 04:22:07,093 INFO    t2o.engine.export: exported 153 translated images to runs\cascade-cyclegan-control\stage0\translated\val (normalize=clamp)
initialize network with normal
initialize network with normal
initialize network with normal
initialize network with normal
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
2026-11-15 04:22:57,756 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8722  R 0.7474  mAP50 0.8124  mAP50-95 0.5471
2026-11-15 04:23:06,771 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 15.75  SSIM 0.5476  LPIPS 0.3046  FID 73.58  KID 0.0078+-0.0042
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11n summary: 182 layers, 2,590,815 parameters, 2,590,799 gradients, 6.5 GFLOPs
Starting training for 50 epochs...
50 epochs completed in 0.105 hours.
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11n summary (fused): 101 layers, 2,583,127 parameters, 0 gradients, 6.4 GFLOPs
2026-11-15 04:30:30,672 INFO    t2o.engine.detector_stage: detector stage0: P 0.9364  R 0.8164  mAP50 0.9039  mAP50-95 0.6203
2026-11-15 04:30:30,683 INFO    t2o.engine.loop: stage 0 done: task_weight=0.0 epochs=200
2026-11-15 04:30:30,683 INFO    t2o.cli: stage 0 (task_weight=0.0): 200 epochs | zero-shot mAP50 0.8124 | LPIPS 0.3046 | fine-tuned mAP50 0.9039
=== FINISHED 2026-11-15 04:30:33 +03:00 ===
```

Loop:

```
2026-11-14 18:01:56,087 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\paired
2026-11-14 18:01:56,206 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\paired\train\visible\images: 600 paired samples, 600 annotated
2026-11-14 18:01:56,769 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\paired
initialize network with normal
initialize network with normal
initialize network with normal
initialize network with normal
2026-11-14 18:01:57,677 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\paired\train\visible\images: 600 paired samples, 600 annotated
2026-11-14 18:01:57,677 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\paired\train\visible\images: visible targets unpaired by a permutation seeded 0
2026-11-14 18:01:57,824 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\paired\val\visible\images: 153 paired samples, 153 annotated
2026-11-14 18:06:11,712 INFO    t2o.engine.trainer: epoch 0 took 253.9 s
2026-11-14 18:06:11,712 INFO    t2o.engine.trainer: epoch 0 peak memory: 10.54 GiB allocated, 11.48 GiB reserved of 39.70 GiB
2026-11-15 07:24:33,888 INFO    t2o.engine.trainer: epoch 199 took 218.5 s
2026-11-15 07:24:33,888 INFO    t2o.engine.trainer: epoch 199 peak memory: 10.55 GiB allocated, 11.45 GiB reserved of 39.70 GiB
2026-11-15 07:24:34,728 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\paired\train\visible\images: 600 paired samples, 600 annotated
2026-11-15 07:25:27,799 INFO    t2o.engine.export: exported 600 translated images to runs\cascade-cyclegan-loop\stage0\translated\train (normalize=clamp)
2026-11-15 07:25:27,913 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\paired\val\visible\images: 153 paired samples, 153 annotated
2026-11-15 07:25:45,817 INFO    t2o.engine.export: exported 153 translated images to runs\cascade-cyclegan-loop\stage0\translated\val (normalize=clamp)
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
2026-11-15 07:26:21,758 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8917  R 0.7603  mAP50 0.8332  mAP50-95 0.5581
2026-11-15 07:26:29,884 INFO    t2o.metrics.fidelity: fidelity val vs visible (153 images): PSNR 15.65  SSIM 0.5493  LPIPS 0.3013  FID 76.08  KID 0.0125+-0.0049
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11n summary: 182 layers, 2,590,815 parameters, 2,590,799 gradients, 6.5 GFLOPs
Starting training for 50 epochs...
50 epochs completed in 0.102 hours.
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11n summary (fused): 101 layers, 2,583,127 parameters, 0 gradients, 6.4 GFLOPs
2026-11-15 07:33:27,444 INFO    t2o.engine.detector_stage: detector stage0: P 0.9050  R 0.8381  mAP50 0.8920  mAP50-95 0.6232
2026-11-15 07:33:27,455 INFO    t2o.engine.loop: stage 0 done: task_weight=1.0 epochs=200
2026-11-15 07:33:27,455 INFO    t2o.cli: stage 0 (task_weight=1.0): 200 epochs | zero-shot mAP50 0.8332 | LPIPS 0.3013 | fine-tuned mAP50 0.8920
=== FINISHED 2026-11-15 07:33:29 +03:00 ===
```

### Step 5 — faithfulness

Everything after the provenance block, verbatim. Control:

```
2026-11-15 04:31:02,292 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-15 04:31:17,684 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.1770  missed-object 0.1868  consistency 0.7890
2026-11-15 04:31:17,684 INFO    t2o.cli:   false-object rate       0.1770  (LOWER better)
2026-11-15 04:31:17,684 INFO    t2o.cli:   missed-object rate      0.1868  (LOWER better)
2026-11-15 04:31:17,684 INFO    t2o.cli:   detection-consistency   0.7890  (higher better)
2026-11-15 04:31:17,684 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-15 04:31:17,684 INFO    t2o.cli:     Fuse           false 0.1562  missed 0.1964
2026-11-15 04:31:17,684 INFO    t2o.cli:     Pole           false 0.1006  missed 0.0654
2026-11-15 04:31:17,684 INFO    t2o.cli:     Switch         false 0.3636  missed 0.4697
2026-11-15 04:31:17,684 INFO    t2o.cli:     Transformer    false 0.2955  missed 0.1389
2026-11-15 04:31:17,720 INFO    t2o.engine.loop: recorded faithfulness into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-cyclegan-control\metrics.json stage 0
=== FINISHED 2026-11-15 04:31:19 +03:00 ===
```

Loop:

```
2026-11-15 07:33:50,360 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-15 07:33:59,881 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.1330  missed-object 0.1678  consistency 0.8066
2026-11-15 07:33:59,882 INFO    t2o.cli:   false-object rate       0.1330  (LOWER better)
2026-11-15 07:33:59,882 INFO    t2o.cli:   missed-object rate      0.1678  (LOWER better)
2026-11-15 07:33:59,882 INFO    t2o.cli:   detection-consistency   0.8066  (higher better)
2026-11-15 07:33:59,883 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-15 07:33:59,883 INFO    t2o.cli:     Fuse           false 0.1139  missed 0.1667
2026-11-15 07:33:59,883 INFO    t2o.cli:     Pole           false 0.0637  missed 0.0392
2026-11-15 07:33:59,883 INFO    t2o.cli:     Switch         false 0.3077  missed 0.4545
2026-11-15 07:33:59,883 INFO    t2o.cli:     Transformer    false 0.2564  missed 0.1944
2026-11-15 07:33:59,905 INFO    t2o.engine.loop: recorded faithfulness into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-cyclegan-loop\metrics.json stage 0
=== FINISHED 2026-11-15 07:34:01 +03:00 ===
```

### Step 6 — metrics dump, and the share the loop run reached

The metrics log prints all three `metrics.json` files whole: 6,234 lines, almost all of them
per-epoch losses. Below is each file's stage summary, verbatim, from its `"detector"` key to the
closing `]`:

```
runs\cascade-cyclegan-control\metrics.json
    "detector": {
      "weights": "D:\\Atick\\GitHub\\Thermal-To-Optical-Translation\\runs\\cascade-cyclegan-control\\stage0\\detector\\stage0\\weights\\best.pt",
      "precision": 0.936376104805908,
      "recall": 0.8164429873873085,
      "map50": 0.9039340929620678,
      "map50_95": 0.6202753263390194
    },
    "zero_shot": {
      "precision": 0.8721894302791616,
      "recall": 0.7474208758732976,
      "map50": 0.8123513925381787,
      "map50_95": 0.5471140387915442,
      "per_class_ap50": {
        "Fuse": 0.8694785316217732,
        "Pole": 0.9669547443726034,
        "Switch": 0.5443257933783132,
        "Transformer": 0.8686465007800245
      },
      "per_class_ap50_95": {
        "Fuse": 0.5685942939959754,
        "Pole": 0.6974928664290465,
        "Switch": 0.2611951698868852,
        "Transformer": 0.6611738248542698
      }
    },
    "fidelity": {
      "psnr": 15.750655174255371,
      "ssim": 0.5475533604621887,
      "lpips": 0.3045540452003479,
      "fid": 73.58313751220703,
      "kid_mean": 0.0077845072373747826,
      "kid_std": 0.004195885267108679
    },
    "faithfulness": {
      "false_object_rate": 0.17703349282296652,
      "missed_object_rate": 0.1867612293144208,
      "detection_consistency": 0.789010989010989,
      "false_object_rate_by_class": {
        "Connector": 0.0,
        "Fuse": 0.15625,
        "Pole": 0.10062893081761007,
        "Switch": 0.36363636363636365,
        "Transformer": 0.29545454545454547
      },
      "missed_object_rate_by_class": {
        "Connector": 0.0,
        "Fuse": 0.19642857142857142,
        "Pole": 0.06535947712418301,
        "Switch": 0.4696969696969697,
        "Transformer": 0.1388888888888889
      }
    }
  }
]
runs\cascade-cyclegan-loop\metrics.json
    "detector": {
      "weights": "D:\\Atick\\GitHub\\Thermal-To-Optical-Translation\\runs\\cascade-cyclegan-loop\\stage0\\detector\\stage0\\weights\\best.pt",
      "precision": 0.9049591715284971,
      "recall": 0.8380734869705458,
      "map50": 0.8920140929944527,
      "map50_95": 0.6232117925881198
    },
    "zero_shot": {
      "precision": 0.8917100687910176,
      "recall": 0.7602601520868378,
      "map50": 0.8332108867182777,
      "map50_95": 0.5580818109891726,
      "per_class_ap50": {
        "Fuse": 0.8773675821352321,
        "Pole": 0.9758732711071525,
        "Switch": 0.6024214596953603,
        "Transformer": 0.8771812339353656
      },
      "per_class_ap50_95": {
        "Fuse": 0.5819430351411286,
        "Pole": 0.6721081929028337,
        "Switch": 0.2944672517673665,
        "Transformer": 0.6838087641453618
      }
    },
    "fidelity": {
      "psnr": 15.645277976989746,
      "ssim": 0.5492994785308838,
      "lpips": 0.3013231158256531,
      "fid": 76.08078002929688,
      "kid_mean": 0.012472419068217278,
      "kid_std": 0.004870269913226366
    },
    "faithfulness": {
      "false_object_rate": 0.1330049261083744,
      "missed_object_rate": 0.16784869976359337,
      "detection_consistency": 0.8065934065934066,
      "false_object_rate_by_class": {
        "Connector": 0.0,
        "Fuse": 0.11392405063291139,
        "Pole": 0.06369426751592357,
        "Switch": 0.3076923076923077,
        "Transformer": 0.2564102564102564
      },
      "missed_object_rate_by_class": {
        "Connector": 0.0,
        "Fuse": 0.16666666666666666,
        "Pole": 0.0392156862745098,
        "Switch": 0.45454545454545453,
        "Transformer": 0.19444444444444445
      }
    }
  }
]
runs\cascade-cyclegan-probe\metrics.json
    "detector": null,
    "zero_shot": null,
    "fidelity": null
  }
]
```

**The loop run's share, computed on the Mac (not a server log).** The metrics dump was split back
into the three `metrics.json` files, each in its own directory in a scratch folder, with the
committed `experiments/cascade_cyclegan_loop.yaml` as the `config.yaml` stub `loss_share.py`
reads: `d4b5781`'s for the probe, `e11380b`'s for the loop run. `grad_scale` feeds only the
`lambda_eff` column; the share is `w × mean(loss_det) / mean(loss_total)` over the logged epochs.
The first line reproduces the server's step-3 line to every printed digit, which checks the
extraction. Then `scripts/loss_share.py --runs <dir>` with `--first-epochs 25`, `--first-epochs
100` and no flag (logger prefix trimmed):

```
__main__: stage runs  epochs     w lambda_eff     loss_l2  loss_lpips    loss_gan  loss_cycle    loss_idt    loss_det  loss_total     w*det   share
__main__:     0    1      25   1.0     0.1500          --          --      0.7480      2.0981      0.9616      0.3416      4.1493    0.3416    8.2%
__main__:     0    1      25   1.0     0.6000          --          --      0.9286      2.3997      1.1211      1.4805      5.9299    1.4805   25.0%
__main__:     0    1     100   1.0     0.6000          --          --      0.8936      1.6389      0.7637      1.1735      4.4697    1.1735   26.3%
__main__:     0    1     200   1.0     0.6000          --          --      0.9456      1.2838      0.5863      1.0411      3.8567    1.0411   27.0%
```

The first line is the probe (`--first-epochs 25`); the other three are `runs/cascade-cyclegan-loop`.

Re-render without re-running: `uv run python scripts/loss_share.py --runs runs/cascade-cyclegan-loop --first-epochs 25`
on the server, and `uv run t2o aggregate --runs 'runs/cascade-cyclegan-*' --stage 0 --metric zero_shot.map50 fidelity.lpips faithfulness.false_object_rate faithfulness.missed_object_rate`.

## Findings

### F234 — A trained CycleGAN stage 2 adds no zero-shot detection over stage 1 alone: control −0.0005, loop −0.0216, on seed 0

The comparison is the cascade's stage-2 val output against stage 1's own stage-3 val export, same
seed, same `yolo11s` judge, same real val labels. Stage 1's values are record 027's CSV, which
record 037 confirms is the run's `zero_shot.map50`:

| arm | stage 1 alone (record 027) | cascade (this run) | cascade − stage 1 |
| --- | --- | --- | --- |
| control | 0.8128 | 0.8124 | −0.0005 |
| loop | 0.8548 | 0.8332 | −0.0216 |

- **The detection term did not lift stage 2 above its input.** The loop arm trained against the
  in-loop detector for 200 epochs at a 25–27% share (F238) and still scored below the pix2pix
  frames it started from.
- **Both differences are inside single-run noise.** Two nominally identical λ = 0 runs differed by
  0.0591 under this judge, the noise floor of F13, and record 007's loop arm has sd 0.0140 at stage 3. At one seed
  per arm this is a reading: the cascade shows no gain. It does not show a loss.
- Stage 2 is trained on pix2pix's train outputs, which pix2pix saw in training, and scored on its
  val outputs, which it did not. A stage 2 that learned to polish in-sample frames could score
  lower on val for that reason alone. This run cannot tell.

### F235 — After the cascade, loop − control is +0.0209 zero-shot mAP50, half stage 1's seed-0 gap of +0.0420

Loop − control, same seed: stage 1 alone 0.8548 − 0.8128 = +0.0420; after the cascade
0.8332 − 0.8124 = +0.0209. On Switch, the class record 007 could not claim (F45), the cascade's
zero-shot AP50 is 0.6024 (loop) against 0.5443 (control), +0.0581. The adapted arm goes the other
way: fine-tuned mAP50 0.8920 (loop) against 0.9039 (control), −0.0119.

- **The arms differ in two places** (plan Q5): the stage-1 checkpoint and stage 2's detection term.
  So the +0.0209 cannot be split between them, and the halving cannot be pinned on either stage.
- It is one seed pair. Record 007's paired stage-3 effect was +0.0512 with CI [+.025, +.081],
  so seed 0's +0.0420 sits inside that range.

### F236 — Stage 2 does not bring the frames closer to real optical by LPIPS: 0.3046 control, 0.3013 loop, against stage-1 arm means 0.2905 ± .0054 and 0.3002 ± .0106

The professor's premise was that the unpaired stage would move stage 1's output closer to real
optical frames. LPIPS against the paired real val frame, from this run, against record 007's
six-seed stage-3 arm means (seed 0's own LPIPS is not in any record):

| arm | stage 1 arm mean ± sd (record 007) | cascade | difference |
| --- | --- | --- | --- |
| control | 0.2905 ± .0054 | 0.3046 | +0.0141 (2.6 sd worse) |
| loop | 0.3002 ± .0106 | 0.3013 | +0.0011 |

FID is 73.58 (control) and 76.08 (loop), KID 0.0078 ± 0.0042 and 0.0125 ± 0.0049. No record gives
stage 1's seed-0 FID on the custom set, so FID has no comparison here. LPIPS is a paired metric,
and stage 2's train targets were unpaired, so a paired-distance score may undersell a change in
style. On the one metric that has a baseline, it moved no closer.

### F237 — Faithfulness holds through the trained stage 2: every pooled rate is within 1.8 sd of the stage-1 arm means, most on the better side

The `yolo11s` judge on real val, against record 008's six-seed stage-3 arm means (seed 0's own
rates are not in any record):

| arm | rate | stage 1 arm mean ± sd (record 008) | cascade | difference |
| --- | --- | --- | --- | --- |
| control | false-object | 0.1918 ± 0.0245 | 0.1770 | −0.0148 |
| control | missed-object | 0.1962 ± 0.0226 | 0.1868 | −0.0094 |
| control | consistency | 0.7677 ± 0.0220 | 0.7890 | +0.0213 |
| loop | false-object | 0.1629 ± 0.0169 | 0.1330 | −0.0299 |
| loop | missed-object | 0.1592 ± 0.0118 | 0.1678 | +0.0086 |
| loop | consistency | 0.7968 ± 0.0091 | 0.8066 | +0.0098 |

- Five of the six move the better way. The one that does not, the loop's missed-object rate, is
  0.7 sd worse. Its largest move, the loop's false-object rate, is 1.8 sd better.
- Loop − control after the cascade: false −0.0440, missed −0.0189, consistency +0.0176. Record
  008's paired stage-1 differences point the same way on all three: −0.0289, −0.0370, +0.0291.
- **Switch stays the weak class**: missed 0.4697 (control) and 0.4545 (loop), false 0.3636 and
  0.3077.
- Zero-shot PiSA-SR (record 035, F221) raised every rate. This run does not, but the two are not
  comparable number for number: record 035 used the `rtdetr` judge and nine paired seeds, and this
  run has one seed and no paired stage-1 value. The direction is the only thing that compares.

### F238 — The probe's straight-line projection held for CycleGAN: the loop run reached 25.0% over epochs 1–25 against a projected 26%, and 27.0% over all 200, but raw `loss_det` rose rather than fell

CYCLEGAN-CASCADE-05 projected 26% at `grad_scale` 0.6 from the probe's 8.2% at 0.15. Over the
same 25-epoch window, the loop run reached **25.0%** (×0.96 of the projection). Over the first
100 epochs it reached 26.3%, and over all 200, 27.0%. The share stayed inside the 20–30% band in
every window read.

- **The mechanism F29 predicted did not appear.** F29 expected a stronger dose to drive the raw
  detection loss down, so the share would land below its projection. The share did land below,
  by 1 pp. But raw `loss_det` over epochs 1–25 *rose*, from 2.28 at 0.15 (0.3416 / 0.15) to 2.47
  at 0.6 (1.4805 / 0.6), +8.4%. The other three terms rose more: gan + cycle + identity
  3.81 → 4.45, +16.9%. The share fell short of the projection because the rest of the objective
  grew, not because the detection loss shrank.
- Against earlier projections: pix2pix's 25-epoch probe landed at 10.9 / 16.6 / 22.5% against a
  predicted 12 / 21 / 27% (F30), and record 024's realised share was ×0.58 of its projection at
  stage 3 (F175). CycleGAN's ×0.96 is closer than either, from one run of a different backbone.
- These shares come from the Mac's run of `loss_share.py` on the metrics dump, not from a server
  log (Results, step 6).

### F239 — Batch-1 CycleGAN uses about a quarter of an A100 and takes 10 h 17 min (control) or 13 h 21 min (loop) for 200 epochs

The plan and runbook called the cost unmeasured. Peak memory was 10.47–10.55 GiB allocated and at
most 11.48 GiB reserved, of 39.70 GiB, in all three runs. The probe was also the memory check
for the heavier arm, and it passed. Mean epoch time was 185.1 s for control and 240.2 s for loop
(×1.30, the in-loop detector's forward and backward pass). The probe, running the loop config
alone on `cuda:0`, averaged 223.5 s. The loop run's epochs barely sped up once control finished
(241.3 s → 236.2 s), so the two arms sharing the machine is not what made the loop arm slower.
The faithfulness and stage-2 export steps each cost under a minute.

## Provenance caveats

- **Paper-2 exploration, one seed per arm, no significance test** (plan Q1). Every comparison
  above is a reading at n = 1, and none enters the TIM paper.
- **One arm per card** (plan Q3), against record 007's split-by-seed rule. With one seed there is
  no paired test for a GPU difference to bias, but the two arms' timings are not comparable as
  hardware-matched.
- **The arms differ in their stage-1 checkpoint as well as the detection term** (plan Q5), so stage
  2's share of any loop − control gap cannot be separated out (F235).
- **Stage 2 trains on pix2pix's in-sample train outputs but is scored on its out-of-sample val
  outputs.** The train inputs are cleaner than anything stage 2 is scored on (F234).
- **The stage-1 baselines come from other records.** Seed 0's zero-shot mAP50 is record 027's CSV
  (same judge and split). LPIPS and faithfulness have no seed-0 value in any record, so F236 and
  F237 compare one run against six-seed arm means, not against its own seed.
- **Two runbook logs are missing**: the post-run `loss_share.py` on the loop run, over 25 epochs and
  over all 200. F238's realised shares were computed on the Mac from the metrics dump, with
  committed config files standing in for the run's own `config.yaml` snapshot. The snapshot was
  not in any log. The probe line reproducing exactly is the check on the extraction, and
  `grad_scale` does not enter the share.
- **Three log filenames carry the campaign's start date, not the step's.** control-faithfulness,
  loop-faithfulness and metrics ran on 2026-11-15 (their `captured` lines) but are named
  `2026-11-14-…`. The contents are unaffected.
- **No config hash appears in any log**, so the two arms' hashes are not recorded here.
- The custom train split logs `FLIR1292.png: 1 duplicate labels removed` in both arms' stage-2
  detector fine-tune. It touches the adapted arm only, and both arms equally.

## Next

The trained stage 2 did what the zero-shot one did not: it left faithfulness intact (F237). But
it also added nothing to zero-shot detection (F234) and moved no closer to optical by LPIPS
(F236). On this evidence the cascade's case for paper 2 rests on something these metrics do not
measure, such as appearance. A side-by-side look at stage-1 and stage-2 val frames for the same
images would show whether CycleGAN changed the style at all, or learned something close to an
identity map. Prediction: if it is close to identity,
the per-frame mean absolute change from the stage-1 input will be smaller than PiSA-SR's
0.033 (record 035). That would also explain why nothing moved.

More seeds and a gated TIM attempt are out of scope (plan, Out of Scope).
