# 026 — Out-of-family RT-DETR judge: three judges trained and gated, every E3 stage-3 export re-scored for Transfer and Faithfulness

**Date:** 2026-10-19 · **Task:** OUT-OF-FAMILY-JUDGE-05 · **Machine:** VSIBMEOT05 (Windows Server 2022, 2× A100-SXM4-40GB, 128 GB RAM; user-stated 2026-10-20, hostname and GPUs also in the readouts' provenance section); FLIR judge and FLIR re-score on `cuda:0`, everything else on `cuda:1` (smoke runs on `cuda:0`) · **Wall clock:** judge training 5.91 GPU-h by ultralytics' own counts (FLIR 4.566 h on `cuda:0`; msrs-day 0.600 h then custom 0.740 h on `cuda:1`); smoke runs 0.027 h + 0.020 h; re-scoring about 1.3 h of logged work (msrs-day 16:35–16:56, FLIR 16:34–17:01, e3b 17:02–17:20, e3t 20:02–20:11, plus FLIR's last run at 20:02–20:04); readouts at 20:47–20:50
**git_sha:** `8fe8fea` (`origin/main` when the runbook was given; the runbook's step 0 checked it with `git log -1`, and both readouts print it as `git HEAD` with two untracked files, `probe-report.txt` and `splits/yolo_rgbt_29_jul.json`). It was not printed by the smoke, training or re-score runs themselves · **W&B:** not recorded · **Log:** `logs/2026-10-19-out-of-family-judge-*.txt` (fifteen files: `smoke-{a,b}`, `train-{flir,msrs-day,custom}-rtdert-l`, `gate`, `sanity`, `rescore-{flir,msrs-day,e3b,e3t}`, `report-{flir,msrs-day}`, `aggregate-{e3b,e3t}`) and `logs/2026-10-19-runs-{e3b,e3t}-judge-tidy.csv`

## Command

The runbook below, run exactly as given (user-confirmed, 2026-10-20). No log carries a
`=== PROVENANCE ===` block, so no log prints the command line itself. One deviation is known: the
first step-5 launch failed on `Get-Runs e3f 0..5` (PowerShell passed `0..5` as a string), before
anything was scored, and was relaunched as `Get-Runs e3f (0..5)`. ultralytics' resolved training arguments, quoted under
Configuration, agree with the runbook on every flag the runbook sets.

```powershell
$FLIR     = "dataset/processed/flir/data.yaml"
$MSRS     = "dataset/processed/msrs-day/data.yaml"
$MSRS_VIS = "runs/gate/msrs-day/manifests/visible/data.yaml"
$CUSTOM   = ((Select-String runs/reference-yolo11s/args.yaml -Pattern '^data:').Line -replace '^data:\s*','')

# 1. smoke, twice, same seed
uv run t2o train-detector --data $MSRS_VIS --init-weights rtdetr-l.pt --epochs 3 `
  --seed 1 --workers 8 --out "runs/smoke-rtdetr-msrs-day-$tag" --device cuda:0

# 2. one judge per dataset, 100 epochs, seed 1
uv run t2o train-detector --data $FLIR     --init-weights rtdetr-l.pt --epochs 100 --seed 1 --workers 8 --out runs/reference-flir-rtdetr-l     --device cuda:0
uv run t2o train-detector --data $MSRS_VIS --init-weights rtdetr-l.pt --epochs 100 --seed 1 --workers 8 --out runs/reference-msrs-day-rtdetr-l --device cuda:1
uv run t2o train-detector --data $CUSTOM   --init-weights rtdetr-l.pt --epochs 100 --seed 1 --workers 8 --out runs/reference-rtdetr-l          --device cuda:1

# 3. gate: yolo11s reference then RT-DETR judge, same command, per dataset
uv run t2o evaluate --weights "$w/weights/best.pt" --data $pair[0] --device 0

# 4. sanity: yolo11s on one export vs the zero-shot number the loop stored
uv run t2o evaluate --weights runs/reference-flir-yolo11s/weights/best.pt `
  --data runs/e3f-control-s0/stage3/translated/data.yaml --device 0

# 5. re-score, per run: two passes, both written to judges.rtdetr.*
uv run t2o evaluate --weights $Weights --data "$x/data.yaml" --write-back --judge rtdetr --device $Card
uv run t2o faithfulness --translated $x --data $Data --weights $Weights --write-back --judge rtdetr --device "cuda:$Card"
#   e3f (0..5) on card 0; e3md (0..5), e3b (0..5), e3t 0,1,3 on card 1

# 6. readouts
uv run python scripts/campaign_report.py --campaign flir --runs 'runs/e3f-*' --stage 3 `
  --primary-classes bicycle car person --data $FLIR --judge rtdetr
uv run python scripts/campaign_report.py --campaign msrs-day `
  --runs 'runs/e3md-control-s*' 'runs/e3md-loop-s*' --stage 3 `
  --primary-classes car_stop color_cone --data $MSRS --judge rtdetr
uv run t2o aggregate --runs 'runs/e3b-*' --stage 3 --metric $M --csv runs/e3b-judge-tidy.csv
uv run t2o aggregate --runs (Get-Runs e3t 0,1,3) --stage 3 --metric $M --csv runs/e3t-judge-tidy.csv
```

`$M` is the yolo11s and `judges.rtdetr` zero-shot and faithfulness metrics that the two aggregate
logs print. Both reports echo the `t2o aggregate` command they ran, which is quoted in Results.

## Environment

No `=== PROVENANCE ===` block. The machine is the project's usual server, described by the user
on 2026-10-20: VSIBMEOT05, Windows Server 2022, 2× A100-SXM4-40GB, 128 GB RAM. Record 024's machine
snapshot of the same host adds an AMD EPYC 7763 and driver 582.16. Nothing here was captured at run
time. The two readouts' own provenance sections describe the machine at readout. The FLIR report's,
verbatim:

```
CAMPAIGN REPORT -- runs/e3f-*

==============================================================================
== 1. PROVENANCE
==============================================================================
generated      2026-10-19T17:47:26Z
host           VSIBMEOT05  (Windows 2022Server)
git HEAD       8fe8feae889bf599d23f0a73df7227314a94fe21
git status     ?? probe-report.txt
?? splits/yolo_rgbt_29_jul.json
gpus           NVIDIA A100-SXM4-40GB
NVIDIA A100-SXM4-40GB
runs matched   12
data           dataset\processed\flir\data.yaml
  path:        D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir
  names:       ['bicycle', 'car', 'dog', 'person']   <- the per-class key spelling
  train/val:   train/visible/images | val/visible/images
  train labels: {   "labels": "de-rolled onto the thermal frame",   "source": "flir/labels (visible side)",   "calibration_digest": "210142740cdba163",   "calibration_git_sha": "739796f",   "direction": "visible_to_infrared = inv(homography)",   "box_transform": "corners",   "files": 4129,   "boxes_dropped": 1 }
  val labels: {   "labels": "de-rolled onto the thermal frame",   "source": "flir/labels (visible side)",   "calibration_digest": "210142740cdba163",   "calibration_git_sha": "739796f",   "direction": "visible_to_infrared = inv(homography)",   "box_transform": "corners",   "files": 1013,   "boxes_dropped": 0 }
```

The msrs-day report's is identical apart from `generated 2026-10-19T17:48:16Z`, the run globs, and
the `data` block. Every training and scoring log opens with
`Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130`.

## Configuration

- **Judge**: one RT-DETR-l per dataset, `--init-weights rtdetr-l.pt` (ultralytics' COCO-pretrained
  checkpoint), seed 1, 100 epochs, batch 16, `imgsz` 640, `deterministic=True`, trained on the same
  visible train split as that dataset's yolo11s reference (plan Q2, Q3). Judge name `rtdetr`.
  Out of family: the loop has only ever used `yolo11n` in-loop and `yolo11s` as reference.
- **ultralytics' resolved arguments** (from each training log's `engine\trainer:` line):
  - FLIR: `data=dataset\processed\flir\data.yaml`, `device=0`, `name=reference-flir-rtdetr-l`
  - msrs-day: `data=runs\gate\msrs-day\manifests\visible\data.yaml`, `device=1`, `name=reference-msrs-day-rtdetr-l`
  - custom: `data=D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml`, `device=1`, `name=reference-rtdetr-l`
  - all three: `model=rtdetr-l.pt`, `epochs=100`, `seed=1`, `deterministic=True`, `batch=16`,
    `imgsz=640`, `patience=100`, `workers=8`
- **Gate (plan Q5)**: a judge scores nothing unless its visible-val mAP50 is at least 90% of the
  yolo11s reference's, both scored by the same `t2o evaluate` command. References:
  `reference-flir-yolo11s`, `reference-msrs-day-yolo11s-s1` (V seed 1 of record 021),
  `reference-yolo11s` (record 004).
- **Sanity check**: `t2o evaluate` with the FLIR yolo11s reference on `e3f-control-s0`'s stage-3
  export gives mAP50 0.1872. The loop stored 0.187177 for that run and stage (the FLIR report's CSV
  row). The scoring path reproduces the stored number.
- **Re-scored**: every stage-3 export of the four E3 cells: `e3b` (record 007, pix2pix, custom set,
  seeds 0–5), `e3t` (record 010, turbo, custom set, seeds 0, 1, 3), `e3f` (record 016, FLIR, seeds
  0–5), `e3md` (record 024, msrs-day, seeds 0–5). That makes 42 exports and 84 write-backs, each
  logged as `recorded judge rtdetr into ... stage 3` (24 + 12 + 24 + 24). Stages 0–2 were not
  scored by the judge, so there is no stage-0 null under it.
- **Endpoints are unchanged from each cell's own record.** All-class mAP50 for the custom set,
  3-class (bicycle, car, person) for FLIR, and 2-class (car_stop, color_cone) for msrs-day.
- **The yolo11s keys were untouched.** Every yolo11s stage-3 headline in the readouts equals its
  record's: e3b +0.0512 (F35), e3t +0.0306 (F72), e3f +0.3512 (F136), e3md +0.1672 (F168); e3b
  false-object −0.0289 (F49).

## Results

ANSI colour escapes are stripped below. Everything else, including the mangled progress-bar
characters, is as the logs hold it.

**Smoke runs** (`smoke-a`, then `smoke-b`; same command, same seed, same card), from
`Starting training` to the end:

```
Starting training for 3 epochs...

      Epoch    GPU_mem  giou_loss   cls_loss    l1_loss  Instances       Size
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torch\autograd\graph.py:882: UserWarning: grid_sampler_2d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\aten\src\ATen\Context.cpp:164.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
        1/3      12.2G      1.034      2.037     0.3549         59        640: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 34/34 1.2it/s 27.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 6/6 1.8it/s 3.3s
                   all        179        976      0.893      0.237      0.391      0.254

      Epoch    GPU_mem  giou_loss   cls_loss    l1_loss  Instances       Size
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torch\autograd\graph.py:882: UserWarning: grid_sampler_2d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\aten\src\ATen\Context.cpp:164.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
        2/3        13G      0.724     0.5655     0.2206         40        640: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 34/34 2.2it/s 15.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 6/6 4.3it/s 1.4s
                   all        179        976      0.712      0.621      0.642      0.382

      Epoch    GPU_mem  giou_loss   cls_loss    l1_loss  Instances       Size
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torch\autograd\graph.py:882: UserWarning: grid_sampler_2d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\aten\src\ATen\Context.cpp:164.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
        3/3      13.1G     0.6633     0.4857     0.1774         75        640: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 34/34 2.2it/s 15.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 6/6 4.8it/s 1.2s
                   all        179        976      0.663      0.654      0.678      0.412

3 epochs completed in 0.027 hours.
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\smoke-rtdetr-msrs-day-a\weights\last.pt, 66.2MB
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\smoke-rtdetr-msrs-day-a\weights\best.pt, 66.2MB

Validating D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\smoke-rtdetr-msrs-day-a\weights\best.pt...
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
rt-detr-l summary: 315 layers, 31,994,015 parameters, 0 gradients, 105.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 6/6 5.9it/s 1.0s
                   all        179        976       0.67      0.654       0.68       0.41
Speed: 0.1ms preprocess, 2.8ms inference, 0.0ms loss, 0.3ms postprocess per image
2026-10-19 10:57:17,714 INFO    t2o.engine.detector_stage: detector smoke-rtdetr-msrs-day-a: P 0.6700  R 0.6544  mAP50 0.6798  mAP50-95 0.4103
2026-10-19 10:57:17,714 INFO    t2o.cli: detector weights: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\smoke-rtdetr-msrs-day-a\weights\best.pt
2026-10-19 10:57:17,718 INFO    t2o.cli:   mAP50 0.6798  mAP50-95 0.4103
```

```
Starting training for 3 epochs...

      Epoch    GPU_mem  giou_loss   cls_loss    l1_loss  Instances       Size
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torch\autograd\graph.py:882: UserWarning: grid_sampler_2d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\aten\src\ATen\Context.cpp:164.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
        1/3      12.2G      1.008      2.042     0.3465         59        640: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 34/34 1.6it/s 21.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 6/6 1.9it/s 3.2s
                   all        179        976      0.867      0.259      0.346      0.216

      Epoch    GPU_mem  giou_loss   cls_loss    l1_loss  Instances       Size
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torch\autograd\graph.py:882: UserWarning: grid_sampler_2d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\aten\src\ATen\Context.cpp:164.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
        2/3        13G     0.7764     0.5593     0.2444         40        640: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 34/34 2.2it/s 15.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 6/6 5.2it/s 1.2s
                   all        179        976      0.655      0.478      0.512      0.318

      Epoch    GPU_mem  giou_loss   cls_loss    l1_loss  Instances       Size
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torch\autograd\graph.py:882: UserWarning: grid_sampler_2d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\aten\src\ATen\Context.cpp:164.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
        3/3      13.1G     0.6943     0.5069     0.1897         75        640: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 34/34 2.3it/s 15.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 6/6 4.7it/s 1.3s
                   all        179        976      0.707      0.599      0.646      0.394

3 epochs completed in 0.020 hours.
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\smoke-rtdetr-msrs-day-b\weights\last.pt, 66.2MB
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\smoke-rtdetr-msrs-day-b\weights\best.pt, 66.2MB

Validating D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\smoke-rtdetr-msrs-day-b\weights\best.pt...
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
rt-detr-l summary: 315 layers, 31,994,015 parameters, 0 gradients, 105.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 6/6 6.5it/s 0.9s
                   all        179        976      0.724      0.585      0.647      0.393
Speed: 0.1ms preprocess, 2.7ms inference, 0.0ms loss, 0.2ms postprocess per image
2026-10-19 11:00:27,927 INFO    t2o.engine.detector_stage: detector smoke-rtdetr-msrs-day-b: P 0.7240  R 0.5847  mAP50 0.6468  mAP50-95 0.3926
2026-10-19 11:00:27,927 INFO    t2o.cli: detector weights: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\smoke-rtdetr-msrs-day-b\weights\best.pt
2026-10-19 11:00:27,927 INFO    t2o.cli:   mAP50 0.6468  mAP50-95 0.3926
```

**Judge training, end of each log** (FLIR, msrs-day, custom):

```
100 epochs completed in 4.566 hours.
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-flir-rtdetr-l\weights\last.pt, 66.2MB
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-flir-rtdetr-l\weights\best.pt, 66.2MB

Validating D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-flir-rtdetr-l\weights\best.pt...
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
rt-detr-l summary: 315 layers, 31,991,960 parameters, 0 gradients, 105.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 32/32 5.3it/s 6.1s
                   all       1013       8601      0.748      0.533      0.601      0.278
Speed: 0.1ms preprocess, 2.8ms inference, 0.0ms loss, 0.3ms postprocess per image
2026-10-19 15:45:42,380 INFO    t2o.engine.detector_stage: detector reference-flir-rtdetr-l: P 0.7477  R 0.5334  mAP50 0.6010  mAP50-95 0.2782
2026-10-19 15:45:42,386 INFO    t2o.cli: detector weights: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-flir-rtdetr-l\weights\best.pt
2026-10-19 15:45:42,386 INFO    t2o.cli:   mAP50 0.6010  mAP50-95 0.2782
```

```
100 epochs completed in 0.600 hours.
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-msrs-day-rtdetr-l\weights\last.pt, 66.2MB
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-msrs-day-rtdetr-l\weights\best.pt, 66.2MB

Validating D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-msrs-day-rtdetr-l\weights\best.pt...
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
rt-detr-l summary: 315 layers, 31,994,015 parameters, 0 gradients, 105.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 6.5it/s 0.9s
                   all        179        976      0.852      0.734      0.795      0.514
Speed: 0.1ms preprocess, 2.8ms inference, 0.0ms loss, 0.3ms postprocess per image
2026-10-19 11:48:05,818 INFO    t2o.engine.detector_stage: detector reference-msrs-day-rtdetr-l: P 0.8520  R 0.7339  mAP50 0.7945  mAP50-95 0.5138
2026-10-19 11:48:05,818 INFO    t2o.cli: detector weights: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-msrs-day-rtdetr-l\weights\best.pt
2026-10-19 11:48:05,828 INFO    t2o.cli:   mAP50 0.7945  mAP50-95 0.5138
```

```
100 epochs completed in 0.740 hours.
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-rtdetr-l\weights\last.pt, 66.2MB
Optimizer stripped from D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-rtdetr-l\weights\best.pt, 66.2MB

Validating D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-rtdetr-l\weights\best.pt...
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
rt-detr-l summary: 315 layers, 31,994,015 parameters, 0 gradients, 105.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 4.3it/s 1.2s
                   all        153        423      0.931      0.904      0.937      0.709
Speed: 0.1ms preprocess, 3.3ms inference, 0.0ms loss, 0.4ms postprocess per image
2026-10-19 13:06:18,215 INFO    t2o.engine.detector_stage: detector reference-rtdetr-l: P 0.9308  R 0.9044  mAP50 0.9366  mAP50-95 0.7087
2026-10-19 13:06:18,220 INFO    t2o.cli: detector weights: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\reference-rtdetr-l\weights\best.pt
2026-10-19 13:06:18,221 INFO    t2o.cli:   mAP50 0.9366  mAP50-95 0.7087
```

**FLIR judge training, epochs 45–48 and the first NaN recovery** (epoch 1's validation is
0.758 / 0.532 / 0.601 / 0.278, the log's first `all` row):

```
     45/100      12.9G     0.5728     0.4536    0.08244         20        640: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 259/259 2.2it/s 1:58
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 32/32 4.0it/s 7.9s
                   all       1013       8601      0.561      0.486      0.501      0.243

      Epoch    GPU_mem  giou_loss   cls_loss    l1_loss  Instances       Size
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torch\autograd\graph.py:882: UserWarning: grid_sampler_2d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\aten\src\ATen\Context.cpp:164.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
     46/100      12.9G     0.5712     0.4527    0.08187          4        640: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 259/259 2.3it/s 1:55
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 32/32 4.1it/s 7.9s
                   all       1013       8601      0.543      0.525      0.504      0.244

      Epoch    GPU_mem  giou_loss   cls_loss    l1_loss  Instances       Size
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torch\autograd\graph.py:882: UserWarning: grid_sampler_2d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\aten\src\ATen\Context.cpp:164.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
     47/100      12.9G        nan        nan        nan         35        640: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 259/259 2.1it/s 2:04
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 32/32 5.4it/s 5.9s
                   all       1013       8601          0          0          0          0

      Epoch    GPU_mem  giou_loss   cls_loss    l1_loss  Instances       Size
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torch\autograd\graph.py:882: UserWarning: grid_sampler_2d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\aten\src\ATen\Context.cpp:164.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
     48/100      12.9G      0.561     0.4507    0.08002          5        640: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 259/259 2.1it/s 2:04
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 32/32 5.7it/s 5.6s
...
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
     76/100      12.9G        nan        nan        nan         15        640: 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 259/259 2.1it/s 2:01
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 32/32 5.6it/s 5.7s
                   all       1013       8601          0          0          0          0
WARNING Loss NaN/Inf detected (attempt 1/3), recovering from last.pt...
```

**Gate** (yolo11s then RT-DETR, per dataset: FLIR, msrs-day, custom), the whole log:

```
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,348 parameters, 0 gradients, 21.4 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 506.3179.6 MB/s, size: 112.0 KB)

val: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\val\visible\images\FLIR_09042.jpg: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\val\visible\images\FLIR_09055.jpg: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\val\visible\images\FLIR_09653.jpg: 1 duplicate labels removed
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 64/64 10.8it/s 5.9s
                   all       1013       8601      0.779      0.511      0.552      0.277
Speed: 1.1ms preprocess, 1.5ms inference, 0.0ms loss, 1.1ms postprocess per image
2026-10-19 16:12:16,841 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7788  R 0.5107  mAP50 0.5523  mAP50-95 0.2767
2026-10-19 16:12:16,841 INFO    t2o.cli:   bicycle              AP50=0.4922
2026-10-19 16:12:16,841 INFO    t2o.cli:   car                  AP50=0.8176
2026-10-19 16:12:16,841 INFO    t2o.cli:   dog                  AP50=0.2396
2026-10-19 16:12:16,843 INFO    t2o.cli:   person               AP50=0.6599
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
rt-detr-l summary: 315 layers, 31,991,960 parameters, 0 gradients, 105.3 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 686.4179.4 MB/s, size: 112.4 KB)

val: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\val\visible\images\FLIR_09042.jpg: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\val\visible\images\FLIR_09055.jpg: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\val\visible\images\FLIR_09653.jpg: 1 duplicate labels removed
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 64/64 7.1it/s 9.1s
                   all       1013       8601      0.754      0.532      0.601      0.278
Speed: 1.7ms preprocess, 4.8ms inference, 0.0ms loss, 0.2ms postprocess per image
2026-10-19 16:13:35,781 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7536  R 0.5322  mAP50 0.6009  mAP50-95 0.2784
2026-10-19 16:13:35,791 INFO    t2o.cli:   bicycle              AP50=0.4844
2026-10-19 16:13:35,791 INFO    t2o.cli:   car                  AP50=0.7931
2026-10-19 16:13:35,791 INFO    t2o.cli:   dog                  AP50=0.4522
2026-10-19 16:13:35,791 INFO    t2o.cli:   person               AP50=0.6739
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 1267.3332.3 MB/s, size: 382.4 KB)

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 12/12 8.4it/s 1.4s
                   all        179        976       0.87       0.71      0.784       0.52
Speed: 1.2ms preprocess, 2.5ms inference, 0.0ms loss, 1.2ms postprocess per image
2026-10-19 16:14:35,715 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8698  R 0.7104  mAP50 0.7841  mAP50-95 0.5199
2026-10-19 16:14:35,715 INFO    t2o.cli:   bike                 AP50=0.6272
2026-10-19 16:14:35,715 INFO    t2o.cli:   car                  AP50=0.9399
2026-10-19 16:14:35,715 INFO    t2o.cli:   car_stop             AP50=0.9035
2026-10-19 16:14:35,715 INFO    t2o.cli:   color_cone           AP50=0.7439
2026-10-19 16:14:35,715 INFO    t2o.cli:   person               AP50=0.7057
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
rt-detr-l summary: 315 layers, 31,994,015 parameters, 0 gradients, 105.4 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 1238.4462.1 MB/s, size: 381.1 KB)

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 12/12 7.1it/s 1.7s
                   all        179        976       0.86      0.731      0.798      0.516
Speed: 1.5ms preprocess, 5.4ms inference, 0.0ms loss, 0.2ms postprocess per image
2026-10-19 16:15:46,790 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8605  R 0.7306  mAP50 0.7979  mAP50-95 0.5161
2026-10-19 16:15:46,790 INFO    t2o.cli:   bike                 AP50=0.6532
2026-10-19 16:15:46,790 INFO    t2o.cli:   car                  AP50=0.9433
2026-10-19 16:15:46,790 INFO    t2o.cli:   car_stop             AP50=0.9143
2026-10-19 16:15:46,790 INFO    t2o.cli:   color_cone           AP50=0.7404
2026-10-19 16:15:46,790 INFO    t2o.cli:   person               AP50=0.7382
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,735 parameters, 0 gradients, 21.4 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 956.7153.3 MB/s, size: 329.3 KB)

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 10/10 1.4it/s 7.1s
                   all        153        423      0.893      0.903      0.937      0.683
Speed: 1.3ms preprocess, 2.9ms inference, 0.0ms loss, 1.2ms postprocess per image
2026-10-19 16:16:16,688 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8927  R 0.9030  mAP50 0.9366  mAP50-95 0.6831
2026-10-19 16:16:16,692 INFO    t2o.cli:   Fuse                 AP50=0.9746
2026-10-19 16:16:16,692 INFO    t2o.cli:   Pole                 AP50=0.9719
2026-10-19 16:16:16,692 INFO    t2o.cli:   Switch               AP50=0.8507
2026-10-19 16:16:16,692 INFO    t2o.cli:   Transformer          AP50=0.9492
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
rt-detr-l summary: 315 layers, 31,994,015 parameters, 0 gradients, 105.4 GFLOPs
val: Fast image access  (ping: 0.10.0 ms, read: 882.3379.9 MB/s, size: 367.8 KB)

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 10/10 5.4it/s 1.8s
                   all        153        423      0.924      0.912      0.937      0.707
Speed: 1.8ms preprocess, 6.4ms inference, 0.0ms loss, 0.3ms postprocess per image
2026-10-19 16:17:27,094 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9241  R 0.9117  mAP50 0.9368  mAP50-95 0.7071
2026-10-19 16:17:27,096 INFO    t2o.cli:   Fuse                 AP50=0.9532
2026-10-19 16:17:27,096 INFO    t2o.cli:   Pole                 AP50=0.9535
2026-10-19 16:17:27,096 INFO    t2o.cli:   Switch               AP50=0.8920
2026-10-19 16:17:27,096 INFO    t2o.cli:   Transformer          AP50=0.9485
```

**Sanity check**, the whole log:

```
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11s summary (fused): 101 layers, 9,414,348 parameters, 0 gradients, 21.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 41.25.4 MB/s, size: 641.8 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/

val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-control-s0\stage3\translated\val\images\FLIR_09042.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-control-s0\stage3\translated\val\images\FLIR_09055.png: 1 duplicate labels removed
val: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3f-control-s0\stage3\translated\val\images\FLIR_09653.png: 1 duplicate labels removed
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöüΓöü 64/64 7.7it/s 8.3s
                   all       1013       8601      0.644      0.182      0.187      0.087
Speed: 1.6ms preprocess, 1.9ms inference, 0.0ms loss, 1.6ms postprocess per image
2026-10-19 16:21:11,130 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.6435  R 0.1822  mAP50 0.1872  mAP50-95 0.0870
2026-10-19 16:21:11,130 INFO    t2o.cli:   bicycle              AP50=0.1408
2026-10-19 16:21:11,130 INFO    t2o.cli:   car                  AP50=0.4925
2026-10-19 16:21:11,130 INFO    t2o.cli:   dog                  AP50=0.0003
2026-10-19 16:21:11,130 INFO    t2o.cli:   person               AP50=0.1151
```

**FLIR readout.** These are the per-run CSV header and the stage-3 rows. Stages 0–2 carry no judge
columns, and their yolo11s values are record 016's.

```
run,seed,arm,stage,task_weight,zero_shot.primary_map50,zero_shot.primary_n_classes,zero_shot.map50,zero_shot.map50_95,zero_shot.precision,zero_shot.recall,zero_shot.per_class_ap50.bicycle,zero_shot.per_class_ap50.car,zero_shot.per_class_ap50.dog,zero_shot.per_class_ap50.person,judges.rtdetr.zero_shot.primary_map50,judges.rtdetr.zero_shot.primary_n_classes,judges.rtdetr.zero_shot.map50,judges.rtdetr.zero_shot.map50_95,judges.rtdetr.zero_shot.precision,judges.rtdetr.zero_shot.recall,judges.rtdetr.zero_shot.per_class_ap50.bicycle,judges.rtdetr.zero_shot.per_class_ap50.car,judges.rtdetr.zero_shot.per_class_ap50.dog,judges.rtdetr.zero_shot.per_class_ap50.person,fidelity.lpips,fidelity.fid,fidelity.psnr,fidelity.ssim,detector.map50,faithfulness.false_object_rate,faithfulness.missed_object_rate,faithfulness.detection_consistency,judges.rtdetr.faithfulness.false_object_rate,judges.rtdetr.faithfulness.missed_object_rate,judges.rtdetr.faithfulness.detection_consistency
e3f-control-s0,0,control,3,0,0.249461,3.000000,0.187177,0.086968,0.643507,0.182201,0.140785,0.492508,0.000325,0.115091,0.386572,3.000000,0.299941,0.146995,0.521963,0.286376,0.160602,0.663824,0.040047,0.335289,0.397022,87.355240,13.699208,0.470360,0.349719,0.253544,0.785797,0.202280,0.741070,0.452348,0.378853
e3f-control-s1,1,control,3,0,0.122640,3.000000,0.091980,0.043649,0.205545,0.093945,0.002745,0.336923,0.000000,0.028253,0.279201,3.000000,0.209842,0.100353,0.391141,0.191616,0.111602,0.522239,0.001766,0.203763,0.475256,115.264114,10.122923,0.364546,0.330791,0.242003,0.873315,0.118015,0.749946,0.595537,0.255383
e3f-control-s2,2,control,3,0,0.049868,3.000000,0.037401,0.016057,0.365409,0.049665,0.007377,0.132133,0.000000,0.010095,0.161710,3.000000,0.122563,0.056979,0.260225,0.141908,0.077474,0.307976,0.005119,0.099681,0.583237,182.420746,10.159594,0.335993,0.307697,0.440890,0.953278,0.044814,0.782124,0.731985,0.164968
e3f-control-s3,3,control,3,0,0.000239,3.000000,0.000179,0.000079,0.001579,0.004059,0.000021,0.000681,0.000000,0.000014,0.077014,3.000000,0.059348,0.024591,0.168477,0.090002,0.055125,0.150531,0.006349,0.025387,0.525262,221.859177,9.487186,0.088155,0.317410,0.963855,0.999651,0.000224,0.774550,0.883659,0.065143
e3f-control-s4,4,control,3,0,0.115542,3.000000,0.086656,0.038264,0.222577,0.149290,0.069102,0.237791,0.000000,0.039733,0.322358,3.000000,0.244989,0.113485,0.420101,0.220285,0.128801,0.535984,0.012882,0.302290,0.442393,92.594994,11.873393,0.384799,0.379026,0.534166,0.926313,0.068954,0.780860,0.527080,0.307558
e3f-control-s5,5,control,3,0,0.003714,3.000000,0.002786,0.000875,0.002031,0.014337,0.005009,0.005300,0.000000,0.000833,0.121102,3.000000,0.109852,0.048528,0.334205,0.142742,0.057785,0.247862,0.076104,0.057658,0.566243,328.395569,9.371244,0.119982,0.305838,0.684211,0.999303,0.000671,0.766260,0.822060,0.102057
e3f-loop-s0,0,loop,3,3,0.477354,3.000000,0.374621,0.178537,0.530815,0.341788,0.237706,0.729614,0.066422,0.464743,0.551846,3.000000,0.418397,0.207077,0.497988,0.358372,0.289965,0.767772,0.018048,0.597802,0.410464,97.513641,12.787950,0.480989,0.428871,0.172449,0.523129,0.416853,0.594675,0.315318,0.422462
e3f-loop-s1,1,loop,3,3,0.466887,3.000000,0.355673,0.168890,0.482534,0.314446,0.248934,0.741155,0.022029,0.410573,0.534786,3.000000,0.402871,0.200199,0.479864,0.353620,0.285722,0.757953,0.007127,0.560684,0.414339,88.829468,12.934482,0.497222,0.425539,0.197782,0.537541,0.414841,0.605606,0.334379,0.411123
e3f-loop-s2,2,loop,3,3,0.427959,3.000000,0.323541,0.156911,0.479465,0.287656,0.209135,0.677370,0.010287,0.397371,0.530506,3.000000,0.411138,0.209768,0.495920,0.385339,0.277489,0.741754,0.053034,0.572276,0.422632,103.400406,12.762724,0.431819,0.437187,0.175889,0.593212,0.350246,0.640109,0.326941,0.418421
e3f-loop-s3,3,loop,3,3,0.425070,3.000000,0.319018,0.152027,0.471488,0.285215,0.209217,0.669841,0.000861,0.396153,0.533379,3.000000,0.429514,0.214821,0.697248,0.360906,0.287372,0.741268,0.117920,0.571495,0.434361,96.199951,11.887878,0.449674,0.437817,0.220660,0.594026,0.358963,0.623408,0.339958,0.405513
e3f-loop-s4,4,loop,3,3,0.414330,3.000000,0.314972,0.149157,0.453704,0.292621,0.184340,0.644352,0.016898,0.414299,0.524785,3.000000,0.405489,0.204850,0.594128,0.360578,0.236667,0.752408,0.047602,0.585280,0.434308,99.427429,12.212207,0.449218,0.437414,0.163588,0.600070,0.345887,0.597296,0.324849,0.411484
e3f-loop-s5,5,loop,3,3,0.437194,3.000000,0.329947,0.159944,0.445577,0.314031,0.180157,0.711024,0.008206,0.420401,0.503497,3.000000,0.377791,0.189437,0.466377,0.335430,0.245903,0.728266,0.000671,0.536322,0.415093,111.382545,14.036967,0.436055,0.415047,0.193813,0.554742,0.388690,0.641563,0.356113,0.403161
```

Section 6 in full:

```
==============================================================================
== 6. PAIRED STATISTICS (t2o aggregate, verbatim)
==============================================================================
$ t2o aggregate --runs runs/e3f-* --metric zero_shot.primary_map50 zero_shot.primary_n_classes zero_shot.map50 zero_shot.map50_95 zero_shot.precision zero_shot.recall zero_shot.per_class_ap50.bicycle zero_shot.per_class_ap50.car zero_shot.per_class_ap50.dog zero_shot.per_class_ap50.person judges.rtdetr.zero_shot.primary_map50 judges.rtdetr.zero_shot.primary_n_classes judges.rtdetr.zero_shot.map50 judges.rtdetr.zero_shot.map50_95 judges.rtdetr.zero_shot.precision judges.rtdetr.zero_shot.recall judges.rtdetr.zero_shot.per_class_ap50.bicycle judges.rtdetr.zero_shot.per_class_ap50.car judges.rtdetr.zero_shot.per_class_ap50.dog judges.rtdetr.zero_shot.per_class_ap50.person fidelity.lpips fidelity.fid fidelity.psnr fidelity.ssim detector.map50 faithfulness.false_object_rate faithfulness.missed_object_rate faithfulness.detection_consistency judges.rtdetr.faithfulness.false_object_rate judges.rtdetr.faithfulness.missed_object_rate judges.rtdetr.faithfulness.detection_consistency --primary-classes bicycle car person --stage 3

12 runs, stages [0, 1, 2, 3]
metric                                           arm      stage   n      mean       std
zero_shot.primary_map50                          control      0   6    0.0722    0.0535
zero_shot.primary_map50                          loop         0   6    0.0987    0.0158
zero_shot.primary_map50                          control      1   6    0.0710    0.0572
zero_shot.primary_map50                          loop         1   6    0.3736    0.0486
zero_shot.primary_map50                          control      2   6    0.0940    0.0472
zero_shot.primary_map50                          loop         2   6    0.4220    0.0224
zero_shot.primary_map50                          control      3   6    0.0902    0.0941
zero_shot.primary_map50                          loop         3   6    0.4415    0.0251
zero_shot.primary_n_classes                      control      0   6    3.0000    0.0000
zero_shot.primary_n_classes                      loop         0   6    3.0000    0.0000
zero_shot.primary_n_classes                      control      1   6    3.0000    0.0000
zero_shot.primary_n_classes                      loop         1   6    3.0000    0.0000
zero_shot.primary_n_classes                      control      2   6    3.0000    0.0000
zero_shot.primary_n_classes                      loop         2   6    3.0000    0.0000
zero_shot.primary_n_classes                      control      3   6    3.0000    0.0000
zero_shot.primary_n_classes                      loop         3   6    3.0000    0.0000
zero_shot.map50                                  control      0   6    0.0541    0.0401
zero_shot.map50                                  loop         0   6    0.0741    0.0119
zero_shot.map50                                  control      1   6    0.0533    0.0429
zero_shot.map50                                  loop         1   6    0.2875    0.0403
zero_shot.map50                                  control      2   6    0.0706    0.0355
zero_shot.map50                                  loop         2   6    0.3212    0.0198
zero_shot.map50                                  control      3   6    0.0677    0.0706
zero_shot.map50                                  loop         3   6    0.3363    0.0237
zero_shot.map50_95                               control      0   6    0.0238    0.0183
zero_shot.map50_95                               loop         0   6    0.0341    0.0059
zero_shot.map50_95                               control      1   6    0.0248    0.0208
zero_shot.map50_95                               loop         1   6    0.1366    0.0201
zero_shot.map50_95                               control      2   6    0.0333    0.0169
zero_shot.map50_95                               loop         2   6    0.1551    0.0083
zero_shot.map50_95                               control      3   6    0.0310    0.0330
zero_shot.map50_95                               loop         3   6    0.1609    0.0110
zero_shot.precision                              control      0   6    0.3164    0.1878
zero_shot.precision                              loop         0   6    0.4234    0.0237
zero_shot.precision                              control      1   6    0.2618    0.1812
zero_shot.precision                              loop         1   6    0.5314    0.1361
zero_shot.precision                              control      2   6    0.1807    0.0892
zero_shot.precision                              loop         2   6    0.5197    0.1108
zero_shot.precision                              control      3   6    0.2401    0.2423
zero_shot.precision                              loop         3   6    0.4773    0.0300
zero_shot.recall                                 control      0   6    0.0635    0.0426
zero_shot.recall                                 loop         0   6    0.0858    0.0099
zero_shot.recall                                 control      1   6    0.0717    0.0475
zero_shot.recall                                 loop         1   6    0.2751    0.0371
zero_shot.recall                                 control      2   6    0.0898    0.0492
zero_shot.recall                                 loop         2   6    0.2994    0.0256
zero_shot.recall                                 control      3   6    0.0822    0.0727
zero_shot.recall                                 loop         3   6    0.3060    0.0217
zero_shot.per_class_ap50.bicycle                 control      0   6    0.0093    0.0088
zero_shot.per_class_ap50.bicycle                 loop         0   6    0.0125    0.0049
zero_shot.per_class_ap50.bicycle                 control      1   6    0.0213    0.0243
zero_shot.per_class_ap50.bicycle                 loop         1   6    0.1503    0.0387
zero_shot.per_class_ap50.bicycle                 control      2   6    0.0259    0.0253
zero_shot.per_class_ap50.bicycle                 loop         2   6    0.2003    0.0260
zero_shot.per_class_ap50.bicycle                 control      3   6    0.0375    0.0570
zero_shot.per_class_ap50.bicycle                 loop         3   6    0.2116    0.0276
zero_shot.per_class_ap50.car                     control      0   6    0.1829    0.1330
zero_shot.per_class_ap50.car                     loop         0   6    0.2542    0.0428
zero_shot.per_class_ap50.car                     control      1   6    0.1649    0.1353
zero_shot.per_class_ap50.car                     loop         1   6    0.6124    0.0770
zero_shot.per_class_ap50.car                     control      2   6    0.2265    0.1169
zero_shot.per_class_ap50.car                     loop         2   6    0.6685    0.0351
zero_shot.per_class_ap50.car                     control      3   6    0.2009    0.1939
zero_shot.per_class_ap50.car                     loop         3   6    0.6956    0.0377
zero_shot.per_class_ap50.dog                     control      0   6    0.0001    0.0001
zero_shot.per_class_ap50.dog                     loop         0   6    0.0000    0.0000
zero_shot.per_class_ap50.dog                     control      1   6    0.0000    0.0001
zero_shot.per_class_ap50.dog                     loop         1   6    0.0291    0.0303
zero_shot.per_class_ap50.dog                     control      2   6    0.0004    0.0009
zero_shot.per_class_ap50.dog                     loop         2   6    0.0188    0.0207
zero_shot.per_class_ap50.dog                     control      3   6    0.0001    0.0001
zero_shot.per_class_ap50.dog                     loop         3   6    0.0208    0.0235
zero_shot.per_class_ap50.person                  control      0   6    0.0243    0.0208
zero_shot.per_class_ap50.person                  loop         0   6    0.0296    0.0131
zero_shot.per_class_ap50.person                  control      1   6    0.0269    0.0236
zero_shot.per_class_ap50.person                  loop         1   6    0.3581    0.0555
zero_shot.per_class_ap50.person                  control      2   6    0.0294    0.0150
zero_shot.per_class_ap50.person                  loop         2   6    0.3973    0.0284
zero_shot.per_class_ap50.person                  control      3   6    0.0323    0.0435
zero_shot.per_class_ap50.person                  loop         3   6    0.4173    0.0251
judges.rtdetr.zero_shot.primary_map50            control      3   6    0.2247    0.1227
judges.rtdetr.zero_shot.primary_map50            loop         3   6    0.5298    0.0158
judges.rtdetr.zero_shot.primary_n_classes        control      3   6    3.0000    0.0000
judges.rtdetr.zero_shot.primary_n_classes        loop         3   6    3.0000    0.0000
judges.rtdetr.zero_shot.map50                    control      3   6    0.1744    0.0917
judges.rtdetr.zero_shot.map50                    loop         3   6    0.4075    0.0175
judges.rtdetr.zero_shot.map50_95                 control      3   6    0.0818    0.0460
judges.rtdetr.zero_shot.map50_95                 loop         3   6    0.2044    0.0088
judges.rtdetr.zero_shot.precision                control      3   6    0.3494    0.1244
judges.rtdetr.zero_shot.precision                loop         3   6    0.5386    0.0899
judges.rtdetr.zero_shot.recall                   control      3   6    0.1788    0.0693
judges.rtdetr.zero_shot.recall                   loop         3   6    0.3590    0.0160
judges.rtdetr.zero_shot.per_class_ap50.bicycle   control      3   6    0.0986    0.0423
judges.rtdetr.zero_shot.per_class_ap50.bicycle   loop         3   6    0.2705    0.0232
judges.rtdetr.zero_shot.per_class_ap50.car       control      3   6    0.4047    0.1984
judges.rtdetr.zero_shot.per_class_ap50.car       loop         3   6    0.7482    0.0140
judges.rtdetr.zero_shot.per_class_ap50.dog       control      3   6    0.0237    0.0292
judges.rtdetr.zero_shot.per_class_ap50.dog       loop         3   6    0.0407    0.0434
judges.rtdetr.zero_shot.per_class_ap50.person    control      3   6    0.1707    0.1300
judges.rtdetr.zero_shot.per_class_ap50.person    loop         3   6    0.5706    0.0211
fidelity.lpips                                   control      0   6    0.6119    0.2425
fidelity.lpips                                   loop         0   6    0.4327    0.0296
fidelity.lpips                                   control      1   6    0.5057    0.0977
fidelity.lpips                                   loop         1   6    0.4574    0.0534
fidelity.lpips                                   control      2   6    0.4740    0.0649
fidelity.lpips                                   loop         2   6    0.4084    0.0102
fidelity.lpips                                   control      3   6    0.4982    0.0728
fidelity.lpips                                   loop         3   6    0.4219    0.0104
fidelity.fid                                     control      0   6  229.2442  126.9289
fidelity.fid                                     loop         0   6  132.8005   40.6259
fidelity.fid                                     control      1   6  193.2944   93.0553
fidelity.fid                                     loop         1   6  125.9171   16.2435
fidelity.fid                                     control      2   6  175.6831  119.4196
fidelity.fid                                     loop         2   6  100.5841    7.7541
fidelity.fid                                     control      3   6  171.3150   93.5309
fidelity.fid                                     loop         3   6   99.4589    7.5512
fidelity.psnr                                    control      0   6    9.4054    4.1475
fidelity.psnr                                    loop         0   6   12.8276    0.5678
fidelity.psnr                                    control      1   6   10.7240    2.5643
fidelity.psnr                                    loop         1   6   12.4334    0.9822
fidelity.psnr                                    control      2   6   11.2256    2.1010
fidelity.psnr                                    loop         2   6   13.4364    0.4609
fidelity.psnr                                    control      3   6   10.7856    1.6846
fidelity.psnr                                    loop         3   6   12.7704    0.7378
fidelity.ssim                                    control      0   6    0.2400    0.1457
fidelity.ssim                                    loop         0   6    0.3738    0.0378
fidelity.ssim                                    control      1   6    0.2997    0.1035
fidelity.ssim                                    loop         1   6    0.3678    0.0428
fidelity.ssim                                    control      2   6    0.3149    0.1550
fidelity.ssim                                    loop         2   6    0.4452    0.0277
fidelity.ssim                                    control      3   6    0.2940    0.1541
fidelity.ssim                                    loop         3   6    0.4575    0.0260
detector.map50                                   control      0   6    0.2635    0.1587
detector.map50                                   loop         0   6    0.3690    0.0204
detector.map50                                   control      1   6    0.3389    0.0434
detector.map50                                   loop         1   6    0.4377    0.0140
detector.map50                                   control      2   6    0.3437    0.0208
detector.map50                                   loop         2   6    0.4449    0.0094
detector.map50                                   control      3   6    0.3317    0.0283
detector.map50                                   loop         3   6    0.4303    0.0091
faithfulness.false_object_rate                   control      3   6    0.5198    0.2752
faithfulness.false_object_rate                   loop         3   6    0.1874    0.0209
faithfulness.missed_object_rate                  control      3   6    0.9229    0.0824
faithfulness.missed_object_rate                  loop         3   6    0.5671    0.0330
faithfulness.detection_consistency               control      3   6    0.0725    0.0776
faithfulness.detection_consistency               loop         3   6    0.3792    0.0320
judges.rtdetr.faithfulness.false_object_rate     control      3   6    0.7658    0.0169
judges.rtdetr.faithfulness.false_object_rate     loop         3   6    0.6171    0.0210
judges.rtdetr.faithfulness.missed_object_rate    control      3   6    0.6688    0.1708
judges.rtdetr.faithfulness.missed_object_rate    loop         3   6    0.3329    0.0141
judges.rtdetr.faithfulness.detection_consistency control      3   6    0.2123    0.1222
judges.rtdetr.faithfulness.detection_consistency loop         3   6    0.4120    0.0074
paired (loop - control), zero_shot.primary_map50
  stage 0 (null) n=6  +0.0266  p=0.2812  95% CI [-0.0052, +0.0613]
  stage 1        n=6  +0.3026  p=0.0312  95% CI [+0.2689, +0.3371]
  stage 2        n=6  +0.3281  p=0.0312  95% CI [+0.3048, +0.3567]
  stage 3        n=6  +0.3512  p=0.0312  95% CI [+0.2919, +0.4053] <-- headline
paired (loop - control), zero_shot.primary_n_classes
  stage 0 (null) n=6  +0.0000  p=1.0000  95% CI [+0.0000, +0.0000]
  stage 1        n=6  +0.0000  p=1.0000  95% CI [+0.0000, +0.0000]
  stage 2        n=6  +0.0000  p=1.0000  95% CI [+0.0000, +0.0000]
  stage 3        n=6  +0.0000  p=1.0000  95% CI [+0.0000, +0.0000] <-- headline
paired (loop - control), zero_shot.map50
  stage 0 (null) n=6  +0.0199  p=0.2812  95% CI [-0.0039, +0.0460]
  stage 1        n=6  +0.2342  p=0.0312  95% CI [+0.2087, +0.2622]
  stage 2        n=6  +0.2506  p=0.0312  95% CI [+0.2314, +0.2719]
  stage 3        n=6  +0.2686  p=0.0312  95% CI [+0.2289, +0.3056] <-- headline
paired (loop - control), zero_shot.map50_95
  stage 0 (null) n=6  +0.0103  p=0.2500  95% CI [-0.0007, +0.0223]
  stage 1        n=6  +0.1118  p=0.0312  95% CI [+0.0994, +0.1247]
  stage 2        n=6  +0.1218  p=0.0312  95% CI [+0.1134, +0.1330]
  stage 3        n=6  +0.1299  p=0.0312  95% CI [+0.1112, +0.1480] <-- headline
paired (loop - control), zero_shot.precision
  stage 0 (null) n=6  +0.1070  p=0.2188  95% CI [-0.0148, +0.2488]
  stage 1        n=6  +0.2696  p=0.0312  95% CI [+0.1392, +0.3945]
  stage 2        n=6  +0.3390  p=0.0312  95% CI [+0.2492, +0.4330]
  stage 3        n=6  +0.2372  p=0.0625  95% CI [+0.0775, +0.3859] <-- headline
paired (loop - control), zero_shot.recall
  stage 0 (null) n=6  +0.0223  p=0.2812  95% CI [-0.0052, +0.0517]
  stage 1        n=6  +0.2034  p=0.0312  95% CI [+0.1665, +0.2426]
  stage 2        n=6  +0.2096  p=0.0312  95% CI [+0.1719, +0.2450]
  stage 3        n=6  +0.2237  p=0.0312  95% CI [+0.1774, +0.2675] <-- headline
paired (loop - control), zero_shot.per_class_ap50.bicycle
  stage 0 (null) n=6  +0.0031  p=0.5938  95% CI [-0.0036, +0.0107]
  stage 1        n=6  +0.1290  p=0.0312  95% CI [+0.1087, +0.1540]
  stage 2        n=6  +0.1744  p=0.0312  95% CI [+0.1550, +0.1988]
  stage 3        n=6  +0.1741  p=0.0312  95% CI [+0.1317, +0.2146] <-- headline
paired (loop - control), zero_shot.per_class_ap50.car
  stage 0 (null) n=6  +0.0712  p=0.2500  95% CI [-0.0002, +0.1494]
  stage 1        n=6  +0.4475  p=0.0312  95% CI [+0.3741, +0.5186]
  stage 2        n=6  +0.4420  p=0.0312  95% CI [+0.3812, +0.5324]
  stage 3        n=6  +0.4947  p=0.0312  95% CI [+0.3656, +0.6226] <-- headline
paired (loop - control), zero_shot.per_class_ap50.dog
  stage 0 (null) n=6  -0.0001  p=1.0000  95% CI [-0.0002, +0.0000]
  stage 1        n=6  +0.0290  p=0.0312  95% CI [+0.0085, +0.0523]
  stage 2        n=6  +0.0184  p=0.0625  95% CI [+0.0054, +0.0339]
  stage 3        n=6  +0.0207  p=0.0312  95% CI [+0.0074, +0.0398] <-- headline
paired (loop - control), zero_shot.per_class_ap50.person
  stage 0 (null) n=6  +0.0053  p=0.7188  95% CI [-0.0154, +0.0275]
  stage 1        n=6  +0.3313  p=0.0312  95% CI [+0.2862, +0.3812]
  stage 2        n=6  +0.3679  p=0.0312  95% CI [+0.3556, +0.3799]
  stage 3        n=6  +0.3849  p=0.0312  95% CI [+0.3678, +0.4018] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.primary_map50
  stage 3        n=6  +0.3051  p=0.0312  95% CI [+0.2228, +0.3871] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.primary_n_classes
  stage 3        n=6  +0.0000  p=1.0000  95% CI [+0.0000, +0.0000] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.map50
  stage 3        n=6  +0.2331  p=0.0312  95% CI [+0.1682, +0.3011] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.map50_95
  stage 3        n=6  +0.1225  p=0.0312  95% CI [+0.0890, +0.1573] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.precision
  stage 3        n=6  +0.1892  p=0.0625  95% CI [+0.0681, +0.3405] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.recall
  stage 3        n=6  +0.1802  p=0.0312  95% CI [+0.1279, +0.2315] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.per_class_ap50.bicycle
  stage 3        n=6  +0.1720  p=0.0312  95% CI [+0.1382, +0.2045] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.per_class_ap50.car
  stage 3        n=6  +0.3435  p=0.0312  95% CI [+0.2105, +0.4765] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.per_class_ap50.dog
  stage 3        n=6  +0.0170  p=0.5312  95% CI [-0.0304, +0.0647] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.per_class_ap50.person
  stage 3        n=6  +0.4000  p=0.0312  95% CI [+0.3167, +0.4798] <-- headline
paired (loop - control), fidelity.lpips
  stage 0 (null) n=6  -0.1792  p=0.0312  95% CI [-0.3679, -0.0288]
  stage 1        n=6  -0.0482  p=0.4062  95% CI [-0.1419, +0.0309]
  stage 2        n=6  -0.0656  p=0.0625  95% CI [-0.1184, -0.0223]
  stage 3        n=6  -0.0764  p=0.0938  95% CI [-0.1292, -0.0257] <-- headline
paired (loop - control), fidelity.fid
  stage 0 (null) n=6  -96.4437  p=0.0312  95% CI [-189.4492, -32.4745]
  stage 1        n=6  -67.3773  p=0.1250  95% CI [-133.3594, -7.0386]
  stage 2        n=6  -75.0990  p=0.0625  95% CI [-171.7105, -11.8807]
  stage 3        n=6  -71.8561  p=0.1250  95% CI [-140.9267, -12.4665] <-- headline
paired (loop - control), fidelity.psnr
  stage 0 (null) n=6  +3.4222  p=0.0625  95% CI [+0.5824, +6.6551]
  stage 1        n=6  +1.7093  p=0.1250  95% CI [+0.1900, +3.5379]
  stage 2        n=6  +2.2108  p=0.0625  95% CI [+0.7775, +3.8738]
  stage 3        n=6  +1.9848  p=0.0938  95% CI [+0.5033, +3.3265] <-- headline
paired (loop - control), fidelity.ssim
  stage 0 (null) n=6  +0.1338  p=0.0312  95% CI [+0.0529, +0.2414]
  stage 1        n=6  +0.0681  p=0.1562  95% CI [+0.0010, +0.1427]
  stage 2        n=6  +0.1303  p=0.0625  95% CI [+0.0408, +0.2512]
  stage 3        n=6  +0.1635  p=0.0312  95% CI [+0.0659, +0.2677] <-- headline
paired (loop - control), detector.map50
  stage 0 (null) n=6  +0.1055  p=0.3438  95% CI [-0.0038, +0.2246]
  stage 1        n=6  +0.0988  p=0.0312  95% CI [+0.0782, +0.1308]
  stage 2        n=6  +0.1012  p=0.0312  95% CI [+0.0854, +0.1169]
  stage 3        n=6  +0.0986  p=0.0312  95% CI [+0.0787, +0.1173] <-- headline
paired (loop - control), faithfulness.false_object_rate
  stage 3        n=6  -0.3324  p=0.0312  95% CI [-0.5347, -0.1477] <-- headline
paired (loop - control), faithfulness.missed_object_rate
  stage 3        n=6  -0.3558  p=0.0312  95% CI [-0.4018, -0.3098] <-- headline
paired (loop - control), faithfulness.detection_consistency
  stage 3        n=6  +0.3068  p=0.0312  95% CI [+0.2628, +0.3507] <-- headline
paired (loop - control), judges.rtdetr.faithfulness.false_object_rate
  stage 3        n=6  -0.1487  p=0.0312  95% CI [-0.1647, -0.1356] <-- headline
paired (loop - control), judges.rtdetr.faithfulness.missed_object_rate
  stage 3        n=6  -0.3359  p=0.0312  95% CI [-0.4504, -0.2234] <-- headline
paired (loop - control), judges.rtdetr.faithfulness.detection_consistency
  stage 3        n=6  +0.1997  p=0.0312  95% CI [+0.1153, +0.2820] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.primary_map50 -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control -0.0011  loop +0.2749  diff +0.2760  p=0.0312  95% CI [+0.2525, +0.2973]
  stage 2        n=6  control +0.0218  loop +0.3233  diff +0.3015  p=0.0312  95% CI [+0.2533, +0.3526]
  stage 3        n=6  control +0.0181  loop +0.3427  diff +0.3247  p=0.0312  95% CI [+0.2659, +0.3846] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.primary_n_classes -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0000  loop +0.0000  diff +0.0000  p=1.0000  95% CI [+0.0000, +0.0000]
  stage 2        n=6  control +0.0000  loop +0.0000  diff +0.0000  p=1.0000  95% CI [+0.0000, +0.0000]
  stage 3        n=6  control +0.0000  loop +0.0000  diff +0.0000  p=1.0000  95% CI [+0.0000, +0.0000] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.map50 -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control -0.0009  loop +0.2134  diff +0.2143  p=0.0312  95% CI [+0.1958, +0.2302]
  stage 2        n=6  control +0.0164  loop +0.2472  diff +0.2307  p=0.0312  95% CI [+0.1937, +0.2687]
  stage 3        n=6  control +0.0136  loop +0.2622  diff +0.2487  p=0.0312  95% CI [+0.2074, +0.2912] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.map50_95 -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0010  loop +0.1025  diff +0.1015  p=0.0312  95% CI [+0.0911, +0.1116]
  stage 2        n=6  control +0.0095  loop +0.1210  diff +0.1115  p=0.0312  95% CI [+0.0931, +0.1307]
  stage 3        n=6  control +0.0072  loop +0.1268  diff +0.1196  p=0.0312  95% CI [+0.0999, +0.1402] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.precision -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control -0.0545  loop +0.1080  diff +0.1626  p=0.0625  95% CI [+0.0601, +0.2696]
  stage 2        n=6  control -0.1357  loop +0.0963  diff +0.2320  p=0.0625  95% CI [+0.0426, +0.3608]
  stage 3        n=6  control -0.0762  loop +0.0539  diff +0.1301  p=0.1875  95% CI [-0.0195, +0.2580] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.recall -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0082  loop +0.1893  diff +0.1810  p=0.0312  95% CI [+0.1670, +0.1973]
  stage 2        n=6  control +0.0263  loop +0.2136  diff +0.1873  p=0.0312  95% CI [+0.1419, +0.2368]
  stage 3        n=6  control +0.0188  loop +0.2201  diff +0.2014  p=0.0312  95% CI [+0.1398, +0.2545] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.per_class_ap50.bicycle -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0120  loop +0.1379  diff +0.1259  p=0.0312  95% CI [+0.1073, +0.1457]
  stage 2        n=6  control +0.0166  loop +0.1879  diff +0.1713  p=0.0312  95% CI [+0.1527, +0.1950]
  stage 3        n=6  control +0.0282  loop +0.1991  diff +0.1709  p=0.0312  95% CI [+0.1279, +0.2130] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.per_class_ap50.car -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control -0.0180  loop +0.3583  diff +0.3763  p=0.0312  95% CI [+0.3123, +0.4419]
  stage 2        n=6  control +0.0436  loop +0.4143  diff +0.3707  p=0.0312  95% CI [+0.2437, +0.5148]
  stage 3        n=6  control +0.0180  loop +0.4414  diff +0.4234  p=0.0312  95% CI [+0.2925, +0.5650] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.per_class_ap50.dog -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control -0.0000  loop +0.0291  diff +0.0291  p=0.0312  95% CI [+0.0085, +0.0524]
  stage 2        n=6  control +0.0004  loop +0.0188  diff +0.0184  p=0.0625  95% CI [+0.0054, +0.0340]
  stage 3        n=6  control +0.0000  loop +0.0208  diff +0.0208  p=0.0312  95% CI [+0.0074, +0.0400] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.per_class_ap50.person -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0026  loop +0.3286  diff +0.3260  p=0.0312  95% CI [+0.2893, +0.3597]
  stage 2        n=6  control +0.0052  loop +0.3677  diff +0.3625  p=0.0312  95% CI [+0.3464, +0.3807]
  stage 3        n=6  control +0.0081  loop +0.3877  diff +0.3796  p=0.0312  95% CI [+0.3643, +0.3952] <-- headline
trajectory (loop - control) of the gain over stage 0, fidelity.lpips -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control -0.1062  loop +0.0247  diff +0.1310  p=0.3125  95% CI [-0.0596, +0.3245]
  stage 2        n=6  control -0.1379  loop -0.0243  diff +0.1136  p=0.4375  95% CI [-0.0552, +0.3070]
  stage 3        n=6  control -0.1137  loop -0.0109  diff +0.1028  p=0.4062  95% CI [-0.0522, +0.3143] <-- headline
trajectory (loop - control) of the gain over stage 0, fidelity.fid -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control -35.9498  loop -6.8834  diff +29.0664  p=0.5938  95% CI [-62.0849, +101.7964]
  stage 2        n=6  control -53.5611  loop -32.2164  diff +21.3447  p=0.7500  95% CI [-103.7095, +134.1320]
  stage 3        n=6  control -57.9292  loop -33.3416  diff +24.5876  p=0.8750  95% CI [-66.9043, +154.4163] <-- headline
trajectory (loop - control) of the gain over stage 0, fidelity.psnr -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +1.3186  loop -0.3943  diff -1.7129  p=0.3750  95% CI [-5.3070, +1.7715]
  stage 2        n=6  control +1.8202  loop +0.6088  diff -1.2114  p=0.5938  95% CI [-5.3054, +2.6643]
  stage 3        n=6  control +1.3802  loop -0.0572  diff -1.4375  p=0.5938  95% CI [-5.0564, +1.2305] <-- headline
trajectory (loop - control) of the gain over stage 0, fidelity.ssim -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0597  loop -0.0060  diff -0.0657  p=0.2812  95% CI [-0.1537, +0.0359]
  stage 2        n=6  control +0.0749  loop +0.0713  diff -0.0036  p=0.9688  95% CI [-0.1486, +0.1524]
  stage 3        n=6  control +0.0540  loop +0.0837  diff +0.0297  p=0.6875  95% CI [-0.1295, +0.1594] <-- headline
trajectory (loop - control) of the gain over stage 0, detector.map50 -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0754  loop +0.0686  diff -0.0068  p=0.9062  95% CI [-0.1382, +0.1148]
  stage 2        n=6  control +0.0802  loop +0.0759  diff -0.0043  p=0.9375  95% CI [-0.1303, +0.1160]
  stage 3        n=6  control +0.0683  loop +0.0613  diff -0.0070  p=1.0000  95% CI [-0.1437, +0.1116] <-- headline

```

**msrs-day readout**, the same two blocks:

```
run,seed,arm,stage,task_weight,zero_shot.primary_map50,zero_shot.primary_n_classes,zero_shot.map50,zero_shot.map50_95,zero_shot.precision,zero_shot.recall,zero_shot.per_class_ap50.bike,zero_shot.per_class_ap50.car,zero_shot.per_class_ap50.car_stop,zero_shot.per_class_ap50.color_cone,zero_shot.per_class_ap50.person,judges.rtdetr.zero_shot.primary_map50,judges.rtdetr.zero_shot.primary_n_classes,judges.rtdetr.zero_shot.map50,judges.rtdetr.zero_shot.map50_95,judges.rtdetr.zero_shot.precision,judges.rtdetr.zero_shot.recall,judges.rtdetr.zero_shot.per_class_ap50.bike,judges.rtdetr.zero_shot.per_class_ap50.car,judges.rtdetr.zero_shot.per_class_ap50.car_stop,judges.rtdetr.zero_shot.per_class_ap50.color_cone,judges.rtdetr.zero_shot.per_class_ap50.person,fidelity.lpips,fidelity.fid,fidelity.psnr,fidelity.ssim,detector.map50,faithfulness.false_object_rate,faithfulness.missed_object_rate,faithfulness.detection_consistency,judges.rtdetr.faithfulness.false_object_rate,judges.rtdetr.faithfulness.missed_object_rate,judges.rtdetr.faithfulness.detection_consistency
e3md-control-s0,0,control,3,0,0.187853,2.000000,0.366795,0.187639,0.685285,0.337391,0.306358,0.750085,0.255659,0.120047,0.401828,0.214290,2.000000,0.361097,0.173713,0.640822,0.359191,0.266348,0.740892,0.224930,0.203650,0.369665,0.389578,107.698227,14.377576,0.471165,0.521631,0.409222,0.579918,0.425852,0.609457,0.543033,0.354325
e3md-control-s1,1,control,3,0,0.189718,2.000000,0.362824,0.181610,0.640750,0.351784,0.302626,0.743591,0.235234,0.144202,0.388468,0.168504,2.000000,0.328795,0.168201,0.608278,0.315014,0.236763,0.685505,0.190219,0.146789,0.384699,0.400415,108.016151,14.183396,0.471795,0.527086,0.436039,0.579918,0.436874,0.657097,0.559426,0.340484
e3md-control-s2,2,control,3,0,0.188682,2.000000,0.347964,0.175008,0.635842,0.324165,0.274280,0.681077,0.273963,0.103402,0.407098,0.191196,2.000000,0.344932,0.165518,0.656770,0.323840,0.273959,0.686305,0.229549,0.152842,0.382005,0.414437,105.724030,13.456588,0.474770,0.536538,0.414925,0.598361,0.420842,0.653936,0.545082,0.352249
e3md-control-s3,3,control,3,0,0.203271,2.000000,0.376136,0.189576,0.652014,0.371088,0.314591,0.735903,0.275027,0.131514,0.423647,0.187831,2.000000,0.364688,0.181049,0.645949,0.351492,0.317710,0.702354,0.207876,0.167787,0.427712,0.389343,103.150581,14.568636,0.477463,0.527408,0.404570,0.546107,0.466934,0.643075,0.519467,0.372318
e3md-control-s4,4,control,3,0,0.204690,2.000000,0.369790,0.188622,0.615467,0.360695,0.280725,0.754374,0.264501,0.144880,0.404467,0.206799,2.000000,0.349385,0.174606,0.571577,0.350766,0.213629,0.714265,0.242099,0.171499,0.405435,0.394001,104.141159,14.402502,0.475510,0.538039,0.432133,0.579918,0.444890,0.643519,0.526639,0.362630
e3md-control-s5,5,control,3,0,0.177780,2.000000,0.369480,0.189121,0.689479,0.344948,0.308399,0.759374,0.241525,0.114035,0.424069,0.186066,2.000000,0.342808,0.169209,0.616851,0.346719,0.228619,0.717718,0.220222,0.151910,0.395569,0.396332,105.358490,14.350080,0.480330,0.527817,0.419619,0.563525,0.458918,0.634726,0.543033,0.355017
e3md-loop-s0,0,loop,3,3,0.363236,2.000000,0.519969,0.282585,0.730978,0.486191,0.461419,0.844178,0.468371,0.258100,0.567774,0.348302,2.000000,0.496201,0.266814,0.753515,0.468355,0.406710,0.808163,0.415204,0.281399,0.569531,0.383836,103.092659,13.746405,0.445510,0.575977,0.347032,0.413934,0.625251,0.542347,0.396516,0.467820
e3md-loop-s1,1,loop,3,3,0.364913,2.000000,0.496129,0.255187,0.712490,0.464364,0.376504,0.814110,0.481122,0.248704,0.560204,0.311616,2.000000,0.461430,0.243881,0.691582,0.451030,0.380298,0.760934,0.339047,0.284186,0.542686,0.402720,105.411629,13.973309,0.459022,0.555085,0.355658,0.428279,0.582164,0.544141,0.418033,0.438754
e3md-loop-s2,2,loop,3,3,0.366478,2.000000,0.515043,0.267104,0.775753,0.478684,0.467999,0.817468,0.456786,0.276170,0.556794,0.345543,2.000000,0.489873,0.256255,0.768410,0.457753,0.411081,0.808629,0.389422,0.301665,0.538568,0.386919,99.611191,13.908452,0.450035,0.554474,0.358708,0.430328,0.598196,0.544234,0.408811,0.458824
e3md-loop-s3,3,loop,3,3,0.351383,2.000000,0.499416,0.267086,0.787968,0.454660,0.421506,0.830596,0.452743,0.250023,0.542210,0.336595,2.000000,0.483085,0.242769,0.737988,0.469685,0.380591,0.812998,0.401581,0.271609,0.548646,0.411481,103.995712,13.832481,0.462758,0.530446,0.300781,0.449795,0.569138,0.528271,0.401639,0.456055
e3md-loop-s4,4,loop,3,3,0.381537,2.000000,0.533022,0.277519,0.754336,0.485946,0.484341,0.834859,0.477142,0.285931,0.582836,0.357173,2.000000,0.494752,0.258219,0.760377,0.460168,0.403364,0.805964,0.382075,0.332270,0.550086,0.391504,100.946861,13.883492,0.455575,0.563505,0.312727,0.419057,0.587174,0.548563,0.404713,0.463668
e3md-loop-s5,5,loop,3,3,0.327724,2.000000,0.488363,0.258366,0.673878,0.467837,0.386957,0.836664,0.426953,0.228494,0.562748,0.326227,2.000000,0.468535,0.249542,0.699748,0.453994,0.335886,0.812449,0.402849,0.249604,0.541889,0.404604,102.218124,14.072066,0.451063,0.542171,0.321696,0.442623,0.570140,0.540783,0.423156,0.452595
```

```
==============================================================================
== 6. PAIRED STATISTICS (t2o aggregate, verbatim)
==============================================================================
$ t2o aggregate --runs runs/e3md-control-s* runs/e3md-loop-s* --metric zero_shot.primary_map50 zero_shot.primary_n_classes zero_shot.map50 zero_shot.map50_95 zero_shot.precision zero_shot.recall zero_shot.per_class_ap50.bike zero_shot.per_class_ap50.car zero_shot.per_class_ap50.car_stop zero_shot.per_class_ap50.color_cone zero_shot.per_class_ap50.person judges.rtdetr.zero_shot.primary_map50 judges.rtdetr.zero_shot.primary_n_classes judges.rtdetr.zero_shot.map50 judges.rtdetr.zero_shot.map50_95 judges.rtdetr.zero_shot.precision judges.rtdetr.zero_shot.recall judges.rtdetr.zero_shot.per_class_ap50.bike judges.rtdetr.zero_shot.per_class_ap50.car judges.rtdetr.zero_shot.per_class_ap50.car_stop judges.rtdetr.zero_shot.per_class_ap50.color_cone judges.rtdetr.zero_shot.per_class_ap50.person fidelity.lpips fidelity.fid fidelity.psnr fidelity.ssim detector.map50 faithfulness.false_object_rate faithfulness.missed_object_rate faithfulness.detection_consistency judges.rtdetr.faithfulness.false_object_rate judges.rtdetr.faithfulness.missed_object_rate judges.rtdetr.faithfulness.detection_consistency --primary-classes car_stop color_cone --stage 3

12 runs, stages [0, 1, 2, 3]
metric                                            arm      stage   n      mean       std
zero_shot.primary_map50                           control      0   6    0.1168    0.0315
zero_shot.primary_map50                           loop         0   6    0.1085    0.0575
zero_shot.primary_map50                           control      1   6    0.1695    0.0430
zero_shot.primary_map50                           loop         1   6    0.3243    0.0248
zero_shot.primary_map50                           control      2   6    0.2001    0.0288
zero_shot.primary_map50                           loop         2   6    0.3516    0.0184
zero_shot.primary_map50                           control      3   6    0.1920    0.0102
zero_shot.primary_map50                           loop         3   6    0.3592    0.0182
zero_shot.primary_n_classes                       control      0   6    2.0000    0.0000
zero_shot.primary_n_classes                       loop         0   6    2.0000    0.0000
zero_shot.primary_n_classes                       control      1   6    2.0000    0.0000
zero_shot.primary_n_classes                       loop         1   6    2.0000    0.0000
zero_shot.primary_n_classes                       control      2   6    2.0000    0.0000
zero_shot.primary_n_classes                       loop         2   6    2.0000    0.0000
zero_shot.primary_n_classes                       control      3   6    2.0000    0.0000
zero_shot.primary_n_classes                       loop         3   6    2.0000    0.0000
zero_shot.map50                                   control      0   6    0.2653    0.0967
zero_shot.map50                                   loop         0   6    0.2631    0.1322
zero_shot.map50                                   control      1   6    0.3161    0.1095
zero_shot.map50                                   loop         1   6    0.4798    0.0125
zero_shot.map50                                   control      2   6    0.3668    0.0277
zero_shot.map50                                   loop         2   6    0.4925    0.0115
zero_shot.map50                                   control      3   6    0.3655    0.0096
zero_shot.map50                                   loop         3   6    0.5087    0.0168
zero_shot.map50_95                                control      0   6    0.1307    0.0522
zero_shot.map50_95                                loop         0   6    0.1302    0.0665
zero_shot.map50_95                                control      1   6    0.1561    0.0598
zero_shot.map50_95                                loop         1   6    0.2486    0.0087
zero_shot.map50_95                                control      2   6    0.1844    0.0130
zero_shot.map50_95                                loop         2   6    0.2566    0.0056
zero_shot.map50_95                                control      3   6    0.1853    0.0058
zero_shot.map50_95                                loop         3   6    0.2680    0.0106
zero_shot.precision                               control      0   6    0.4873    0.1133
zero_shot.precision                               loop         0   6    0.4766    0.2367
zero_shot.precision                               control      1   6    0.5538    0.1345
zero_shot.precision                               loop         1   6    0.7172    0.0396
zero_shot.precision                               control      2   6    0.6498    0.0542
zero_shot.precision                               loop         2   6    0.7147    0.0245
zero_shot.precision                               control      3   6    0.6531    0.0291
zero_shot.precision                               loop         3   6    0.7392    0.0424
zero_shot.recall                                  control      0   6    0.2722    0.0921
zero_shot.recall                                  loop         0   6    0.2612    0.1255
zero_shot.recall                                  control      1   6    0.3171    0.1010
zero_shot.recall                                  loop         1   6    0.4639    0.0229
zero_shot.recall                                  control      2   6    0.3544    0.0251
zero_shot.recall                                  loop         2   6    0.4700    0.0084
zero_shot.recall                                  control      3   6    0.3483    0.0167
zero_shot.recall                                  loop         3   6    0.4729    0.0127
zero_shot.per_class_ap50.bike                     control      0   6    0.1783    0.0945
zero_shot.per_class_ap50.bike                     loop         0   6    0.1739    0.0924
zero_shot.per_class_ap50.bike                     control      1   6    0.2232    0.1032
zero_shot.per_class_ap50.bike                     loop         1   6    0.4114    0.0315
zero_shot.per_class_ap50.bike                     control      2   6    0.2777    0.0418
zero_shot.per_class_ap50.bike                     loop         2   6    0.3968    0.0252
zero_shot.per_class_ap50.bike                     control      3   6    0.2978    0.0163
zero_shot.per_class_ap50.bike                     loop         3   6    0.4331    0.0450
zero_shot.per_class_ap50.car                      control      0   6    0.5792    0.2026
zero_shot.per_class_ap50.car                      loop         0   6    0.5756    0.2885
zero_shot.per_class_ap50.car                      control      1   6    0.6456    0.2170
zero_shot.per_class_ap50.car                      loop         1   6    0.7899    0.0146
zero_shot.per_class_ap50.car                      control      2   6    0.7324    0.0317
zero_shot.per_class_ap50.car                      loop         2   6    0.8015    0.0304
zero_shot.per_class_ap50.car                      control      3   6    0.7374    0.0288
zero_shot.per_class_ap50.car                      loop         3   6    0.8296    0.0116
zero_shot.per_class_ap50.car_stop                 control      0   6    0.1650    0.0460
zero_shot.per_class_ap50.car_stop                 loop         0   6    0.1548    0.0819
zero_shot.per_class_ap50.car_stop                 control      1   6    0.2276    0.0567
zero_shot.per_class_ap50.car_stop                 loop         1   6    0.4101    0.0211
zero_shot.per_class_ap50.car_stop                 control      2   6    0.2695    0.0380
zero_shot.per_class_ap50.car_stop                 loop         2   6    0.4259    0.0180
zero_shot.per_class_ap50.car_stop                 control      3   6    0.2577    0.0166
zero_shot.per_class_ap50.car_stop                 loop         3   6    0.4605    0.0198
zero_shot.per_class_ap50.color_cone               control      0   6    0.0686    0.0245
zero_shot.per_class_ap50.color_cone               loop         0   6    0.0622    0.0427
zero_shot.per_class_ap50.color_cone               control      1   6    0.1113    0.0307
zero_shot.per_class_ap50.color_cone               loop         1   6    0.2385    0.0434
zero_shot.per_class_ap50.color_cone               control      2   6    0.1307    0.0323
zero_shot.per_class_ap50.color_cone               loop         2   6    0.2774    0.0328
zero_shot.per_class_ap50.color_cone               control      3   6    0.1263    0.0168
zero_shot.per_class_ap50.color_cone               loop         3   6    0.2579    0.0206
zero_shot.per_class_ap50.person                   control      0   6    0.3351    0.1455
zero_shot.per_class_ap50.person                   loop         0   6    0.3490    0.1787
zero_shot.per_class_ap50.person                   control      1   6    0.3726    0.1462
zero_shot.per_class_ap50.person                   loop         1   6    0.5490    0.0110
zero_shot.per_class_ap50.person                   control      2   6    0.4239    0.0309
zero_shot.per_class_ap50.person                   loop         2   6    0.5609    0.0151
zero_shot.per_class_ap50.person                   control      3   6    0.4083    0.0137
zero_shot.per_class_ap50.person                   loop         3   6    0.5621    0.0133
judges.rtdetr.zero_shot.primary_map50             control      3   6    0.1924    0.0162
judges.rtdetr.zero_shot.primary_map50             loop         3   6    0.3376    0.0165
judges.rtdetr.zero_shot.primary_n_classes         control      3   6    2.0000    0.0000
judges.rtdetr.zero_shot.primary_n_classes         loop         3   6    2.0000    0.0000
judges.rtdetr.zero_shot.map50                     control      3   6    0.3486    0.0131
judges.rtdetr.zero_shot.map50                     loop         3   6    0.4823    0.0144
judges.rtdetr.zero_shot.map50_95                  control      3   6    0.1720    0.0056
judges.rtdetr.zero_shot.map50_95                  loop         3   6    0.2529    0.0093
judges.rtdetr.zero_shot.precision                 control      3   6    0.6234    0.0312
judges.rtdetr.zero_shot.precision                 loop         3   6    0.7353    0.0324
judges.rtdetr.zero_shot.recall                    control      3   6    0.3412    0.0175
judges.rtdetr.zero_shot.recall                    loop         3   6    0.4602    0.0076
judges.rtdetr.zero_shot.per_class_ap50.bike       control      3   6    0.2562    0.0378
judges.rtdetr.zero_shot.per_class_ap50.bike       loop         3   6    0.3863    0.0280
judges.rtdetr.zero_shot.per_class_ap50.car        control      3   6    0.7078    0.0211
judges.rtdetr.zero_shot.per_class_ap50.car        loop         3   6    0.8015    0.0201
judges.rtdetr.zero_shot.per_class_ap50.car_stop   control      3   6    0.2191    0.0181
judges.rtdetr.zero_shot.per_class_ap50.car_stop   loop         3   6    0.3884    0.0267
judges.rtdetr.zero_shot.per_class_ap50.color_cone control      3   6    0.1657    0.0209
judges.rtdetr.zero_shot.per_class_ap50.color_cone loop         3   6    0.2868    0.0281
judges.rtdetr.zero_shot.per_class_ap50.person     control      3   6    0.3942    0.0205
judges.rtdetr.zero_shot.per_class_ap50.person     loop         3   6    0.5486    0.0111
fidelity.lpips                                    control      0   6    0.4212    0.0555
fidelity.lpips                                    loop         0   6    0.4518    0.1299
fidelity.lpips                                    control      1   6    0.4140    0.0622
fidelity.lpips                                    loop         1   6    0.4202    0.0107
fidelity.lpips                                    control      2   6    0.3862    0.0134
fidelity.lpips                                    loop         2   6    0.4094    0.0086
fidelity.lpips                                    control      3   6    0.3974    0.0094
fidelity.lpips                                    loop         3   6    0.3968    0.0110
fidelity.fid                                      control      0   6  116.6055   12.8928
fidelity.fid                                      loop         0   6  126.8754   44.6513
fidelity.fid                                      control      1   6  111.3943   16.9987
fidelity.fid                                      loop         1   6  109.8880    8.1459
fidelity.fid                                      control      2   6  100.2937    3.3649
fidelity.fid                                      loop         2   6  103.6328    3.0748
fidelity.fid                                      control      3   6  105.6814    1.9184
fidelity.fid                                      loop         3   6  102.5460    2.0924
fidelity.psnr                                     control      0   6   12.4942    2.2846
fidelity.psnr                                     loop         0   6   11.7422    3.1854
fidelity.psnr                                     control      1   6   13.2138    2.4121
fidelity.psnr                                     loop         1   6   13.2637    0.3075
fidelity.psnr                                     control      2   6   14.2450    0.2601
fidelity.psnr                                     loop         2   6   13.7042    0.3788
fidelity.psnr                                     control      3   6   14.2231    0.3951
fidelity.psnr                                     loop         3   6   13.9027    0.1126
fidelity.ssim                                     control      0   6    0.4027    0.0300
fidelity.ssim                                     loop         0   6    0.3926    0.0366
fidelity.ssim                                     control      1   6    0.4354    0.0301
fidelity.ssim                                     loop         1   6    0.4318    0.0172
fidelity.ssim                                     control      2   6    0.4613    0.0062
fidelity.ssim                                     loop         2   6    0.4541    0.0089
fidelity.ssim                                     control      3   6    0.4752    0.0035
fidelity.ssim                                     loop         3   6    0.4540    0.0063
detector.map50                                    control      0   6    0.5109    0.0425
detector.map50                                    loop         0   6    0.4791    0.1261
detector.map50                                    control      1   6    0.5130    0.0401
detector.map50                                    loop         1   6    0.5577    0.0142
detector.map50                                    control      2   6    0.5333    0.0082
detector.map50                                    loop         2   6    0.5643    0.0097
detector.map50                                    control      3   6    0.5298    0.0063
detector.map50                                    loop         3   6    0.5536    0.0159
faithfulness.false_object_rate                    control      3   6    0.4194    0.0125
faithfulness.false_object_rate                    loop         3   6    0.3328    0.0243
faithfulness.missed_object_rate                   control      3   6    0.5746    0.0178
faithfulness.missed_object_rate                   loop         3   6    0.4307    0.0136
faithfulness.detection_consistency                control      3   6    0.4424    0.0182
faithfulness.detection_consistency                loop         3   6    0.5887    0.0210
judges.rtdetr.faithfulness.false_object_rate      control      3   6    0.6403    0.0171
judges.rtdetr.faithfulness.false_object_rate      loop         3   6    0.5414    0.0069
judges.rtdetr.faithfulness.missed_object_rate     control      3   6    0.5394    0.0143
judges.rtdetr.faithfulness.missed_object_rate     loop         3   6    0.4088    0.0101
judges.rtdetr.faithfulness.detection_consistency  control      3   6    0.3562    0.0107
judges.rtdetr.faithfulness.detection_consistency  loop         3   6    0.4563    0.0101
paired (loop - control), zero_shot.primary_map50
  stage 0 (null) n=6  -0.0083  p=0.5938  95% CI [-0.0380, +0.0190]
  stage 1        n=6  +0.1548  p=0.0312  95% CI [+0.1368, +0.1776]
  stage 2        n=6  +0.1515  p=0.0312  95% CI [+0.1218, +0.1806]
  stage 3        n=6  +0.1672  p=0.0312  95% CI [+0.1575, +0.1766] <-- headline
paired (loop - control), zero_shot.primary_n_classes
  stage 0 (null) n=6  +0.0000  p=1.0000  95% CI [+0.0000, +0.0000]
  stage 1        n=6  +0.0000  p=1.0000  95% CI [+0.0000, +0.0000]
  stage 2        n=6  +0.0000  p=1.0000  95% CI [+0.0000, +0.0000]
  stage 3        n=6  +0.0000  p=1.0000  95% CI [+0.0000, +0.0000] <-- headline
paired (loop - control), zero_shot.map50
  stage 0 (null) n=6  -0.0021  p=0.9062  95% CI [-0.0386, +0.0300]
  stage 1        n=6  +0.1637  p=0.0312  95% CI [+0.1154, +0.2452]
  stage 2        n=6  +0.1256  p=0.0312  95% CI [+0.0988, +0.1474]
  stage 3        n=6  +0.1432  p=0.0312  95% CI [+0.1285, +0.1585] <-- headline
paired (loop - control), zero_shot.map50_95
  stage 0 (null) n=6  -0.0004  p=0.9688  95% CI [-0.0176, +0.0160]
  stage 1        n=6  +0.0925  p=0.0312  95% CI [+0.0656, +0.1355]
  stage 2        n=6  +0.0722  p=0.0312  95% CI [+0.0588, +0.0820]
  stage 3        n=6  +0.0827  p=0.0312  95% CI [+0.0750, +0.0904] <-- headline
paired (loop - control), zero_shot.precision
  stage 0 (null) n=6  -0.0106  p=0.9062  95% CI [-0.1197, +0.0737]
  stage 1        n=6  +0.1634  p=0.0625  95% CI [+0.0521, +0.2753]
  stage 2        n=6  +0.0649  p=0.0938  95% CI [+0.0140, +0.1116]
  stage 3        n=6  +0.0861  p=0.0625  95% CI [+0.0349, +0.1275] <-- headline
paired (loop - control), zero_shot.recall
  stage 0 (null) n=6  -0.0110  p=0.6562  95% CI [-0.0439, +0.0160]
  stage 1        n=6  +0.1468  p=0.0312  95% CI [+0.0983, +0.2181]
  stage 2        n=6  +0.1156  p=0.0312  95% CI [+0.0942, +0.1305]
  stage 3        n=6  +0.1246  p=0.0312  95% CI [+0.1051, +0.1425] <-- headline
paired (loop - control), zero_shot.per_class_ap50.bike
  stage 0 (null) n=6  -0.0044  p=0.8750  95% CI [-0.0390, +0.0324]
  stage 1        n=6  +0.1882  p=0.0312  95% CI [+0.1231, +0.2684]
  stage 2        n=6  +0.1191  p=0.0312  95% CI [+0.0841, +0.1510]
  stage 3        n=6  +0.1353  p=0.0312  95% CI [+0.0953, +0.1778] <-- headline
paired (loop - control), zero_shot.per_class_ap50.car
  stage 0 (null) n=6  -0.0036  p=1.0000  95% CI [-0.0789, +0.0638]
  stage 1        n=6  +0.1443  p=0.0312  95% CI [+0.0485, +0.3230]
  stage 2        n=6  +0.0691  p=0.0312  95% CI [+0.0442, +0.0963]
  stage 3        n=6  +0.0922  p=0.0312  95% CI [+0.0778, +0.1115] <-- headline
paired (loop - control), zero_shot.per_class_ap50.car_stop
  stage 0 (null) n=6  -0.0102  p=0.8438  95% CI [-0.0508, +0.0191]
  stage 1        n=6  +0.1825  p=0.0312  95% CI [+0.1483, +0.2254]
  stage 2        n=6  +0.1564  p=0.0312  95% CI [+0.1133, +0.1924]
  stage 3        n=6  +0.2029  p=0.0312  95% CI [+0.1857, +0.2237] <-- headline
paired (loop - control), zero_shot.per_class_ap50.color_cone
  stage 0 (null) n=6  -0.0064  p=0.7500  95% CI [-0.0276, +0.0250]
  stage 1        n=6  +0.1272  p=0.0312  95% CI [+0.1087, +0.1465]
  stage 2        n=6  +0.1466  p=0.0312  95% CI [+0.1236, +0.1796]
  stage 3        n=6  +0.1316  p=0.0312  95% CI [+0.1151, +0.1511] <-- headline
paired (loop - control), zero_shot.per_class_ap50.person
  stage 0 (null) n=6  +0.0138  p=0.5938  95% CI [-0.0277, +0.0572]
  stage 1        n=6  +0.1763  p=0.0312  95% CI [+0.1040, +0.2920]
  stage 2        n=6  +0.1369  p=0.0312  95% CI [+0.1005, +0.1625]
  stage 3        n=6  +0.1538  p=0.0312  95% CI [+0.1368, +0.1693] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.primary_map50
  stage 3        n=6  +0.1451  p=0.0312  95% CI [+0.1395, +0.1504] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.primary_n_classes
  stage 3        n=6  +0.0000  p=1.0000  95% CI [+0.0000, +0.0000] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.map50
  stage 3        n=6  +0.1337  p=0.0312  95% CI [+0.1259, +0.1414] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.map50_95
  stage 3        n=6  +0.0809  p=0.0312  95% CI [+0.0720, +0.0886] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.precision
  stage 3        n=6  +0.1119  p=0.0312  95% CI [+0.0893, +0.1455] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.recall
  stage 3        n=6  +0.1190  p=0.0312  95% CI [+0.1101, +0.1283] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.per_class_ap50.bike
  stage 3        n=6  +0.1302  p=0.0312  95% CI [+0.0975, +0.1598] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.per_class_ap50.car
  stage 3        n=6  +0.0937  p=0.0312  95% CI [+0.0786, +0.1087] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.per_class_ap50.car_stop
  stage 3        n=6  +0.1692  p=0.0312  95% CI [+0.1528, +0.1851] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.per_class_ap50.color_cone
  stage 3        n=6  +0.1210  p=0.0312  95% CI [+0.0973, +0.1444] <-- headline
paired (loop - control), judges.rtdetr.zero_shot.per_class_ap50.person
  stage 3        n=6  +0.1544  p=0.0312  95% CI [+0.1370, +0.1748] <-- headline
paired (loop - control), fidelity.lpips
  stage 0 (null) n=6  +0.0307  p=0.5000  95% CI [-0.0110, +0.0950]
  stage 1        n=6  +0.0062  p=0.8438  95% CI [-0.0375, +0.0308]
  stage 2        n=6  +0.0231  p=0.0312  95% CI [+0.0141, +0.0339]
  stage 3        n=6  -0.0005  p=0.9375  95% CI [-0.0132, +0.0111] <-- headline
paired (loop - control), fidelity.fid
  stage 0 (null) n=6  +10.2699  p=0.8125  95% CI [-6.4844, +37.8391]
  stage 1        n=6  -1.5063  p=0.8125  95% CI [-9.7560, +5.6671]
  stage 2        n=6  +3.3391  p=0.0625  95% CI [+1.4448, +5.3307]
  stage 3        n=6  -3.1354  p=0.0625  95% CI [-4.7747, -1.3113] <-- headline
paired (loop - control), fidelity.psnr
  stage 0 (null) n=6  -0.7520  p=0.2188  95% CI [-1.7569, +0.1121]
  stage 1        n=6  +0.0500  p=1.0000  95% CI [-0.9291, +1.8263]
  stage 2        n=6  -0.5409  p=0.0312  95% CI [-0.7971, -0.2955]
  stage 3        n=6  -0.3204  p=0.1250  95% CI [-0.5948, +0.0219] <-- headline
paired (loop - control), fidelity.ssim
  stage 0 (null) n=6  -0.0102  p=0.3125  95% CI [-0.0274, +0.0083]
  stage 1        n=6  -0.0036  p=0.5938  95% CI [-0.0167, +0.0106]
  stage 2        n=6  -0.0073  p=0.1875  95% CI [-0.0152, +0.0008]
  stage 3        n=6  -0.0212  p=0.0312  95% CI [-0.0258, -0.0164] <-- headline
paired (loop - control), detector.map50
  stage 0 (null) n=6  -0.0318  p=0.7500  95% CI [-0.1051, +0.0121]
  stage 1        n=6  +0.0447  p=0.0312  95% CI [+0.0265, +0.0720]
  stage 2        n=6  +0.0311  p=0.0312  95% CI [+0.0251, +0.0384]
  stage 3        n=6  +0.0239  p=0.0312  95% CI [+0.0124, +0.0370] <-- headline
paired (loop - control), faithfulness.false_object_rate
  stage 3        n=6  -0.0867  p=0.0312  95% CI [-0.1044, -0.0687] <-- headline
paired (loop - control), faithfulness.missed_object_rate
  stage 3        n=6  -0.1440  p=0.0312  95% CI [-0.1634, -0.1212] <-- headline
paired (loop - control), faithfulness.detection_consistency
  stage 3        n=6  +0.1463  p=0.0312  95% CI [+0.1196, +0.1735] <-- headline
paired (loop - control), judges.rtdetr.faithfulness.false_object_rate
  stage 3        n=6  -0.0989  p=0.0312  95% CI [-0.1103, -0.0842] <-- headline
paired (loop - control), judges.rtdetr.faithfulness.missed_object_rate
  stage 3        n=6  -0.1306  p=0.0312  95% CI [-0.1397, -0.1223] <-- headline
paired (loop - control), judges.rtdetr.faithfulness.detection_consistency
  stage 3        n=6  +0.1001  p=0.0312  95% CI [+0.0925, +0.1070] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.primary_map50 -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0527  loop +0.2158  diff +0.1631  p=0.0312  95% CI [+0.1218, +0.2128]
  stage 2        n=6  control +0.0833  loop +0.2431  diff +0.1598  p=0.0312  95% CI [+0.1255, +0.1952]
  stage 3        n=6  control +0.0752  loop +0.2507  diff +0.1755  p=0.0312  95% CI [+0.1402, +0.2111] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.primary_n_classes -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0000  loop +0.0000  diff +0.0000  p=1.0000  95% CI [+0.0000, +0.0000]
  stage 2        n=6  control +0.0000  loop +0.0000  diff +0.0000  p=1.0000  95% CI [+0.0000, +0.0000]
  stage 3        n=6  control +0.0000  loop +0.0000  diff +0.0000  p=1.0000  95% CI [+0.0000, +0.0000] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.map50 -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0508  loop +0.2167  diff +0.1658  p=0.0312  95% CI [+0.0914, +0.2814]
  stage 2        n=6  control +0.1016  loop +0.2294  diff +0.1278  p=0.0312  95% CI [+0.0900, +0.1662]
  stage 3        n=6  control +0.1002  loop +0.2455  diff +0.1453  p=0.0312  95% CI [+0.1101, +0.1885] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.map50_95 -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0254  loop +0.1183  diff +0.0929  p=0.0312  95% CI [+0.0559, +0.1486]
  stage 2        n=6  control +0.0538  loop +0.1264  diff +0.0726  p=0.0312  95% CI [+0.0541, +0.0929]
  stage 3        n=6  control +0.0546  loop +0.1377  diff +0.0831  p=0.0312  95% CI [+0.0664, +0.1015] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.precision -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0665  loop +0.2406  diff +0.1740  p=0.1250  95% CI [+0.0043, +0.3921]
  stage 2        n=6  control +0.1625  loop +0.2381  diff +0.0756  p=0.1562  95% CI [-0.0008, +0.1530]
  stage 3        n=6  control +0.1659  loop +0.2626  diff +0.0967  p=0.2188  95% CI [-0.0180, +0.2310] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.recall -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0448  loop +0.2027  diff +0.1578  p=0.0312  95% CI [+0.0933, +0.2598]
  stage 2        n=6  control +0.0822  loop +0.2088  diff +0.1266  p=0.0312  95% CI [+0.0963, +0.1658]
  stage 3        n=6  control +0.0761  loop +0.2117  diff +0.1356  p=0.0312  95% CI [+0.1097, +0.1781] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.per_class_ap50.bike -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0449  loop +0.2374  diff +0.1925  p=0.0312  95% CI [+0.1235, +0.2784]
  stage 2        n=6  control +0.0994  loop +0.2228  diff +0.1235  p=0.0312  95% CI [+0.0879, +0.1602]
  stage 3        n=6  control +0.1195  loop +0.2592  diff +0.1397  p=0.0312  95% CI [+0.1079, +0.1742] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.per_class_ap50.car -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0664  loop +0.2143  diff +0.1479  p=0.2500  95% CI [-0.0064, +0.3994]
  stage 2        n=6  control +0.1532  loop +0.2259  diff +0.0727  p=0.2188  95% CI [-0.0047, +0.1528]
  stage 3        n=6  control +0.1582  loop +0.2541  diff +0.0958  p=0.0938  95% CI [+0.0228, +0.1859] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.per_class_ap50.car_stop -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0626  loop +0.2553  diff +0.1927  p=0.0312  95% CI [+0.1378, +0.2736]
  stage 2        n=6  control +0.1045  loop +0.2711  diff +0.1666  p=0.0312  95% CI [+0.1222, +0.2082]
  stage 3        n=6  control +0.0926  loop +0.3057  diff +0.2130  p=0.0312  95% CI [+0.1815, +0.2490] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.per_class_ap50.color_cone -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0427  loop +0.1763  diff +0.1336  p=0.0312  95% CI [+0.0874, +0.1683]
  stage 2        n=6  control +0.0621  loop +0.2151  diff +0.1530  p=0.0312  95% CI [+0.1106, +0.1899]
  stage 3        n=6  control +0.0577  loop +0.1957  diff +0.1380  p=0.0312  95% CI [+0.0945, +0.1769] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.per_class_ap50.person -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0375  loop +0.2000  diff +0.1625  p=0.0312  95% CI [+0.0635, +0.3074]
  stage 2        n=6  control +0.0888  loop +0.2119  diff +0.1231  p=0.0312  95% CI [+0.0776, +0.1719]
  stage 3        n=6  control +0.0731  loop +0.2131  diff +0.1400  p=0.0312  95% CI [+0.0998, +0.1846] <-- headline
trajectory (loop - control) of the gain over stage 0, fidelity.lpips -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control -0.0072  loop -0.0317  diff -0.0245  p=0.9688  95% CI [-0.1317, +0.0380]
  stage 2        n=6  control -0.0349  loop -0.0425  diff -0.0075  p=0.9375  95% CI [-0.0754, +0.0394]
  stage 3        n=6  control -0.0238  loop -0.0550  diff -0.0312  p=0.7188  95% CI [-0.1075, +0.0164] <-- headline
trajectory (loop - control) of the gain over stage 0, fidelity.fid -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control -5.2112  loop -16.9875  diff -11.7763  p=0.9062  95% CI [-46.4643, +9.1821]
  stage 2        n=6  control -16.3118  loop -23.2426  diff -6.9308  p=0.9375  95% CI [-34.9266, +10.7609]
  stage 3        n=6  control -10.9241  loop -24.3294  diff -13.4053  p=0.5938  95% CI [-42.4121, +4.4306] <-- headline
trajectory (loop - control) of the gain over stage 0, fidelity.psnr -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.7196  loop +1.5215  diff +0.8020  p=0.8750  95% CI [-0.9400, +3.5611]
  stage 2        n=6  control +1.7508  loop +1.9620  diff +0.2112  p=0.7188  95% CI [-0.5638, +1.0580]
  stage 3        n=6  control +1.7289  loop +2.1605  diff +0.4316  p=0.6875  95% CI [-0.6048, +1.7439] <-- headline
trajectory (loop - control) of the gain over stage 0, fidelity.ssim -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0326  loop +0.0392  diff +0.0066  p=0.7188  95% CI [-0.0149, +0.0324]
  stage 2        n=6  control +0.0586  loop +0.0615  diff +0.0029  p=0.7188  95% CI [-0.0100, +0.0157]
  stage 3        n=6  control +0.0724  loop +0.0614  diff -0.0110  p=0.4062  95% CI [-0.0311, +0.0085] <-- headline
trajectory (loop - control) of the gain over stage 0, detector.map50 -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control +0.0021  loop +0.0786  diff +0.0765  p=0.0312  95% CI [+0.0227, +0.1735]
  stage 2        n=6  control +0.0224  loop +0.0852  diff +0.0628  p=0.0625  95% CI [+0.0148, +0.1429]
  stage 3        n=6  control +0.0189  loop +0.0745  diff +0.0556  p=0.0625  95% CI [+0.0109, +0.1272] <-- headline

```

**Custom set, pix2pix (e3b)**, the whole log:

```
2026-10-19 20:49:20,661 INFO    t2o.cli: 12 runs, stages [0, 1, 2, 3]
2026-10-19 20:49:20,661 INFO    t2o.cli: metric                                             arm      stage   n      mean       std
2026-10-19 20:49:20,661 INFO    t2o.cli: zero_shot.map50                                    control      0   6    0.7579    0.0372
2026-10-19 20:49:20,661 INFO    t2o.cli: zero_shot.map50                                    loop         0   6    0.7182    0.0907
2026-10-19 20:49:20,661 INFO    t2o.cli: zero_shot.map50                                    control      1   6    0.7519    0.0238
2026-10-19 20:49:20,661 INFO    t2o.cli: zero_shot.map50                                    loop         1   6    0.7799    0.0887
2026-10-19 20:49:20,661 INFO    t2o.cli: zero_shot.map50                                    control      2   6    0.8071    0.0263
2026-10-19 20:49:20,661 INFO    t2o.cli: zero_shot.map50                                    loop         2   6    0.8427    0.0215
2026-10-19 20:49:20,661 INFO    t2o.cli: zero_shot.map50                                    control      3   6    0.7975    0.0330
2026-10-19 20:49:20,664 INFO    t2o.cli: zero_shot.map50                                    loop         3   6    0.8487    0.0140
2026-10-19 20:49:20,664 INFO    t2o.cli: judges.rtdetr.zero_shot.map50                      control      3   6    0.8238    0.0187
2026-10-19 20:49:20,664 INFO    t2o.cli: judges.rtdetr.zero_shot.map50                      loop         3   6    0.8673    0.0137
2026-10-19 20:49:20,665 INFO    t2o.cli: judges.rtdetr.zero_shot.map50_95                   control      3   6    0.5713    0.0199
2026-10-19 20:49:20,665 INFO    t2o.cli: judges.rtdetr.zero_shot.map50_95                   loop         3   6    0.6015    0.0083
2026-10-19 20:49:20,665 INFO    t2o.cli: judges.rtdetr.zero_shot.per_class_ap50.Fuse        control      3   6    0.8276    0.0233
2026-10-19 20:49:20,665 INFO    t2o.cli: judges.rtdetr.zero_shot.per_class_ap50.Fuse        loop         3   6    0.8695    0.0119
2026-10-19 20:49:20,665 INFO    t2o.cli: judges.rtdetr.zero_shot.per_class_ap50.Pole        control      3   6    0.9572    0.0096
2026-10-19 20:49:20,665 INFO    t2o.cli: judges.rtdetr.zero_shot.per_class_ap50.Pole        loop         3   6    0.9585    0.0044
2026-10-19 20:49:20,665 INFO    t2o.cli: judges.rtdetr.zero_shot.per_class_ap50.Switch      control      3   6    0.6394    0.0335
2026-10-19 20:49:20,665 INFO    t2o.cli: judges.rtdetr.zero_shot.per_class_ap50.Switch      loop         3   6    0.7323    0.0464
2026-10-19 20:49:20,665 INFO    t2o.cli: judges.rtdetr.zero_shot.per_class_ap50.Transformer control      3   6    0.8710    0.0536
2026-10-19 20:49:20,666 INFO    t2o.cli: judges.rtdetr.zero_shot.per_class_ap50.Transformer loop         3   6    0.9091    0.0197
2026-10-19 20:49:20,666 INFO    t2o.cli: faithfulness.false_object_rate                     control      3   6    0.1918    0.0245
2026-10-19 20:49:20,666 INFO    t2o.cli: faithfulness.false_object_rate                     loop         3   6    0.1629    0.0169
2026-10-19 20:49:20,666 INFO    t2o.cli: faithfulness.missed_object_rate                    control      3   6    0.1962    0.0226
2026-10-19 20:49:20,666 INFO    t2o.cli: faithfulness.missed_object_rate                    loop         3   6    0.1592    0.0118
2026-10-19 20:49:20,666 INFO    t2o.cli: faithfulness.detection_consistency                 control      3   6    0.7677    0.0220
2026-10-19 20:49:20,666 INFO    t2o.cli: faithfulness.detection_consistency                 loop         3   6    0.7968    0.0091
2026-10-19 20:49:20,667 INFO    t2o.cli: judges.rtdetr.faithfulness.false_object_rate       control      3   6    0.1928    0.0179
2026-10-19 20:49:20,667 INFO    t2o.cli: judges.rtdetr.faithfulness.false_object_rate       loop         3   6    0.1751    0.0132
2026-10-19 20:49:20,667 INFO    t2o.cli: judges.rtdetr.faithfulness.missed_object_rate      control      3   6    0.1659    0.0094
2026-10-19 20:49:20,667 INFO    t2o.cli: judges.rtdetr.faithfulness.missed_object_rate      loop         3   6    0.1359    0.0127
2026-10-19 20:49:20,667 INFO    t2o.cli: judges.rtdetr.faithfulness.detection_consistency   control      3   6    0.8076    0.0075
2026-10-19 20:49:20,667 INFO    t2o.cli: judges.rtdetr.faithfulness.detection_consistency   loop         3   6    0.8384    0.0143
2026-10-19 20:49:20,667 INFO    t2o.cli: paired (loop - control), zero_shot.map50
2026-10-19 20:49:20,668 INFO    t2o.cli:   stage 0 (null) n=6  -0.0397  p=0.4688  95% CI [-0.1265, +0.0373]
2026-10-19 20:49:20,668 INFO    t2o.cli:   stage 1        n=6  +0.0280  p=0.5000  95% CI [-0.0513, +0.0783]
2026-10-19 20:49:20,668 INFO    t2o.cli:   stage 2        n=6  +0.0357  p=0.0625  95% CI [+0.0133, +0.0549]
2026-10-19 20:49:20,668 INFO    t2o.cli:   stage 3        n=6  +0.0512  p=0.0312  95% CI [+0.0248, +0.0809] <-- headline
2026-10-19 20:49:20,668 INFO    t2o.cli: paired (loop - control), judges.rtdetr.zero_shot.map50
2026-10-19 20:49:20,668 INFO    t2o.cli:   stage 3        n=6  +0.0435  p=0.0312  95% CI [+0.0212, +0.0653] <-- headline
2026-10-19 20:49:20,668 INFO    t2o.cli: paired (loop - control), judges.rtdetr.zero_shot.map50_95
2026-10-19 20:49:20,672 INFO    t2o.cli:   stage 3        n=6  +0.0302  p=0.0625  95% CI [+0.0114, +0.0484] <-- headline
2026-10-19 20:49:20,673 INFO    t2o.cli: paired (loop - control), judges.rtdetr.zero_shot.per_class_ap50.Fuse
2026-10-19 20:49:20,673 INFO    t2o.cli:   stage 3        n=6  +0.0419  p=0.0312  95% CI [+0.0208, +0.0640] <-- headline
2026-10-19 20:49:20,673 INFO    t2o.cli: paired (loop - control), judges.rtdetr.zero_shot.per_class_ap50.Pole
2026-10-19 20:49:20,673 INFO    t2o.cli:   stage 3        n=6  +0.0013  p=0.7500  95% CI [-0.0046, +0.0074] <-- headline
2026-10-19 20:49:20,673 INFO    t2o.cli: paired (loop - control), judges.rtdetr.zero_shot.per_class_ap50.Switch
2026-10-19 20:49:20,673 INFO    t2o.cli:   stage 3        n=6  +0.0929  p=0.0625  95% CI [+0.0325, +0.1444] <-- headline
2026-10-19 20:49:20,674 INFO    t2o.cli: paired (loop - control), judges.rtdetr.zero_shot.per_class_ap50.Transformer
2026-10-19 20:49:20,674 INFO    t2o.cli:   stage 3        n=6  +0.0381  p=0.2500  95% CI [-0.0063, +0.0831] <-- headline
2026-10-19 20:49:20,674 INFO    t2o.cli: paired (loop - control), faithfulness.false_object_rate
2026-10-19 20:49:20,674 INFO    t2o.cli:   stage 3        n=6  -0.0289  p=0.1562  95% CI [-0.0559, +0.0039] <-- headline
2026-10-19 20:49:20,674 INFO    t2o.cli: paired (loop - control), faithfulness.missed_object_rate
2026-10-19 20:49:20,674 INFO    t2o.cli:   stage 3        n=6  -0.0370  p=0.0312  95% CI [-0.0532, -0.0209] <-- headline
2026-10-19 20:49:20,674 INFO    t2o.cli: paired (loop - control), faithfulness.detection_consistency
2026-10-19 20:49:20,674 INFO    t2o.cli:   stage 3        n=6  +0.0291  p=0.0312  95% CI [+0.0121, +0.0450] <-- headline
2026-10-19 20:49:20,674 INFO    t2o.cli: paired (loop - control), judges.rtdetr.faithfulness.false_object_rate
2026-10-19 20:49:20,674 INFO    t2o.cli:   stage 3        n=6  -0.0177  p=0.1562  95% CI [-0.0324, +0.0014] <-- headline
2026-10-19 20:49:20,674 INFO    t2o.cli: paired (loop - control), judges.rtdetr.faithfulness.missed_object_rate
2026-10-19 20:49:20,674 INFO    t2o.cli:   stage 3        n=6  -0.0299  p=0.0312  95% CI [-0.0394, -0.0173] <-- headline
2026-10-19 20:49:20,674 INFO    t2o.cli: paired (loop - control), judges.rtdetr.faithfulness.detection_consistency
2026-10-19 20:49:20,674 INFO    t2o.cli:   stage 3        n=6  +0.0309  p=0.0312  95% CI [+0.0200, +0.0418] <-- headline
2026-10-19 20:49:20,674 INFO    t2o.cli: trajectory (loop - control) of the gain over stage 0, zero_shot.map50 -- sensitivity analysis, not the pre-registered endpoint
2026-10-19 20:49:20,675 INFO    t2o.cli:   stage 1        n=6  control -0.0060  loop +0.0617  diff +0.0677  p=0.3125  95% CI [-0.0607, +0.1842]
2026-10-19 20:49:20,675 INFO    t2o.cli:   stage 2        n=6  control +0.0491  loop +0.1245  diff +0.0754  p=0.1562  95% CI [+0.0031, +0.1596]
2026-10-19 20:49:20,675 INFO    t2o.cli:   stage 3        n=6  control +0.0396  loop +0.1305  diff +0.0909  p=0.0938  95% CI [+0.0192, +0.1812] <-- headline
2026-10-19 20:49:20,680 INFO    t2o.cli: wrote runs\e3b-judge-tidy.csv
```

**Custom set, turbo (e3t)**, the whole log:

```
2026-10-19 20:50:24,914 INFO    t2o.cli: 6 runs, stages [0, 1, 2, 3]
2026-10-19 20:50:24,914 INFO    t2o.cli: metric                                             arm      stage   n      mean       std
2026-10-19 20:50:24,914 INFO    t2o.cli: zero_shot.map50                                    control      0   3    0.8599    0.0293
2026-10-19 20:50:24,914 INFO    t2o.cli: zero_shot.map50                                    loop         0   3    0.8576    0.0286
2026-10-19 20:50:24,923 INFO    t2o.cli: zero_shot.map50                                    control      1   3    0.8474    0.0093
2026-10-19 20:50:24,924 INFO    t2o.cli: zero_shot.map50                                    loop         1   3    0.8748    0.0093
2026-10-19 20:50:24,924 INFO    t2o.cli: zero_shot.map50                                    control      2   3    0.8439    0.0138
2026-10-19 20:50:24,924 INFO    t2o.cli: zero_shot.map50                                    loop         2   3    0.8674    0.0070
2026-10-19 20:50:24,925 INFO    t2o.cli: zero_shot.map50                                    control      3   3    0.8326    0.0161
2026-10-19 20:50:24,925 INFO    t2o.cli: zero_shot.map50                                    loop         3   3    0.8632    0.0115
2026-10-19 20:50:24,925 INFO    t2o.cli: judges.rtdetr.zero_shot.map50                      control      3   3    0.8456    0.0149
2026-10-19 20:50:24,925 INFO    t2o.cli: judges.rtdetr.zero_shot.map50                      loop         3   3    0.8760    0.0165
2026-10-19 20:50:24,925 INFO    t2o.cli: judges.rtdetr.zero_shot.map50_95                   control      3   3    0.5897    0.0161
2026-10-19 20:50:24,925 INFO    t2o.cli: judges.rtdetr.zero_shot.map50_95                   loop         3   3    0.6146    0.0118
2026-10-19 20:50:24,926 INFO    t2o.cli: judges.rtdetr.zero_shot.per_class_ap50.Fuse        control      3   3    0.8672    0.0151
2026-10-19 20:50:24,926 INFO    t2o.cli: judges.rtdetr.zero_shot.per_class_ap50.Fuse        loop         3   3    0.8773    0.0228
2026-10-19 20:50:24,926 INFO    t2o.cli: judges.rtdetr.zero_shot.per_class_ap50.Pole        control      3   3    0.9586    0.0124
2026-10-19 20:50:24,926 INFO    t2o.cli: judges.rtdetr.zero_shot.per_class_ap50.Pole        loop         3   3    0.9522    0.0119
2026-10-19 20:50:24,926 INFO    t2o.cli: judges.rtdetr.zero_shot.per_class_ap50.Switch      control      3   3    0.6607    0.0164
2026-10-19 20:50:24,926 INFO    t2o.cli: judges.rtdetr.zero_shot.per_class_ap50.Switch      loop         3   3    0.7269    0.0652
2026-10-19 20:50:24,927 INFO    t2o.cli: judges.rtdetr.zero_shot.per_class_ap50.Transformer control      3   3    0.8957    0.0461
2026-10-19 20:50:24,927 INFO    t2o.cli: judges.rtdetr.zero_shot.per_class_ap50.Transformer loop         3   3    0.9475    0.0124
2026-10-19 20:50:24,927 INFO    t2o.cli: faithfulness.false_object_rate                     control      3   3    0.1836    0.0250
2026-10-19 20:50:24,927 INFO    t2o.cli: faithfulness.false_object_rate                     loop         3   3    0.1642    0.0268
2026-10-19 20:50:24,927 INFO    t2o.cli: faithfulness.missed_object_rate                    control      3   3    0.1552    0.0181
2026-10-19 20:50:24,927 INFO    t2o.cli: faithfulness.missed_object_rate                    loop         3   3    0.1166    0.0098
2026-10-19 20:50:24,928 INFO    t2o.cli: faithfulness.detection_consistency                 control      3   3    0.8064    0.0255
2026-10-19 20:50:24,928 INFO    t2o.cli: faithfulness.detection_consistency                 loop         3   3    0.8383    0.0064
2026-10-19 20:50:24,928 INFO    t2o.cli: judges.rtdetr.faithfulness.false_object_rate       control      3   3    0.1977    0.0315
2026-10-19 20:50:24,928 INFO    t2o.cli: judges.rtdetr.faithfulness.false_object_rate       loop         3   3    0.2017    0.0289
2026-10-19 20:50:24,928 INFO    t2o.cli: judges.rtdetr.faithfulness.missed_object_rate      control      3   3    0.1434    0.0112
2026-10-19 20:50:24,929 INFO    t2o.cli: judges.rtdetr.faithfulness.missed_object_rate      loop         3   3    0.1261    0.0119
2026-10-19 20:50:24,929 INFO    t2o.cli: judges.rtdetr.faithfulness.detection_consistency   control      3   3    0.8344    0.0038
2026-10-19 20:50:24,929 INFO    t2o.cli: judges.rtdetr.faithfulness.detection_consistency   loop         3   3    0.8613    0.0045
2026-10-19 20:50:24,929 INFO    t2o.cli: paired (loop - control), zero_shot.map50
2026-10-19 20:50:24,930 INFO    t2o.cli:   stage 0 (null) n=3  -0.0023  p=1.0000  95% CI [-0.0601, +0.0276]
2026-10-19 20:50:24,930 INFO    t2o.cli:   stage 1        n=3  +0.0274  p=0.2500  95% CI [+0.0100, +0.0383]
2026-10-19 20:50:24,930 INFO    t2o.cli:   stage 2        n=3  +0.0235  p=0.2500  95% CI [+0.0048, +0.0359]
2026-10-19 20:50:24,931 INFO    t2o.cli:   stage 3        n=3  +0.0306  p=0.2500  95% CI [+0.0121, +0.0459] <-- headline
2026-10-19 20:50:24,931 INFO    t2o.cli: paired (loop - control), judges.rtdetr.zero_shot.map50
2026-10-19 20:50:24,931 INFO    t2o.cli:   stage 3        n=3  +0.0304  p=0.2500  95% CI [+0.0023, +0.0525] <-- headline
2026-10-19 20:50:24,931 INFO    t2o.cli: paired (loop - control), judges.rtdetr.zero_shot.map50_95
2026-10-19 20:50:24,932 INFO    t2o.cli:   stage 3        n=3  +0.0249  p=0.2500  95% CI [+0.0111, +0.0404] <-- headline
2026-10-19 20:50:24,932 INFO    t2o.cli: paired (loop - control), judges.rtdetr.zero_shot.per_class_ap50.Fuse
2026-10-19 20:50:24,932 INFO    t2o.cli:   stage 3        n=3  +0.0100  p=0.7500  95% CI [-0.0187, +0.0365] <-- headline
2026-10-19 20:50:24,932 INFO    t2o.cli: paired (loop - control), judges.rtdetr.zero_shot.per_class_ap50.Pole
2026-10-19 20:50:24,933 INFO    t2o.cli:   stage 3        n=3  -0.0064  p=0.7500  95% CI [-0.0204, +0.0072] <-- headline
2026-10-19 20:50:24,933 INFO    t2o.cli: paired (loop - control), judges.rtdetr.zero_shot.per_class_ap50.Switch
2026-10-19 20:50:24,933 INFO    t2o.cli:   stage 3        n=3  +0.0662  p=0.5000  95% CI [-0.0229, +0.1156] <-- headline
2026-10-19 20:50:24,933 INFO    t2o.cli: paired (loop - control), judges.rtdetr.zero_shot.per_class_ap50.Transformer
2026-10-19 20:50:24,933 INFO    t2o.cli:   stage 3        n=3  +0.0518  p=0.5000  95% CI [-0.0042, +0.0881] <-- headline
2026-10-19 20:50:24,934 INFO    t2o.cli: paired (loop - control), faithfulness.false_object_rate
2026-10-19 20:50:24,934 INFO    t2o.cli:   stage 3        n=3  -0.0193  p=0.5000  95% CI [-0.0531, +0.0209] <-- headline
2026-10-19 20:50:24,934 INFO    t2o.cli: paired (loop - control), faithfulness.missed_object_rate
2026-10-19 20:50:24,934 INFO    t2o.cli:   stage 3        n=3  -0.0386  p=0.2500  95% CI [-0.0662, -0.0236] <-- headline
2026-10-19 20:50:24,935 INFO    t2o.cli: paired (loop - control), faithfulness.detection_consistency
2026-10-19 20:50:24,935 INFO    t2o.cli:   stage 3        n=3  +0.0319  p=0.2500  95% CI [+0.0128, +0.0574] <-- headline
2026-10-19 20:50:24,935 INFO    t2o.cli: paired (loop - control), judges.rtdetr.faithfulness.false_object_rate
2026-10-19 20:50:24,935 INFO    t2o.cli:   stage 3        n=3  +0.0040  p=0.7500  95% CI [-0.0461, +0.0538] <-- headline
2026-10-19 20:50:24,935 INFO    t2o.cli: paired (loop - control), judges.rtdetr.faithfulness.missed_object_rate
2026-10-19 20:50:24,936 INFO    t2o.cli:   stage 3        n=3  -0.0173  p=0.2500  95% CI [-0.0284, -0.0024] <-- headline
2026-10-19 20:50:24,936 INFO    t2o.cli: paired (loop - control), judges.rtdetr.faithfulness.detection_consistency
2026-10-19 20:50:24,936 INFO    t2o.cli:   stage 3        n=3  +0.0269  p=0.2500  95% CI [+0.0240, +0.0305] <-- headline
2026-10-19 20:50:24,936 INFO    t2o.cli: trajectory (loop - control) of the gain over stage 0, zero_shot.map50 -- sensitivity analysis, not the pre-registered endpoint
2026-10-19 20:50:24,936 INFO    t2o.cli:   stage 1        n=3  control -0.0124  loop +0.0172  diff +0.0297  p=0.2500  95% CI [+0.0063, +0.0701]
2026-10-19 20:50:24,937 INFO    t2o.cli:   stage 2        n=3  control -0.0159  loop +0.0098  diff +0.0258  p=0.2500  95% CI [+0.0042, +0.0648]
2026-10-19 20:50:24,937 INFO    t2o.cli:   stage 3        n=3  control -0.0273  loop +0.0056  diff +0.0329  p=0.7500  95% CI [-0.0135, +0.1060] <-- headline
2026-10-19 20:50:24,943 INFO    t2o.cli: wrote runs\e3t-judge-tidy.csv
```

Sections 2, 3, 5, 7 and 8 of the two reports cover the translator campaigns. They are unchanged
from records 016 and 024 and are in the logs.

Re-render without re-running (on the server): the step-6 commands above. They only read
`metrics.json`.

## Findings

### F181 — Every RT-DETR-l judge clears the 90% gate, by at least 2.8× the smoke spread, and matches yolo11s on each dataset's primary classes

The gate compares visible-val mAP50 for the RT-DETR judge against the yolo11s reference, both
scored by the same command:

| dataset | yolo11s | RT-DETR-l | ratio | 90% line | margin over the line |
| --- | --- | --- | --- | --- | --- |
| FLIR | 0.5523 | 0.6009 | 108.8% | 0.4971 | +0.1038 |
| msrs-day | 0.7841 | 0.7979 | 101.8% | 0.7057 | +0.0922 |
| custom | 0.9366 | 0.9368 | 100.0% | 0.8429 | +0.0939 |

- **Every judge passes.** The smallest margin, msrs-day's +0.0922, is 2.8× F188's 0.0330
  same-seed spread, so no judge sits where a training redraw could plausibly flip the verdict.
- **FLIR's all-class lead is dog**, at 0.4522 against 0.2396. On the three primary classes the
  judge is level with yolo11s: 0.6505 against 0.6566 (99.1%).
- **msrs-day** on the two passive classes: 0.8274 against 0.8237.

### F182 — Transfer holds on all four cells: RT-DETR sees a stage-3 loop − control gain on every pre-registered endpoint, at the sign-flip floor, keeping 85–99% of the yolo11s gain

The paired stage-3 difference under the RT-DETR judge, against the yolo11s difference on the same
exports:

| cell | endpoint | RT-DETR | p | 95% CI | yolo11s | RT / yolo |
| --- | --- | --- | --- | --- | --- | --- |
| e3b (custom, pix2pix, n = 6) | all-class mAP50 | +0.0435 | .031 | [+0.0212, +0.0653] | +0.0512 | 85% |
| e3t (custom, turbo, n = 3) | all-class mAP50 | +0.0304 | .250 | [+0.0023, +0.0525] | +0.0306 | 99% |
| e3f (FLIR, n = 6) | 3-class mAP50 | +0.3051 | .031 | [+0.2228, +0.3871] | +0.3512 | 87% |
| e3md (msrs-day, n = 6) | 2-class mAP50 | +0.1451 | .031 | [+0.1395, +0.1504] | +0.1672 | 87% |

- **Every p is its cell's floor**: .031 at n = 6, and .250 at n = 3. Every bootstrap CI lies above
  zero. e3t meets goal.md's "≥3 seeds" exactly, so it cannot go below p = .25.
- **The gain is smaller under RT-DETR on every cell**, by 1–15%. One judge seed (plan Q3) cannot
  separate a family-specific share from judge-to-judge noise.
- **Per class it holds too.** msrs-day car_stop +0.1692 and color_cone +0.1210, both p = .031.
  Every FLIR per-class difference is p = .031 except dog: +0.0170, p = .531, the class section 8
  of the report pre-registered as noise.
- **The Switch difference matches F45**: +0.0929 (p = .0625) under RT-DETR, against F45's +0.1035
  (p = .094).

### F183 — On FLIR, RT-DETR lifts both arms and the worst loop seed still beats the best control seed; the raw-thermal floor under RT-DETR is unmeasured

3-class stage-3 mAP50 on FLIR, by judge:

| arm | yolo11s | RT-DETR-l |
| --- | --- | --- |
| control (mean) | 0.0902 | 0.2247 |
| loop (mean) | 0.4415 | 0.5298 |

- **The collapsed controls move most.** The two control seeds yolo11s scores as collapsed, s3 at
  0.0002 and s5 at 0.0037, score 0.0770 and 0.1211 under RT-DETR.
- **The worst loop seed beats the best control seed**: 0.5035 (s5) against 0.3866 (s0). This is
  F139's ordering under a second family.
- **Against each judge's own real-visible ceiling**, the loop reaches 81% (0.5298 / 0.6505) under
  RT-DETR, against 67% (0.4415 / 0.6566) under yolo11s.
- **F137's reading cannot be re-run under this judge.** That reading has the loop landing on the
  raw-thermal floor, 0.4415 against 0.4499. RT-DETR never scored raw thermal, so whether its loop
  score clears its own floor is open.

### F184 — Faithfulness under RT-DETR: no cell has a rate significantly worse for the loop, and the per-cell pattern reproduces under yolo11s

Paired stage-3 loop − control under RT-DETR (false-object, missed-object, consistency):

| cell | false-object | missed-object | consistency |
| --- | --- | --- | --- |
| e3b | −0.0177 (p = .156) | −0.0299 (.031) | +0.0309 (.031) |
| e3t | +0.0040 (p = .750) | −0.0173 (.250) | +0.0269 (.250) |
| e3f | −0.1487 (.031) | −0.3359 (.031) | +0.1997 (.031) |
| e3md | −0.0989 (.031) | −0.1306 (.031) | +0.1001 (.031) |

- **e3b repeats F50's pattern.** Its false-object rate is the one rate not at the floor, −0.0177
  (p = .156) here against −0.0289 (p = .156) under yolo11s.
- **FLIR and msrs-day favour the loop on all three rates**, as F143 and F171 found under yolo11s.
- **e3t's false-object rate is the only one pointing against the loop.** It is +0.0040, with CI
  [−0.0461, +0.0538], p = .75, at n = 3.

### F185 — Absolute faithfulness rates depend on the judge, in both directions: on the same exports RT-DETR's loop false-object rate is 3.3× yolo11s's on FLIR, and its missed-object rate is lower

Stage-3 loop-arm means on the same exports:

| cell | false, yolo11s | false, RT-DETR | missed, yolo11s | missed, RT-DETR |
| --- | --- | --- | --- | --- |
| e3b | 0.1629 | 0.1751 | 0.1592 | 0.1359 |
| e3f | 0.1874 | 0.6171 | 0.5671 | 0.3329 |
| e3md | 0.3328 | 0.5414 | 0.4307 | 0.4088 |

- **The gap is large on the public sets.** On FLIR the false-object rate is 3.3× yolo11s's, and on
  msrs-day 1.6×. On the custom set the two judges are within 0.025.
- **What follows:**
  - An absolute "low" bar, or a margin fixed under one judge, does not carry over to another.
  - Paired differences are comparable across judges. Levels are not.
- **Open.** RT-DETR's own rates on real visible frames were not measured, so a 0.62 FLIR
  false-object rate cannot yet be split between the judge and the translation.
- **A hypothesis, not tested:** RT-DETR runs at a different operating point under the same
  faithfulness settings, with more boxes, so more false and fewer missed.

### F186 — The control arm's seed-to-seed sd under RT-DETR (plan Q4's margin rule) is 0.004–0.032 on every rate except FLIR's missed-object (0.1708) and consistency (0.1222)

Stage-3 control-arm std, as `t2o aggregate` prints it:

| cell (n) | false-object | missed-object | consistency |
| --- | --- | --- | --- |
| e3b (6) | 0.0179 | 0.0094 | 0.0075 |
| e3t (3) | 0.0315 | 0.0112 | 0.0038 |
| e3f (6) | 0.0169 | 0.1708 | 0.1222 |
| e3md (6) | 0.0171 | 0.0143 | 0.0107 |

- **FLIR's missed-object sd is 12× msrs-day's** (0.1708 against 0.0143), and its consistency sd 11×.
  - The two seeds yolo11s scores as collapsed, s3 and s5, sit at 0.884 and 0.822 missed-object.
  - The other four span 0.452–0.732.
  - The FLIR control is a mix of working and collapsed translators, and a margin taken from its sd
    is about ten times looser there than anywhere else.
- **e3t has three control seeds, not six**, so Q4's "six val seeds" cannot be met on turbo.

### F187 — The FLIR RT-DETR judge is its epoch-1 checkpoint: training went NaN at epoch 47, scored 0 on every later epoch, and mAP50 only fell before that

- **The best epoch was the first.** Epoch 1 validates at 0.601 mAP50 / 0.278 mAP50-95. No later
  epoch reaches either number.
- **mAP50 drifts down until the collapse**: 0.551 at epoch 3, and 0.501–0.504 at epochs 45–46.
- **Epoch 47 is the first with NaN losses**, and every validation from 47 to 100 is 0 / 0 / 0 / 0.
  60 epoch lines carry NaN losses.
- **ultralytics' "Loss NaN/Inf detected … recovering from last.pt" fired 24 times**, the first at
  epoch 76. None restored a non-zero validation.
- **`best.pt` is the epoch-1 checkpoint**: 0.6010 / 0.2782 at the final validation. The FLIR judge
  is therefore COCO-pretrained RT-DETR-l plus one epoch on FLIR visible. 4.57 h of training bought
  nothing past the first 2–3 minutes.
- **The other two judges trained without a NaN**: msrs-day 0.7945 and custom 0.9366 at final
  validation.
- **This matches FLIR's own history.** FLIR's yolo11s judge also peaked early (epoch 46, −8.1% by
  the end, per F163). FLIR visible training degrades both families after their early epochs.

### F188 — RT-DETR training is not reproducible under `deterministic=True`: the same command and seed on the same card give 0.6798 and 0.6468 mAP50 after 3 epochs

- **Setup**: `smoke-a` and `smoke-b`, both msrs-day visible, seed 1, `cuda:0`, 3 epochs.
- **The losses differ from epoch 1**: giou 1.034 against 1.008, and cls 2.037 against 2.042.
- **The checkpoints differ by 0.0330 mAP50**: `best.pt` scores 0.6798 against 0.6468, and
  0.4103 against 0.3926 mAP50-95.
- **The warning names the cause.** Each epoch's backward pass warns that
  `grid_sampler_2d_backward_cuda does not have a deterministic implementation`.
- **What follows:**
  - Each judge checkpoint cannot be reproduced; a retrain is a different judge.
  - With one judge seed (Q3), there is a judge-training spread that at 3 epochs is about 0.03
    mAP50, unmeasured at 100.

## Provenance caveats

- **No log carries a provenance block.** The commit, command and machine above come from the
  runbook, the readouts' provenance sections and the user's account on 2026-10-20. None was printed
  by the smoke, training or re-score runs themselves. Python, torch and ultralytics versions are
  the ones every log prints.
- **The judges scored stage 3 only.** There is no stage-0 null under RT-DETR. F182 leans on each
  cell's yolo11s stage-0 null having held (F35, F72, F136, F168), not on a null of its own.
- **One judge seed per dataset** (plan Q3), and F188 shows a retrain would not reproduce it.
- **The FLIR judge is its epoch-1 checkpoint** (F187). It passed the gate, so it scores, but it is
  not "a judge trained for 100 epochs" as the plan read.
- **Re-score timing.** Both re-score windows went silent at about the same time and resumed within
  10 s of each other at 20:02:
  - FLIR's window wrote `e3f-control-s5` at 17:00:52 and `e3f-loop-s5` at 20:02:20.
  - The other window finished e3b at 17:19:35 and logged e3t's first line at 20:02:11.

  Nothing is logged in the gap. Every run has both write-backs, and the scores are unaffected.
  The user reports that the server's terminals, reached over Remote Desktop, sometimes freeze until
  Enter is pressed in them. That fits both windows stalling together. Windows console QuickEdit
  mode, where a click starts a selection that blocks console writes, is a likely mechanism, but it is
  not confirmed.
- **The training logs are named `-rtdert-l`**, not the runbook's `-rtdetr-l`. The two tidy CSVs
  sit in `logs/` as `2026-10-19-runs-*-judge-tidy.csv`, outside the
  `<date>-<task>-<what>.txt` convention.
- **ANSI colour escapes were stripped** from the quoted log blocks.
- **The custom set's data path** (`D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul`) sits
  outside the repo. It was read from the yolo11s reference's `args.yaml`, as record 004's judge
  used it, and it has no committed split file.

## Next

- **ADR-01 (OUT-OF-FAMILY-JUDGE-06) now has its inputs**: F186's control-sd table, the gate verdicts
  in F181, and the fact that the test split is still unread. Two calls it has to face:
  - FLIR's 0.17 missed-object sd (F186) makes a margin there about ten times looser than on the
    other datasets.
  - Turbo has only three control seeds.
- **Prediction, posted now.** If RT-DETR scores raw FLIR thermal, its 3-class floor will sit above
  yolo11s's 0.4499. That is the same lift F183 shows on the translated control. Whether the loop's
  0.5298 clears that floor is the open half of F137.
- **The same pass on real visible frames** gives RT-DETR's own faithfulness rates, which F185
  needs before any absolute rate is read as the translation's.
