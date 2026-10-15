# 024 — msrs-day twelve-run cell: the loop beats its control by +0.167 on the passive classes and stays 0.204 below the thermal-trained detector

**Date:** 2026-10-15 (readout; the launch date is not recorded) · **Task:** MSRS-DAY-CAMPAIGN-01 · **Machine:** VSIBMEOT05 (Windows Server 2022 Standard, AMD EPYC 7763 64 cores, 128 GB RAM), 2× A100-SXM4-40GB, driver 582.16 — two-up, one run per card, each pair on one card (s0–s2 `cuda:0`, s3–s5 `cuda:1`) · **Wall clock:** 101.6 GPU-h summed over the twelve spans; a run 7.98–8.40 h control, 8.50–8.87 h loop; the C2 pass 5.5 min (09:12:48 → 09:18:17)
**git_sha:** launch not recorded; the report ran at `3f1394a` (`git status`: two untracked files, `probe-report.txt` and `splits/yolo_rgbt_29_jul.json`), and a machine snapshot an hour after readout reads `pre-spec-migration-99-g3f1394a`, no `-dirty` · **W&B:** project `thermal-to-optical`, group `e3-msrs-day` — read off the resolved config (`runtime.wandb: True` on all twelve), not checked in the W&B UI · **Log:** `logs/2026-10-15-msrs-day-campaign-report.txt` (the report); `logs/2026-10-15-msrs-day-campaign-faithfulness.txt` (C2). The twelve run logs were not passed in.

## Command

**The launch: not recorded.** The twelve run logs carry the provenance blocks and were not passed to
this record. What was handed over for the launch is the runbook below, which is not a transcript of
what was typed. The twelve `config.yaml` snapshots agree with it on every key, with one addition:
`runtime.wandb` is `True` on all twelve, so `--wandb` was added to the runbook's flag list.

```powershell
$DATA   = "dataset/processed/msrs-day/data.yaml"
$JUDGE  = "runs/reference-msrs-day-yolo11s-s1/weights/best.pt"
$INLOOP = "runs/inloop-msrs-day-yolo11n/weights/best.pt"
# Start-Cell <seeds> <device>: per seed, control then loop
uv run t2o loop --config "experiments/e3_pix2pix_$arm.yaml" `
    --data $DATA --in-loop-weights $INLOOP --eval-init-weights $INLOOP `
    --reference-weights $JUDGE --seed $s --workers 8 `
    --name "e3md-$arm-s$s" --group e3-msrs-day --resume --device $Device
# window A: Start-Cell 0,1,2 cuda:0    window B: Start-Cell 3,4,5 cuda:1
```

The readout, as handed over:

```powershell
foreach ($arm in 'control','loop') { foreach ($s in 0,1,2,3,4,5) {
  uv run t2o faithfulness --translated "runs/e3md-$arm-s$s/stage3/translated" `
    --data $DATA --weights $JUDGE --write-back --device cuda:0 2>&1 |
  Tee-Object -Append "logs/$d-msrs-day-campaign-faithfulness.txt"
} }

uv run python scripts/campaign_report.py --campaign msrs-day `
    --runs 'runs/e3md-control-s*' 'runs/e3md-loop-s*' --stage 3 `
    --primary-classes car_stop color_cone --data $DATA 2>&1 |
  Tee-Object "logs/$d-msrs-day-campaign-report.txt"
```

## Environment

No `=== PROVENANCE ===` block: neither readout log carries one, and the run logs were not passed in.
Two snapshots stand in. Both describe the machine **after** the cell, not at launch.

A machine snapshot captured on the server at 2026-10-15 10:29:27 +03:00, about an hour after the
report (06:30:05Z = 09:30:05 +03:00). It was pasted into the session, not saved under `logs/`.
Verbatim:

```
captured: 2026-10-15 10:29:27 +03:00
hostname: VSIBMEOT05
os: Microsoft Windows Server 2022 Standard 10.0.20348 (build 20348)
cpu: AMD EPYC 7763 64-Core Processor
cpu_logical_cores: 64
ram_total: 128.0 GB
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16; NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16
cuda_driver: 13.0
git_sha: pre-spec-migration-99-g3f1394a
working_dir: D:\Atick\GitHub\Thermal-To-Optical-Translation
```

The report's own header and provenance section, verbatim:

```
CAMPAIGN REPORT -- runs/e3md-control-s*, runs/e3md-loop-s*

==============================================================================
== 1. PROVENANCE
==============================================================================
generated      2026-10-15T06:30:05Z
host           VSIBMEOT05  (Windows 2022Server)
git HEAD       3f1394a20d860a68471cd772a4215484cef6339a
git status     ?? probe-report.txt
?? splits/yolo_rgbt_29_jul.json
gpus           NVIDIA A100-SXM4-40GB
NVIDIA A100-SXM4-40GB
runs matched   12
data           dataset\processed\msrs-day\data.yaml
  path:        D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\msrs-day
  names:       ['car', 'person', 'bike', 'car_stop', 'color_cone']   <- the per-class key spelling
  train/val:   train/visible/images | val/visible/images
  train labels: absent
  val labels: absent
```

## Configuration

- **Twelve runs**: `experiments/e3_pix2pix_{control,loop}.yaml` × seeds 0–5. Control at
  `task_weights` 0/0/0/0, loop at 0/1/2/3, `grad_scale: 0.15` unchanged from the custom set and
  FLIR (record 023's Next). 100 epochs a stage. All 536 msrs-day train pairs, 179 val pairs, 5
  classes.
- **Three detector roles.** In-loop and eval-init: a 5-class `yolo11n` trained on msrs-day visible.
  Judge: `reference-msrs-day-yolo11s-s1`, which is V's seed 1 from record 021 (all-class 0.7841 there).
- **The endpoint, fixed before the cell launched**, is section 8 below, printed by the report
  from `campaign_report.py`'s `MSRS_DAY_PRE_REGISTERED` (committed in 5324522, before launch).
- **C2 (faithfulness) under the judge**, written back into `metrics.json` before the report ran, so
  its paired test is the same one the headline uses. Stage 3 only, as on FLIR.

The report's confound check and resolved config, verbatim:

```
==============================================================================
== 3. CONFIG VARIANCE -- the confound check
==============================================================================
keys whose value differs across the campaign, grouped by value:
  coupling.task_weights
      [0.0, 0.0, 0.0, 0.0]         e3md-control-s0 e3md-control-s1 e3md-control-s2 e3md-control-s3 e3md-control-s4 e3md-control-s5
      [0.0, 1.0, 2.0, 3.0]         e3md-loop-s0 e3md-loop-s1 e3md-loop-s2 e3md-loop-s3 e3md-loop-s4 e3md-loop-s5
  runtime.device
      'cuda:0'                     e3md-control-s0 e3md-control-s1 e3md-control-s2 e3md-loop-s0 e3md-loop-s1 e3md-loop-s2
      'cuda:1'                     e3md-control-s3 e3md-control-s4 e3md-control-s5 e3md-loop-s3 e3md-loop-s4 e3md-loop-s5
  runtime.name
      12 distinct values, one per run
  train.seed
      0                            e3md-control-s0 e3md-loop-s0
      1                            e3md-control-s1 e3md-loop-s1
      2                            e3md-control-s2 e3md-loop-s2
      3                            e3md-control-s3 e3md-loop-s3
      4                            e3md-control-s4 e3md-loop-s4
      5                            e3md-control-s5 e3md-loop-s5

verdict: 0 unexpected varying key(s): none
pairs split across cards: none (a pair must share one card, or the GPU difference lands in the paired contrast)

fully resolved config of e3md-control-s0 (the reconstructed launch):
  coupling.grad_scale                          0.15
  coupling.reward_target                       None
  coupling.task_weights                        [0.0, 0.0, 0.0, 0.0]
  data.annotation_fraction                     1.0
  data.annotation_seed                         0
  data.manifest                                'dataset\\processed\\msrs-day\\data.yaml'
  data.max_train_images                        None
  data.subset_seed                             0
  data.val_loss_images                         None
  detector.evaluation.batch                    16
  detector.evaluation.epochs                   50
  detector.evaluation.imgsz                    640
  detector.evaluation.init_weights             'runs\\inloop-msrs-day-yolo11n\\weights\\best.pt'
  detector.in_loop.imgsz                       640
  detector.in_loop.weights                     'runs\\inloop-msrs-day-yolo11n\\weights\\best.pt'
  detector.reference.batch                     16
  detector.reference.imgsz                     640
  detector.reference.weights                   'runs\\reference-msrs-day-yolo11s-s1\\weights\\best.pt'
  export.normalize                             'clamp'
  loss.gan                                     1.0
  loss.l2                                      1.0
  loss.lpips                                   5.0
  metrics.conf_threshold                       0.25
  metrics.iou_threshold                        0.5
  metrics.kid_subset_size                      50
  metrics.lpips_net                            'alex'
  runtime.device                               'cuda:0'
  runtime.group                                'e3-msrs-day'
  runtime.name                                 'e3md-control-s0'
  runtime.run_dir                              'runs'
  runtime.wandb                                True
  runtime.wandb_project                        'thermal-to-optical'
  runtime.workers                              8
  train.amp                                    True
  train.amp_dtype                              'bfloat16'
  train.batch_size                             8
  train.crop                                   None
  train.epochs_per_stage                       100
  train.hflip                                  0.5
  train.lr                                     0.0002
  train.lr_gamma                               0.95
  train.seed                                   0
  translator.backbone                          'pix2pix'
  translator.gan_mode                          'vanilla'
  translator.ndf                               64
  translator.net_d                             'basic'
  translator.net_g                             'resnet_9blocks'
  translator.ngf                               64
```

The pre-registration as the report printed it, verbatim:

```
==============================================================================
== 8. PRE-REGISTERED READINGS -- written before these numbers
==============================================================================
primary endpoint   paired stage-3 difference in the 2-class mAP50 (car_stop/color_cone),
                   exact two-sided sign-flip p over 2^6 assignments. Never a bootstrap CI
                   excluding zero (PLAN.md º12); CIs are descriptive spread only.
null control       the stage-0 paired difference -- lambda = 0 in BOTH arms there.
passive bar        the loop's stage-3 2-class mAP50 is READ against T = 0.5628 (thermal-trained
                   yolo11s on thermal, F161). A reading, not a second test. Below T, a win over
                   control is the zero-label route only; the passive claim needs the loop above
                   T, towards V = 0.8251.
car_stop           V - T +0.2132; color_cone +0.3114 (F159). Both passive GO.
warm classes       car/person/bike V - T +0.0323/+0.0650/+0.0999 (F160). Bike sits on the 0.10
                   line. A gain there is domain gap, not the passive claim.
noise floor        0.059 mAP50 (custom set, M1.2) and 0.048 (FLIR's A/B probe pair).
                   An effect below these is not an effect.
headroom           +0.7725 2-class (ceiling 0.8251 / floor 0.0526), PASS at 1.9x the 0.40
                   line (F158). FLIR was +0.2066.
cost               ~117 GPU-h planned; FLIR's 600-pair cell cost 116.8 (F146). 536 pairs here,
                   all of them, not matched to 600.
dose               grad_scale 0.15 was calibrated on custom and FLIR (F144). Unmeasured on
                   msrs-day before loss_share.py's probe; section 7 is the realised dose.
confound           MSRS registration is assumed, not measured (021). A residual offset hits
                   car_stop's ~4x7 px boxes hardest and inflates V - T there.
```

## Results

The report's sections 2 and 4–7, verbatim:

```
==============================================================================
== 2. COMPLETENESS -- read this first
==============================================================================
run                          seed arm      stages         epochs/stage     task_weights
e3md-control-s0                 0 control  [0, 1, 2, 3]   [99, 100, 100, 100] [0.0, 0.0, 0.0, 0.0]
e3md-control-s1                 1 control  [0, 1, 2, 3]   [100, 100, 100, 100] [0.0, 0.0, 0.0, 0.0]
e3md-control-s2                 2 control  [0, 1, 2, 3]   [100, 100, 100, 100] [0.0, 0.0, 0.0, 0.0]
e3md-control-s3                 3 control  [0, 1, 2, 3]   [97, 100, 100, 100] [0.0, 0.0, 0.0, 0.0]
e3md-control-s4                 4 control  [0, 1, 2, 3]   [100, 100, 100, 100] [0.0, 0.0, 0.0, 0.0]
e3md-control-s5                 5 control  [0, 1, 2, 3]   [100, 100, 100, 100] [0.0, 0.0, 0.0, 0.0]
e3md-loop-s0                    0 loop     [0, 1, 2, 3]   [100, 100, 100, 100] [0.0, 1.0, 2.0, 3.0]
e3md-loop-s1                    1 loop     [0, 1, 2, 3]   [100, 100, 100, 100] [0.0, 1.0, 2.0, 3.0]
e3md-loop-s2                    2 loop     [0, 1, 2, 3]   [100, 100, 100, 100] [0.0, 1.0, 2.0, 3.0]
e3md-loop-s3                    3 loop     [0, 1, 2, 3]   [100, 100, 100, 100] [0.0, 1.0, 2.0, 3.0]
e3md-loop-s4                    4 loop     [0, 1, 2, 3]   [100, 100, 100, 100] [0.0, 1.0, 2.0, 3.0]
e3md-loop-s5                    5 loop     [0, 1, 2, 3]   [100, 100, 100, 100] [0.0, 1.0, 2.0, 3.0]

verdict: 12 runs, every run reached stage 3: True

==============================================================================
== 4. PER-RUN x PER-STAGE ROWS (csv)
==============================================================================
run,seed,arm,stage,task_weight,zero_shot.primary_map50,zero_shot.primary_n_classes,zero_shot.map50,zero_shot.map50_95,zero_shot.precision,zero_shot.recall,zero_shot.per_class_ap50.bike,zero_shot.per_class_ap50.car,zero_shot.per_class_ap50.car_stop,zero_shot.per_class_ap50.color_cone,zero_shot.per_class_ap50.person,fidelity.lpips,fidelity.fid,fidelity.psnr,fidelity.ssim,detector.map50,faithfulness.false_object_rate,faithfulness.missed_object_rate,faithfulness.detection_consistency
e3md-control-s0,0,control,0,0,0.105225,2.000000,0.284239,0.140563,0.503712,0.295967,0.191065,0.640956,0.156148,0.054302,0.378725,0.392781,114.152237,13.561300,0.419000,0.534391,,,
e3md-control-s0,0,control,1,0,0.197209,2.000000,0.358631,0.180368,0.718744,0.343014,0.265802,0.729258,0.266305,0.128112,0.403676,0.386547,110.202484,14.183418,0.454199,0.521611,,,
e3md-control-s0,0,control,2,0,0.206100,2.000000,0.360298,0.184656,0.614322,0.356733,0.260829,0.731829,0.285324,0.126876,0.396631,0.387666,100.256454,14.396457,0.464755,0.535601,,,
e3md-control-s0,0,control,3,0,0.187853,2.000000,0.366795,0.187639,0.685285,0.337391,0.306358,0.750085,0.255659,0.120047,0.401828,0.389578,107.698227,14.377576,0.471165,0.521631,0.409222,0.579918,0.425852
e3md-control-s1,1,control,0,0,0.116183,2.000000,0.293964,0.142681,0.536033,0.289607,0.230632,0.626510,0.138211,0.094155,0.380312,0.394912,106.976501,13.770079,0.431325,0.520048,,,
e3md-control-s1,1,control,1,0,0.181146,2.000000,0.347494,0.171424,0.612310,0.343619,0.244370,0.707053,0.231587,0.130705,0.423754,0.384446,100.993828,14.486850,0.456601,0.541867,,,
e3md-control-s1,1,control,2,0,0.170900,2.000000,0.340429,0.174708,0.575192,0.342511,0.264251,0.697184,0.235166,0.106634,0.398908,0.389115,102.273911,14.383972,0.469192,0.544587,,,
e3md-control-s1,1,control,3,0,0.189718,2.000000,0.362824,0.181610,0.640750,0.351784,0.302626,0.743591,0.235234,0.144202,0.388468,0.400415,108.016151,14.183396,0.471795,0.527086,0.436039,0.579918,0.436874
e3md-control-s2,2,control,0,0,0.070351,2.000000,0.074763,0.028543,0.258893,0.090751,0.018700,0.169329,0.103597,0.037104,0.045084,0.530776,141.110825,8.133429,0.355651,0.428520,,,
e3md-control-s2,2,control,1,0,0.083585,2.000000,0.094197,0.035819,0.310121,0.113704,0.021215,0.204358,0.114983,0.052186,0.078240,0.540076,142.744934,8.303067,0.376306,0.433433,,,
e3md-control-s2,2,control,2,0,0.195124,2.000000,0.355421,0.175107,0.694475,0.331318,0.256406,0.690920,0.295356,0.094892,0.439533,0.393861,104.019547,14.003051,0.454651,0.522766,,,
e3md-control-s2,2,control,3,0,0.188682,2.000000,0.347964,0.175008,0.635842,0.324165,0.274280,0.681077,0.273963,0.103402,0.407098,0.414437,105.724030,13.456588,0.474770,0.536538,0.414925,0.598361,0.420842
e3md-control-s3,3,control,0,0,0.168159,2.000000,0.348349,0.178090,0.545126,0.355515,0.245447,0.704195,0.238560,0.097758,0.455786,0.384072,110.042763,13.831805,0.416571,0.546842,,,
e3md-control-s3,3,control,1,0,0.192547,2.000000,0.384697,0.198167,0.573103,0.382111,0.314358,0.756109,0.262407,0.122687,0.467925,0.393513,101.155205,14.151528,0.449827,0.520369,,,
e3md-control-s3,3,control,2,0,0.253248,2.000000,0.420580,0.209707,0.720991,0.402886,0.360853,0.760436,0.325194,0.181302,0.475115,0.359471,94.257378,14.625622,0.465726,0.534509,,,
e3md-control-s3,3,control,3,0,0.203271,2.000000,0.376136,0.189576,0.652014,0.371088,0.314591,0.735903,0.275027,0.131514,0.423647,0.389343,103.150581,14.568636,0.477463,0.527408,0.404570,0.546107,0.466934
e3md-control-s4,4,control,0,0,0.119999,2.000000,0.278207,0.137678,0.523717,0.303654,0.116602,0.660035,0.164120,0.075877,0.374399,0.399260,119.660118,13.905208,0.418562,0.506772,,,
e3md-control-s4,4,control,1,0,0.190175,2.000000,0.348656,0.168628,0.557046,0.346879,0.227699,0.722026,0.250003,0.130348,0.413203,0.399233,116.689133,13.939633,0.435453,0.522111,,,
e3md-control-s4,4,control,2,0,0.195242,2.000000,0.366297,0.180824,0.627069,0.345132,0.276210,0.763079,0.234316,0.156168,0.401711,0.392135,101.550682,14.013448,0.453888,0.537491,,,
e3md-control-s4,4,control,3,0,0.204690,2.000000,0.369790,0.188622,0.615467,0.360695,0.280725,0.754374,0.264501,0.144880,0.404467,0.394001,104.141159,14.402502,0.475510,0.538039,0.432133,0.579918,0.444890
e3md-control-s5,5,control,0,0,0.120998,2.000000,0.312014,0.156412,0.556107,0.297780,0.267534,0.674109,0.189393,0.052604,0.376432,0.425244,107.690620,11.763383,0.375321,0.528604,,,
e3md-control-s5,5,control,1,0,0.172189,2.000000,0.362768,0.182175,0.551506,0.373015,0.265859,0.754594,0.240406,0.103973,0.449009,0.380041,96.580284,14.218051,0.439790,0.538439,,,
e3md-control-s5,5,control,2,0,0.180147,2.000000,0.358071,0.181613,0.666724,0.347626,0.247590,0.750820,0.241670,0.118623,0.431652,0.395205,99.404350,14.047576,0.459886,0.524622,,,
e3md-control-s5,5,control,3,0,0.177780,2.000000,0.369480,0.189121,0.689479,0.344948,0.308399,0.759374,0.241525,0.114035,0.424069,0.396332,105.358490,14.350080,0.480330,0.527817,0.419619,0.563525,0.458918
e3md-loop-s0,0,loop,0,0,0.109096,2.000000,0.315481,0.158826,0.589189,0.327654,0.223083,0.723824,0.185523,0.032669,0.412308,0.391192,108.311081,13.065484,0.405355,0.535496,,,
e3md-loop-s0,0,loop,1,1,0.330834,2.000000,0.479260,0.247986,0.670244,0.477144,0.384650,0.808637,0.435045,0.226623,0.541347,0.413347,113.100121,13.228451,0.424209,0.546204,,,
e3md-loop-s0,0,loop,2,2,0.341521,2.000000,0.490146,0.257533,0.723756,0.473797,0.370608,0.829850,0.423656,0.259385,0.567231,0.412996,107.193878,13.497461,0.445139,0.561391,,,
e3md-loop-s0,0,loop,3,3,0.363236,2.000000,0.519969,0.282585,0.730978,0.486191,0.461419,0.844178,0.468371,0.258100,0.567774,0.383836,103.092659,13.746405,0.445510,0.575977,0.347032,0.413934,0.625251
e3md-loop-s1,1,loop,0,0,0.114915,2.000000,0.257404,0.119149,0.465940,0.273660,0.163990,0.567765,0.134053,0.095777,0.325437,0.430286,116.992760,12.189564,0.393604,0.524391,,,
e3md-loop-s1,1,loop,1,1,0.350841,2.000000,0.492230,0.251770,0.677184,0.495031,0.413687,0.779367,0.435231,0.266451,0.566412,0.411627,105.757355,13.568687,0.445229,0.577416,,,
e3md-loop-s1,1,loop,2,2,0.383061,2.000000,0.506262,0.264013,0.724215,0.477801,0.384796,0.819316,0.435672,0.330450,0.561076,0.394783,105.821754,14.081296,0.452263,0.572318,,,
e3md-loop-s1,1,loop,3,3,0.364913,2.000000,0.496129,0.255187,0.712490,0.464364,0.376504,0.814110,0.481122,0.248704,0.560204,0.402720,105.411629,13.973309,0.459022,0.555085,0.355658,0.428279,0.582164
e3md-loop-s2,2,loop,0,0,0.000500,2.000000,0.000545,0.000122,0.009313,0.008484,0.000000,0.001724,0.000000,0.001000,0.000000,0.714033,217.336624,5.341686,0.320546,0.221885,,,
e3md-loop-s2,2,loop,1,1,0.286971,2.000000,0.455970,0.231435,0.718446,0.426522,0.385500,0.785886,0.394822,0.179121,0.534522,0.438962,124.721718,12.705272,0.403641,0.542054,,,
e3md-loop-s2,2,loop,2,2,0.340048,2.000000,0.473973,0.248876,0.666542,0.463704,0.370247,0.747597,0.431474,0.248623,0.571925,0.410319,106.120735,13.038949,0.443148,0.569435,,,
e3md-loop-s2,2,loop,3,3,0.366478,2.000000,0.515043,0.267104,0.775753,0.478684,0.467999,0.817468,0.456786,0.276170,0.556794,0.386919,99.611191,13.908452,0.450035,0.554474,0.358708,0.430328,0.598196
e3md-loop-s3,3,loop,0,0,0.147382,2.000000,0.332865,0.167944,0.544542,0.330646,0.204304,0.709363,0.219553,0.075211,0.455892,0.386742,102.296883,13.375084,0.421329,0.527335,,,
e3md-loop-s3,3,loop,1,1,0.346923,2.000000,0.486812,0.254189,0.766491,0.460858,0.391301,0.800222,0.404804,0.289041,0.548692,0.426157,101.615700,13.276324,0.451019,0.548958,,,
e3md-loop-s3,3,loop,2,2,0.343062,2.000000,0.486582,0.251547,0.730514,0.469932,0.406555,0.809090,0.391733,0.294391,0.531144,0.406400,100.081451,13.936938,0.459238,0.570828,,,
e3md-loop-s3,3,loop,3,3,0.351383,2.000000,0.499416,0.267086,0.787968,0.454660,0.421506,0.830596,0.452743,0.250023,0.542210,0.411481,103.995712,13.832481,0.462758,0.530446,0.300781,0.449795,0.569138
e3md-loop-s4,4,loop,0,0,0.113302,2.000000,0.328901,0.165367,0.624844,0.307707,0.184194,0.756556,0.175633,0.050972,0.477148,0.376854,112.537468,13.815270,0.413446,0.536066,,,
e3md-loop-s4,4,loop,1,1,0.326671,2.000000,0.480933,0.253545,0.756852,0.456393,0.427822,0.768797,0.383782,0.269560,0.554705,0.418095,106.998749,13.521207,0.427304,0.570151,,,
e3md-loop-s4,4,loop,2,2,0.365061,2.000000,0.501156,0.260620,0.730390,0.456753,0.420127,0.785234,0.443698,0.286424,0.570298,0.410682,101.374733,13.861137,0.458378,0.565720,,,
e3md-loop-s4,4,loop,3,3,0.381537,2.000000,0.533022,0.277519,0.754336,0.485946,0.484341,0.834859,0.477142,0.285931,0.582836,0.391504,100.946861,13.883492,0.455575,0.563505,0.312727,0.419057,0.587174
e3md-loop-s5,5,loop,0,0,0.166037,2.000000,0.343501,0.169949,0.625868,0.319287,0.268074,0.694311,0.214265,0.117809,0.423045,0.411847,103.777740,12.665952,0.401246,0.529484,,,
e3md-loop-s5,5,loop,1,1,0.303640,2.000000,0.483404,0.252474,0.713799,0.467471,0.465300,0.796405,0.407117,0.200162,0.548036,0.412725,107.134193,13.282326,0.439466,0.561273,,,
e3md-loop-s5,5,loop,2,2,0.337071,2.000000,0.496772,0.256940,0.713038,0.477951,0.428349,0.817855,0.429259,0.244883,0.563513,0.421037,101.204239,13.809188,0.466395,0.546273,,,
e3md-loop-s5,5,loop,3,3,0.327724,2.000000,0.488363,0.258366,0.673878,0.467837,0.386957,0.836664,0.426953,0.228494,0.562748,0.404604,102.218124,14.072066,0.451063,0.542171,0.321696,0.442623,0.570140

==============================================================================
== 5. WALL CLOCK
==============================================================================
run                          stage epochs  median_s  train_h  bound_h
e3md-control-s0                  0     99    45.324     1.27     0.18
e3md-control-s0                  1    100    62.175     1.73     0.18
e3md-control-s0                  2    100    75.752     2.12     0.18
e3md-control-s0                  3    100    91.050     2.53     0.17
e3md-control-s0              TOTAL span 8.37 h (config.yaml -> last stage's last file)
e3md-control-s1                  0    100    46.885     1.30     0.17
e3md-control-s1                  1    100    60.716     1.67     0.17
e3md-control-s1                  2    100    74.058     2.04     0.17
e3md-control-s1                  3    100    86.752     2.41     0.16
e3md-control-s1              TOTAL span 8.08 h (config.yaml -> last stage's last file)
e3md-control-s2                  0    100    46.756     1.31     0.20
e3md-control-s2                  1    100    59.143     1.70     0.19
e3md-control-s2                  2    100    75.460     2.12     0.19
e3md-control-s2                  3    100    88.817     2.51     0.17
e3md-control-s2              TOTAL span 8.40 h (config.yaml -> last stage's last file)
e3md-control-s3                  0     97    45.099     1.24     0.19
e3md-control-s3                  1    100    61.752     1.72     0.17
e3md-control-s3                  2    100    76.136     2.13     0.18
e3md-control-s3                  3    100    91.385     2.54     0.17
e3md-control-s3              TOTAL span 8.34 h (config.yaml -> last stage's last file)
e3md-control-s4                  0    100    47.283     1.29     0.17
e3md-control-s4                  1    100    61.404     1.68     0.17
e3md-control-s4                  2    100    74.635     2.06     0.16
e3md-control-s4                  3    100    88.110     2.43     0.16
e3md-control-s4              TOTAL span 8.13 h (config.yaml -> last stage's last file)
e3md-control-s5                  0    100    45.119     1.27     0.17
e3md-control-s5                  1    100    58.410     1.65     0.17
e3md-control-s5                  2    100    73.530     2.03     0.16
e3md-control-s5                  3    100    84.694     2.37     0.16
e3md-control-s5              TOTAL span 7.98 h (config.yaml -> last stage's last file)
e3md-loop-s0                     0    100    45.487     1.29     0.20
e3md-loop-s0                     1    100    67.605     1.91     0.20
e3md-loop-s0                     2    100    80.784     2.28     0.20
e3md-loop-s0                     3    100    94.751     2.62     0.17
e3md-loop-s0                 TOTAL span 8.87 h (config.yaml -> last stage's last file)
e3md-loop-s1                     0    100    48.326     1.31     0.18
e3md-loop-s1                     1    100    64.710     1.82     0.18
e3md-loop-s1                     2    100    83.881     2.27     0.19
e3md-loop-s1                     3    100    94.980     2.66     0.19
e3md-loop-s1                 TOTAL span 8.82 h (config.yaml -> last stage's last file)
e3md-loop-s2                     0    100    44.668     1.27     0.17
e3md-loop-s2                     1    100    65.020     1.82     0.17
e3md-loop-s2                     2    100    78.933     2.20     0.17
e3md-loop-s2                     3    100    91.419     2.55     0.15
e3md-loop-s2                 TOTAL span 8.50 h (config.yaml -> last stage's last file)
e3md-loop-s3                     0    100    45.273     1.28     0.20
e3md-loop-s3                     1    100    67.106     1.90     0.19
e3md-loop-s3                     2    100    80.478     2.25     0.19
e3md-loop-s3                     3    100    94.185     2.60     0.16
e3md-loop-s3                 TOTAL span 8.78 h (config.yaml -> last stage's last file)
e3md-loop-s4                     0    100    48.218     1.31     0.18
e3md-loop-s4                     1    100    63.447     1.80     0.17
e3md-loop-s4                     2    100    82.863     2.25     0.18
e3md-loop-s4                     3    100    94.613     2.64     0.18
e3md-loop-s4                 TOTAL span 8.72 h (config.yaml -> last stage's last file)
e3md-loop-s5                     0    100    45.312     1.26     0.18
e3md-loop-s5                     1    100    66.379     1.84     0.17
e3md-loop-s5                     2    100    79.754     2.22     0.17
e3md-loop-s5                     3    100    92.929     2.60     0.16
e3md-loop-s5                 TOTAL span 8.58 h (config.yaml -> last stage's last file)

two-up contention -- no solo probe recorded for this campaign

per-stage growth, last stage's training / stage 0's: median x2.01 over 12 runs (step 0 measured ~+20%/stage on the custom set and never explained it)

==============================================================================
== 6. PAIRED STATISTICS (t2o aggregate, verbatim)
==============================================================================
$ t2o aggregate --runs runs/e3md-control-s* runs/e3md-loop-s* --metric zero_shot.primary_map50 zero_shot.primary_n_classes zero_shot.map50 zero_shot.map50_95 zero_shot.precision zero_shot.recall zero_shot.per_class_ap50.bike zero_shot.per_class_ap50.car zero_shot.per_class_ap50.car_stop zero_shot.per_class_ap50.color_cone zero_shot.per_class_ap50.person fidelity.lpips fidelity.fid fidelity.psnr fidelity.ssim detector.map50 faithfulness.false_object_rate faithfulness.missed_object_rate faithfulness.detection_consistency --primary-classes car_stop color_cone --stage 3

12 runs, stages [0, 1, 2, 3]
metric                              arm      stage   n      mean       std
zero_shot.primary_map50             control      0   6    0.1168    0.0315
zero_shot.primary_map50             loop         0   6    0.1085    0.0575
zero_shot.primary_map50             control      1   6    0.1695    0.0430
zero_shot.primary_map50             loop         1   6    0.3243    0.0248
zero_shot.primary_map50             control      2   6    0.2001    0.0288
zero_shot.primary_map50             loop         2   6    0.3516    0.0184
zero_shot.primary_map50             control      3   6    0.1920    0.0102
zero_shot.primary_map50             loop         3   6    0.3592    0.0182
zero_shot.primary_n_classes         control      0   6    2.0000    0.0000
zero_shot.primary_n_classes         loop         0   6    2.0000    0.0000
zero_shot.primary_n_classes         control      1   6    2.0000    0.0000
zero_shot.primary_n_classes         loop         1   6    2.0000    0.0000
zero_shot.primary_n_classes         control      2   6    2.0000    0.0000
zero_shot.primary_n_classes         loop         2   6    2.0000    0.0000
zero_shot.primary_n_classes         control      3   6    2.0000    0.0000
zero_shot.primary_n_classes         loop         3   6    2.0000    0.0000
zero_shot.map50                     control      0   6    0.2653    0.0967
zero_shot.map50                     loop         0   6    0.2631    0.1322
zero_shot.map50                     control      1   6    0.3161    0.1095
zero_shot.map50                     loop         1   6    0.4798    0.0125
zero_shot.map50                     control      2   6    0.3668    0.0277
zero_shot.map50                     loop         2   6    0.4925    0.0115
zero_shot.map50                     control      3   6    0.3655    0.0096
zero_shot.map50                     loop         3   6    0.5087    0.0168
zero_shot.map50_95                  control      0   6    0.1307    0.0522
zero_shot.map50_95                  loop         0   6    0.1302    0.0665
zero_shot.map50_95                  control      1   6    0.1561    0.0598
zero_shot.map50_95                  loop         1   6    0.2486    0.0087
zero_shot.map50_95                  control      2   6    0.1844    0.0130
zero_shot.map50_95                  loop         2   6    0.2566    0.0056
zero_shot.map50_95                  control      3   6    0.1853    0.0058
zero_shot.map50_95                  loop         3   6    0.2680    0.0106
zero_shot.precision                 control      0   6    0.4873    0.1133
zero_shot.precision                 loop         0   6    0.4766    0.2367
zero_shot.precision                 control      1   6    0.5538    0.1345
zero_shot.precision                 loop         1   6    0.7172    0.0396
zero_shot.precision                 control      2   6    0.6498    0.0542
zero_shot.precision                 loop         2   6    0.7147    0.0245
zero_shot.precision                 control      3   6    0.6531    0.0291
zero_shot.precision                 loop         3   6    0.7392    0.0424
zero_shot.recall                    control      0   6    0.2722    0.0921
zero_shot.recall                    loop         0   6    0.2612    0.1255
zero_shot.recall                    control      1   6    0.3171    0.1010
zero_shot.recall                    loop         1   6    0.4639    0.0229
zero_shot.recall                    control      2   6    0.3544    0.0251
zero_shot.recall                    loop         2   6    0.4700    0.0084
zero_shot.recall                    control      3   6    0.3483    0.0167
zero_shot.recall                    loop         3   6    0.4729    0.0127
zero_shot.per_class_ap50.bike       control      0   6    0.1783    0.0945
zero_shot.per_class_ap50.bike       loop         0   6    0.1739    0.0924
zero_shot.per_class_ap50.bike       control      1   6    0.2232    0.1032
zero_shot.per_class_ap50.bike       loop         1   6    0.4114    0.0315
zero_shot.per_class_ap50.bike       control      2   6    0.2777    0.0418
zero_shot.per_class_ap50.bike       loop         2   6    0.3968    0.0252
zero_shot.per_class_ap50.bike       control      3   6    0.2978    0.0163
zero_shot.per_class_ap50.bike       loop         3   6    0.4331    0.0450
zero_shot.per_class_ap50.car        control      0   6    0.5792    0.2026
zero_shot.per_class_ap50.car        loop         0   6    0.5756    0.2885
zero_shot.per_class_ap50.car        control      1   6    0.6456    0.2170
zero_shot.per_class_ap50.car        loop         1   6    0.7899    0.0146
zero_shot.per_class_ap50.car        control      2   6    0.7324    0.0317
zero_shot.per_class_ap50.car        loop         2   6    0.8015    0.0304
zero_shot.per_class_ap50.car        control      3   6    0.7374    0.0288
zero_shot.per_class_ap50.car        loop         3   6    0.8296    0.0116
zero_shot.per_class_ap50.car_stop   control      0   6    0.1650    0.0460
zero_shot.per_class_ap50.car_stop   loop         0   6    0.1548    0.0819
zero_shot.per_class_ap50.car_stop   control      1   6    0.2276    0.0567
zero_shot.per_class_ap50.car_stop   loop         1   6    0.4101    0.0211
zero_shot.per_class_ap50.car_stop   control      2   6    0.2695    0.0380
zero_shot.per_class_ap50.car_stop   loop         2   6    0.4259    0.0180
zero_shot.per_class_ap50.car_stop   control      3   6    0.2577    0.0166
zero_shot.per_class_ap50.car_stop   loop         3   6    0.4605    0.0198
zero_shot.per_class_ap50.color_cone control      0   6    0.0686    0.0245
zero_shot.per_class_ap50.color_cone loop         0   6    0.0622    0.0427
zero_shot.per_class_ap50.color_cone control      1   6    0.1113    0.0307
zero_shot.per_class_ap50.color_cone loop         1   6    0.2385    0.0434
zero_shot.per_class_ap50.color_cone control      2   6    0.1307    0.0323
zero_shot.per_class_ap50.color_cone loop         2   6    0.2774    0.0328
zero_shot.per_class_ap50.color_cone control      3   6    0.1263    0.0168
zero_shot.per_class_ap50.color_cone loop         3   6    0.2579    0.0206
zero_shot.per_class_ap50.person     control      0   6    0.3351    0.1455
zero_shot.per_class_ap50.person     loop         0   6    0.3490    0.1787
zero_shot.per_class_ap50.person     control      1   6    0.3726    0.1462
zero_shot.per_class_ap50.person     loop         1   6    0.5490    0.0110
zero_shot.per_class_ap50.person     control      2   6    0.4239    0.0309
zero_shot.per_class_ap50.person     loop         2   6    0.5609    0.0151
zero_shot.per_class_ap50.person     control      3   6    0.4083    0.0137
zero_shot.per_class_ap50.person     loop         3   6    0.5621    0.0133
fidelity.lpips                      control      0   6    0.4212    0.0555
fidelity.lpips                      loop         0   6    0.4518    0.1299
fidelity.lpips                      control      1   6    0.4140    0.0622
fidelity.lpips                      loop         1   6    0.4202    0.0107
fidelity.lpips                      control      2   6    0.3862    0.0134
fidelity.lpips                      loop         2   6    0.4094    0.0086
fidelity.lpips                      control      3   6    0.3974    0.0094
fidelity.lpips                      loop         3   6    0.3968    0.0110
fidelity.fid                        control      0   6  116.6055   12.8928
fidelity.fid                        loop         0   6  126.8754   44.6513
fidelity.fid                        control      1   6  111.3943   16.9987
fidelity.fid                        loop         1   6  109.8880    8.1459
fidelity.fid                        control      2   6  100.2937    3.3649
fidelity.fid                        loop         2   6  103.6328    3.0748
fidelity.fid                        control      3   6  105.6814    1.9184
fidelity.fid                        loop         3   6  102.5460    2.0924
fidelity.psnr                       control      0   6   12.4942    2.2846
fidelity.psnr                       loop         0   6   11.7422    3.1854
fidelity.psnr                       control      1   6   13.2138    2.4121
fidelity.psnr                       loop         1   6   13.2637    0.3075
fidelity.psnr                       control      2   6   14.2450    0.2601
fidelity.psnr                       loop         2   6   13.7042    0.3788
fidelity.psnr                       control      3   6   14.2231    0.3951
fidelity.psnr                       loop         3   6   13.9027    0.1126
fidelity.ssim                       control      0   6    0.4027    0.0300
fidelity.ssim                       loop         0   6    0.3926    0.0366
fidelity.ssim                       control      1   6    0.4354    0.0301
fidelity.ssim                       loop         1   6    0.4318    0.0172
fidelity.ssim                       control      2   6    0.4613    0.0062
fidelity.ssim                       loop         2   6    0.4541    0.0089
fidelity.ssim                       control      3   6    0.4752    0.0035
fidelity.ssim                       loop         3   6    0.4540    0.0063
detector.map50                      control      0   6    0.5109    0.0425
detector.map50                      loop         0   6    0.4791    0.1261
detector.map50                      control      1   6    0.5130    0.0401
detector.map50                      loop         1   6    0.5577    0.0142
detector.map50                      control      2   6    0.5333    0.0082
detector.map50                      loop         2   6    0.5643    0.0097
detector.map50                      control      3   6    0.5298    0.0063
detector.map50                      loop         3   6    0.5536    0.0159
faithfulness.false_object_rate      control      3   6    0.4194    0.0125
faithfulness.false_object_rate      loop         3   6    0.3328    0.0243
faithfulness.missed_object_rate     control      3   6    0.5746    0.0178
faithfulness.missed_object_rate     loop         3   6    0.4307    0.0136
faithfulness.detection_consistency  control      3   6    0.4424    0.0182
faithfulness.detection_consistency  loop         3   6    0.5887    0.0210
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

==============================================================================
== 7. LOSS SPACE -- the realised dose
==============================================================================

-- loop arm, 6 run(s)
stage runs  epochs     w lambda_eff     loss_l2  loss_lpips    loss_gan    loss_det  loss_total     w*det   share
    0    6     100   0.0     0.0000      0.0513      2.0874      0.8311          --      2.9697    0.0000    0.0%
    1    6     100   1.0     0.1500      0.0443      2.0484      1.1086      0.4945      3.6958    0.4945   13.4%
    2    6     100   2.0     0.3000      0.0393      1.9232      1.4017      0.3361      4.0363    0.6721   16.7%
    3    6     100   3.0     0.4500      0.0368      1.8258      1.7746      0.2747      4.4611    0.8240   18.5%

-- control arm (terms only), 6 run(s)
stage runs  epochs     w lambda_eff     loss_l2  loss_lpips    loss_gan    loss_det  loss_total     w*det   share
    0    6  97-100   0.0     0.0000      0.0502      2.0609      0.8248          --      2.9359    0.0000    0.0%
    1    6     100   0.0     0.0000      0.0343      1.6789      0.9094          --      2.6226    0.0000    0.0%
    2    6     100   0.0     0.0000      0.0298      1.5601      1.2498          --      2.8397    0.0000    0.0%
    3    6     100   0.0     0.0000      0.0285      1.5262      1.6445          --      3.1992    0.0000    0.0%
```

`logs/2026-10-15-msrs-day-campaign-faithfulness.txt`, whole:

```
2026-10-15 09:12:48,624 INFO    t2o.data.manifest: data.yaml: 5 classes ['car', 'person', 'bike', 'car_stop', 'color_cone'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\msrs-day
2026-10-15 09:13:03,803 INFO    t2o.metrics.faithfulness: faithfulness val on 179 pairs: false-object 0.4092  missed-object 0.5799  consistency 0.4259
2026-10-15 09:13:03,805 INFO    t2o.cli:   false-object rate       0.4092  (LOWER better)
2026-10-15 09:13:03,805 INFO    t2o.cli:   missed-object rate      0.5799  (LOWER better)
2026-10-15 09:13:03,805 INFO    t2o.cli:   detection-consistency   0.4259  (higher better)
2026-10-15 09:13:06,391 INFO    t2o.engine.loop: recorded faithfulness into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-control-s0\metrics.json stage 3
2026-10-15 09:13:21,660 INFO    t2o.data.manifest: data.yaml: 5 classes ['car', 'person', 'bike', 'car_stop', 'color_cone'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\msrs-day
2026-10-15 09:13:33,330 INFO    t2o.metrics.faithfulness: faithfulness val on 179 pairs: false-object 0.4360  missed-object 0.5799  consistency 0.4369
2026-10-15 09:13:33,335 INFO    t2o.cli:   false-object rate       0.4360  (LOWER better)
2026-10-15 09:13:33,335 INFO    t2o.cli:   missed-object rate      0.5799  (LOWER better)
2026-10-15 09:13:33,335 INFO    t2o.cli:   detection-consistency   0.4369  (higher better)
2026-10-15 09:13:36,028 INFO    t2o.engine.loop: recorded faithfulness into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-control-s1\metrics.json stage 3
2026-10-15 09:13:50,759 INFO    t2o.data.manifest: data.yaml: 5 classes ['car', 'person', 'bike', 'car_stop', 'color_cone'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\msrs-day
2026-10-15 09:14:01,402 INFO    t2o.metrics.faithfulness: faithfulness val on 179 pairs: false-object 0.4149  missed-object 0.5984  consistency 0.4208
2026-10-15 09:14:01,402 INFO    t2o.cli:   false-object rate       0.4149  (LOWER better)
2026-10-15 09:14:01,402 INFO    t2o.cli:   missed-object rate      0.5984  (LOWER better)
2026-10-15 09:14:01,402 INFO    t2o.cli:   detection-consistency   0.4208  (higher better)
2026-10-15 09:14:03,676 INFO    t2o.engine.loop: recorded faithfulness into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-control-s2\metrics.json stage 3
2026-10-15 09:14:17,664 INFO    t2o.data.manifest: data.yaml: 5 classes ['car', 'person', 'bike', 'car_stop', 'color_cone'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\msrs-day
2026-10-15 09:14:29,152 INFO    t2o.metrics.faithfulness: faithfulness val on 179 pairs: false-object 0.4046  missed-object 0.5461  consistency 0.4669
2026-10-15 09:14:29,152 INFO    t2o.cli:   false-object rate       0.4046  (LOWER better)
2026-10-15 09:14:29,152 INFO    t2o.cli:   missed-object rate      0.5461  (LOWER better)
2026-10-15 09:14:29,152 INFO    t2o.cli:   detection-consistency   0.4669  (higher better)
2026-10-15 09:14:31,529 INFO    t2o.engine.loop: recorded faithfulness into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-control-s3\metrics.json stage 3
2026-10-15 09:14:46,309 INFO    t2o.data.manifest: data.yaml: 5 classes ['car', 'person', 'bike', 'car_stop', 'color_cone'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\msrs-day
2026-10-15 09:14:56,657 INFO    t2o.metrics.faithfulness: faithfulness val on 179 pairs: false-object 0.4321  missed-object 0.5799  consistency 0.4449
2026-10-15 09:14:56,660 INFO    t2o.cli:   false-object rate       0.4321  (LOWER better)
2026-10-15 09:14:56,660 INFO    t2o.cli:   missed-object rate      0.5799  (LOWER better)
2026-10-15 09:14:56,660 INFO    t2o.cli:   detection-consistency   0.4449  (higher better)
2026-10-15 09:14:59,398 INFO    t2o.engine.loop: recorded faithfulness into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-control-s4\metrics.json stage 3
2026-10-15 09:15:13,277 INFO    t2o.data.manifest: data.yaml: 5 classes ['car', 'person', 'bike', 'car_stop', 'color_cone'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\msrs-day
2026-10-15 09:15:24,590 INFO    t2o.metrics.faithfulness: faithfulness val on 179 pairs: false-object 0.4196  missed-object 0.5635  consistency 0.4589
2026-10-15 09:15:24,590 INFO    t2o.cli:   false-object rate       0.4196  (LOWER better)
2026-10-15 09:15:24,590 INFO    t2o.cli:   missed-object rate      0.5635  (LOWER better)
2026-10-15 09:15:24,590 INFO    t2o.cli:   detection-consistency   0.4589  (higher better)
2026-10-15 09:15:27,371 INFO    t2o.engine.loop: recorded faithfulness into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-control-s5\metrics.json stage 3
2026-10-15 09:15:41,883 INFO    t2o.data.manifest: data.yaml: 5 classes ['car', 'person', 'bike', 'car_stop', 'color_cone'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\msrs-day
2026-10-15 09:15:53,707 INFO    t2o.metrics.faithfulness: faithfulness val on 179 pairs: false-object 0.3470  missed-object 0.4139  consistency 0.6253
2026-10-15 09:15:53,707 INFO    t2o.cli:   false-object rate       0.3470  (LOWER better)
2026-10-15 09:15:53,707 INFO    t2o.cli:   missed-object rate      0.4139  (LOWER better)
2026-10-15 09:15:53,707 INFO    t2o.cli:   detection-consistency   0.6253  (higher better)
2026-10-15 09:15:55,858 INFO    t2o.engine.loop: recorded faithfulness into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-loop-s0\metrics.json stage 3
2026-10-15 09:16:09,646 INFO    t2o.data.manifest: data.yaml: 5 classes ['car', 'person', 'bike', 'car_stop', 'color_cone'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\msrs-day
2026-10-15 09:16:20,847 INFO    t2o.metrics.faithfulness: faithfulness val on 179 pairs: false-object 0.3557  missed-object 0.4283  consistency 0.5822
2026-10-15 09:16:20,847 INFO    t2o.cli:   false-object rate       0.3557  (LOWER better)
2026-10-15 09:16:20,847 INFO    t2o.cli:   missed-object rate      0.4283  (LOWER better)
2026-10-15 09:16:20,847 INFO    t2o.cli:   detection-consistency   0.5822  (higher better)
2026-10-15 09:16:23,591 INFO    t2o.engine.loop: recorded faithfulness into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-loop-s1\metrics.json stage 3
2026-10-15 09:16:36,894 INFO    t2o.data.manifest: data.yaml: 5 classes ['car', 'person', 'bike', 'car_stop', 'color_cone'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\msrs-day
2026-10-15 09:16:48,391 INFO    t2o.metrics.faithfulness: faithfulness val on 179 pairs: false-object 0.3587  missed-object 0.4303  consistency 0.5982
2026-10-15 09:16:48,391 INFO    t2o.cli:   false-object rate       0.3587  (LOWER better)
2026-10-15 09:16:48,391 INFO    t2o.cli:   missed-object rate      0.4303  (LOWER better)
2026-10-15 09:16:48,391 INFO    t2o.cli:   detection-consistency   0.5982  (higher better)
2026-10-15 09:16:50,843 INFO    t2o.engine.loop: recorded faithfulness into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-loop-s2\metrics.json stage 3
2026-10-15 09:17:04,888 INFO    t2o.data.manifest: data.yaml: 5 classes ['car', 'person', 'bike', 'car_stop', 'color_cone'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\msrs-day
2026-10-15 09:17:16,596 INFO    t2o.metrics.faithfulness: faithfulness val on 179 pairs: false-object 0.3008  missed-object 0.4498  consistency 0.5691
2026-10-15 09:17:16,598 INFO    t2o.cli:   false-object rate       0.3008  (LOWER better)
2026-10-15 09:17:16,598 INFO    t2o.cli:   missed-object rate      0.4498  (LOWER better)
2026-10-15 09:17:16,598 INFO    t2o.cli:   detection-consistency   0.5691  (higher better)
2026-10-15 09:17:19,153 INFO    t2o.engine.loop: recorded faithfulness into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-loop-s3\metrics.json stage 3
2026-10-15 09:17:33,720 INFO    t2o.data.manifest: data.yaml: 5 classes ['car', 'person', 'bike', 'car_stop', 'color_cone'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\msrs-day
2026-10-15 09:17:45,185 INFO    t2o.metrics.faithfulness: faithfulness val on 179 pairs: false-object 0.3127  missed-object 0.4191  consistency 0.5872
2026-10-15 09:17:45,185 INFO    t2o.cli:   false-object rate       0.3127  (LOWER better)
2026-10-15 09:17:45,185 INFO    t2o.cli:   missed-object rate      0.4191  (LOWER better)
2026-10-15 09:17:45,185 INFO    t2o.cli:   detection-consistency   0.5872  (higher better)
2026-10-15 09:17:47,674 INFO    t2o.engine.loop: recorded faithfulness into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-loop-s4\metrics.json stage 3
2026-10-15 09:18:03,106 INFO    t2o.data.manifest: data.yaml: 5 classes ['car', 'person', 'bike', 'car_stop', 'color_cone'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\msrs-day
2026-10-15 09:18:14,444 INFO    t2o.metrics.faithfulness: faithfulness val on 179 pairs: false-object 0.3217  missed-object 0.4426  consistency 0.5701
2026-10-15 09:18:14,444 INFO    t2o.cli:   false-object rate       0.3217  (LOWER better)
2026-10-15 09:18:14,444 INFO    t2o.cli:   missed-object rate      0.4426  (LOWER better)
2026-10-15 09:18:14,444 INFO    t2o.cli:   detection-consistency   0.5701  (higher better)
2026-10-15 09:18:17,035 INFO    t2o.engine.loop: recorded faithfulness into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-loop-s5\metrics.json stage 3
```

Re-render without re-running (on the server): the `campaign_report.py` command above. It only reads
the twelve run directories.

## Findings

Gate numbers below (V, T, J and their per-class values) are record 021's three-seed means. The
campaign's judge is V's seed 1 alone.

### F168 — The pre-registered endpoint is positive at the sign-flip floor: 2-class (car_stop, color_cone) mAP50 loop − control +0.1672, p = .031, against a stage-0 null of −0.0083 that held

- **Stage 3:** +0.1672, p = .031 (the n = 6 floor), CI [+0.1575, +0.1766]. Every loop seed
  (0.3277–0.3815) beats every control seed (0.1778–0.2047).
- **Stage-0 null** (λ = 0 in both arms): −0.0083, p = .594. The stage-3 effect is 20× its size.
- **Against the noise floors** (0.059 custom, 0.048 FLIR): 2.8× and 3.5×.
- **Trajectory** (gain over stage 0, a sensitivity analysis): +0.1755, p = .031.
- **Against the two earlier cells:** custom +0.0512 (F35), FLIR +0.3512 on its 3-class endpoint
  (F136). The direction replicates on a third dataset.
- **It is not only the primary classes.** All-class zero-shot mAP50 +0.1432, mAP50-95 +0.0827,
  recall +0.1246, each p = .031. Precision +0.0861, p = .0625.

### F169 — The passive bar is not met: the loop's stage-3 2-class mAP50 is 0.3592, 0.2036 below the thermal-trained T = 0.5628; the win is the zero-label route only

- **The reading, as pre-registered** (a reading, not a second test): loop 0.3592 ± 0.0182 against
  T 0.5628 ± 0.0218. The best loop seed (0.3815) is still 0.18 below T.
- **On F161's scale.** Of the 2-class headroom V − J = +0.7725, T sits at 66%. The loop sits at
  40%, the control at 18%.
- **Per passive class:** car_stop 0.4605 vs T 0.6772 (−0.2167); color_cone 0.2579 vs T 0.4483
  (−0.1904).
- **Against the floor, unlike FLIR.** The loop is 6.8× the raw-thermal floor J = 0.0526 (+0.3066).
  On FLIR it landed on the floor (F137).
- **What this means.** Translation with the detector loss lifts a visible-trained detector a long
  way on thermal without thermal labels. It does not get past what a detector trained on labelled
  thermal already does. The passive claim, colour recovered beyond what thermal carries, is not
  supported on msrs-day by pix2pix at this dose.

### F170 — The gain is not specific to the passive classes: the loop recovers 54–64% of the domain gap on car_stop and color_cone, 69–91% on the warm classes

- **Stage-3 paired gains**, all p = .031: car_stop +0.2029, color_cone +0.1316; car +0.0922,
  person +0.1538, bike +0.1353. color_cone gains less than person or bike.
- **Share of each class's domain gap T − J that the loop recovers**, (loop − J) / (T − J):

  | class | loop | T | loop − T | loop's share of T − J | control's share |
  | --- | --- | --- | --- | --- | --- |
  | car | 0.8296 | 0.9038 | −0.0742 | 91% | 80% |
  | person | 0.5621 | 0.6509 | −0.0888 | 69% | 16% |
  | bike | 0.4331 | 0.5578 | −0.1247 | 74% | 47% |
  | car_stop | 0.4605 | 0.6772 | −0.2167 | 64% | 31% |
  | color_cone | 0.2579 | 0.4483 | −0.1904 | 54% | 22% |

- **Against F160.** The sensor gap V − T was 4.0× larger on the passive classes than on the warm
  ones. Here the loop is below T on every class, and furthest below it on the two passive ones.
  The passive classes are where it falls shortest.

### F171 — C2 favours the loop on all three rates (p = .031 each), and the worst loop seed beats the best control seed on each; absolute rates stay high

- **False-object rate:** 0.3328 loop vs 0.4194 control, −0.0867. Worst loop seed 0.3587, best
  control 0.4046.
- **Missed-object rate:** 0.4307 vs 0.5746, −0.1440. Worst loop 0.4498, best control 0.5461.
- **Detection consistency:** 0.5887 vs 0.4424, +0.1463. Worst loop 0.5691, best control 0.4669.
- **Against FLIR (F143):** the same result, all three rates at the floor, worst loop beating best
  control.
- **Not "low".** The loop's frames still yield a false object at 33% and miss 43% against the real
  visible frames. FLIR's loop read 19% false and 57% missed (F143).
- **No stage-0 C2 null.** Only stage 3 was scored, as on FLIR.

### F172 — No fidelity cost from λ except SSIM: LPIPS, FID and PSNR are null at stage 3, SSIM is −0.0212 (p = .031) with a null trajectory, and every stage-0 null held

- **Stage-3 paired:** LPIPS −0.0005 (p = .938), FID −3.14 (p = .0625), PSNR −0.32 dB (p = .125),
  SSIM −0.0212 (p = .031, 4.5% of the control's 0.4752).
- **Stage-0 nulls held on all four** (p = .22–.81), unlike FLIR where all three failed (F141).
- **Trajectories are null on all four** (p ≥ .41). SSIM's −0.0110 trajectory (p = .406) does not
  separate its stage-3 gap from the stage-0 offset of −0.0102.

### F173 — `detector.map50` (the fine-tuned 5-class `yolo11n`) moves +0.0239 at stage 3 (p = .031) against a −0.0318 stage-0 null, trajectory +0.0556 (p = .0625)

- **Against FLIR (F142):** there the stage-3 gap equalled its stage-0 null and the trajectory was
  −0.0070, p = 1.000. Here the stage-0 null favours the control and stage 3 favours the loop. The
  trajectory is one step off the floor.
- **Size.** Fine-tuning on translated frames with labels brings both arms to 0.53–0.55 all-class.
  The zero-label gap of F168 (+0.1432 all-class) shrinks to +0.0239 once the detector sees labels.

### F174 — Seed 2 degrades at stage 0 in both arms, and the loop collapses there (0.0005) then recovers by stage 1; at stage 3 the loop arm is not the tighter one

- **Stage 0, λ = 0 in both arms:** `loop-s2` 2-class 0.0005 (LPIPS 0.714, FID 217, PSNR 5.3 dB).
  `control-s2` 0.0704 (LPIPS 0.531), then 0.0836 at stage 1. No other stage-row in either arm is
  below 0.10.
- **Recovery.** `loop-s2` reaches 0.2870 at stage 1 and 0.3665 at stage 3, inside the loop arm's
  range. `control-s2` reaches 0.1951 at stage 2.
- **Against F140:** the same seed degrades in both arms at stage 0, consistent with collapse by draw,
  not by arm.
- **Against F139 and F42:** the loop's sd falls as λ rises (0.0575 → 0.0248 → 0.0184 → 0.0182).
  At stage 3, though, the control is tighter (0.0102). FLIR's loop was 3.7× tighter than its control
  (F139). The custom set's was 2.4× tighter (F42).

### F175 — The realised dose is 13.4 / 16.7 / 18.5%, under the 20–30% band: record 023's 20–30% prediction and F166's ~28% projection are refuted, while the probe's 32.0% upper bound held

- **Probe → campaign, stage by stage:** 17.4 → 13.4% (×0.77), 26.6 → 16.7% (×0.63),
  32.0 → 18.5% (×0.58).
- **At stage 3, ×0.58 is turbo's ×0.53, not pix2pix-custom's ×0.88** (F166).
- **Why the share fell.** From probe to campaign at stage 3, raw `loss_det` fell 53%
  (0.5811 → 0.2747), against FLIR's −37% (F144). `loss_gan` rose 41% (1.2584 → 1.7746). In the
  control arm `loss_gan` doubled across the stages too (0.8248 → 1.6445). So the GAN term grows with
  training length, not with λ.
- **Against the two earlier pix2pix cells at full length:** custom 19.8% (F38), FLIR 21.1% (F144).
  msrs-day's 18.5% lands beside them, although its probe read the hottest of the three.
- **Against record 007's re-tune rule** (below ~10% or above ~40%): inside, so no re-tune is
  indicated. F168's effect landed at this dose anyway.

### F176 — The cell cost 101.6 GPU-h, 7% over record 023's 70–95 prediction and 1.45× F167's floor; per-stage growth ×2.01 in both arms is the term the floor missed

- **Spans:** control 49.3 GPU-h (mean 8.22 h a run), loop 52.3 (8.71 h). Loop over control is
  +6.0% a run, against FLIR's +10.4% (F126).
- **Against the plan and FLIR:** the pre-registration's ~117 GPU-h; FLIR's realised 116.8 (F146).
- **Split:** 93.1 h of measured translator training, 8.5 h of boundaries (0.15–0.20 h each).
  The probe's solo boundaries were 0.16–0.17 h, so two-up barely slowed them.
- **Two-up contention at stage 0:** median epoch 46.1 s across all twelve runs, against the solo
  probe's 40.46 s, ×1.14. This is derived against record 023's single solo run, since the report
  has no solo probe for this campaign. FLIR's was ×1.16 (F145).
- **Per-stage growth:** the report reads ×2.01 (last stage's training ÷ stage 0's, median over 12
  runs). The control arm, at λ = 0 throughout, goes 46.1 → 60.6 → 74.9 → 88.5 s per epoch. So, as
  on FLIR (F145, ×2.03), the growth is not coupling.
- **Why the floor missed.** F167 priced training from the probe's 25-epoch stage means
  (40.5–56.5 s). Those stages carry little of a growth that reaches 88–94 s per epoch by stage 3 at
  100 epochs a stage.

## Provenance caveats

- **The twelve run logs were not passed in.** The launch SHA, the launch date and the exact launch
  command are not recorded. The command section is the runbook, corrected by the snapshots for
  `--wandb`. The machine is the report's `host` line and the post-readout snapshot. Both postdate
  the cell. Nothing on a single-machine server suggests the hardware changed mid-campaign, but no
  launch-time snapshot exists to show it.
- **The snapshot's `git_sha` has no `-dirty` suffix, but untracked files were present.** The report
  saw two untracked files at the same commit an hour earlier. `git describe --dirty` ignores
  untracked files, so "clean" here means the tracked tree only.
- **The report ran at `3f1394a`.** Between the pre-registration's commit (5324522) and `3f1394a`,
  only docs changed. If the server sat anywhere in that range at launch, the code that ran is the
  same. Where it actually sat is unknown.
- **`control-s0` stage 0 records 99 epochs and `control-s3` 97.** These are the first run on each
  card. F149's mechanism applies: `Trainer.resume` restarts at the checkpoint's epoch + 1
  (`src/t2o/engine/trainer.py:356`), and a resume records only the epochs it runs. So both were
  interrupted early in stage 0 and resumed. Why is not recorded.
- **The same two resumes change their spans.** `run_loop` re-snapshots `config.yaml` on every call
  (`src/t2o/engine/loop.py:145`), so both spans count from the relaunch. The minutes before it are
  not in the 101.6 GPU-h.
- **The two resumed stage 0s are not bit-for-bit the runs an uninterrupted launch would make.**
  Stage 0 is λ = 0, and the endpoint is stage 3.
- **W&B** is read off the snapshots, never checked in the UI. For the two resumed runs, the
  pre-relaunch epochs may be missing there.
- **The pre-registration's confound line is stale.** It says MSRS registration is "assumed, not
  measured (021)". The sister repo had already measured it: median EPE 4.63 px over all 361 val
  pairs, and no fixed rig error (rotation +0.05°, centre shift −0.1 / −1.4 px). The residual is
  per-pair scatter (`../Thermal-Image-Registration/TASKS.md:340-356`, `:476-478`). The text was not
  updated before launch and is printed as written. This record does not test how a 4.6 px random
  residual bears on T, J or the loop.
- **V, T and J are three-seed means.** The campaign's judge is V's seed 1 only (all-class 0.7841
  against V's mean of 0.7920). F169 and F170 compare against the means.
- **The contention ratio in F176 is derived here,** not printed by the report. It is against one
  solo run at 25-epoch stages, one seed.
- **C2 has no stage-0 null** (F171), and the C2 log has no provenance block.

## Next

- **The passive claim on msrs-day stands unsupported** (F169, F170). A human picks whether to press
  on it, and with which lever. Candidates, none planned:
  - a stronger backbone (turbo, F77's dose caveat applies);
  - a higher dose, now that the realised share sits under the band (F175);
  - reading the loop against an E8-style direct-thermal sweep on msrs-day, which prices the
    0.3592 in annotated thermal images instead of against T's full train set.
- **The zero-label route replicates on a third dataset** (F168, F171). This is the result to carry
  into the paper's C1 and C2 (legacy numbering) beside the custom and FLIR cells.
- **No prediction is posted.** None of the candidates above is planned yet.
