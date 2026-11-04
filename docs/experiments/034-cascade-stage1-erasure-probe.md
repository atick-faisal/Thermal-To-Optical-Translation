# 034 — Cascade gate, stage 1: the erasure probe puts back none of 234 erased fuses and switches, and the faithfulness rates are re-scored per class

**Date:** 2026-11-03 · **Task:** CASCADE-GATE-03 · **Machine:** VSIBMEOT05, 1× A100 (`cuda:1`) · **Wall clock:** 1 h 24 min end to end (15:30:42 → 16:55:00 +03:00). Per step: erase 30 s (CPU), sanity 1 min 39 s, re-score 5 min 57 s, export 13 min 45 s, completion 5 min 1 s; the readout's start time is not recorded
**git_sha:** `pre-spec-migration-215-g150f9c1` (a clean tree at `150f9c1`, no `-dirty` suffix) · **W&B:** none (no W&B line in any log) · **Log:** `logs/2026-11-03-cascade-gate-03-erase.txt`, `logs/2026-11-03-cascade-gate-03-sanity.txt`, `logs/2026-11-03-cascade-gate-03-rescore.txt`, `logs/2026-11-03-cascade-gate-03-export.txt`, `logs/2026-11-03-cascade-gate-03-completion.txt`, `logs/2026-11-03-cascade-gate-03-readout.txt`

This is the cascade gate's stage-1-only run (plan Q3, ADR-03 in
`docs/features/cascade-gate/decisions.md`). It does two things, both on the custom set's loop arm,
e3b seeds 0–5 and e3t seeds 0, 1, 3, at stage 3. First, it re-scores each run's existing val export
with the `rtdetr` judge, so `metrics.json` gains the per-class missed- and false-object rates.
Second, it erases only the Fuse and Switch boxes from val's thermal frames, re-exports each
translator on those frames, and scores how many erased objects come back. Gate ADR-04
(CASCADE-GATE-04) takes its new margins from the seed spread this run prints. No stage-2 output
exists yet, and none was scored.

## Command

From the provenance blocks, as run. Steps 3–5 each ran their nine runs in one loop, and their
provenance `command:` line is the runbook's label for that loop, `<cell>` and `<s>` included:

```
# 1. erase (CPU)
uv run t2o erase --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --out runs/cascade-gate/erased-custom-val-fuse-switch --split val --classes Fuse Switch

# 2. sanity check, one run, no write-back
uv run t2o completion --translated runs/e3b-loop-s0/stage3/translated --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --weights runs/reference-rtdetr-l/weights/best.pt --device cuda:1

# 3. re-score faithfulness with per-class rates
uv run t2o faithfulness --translated runs/<cell>-loop-s<s>/stage3/translated --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --weights runs/reference-rtdetr-l/weights/best.pt --write-back --judge rtdetr --device cuda:1, for e3b s0-5 and e3t s0,1,3

# 4. export each stage-3 translator on the erased frames
uv run t2o export --config runs/<cell>-loop-s<s>/config.yaml --data runs/cascade-gate/erased-custom-val-fuse-switch/data.yaml --checkpoint runs/<cell>-loop-s<s>/stage3/translator_last.pt --out runs/<cell>-loop-s<s>/stage3/erased-translated --device cuda:1, for e3b s0-5 and e3t s0,1,3

# 5. completion rate, written back
uv run t2o completion --translated runs/<cell>-loop-s<s>/stage3/erased-translated --data runs/cascade-gate/erased-custom-val-fuse-switch/data.yaml --weights runs/reference-rtdetr-l/weights/best.pt --write-back --judge rtdetr --device cuda:1, for e3b s0-5 and e3t s0,1,3
```

**Step 6, the readout, has no provenance block**, so its log does not hold the command that ran.
The script below is step 6 of the runbook the run followed (`RUNBOOK-cascade-gate-stage1.md`,
untracked, to be deleted once this record lands). It is kept here so the readout can be re-run.
That it is exactly what ran is an assumption:

```powershell
function Get-Stage3Judge($run) {
    ((Get-Content "$run/metrics.json" -Raw | ConvertFrom-Json) | Where-Object stage -eq 3).judges.rtdetr
}
$cells = [ordered]@{ e3b = 0..5; e3t = 0,1,3 }
$cols  = 'completion_rate', 'missed_Fuse', 'false_Fuse', 'missed_Switch', 'false_Switch',
         'missed_pooled', 'false_pooled'
foreach ($cell in $cells.Keys) {
    $rows = foreach ($s in $cells[$cell]) {
        $judge = Get-Stage3Judge "runs/$cell-loop-s$s"
        $f     = $judge.faithfulness
        [pscustomobject]@{
            cell = $cell; seed = $s
            completion_rate = $judge.completion.completion_rate
            missed_Fuse     = $f.missed_object_rate_by_class.Fuse
            false_Fuse      = $f.false_object_rate_by_class.Fuse
            missed_Switch   = $f.missed_object_rate_by_class.Switch
            false_Switch    = $f.false_object_rate_by_class.Switch
            missed_pooled   = $f.missed_object_rate
            false_pooled    = $f.false_object_rate
        }
    }
    $rows | Format-Table -AutoSize | Out-String
    foreach ($col in $cols) {
        $v    = $rows.$col
        $mean = ($v | Measure-Object -Average).Average
        $std  = [math]::Sqrt((($v | ForEach-Object { ($_ - $mean) * ($_ - $mean) }) | Measure-Object -Sum).Sum / ($v.Count - 1))
        "{0} {1,-16} n={2}  mean={3:F4}  std={4:F4} (sample, ddof=1)" -f $cell, $col, $v.Count, $mean, $std
    }
}
```

## Environment

The erase log's block. The other four blocks (sanity, re-score, export, completion) differ only in
their `captured` and `command` lines (checked with `diff`). The readout log has none.

```
=== PROVENANCE ===
captured: 2026-11-03 15:30:42 +03:00
hostname: VSIBMEOT05
os: Microsoft Windows Server 2022 Standard 10.0.20348 (build 20348)
cpu: AMD EPYC 7763 64-Core Processor
cpu_logical_cores: 64
ram_total: 128.0 GB
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16; NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16
cuda_driver: 13.0
git_sha: pre-spec-migration-215-g150f9c1
working_dir: D:\Atick\GitHub\Thermal-To-Optical-Translation
command: uv run t2o erase --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --out runs/cascade-gate/erased-custom-val-fuse-switch --split val --classes Fuse Switch
python: 3.12.7
torch: 2.12.1+cu130
cuda_torch_build: 13.0
cudnn: 92000
cuda_available: True
gpu0_torch: NVIDIA A100-SXM4-40GB, 39.7 GiB, sm 8.0
=== END PROVENANCE ===
```

## Configuration

Nothing was trained.

| Axis | Value | Where it is evidenced |
| --- | --- | --- |
| Runs | the custom set's loop arm at stage 3: e3b (pix2pix) seeds 0–5, e3t (turbo) seeds 0, 1, 3. e3t seeds 2, 4, 5 never reached stage 3 (record 010) | `--- runs/... ---` lines in each log |
| Arms | loop only. No control arm, because the cascade is compared against its own stage 1 (plan Q3) | the run list |
| Judge | `rtdetr`: record 026's custom-set RT-DETR-l checkpoint, `runs/reference-rtdetr-l/weights/best.pt` | the commands |
| Scorer settings | `t2o faithfulness` / `t2o completion` defaults: IoU 0.5, conf 0.25, imgsz 640, batch 16 (`src/t2o/cli.py:201-205`) | no flag overrides them in the commands |
| Erasure | OpenCV Telea, boxes grown 4 px, radius 3.0, classes `Fuse` and `Switch` only (ADR-03), val split: 234 boxes from 153 frames. The erased tree's labels hold only those boxes | `erased 234 box(es) from 153 val frame(s) (dilate 4 px, radius 3.0)` |
| Export | each run's `stage3/translator_last.pt` on the erased frames, `normalize=clamp`. Export also writes a `train/` copy of the same 153 frames, which nothing scores | `exported 153 translated images to ...\erased-translated\{train,val}` |
| turbo base | `stabilityai/sd-turbo`, resolved to Hub commit `b261bac6fd2cf515557d5d0707481eafa0485ec2` | the export log's `resolve-cache` lines |
| Readout summary | per-cell mean and sample sd (ddof = 1) over the seeds | the readout's summary lines |

## Results

### Step 1 — erase

Everything after the provenance block, verbatim:

```
2026-11-03 15:30:53,336 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-03 15:31:11,986 INFO    t2o.data.erasure: erased 234 box(es) from 153 val frame(s) (dilate 4 px, radius 3.0) -> D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\data.yaml
2026-11-03 15:31:11,986 INFO    t2o.cli: wrote D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\data.yaml
=== FINISHED 2026-11-03 15:31:12 +03:00 ===
```

### Step 2 — sanity check (unerased export, e3b seed 0)

```
2026-11-03 15:35:54,842 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-03 15:36:05,601 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.8747
2026-11-03 15:36:05,602 INFO    t2o.cli:   completion rate         0.8747  (LOWER better)
expected (1 - stored rtdetr missed-object rate): 0.8747
=== FINISHED 2026-11-03 15:36:07 +03:00 ===
```

### Step 3 — re-score with per-class rates

```
--- runs/e3b-loop-s0 ---
2026-11-03 15:38:52,169 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-03 15:39:07,019 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.1553  missed-object 0.1253  consistency 0.8431
2026-11-03 15:39:07,022 INFO    t2o.cli:   false-object rate       0.1553  (LOWER better)
2026-11-03 15:39:07,023 INFO    t2o.cli:   missed-object rate      0.1253  (LOWER better)
2026-11-03 15:39:07,023 INFO    t2o.cli:   detection-consistency   0.8431  (higher better)
2026-11-03 15:39:07,023 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-03 15:39:07,023 INFO    t2o.cli:     Fuse           false 0.0839  missed 0.1548
2026-11-03 15:39:07,023 INFO    t2o.cli:     Pole           false 0.1152  missed 0.0458
2026-11-03 15:39:07,024 INFO    t2o.cli:     Switch         false 0.2836  missed 0.2727
2026-11-03 15:39:07,024 INFO    t2o.cli:     Transformer    false 0.3333  missed 0.0556
2026-11-03 15:39:07,042 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\metrics.json stage 3
runs/e3b-loop-s0 missed_object_rate: record 026 0.125296  re-scored 0.125296  diff 0.000000
runs/e3b-loop-s0 false_object_rate: record 026 0.155251  re-scored 0.155251  diff 0.000000
--- runs/e3b-loop-s1 ---
2026-11-03 15:39:28,108 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-03 15:39:45,658 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.1865  missed-object 0.1442  consistency 0.8257
2026-11-03 15:39:45,658 INFO    t2o.cli:   false-object rate       0.1865  (LOWER better)
2026-11-03 15:39:45,658 INFO    t2o.cli:   missed-object rate      0.1442  (LOWER better)
2026-11-03 15:39:45,658 INFO    t2o.cli:   detection-consistency   0.8257  (higher better)
2026-11-03 15:39:45,664 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-03 15:39:45,664 INFO    t2o.cli:     Fuse           false 0.1056  missed 0.1429
2026-11-03 15:39:45,664 INFO    t2o.cli:     Pole           false 0.0994  missed 0.0523
2026-11-03 15:39:45,664 INFO    t2o.cli:     Switch         false 0.4143  missed 0.3788
2026-11-03 15:39:45,664 INFO    t2o.cli:     Transformer    false 0.3962  missed 0.1111
2026-11-03 15:39:45,680 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s1\metrics.json stage 3
runs/e3b-loop-s1 missed_object_rate: record 026 0.144208  re-scored 0.144208  diff 0.000000
runs/e3b-loop-s1 false_object_rate: record 026 0.186517  re-scored 0.186517  diff 0.000000
--- runs/e3b-loop-s2 ---
2026-11-03 15:40:06,051 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-03 15:40:25,903 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.1905  missed-object 0.1158  consistency 0.8649
2026-11-03 15:40:25,906 INFO    t2o.cli:   false-object rate       0.1905  (LOWER better)
2026-11-03 15:40:25,906 INFO    t2o.cli:   missed-object rate      0.1158  (LOWER better)
2026-11-03 15:40:25,907 INFO    t2o.cli:   detection-consistency   0.8649  (higher better)
2026-11-03 15:40:25,907 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-03 15:40:25,907 INFO    t2o.cli:     Fuse           false 0.1506  missed 0.1607
2026-11-03 15:40:25,907 INFO    t2o.cli:     Pole           false 0.1030  missed 0.0327
2026-11-03 15:40:25,908 INFO    t2o.cli:     Switch         false 0.3014  missed 0.2273
2026-11-03 15:40:25,908 INFO    t2o.cli:     Transformer    false 0.4138  missed 0.0556
2026-11-03 15:40:25,941 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s2\metrics.json stage 3
runs/e3b-loop-s2 missed_object_rate: record 026 0.115839  re-scored 0.115839  diff 0.000000
runs/e3b-loop-s2 false_object_rate: record 026 0.190476  re-scored 0.190476  diff 0.000000
--- runs/e3b-loop-s3 ---
2026-11-03 15:40:47,524 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-03 15:41:05,618 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.1651  missed-object 0.1395  consistency 0.8301
2026-11-03 15:41:05,623 INFO    t2o.cli:   false-object rate       0.1651  (LOWER better)
2026-11-03 15:41:05,624 INFO    t2o.cli:   missed-object rate      0.1395  (LOWER better)
2026-11-03 15:41:05,624 INFO    t2o.cli:   detection-consistency   0.8301  (higher better)
2026-11-03 15:41:05,624 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-03 15:41:05,624 INFO    t2o.cli:     Fuse           false 0.0753  missed 0.1964
2026-11-03 15:41:05,624 INFO    t2o.cli:     Pole           false 0.1273  missed 0.0588
2026-11-03 15:41:05,624 INFO    t2o.cli:     Switch         false 0.2973  missed 0.2121
2026-11-03 15:41:05,625 INFO    t2o.cli:     Transformer    false 0.3529  missed 0.0833
2026-11-03 15:41:05,637 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s3\metrics.json stage 3
runs/e3b-loop-s3 missed_object_rate: record 026 0.139480  re-scored 0.139480  diff 0.000000
runs/e3b-loop-s3 false_object_rate: record 026 0.165138  re-scored 0.165138  diff 0.000000
--- runs/e3b-loop-s4 ---
2026-11-03 15:41:26,756 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-03 15:41:44,070 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.1787  missed-object 0.1418  consistency 0.8366
2026-11-03 15:41:44,070 INFO    t2o.cli:   false-object rate       0.1787  (LOWER better)
2026-11-03 15:41:44,070 INFO    t2o.cli:   missed-object rate      0.1418  (LOWER better)
2026-11-03 15:41:44,070 INFO    t2o.cli:   detection-consistency   0.8366  (higher better)
2026-11-03 15:41:44,070 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-03 15:41:44,070 INFO    t2o.cli:     Fuse           false 0.1218  missed 0.1845
2026-11-03 15:41:44,070 INFO    t2o.cli:     Pole           false 0.0764  missed 0.0523
2026-11-03 15:41:44,070 INFO    t2o.cli:     Switch         false 0.3472  missed 0.2879
2026-11-03 15:41:44,070 INFO    t2o.cli:     Transformer    false 0.4035  missed 0.0556
2026-11-03 15:41:44,085 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s4\metrics.json stage 3
runs/e3b-loop-s4 missed_object_rate: record 026 0.141844  re-scored 0.141844  diff 0.000000
runs/e3b-loop-s4 false_object_rate: record 026 0.178733  re-scored 0.178733  diff 0.000000
--- runs/e3b-loop-s5 ---
2026-11-03 15:42:04,775 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-03 15:42:22,211 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.1743  missed-object 0.1489  consistency 0.8301
2026-11-03 15:42:22,213 INFO    t2o.cli:   false-object rate       0.1743  (LOWER better)
2026-11-03 15:42:22,213 INFO    t2o.cli:   missed-object rate      0.1489  (LOWER better)
2026-11-03 15:42:22,213 INFO    t2o.cli:   detection-consistency   0.8301  (higher better)
2026-11-03 15:42:22,213 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-03 15:42:22,213 INFO    t2o.cli:     Fuse           false 0.1290  missed 0.1964
2026-11-03 15:42:22,213 INFO    t2o.cli:     Pole           false 0.0833  missed 0.0654
2026-11-03 15:42:22,213 INFO    t2o.cli:     Switch         false 0.3506  missed 0.2424
2026-11-03 15:42:22,213 INFO    t2o.cli:     Transformer    false 0.3333  missed 0.1111
2026-11-03 15:42:22,232 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s5\metrics.json stage 3
runs/e3b-loop-s5 missed_object_rate: record 026 0.148936  re-scored 0.148936  diff 0.000000
runs/e3b-loop-s5 false_object_rate: record 026 0.174312  re-scored 0.174312  diff 0.000000
--- runs/e3t-loop-s0 ---
2026-11-03 15:42:45,145 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-03 15:43:02,882 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.2215  missed-object 0.1277  consistency 0.8627
2026-11-03 15:43:02,884 INFO    t2o.cli:   false-object rate       0.2215  (LOWER better)
2026-11-03 15:43:02,885 INFO    t2o.cli:   missed-object rate      0.1277  (LOWER better)
2026-11-03 15:43:02,885 INFO    t2o.cli:   detection-consistency   0.8627  (higher better)
2026-11-03 15:43:02,885 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-03 15:43:02,885 INFO    t2o.cli:     Fuse           false 0.1796  missed 0.1845
2026-11-03 15:43:02,885 INFO    t2o.cli:     Pole           false 0.1159  missed 0.0523
2026-11-03 15:43:02,885 INFO    t2o.cli:     Switch         false 0.3855  missed 0.2273
2026-11-03 15:43:02,885 INFO    t2o.cli:     Transformer    false 0.4000  missed 0.0000
2026-11-03 15:43:02,901 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s0\metrics.json stage 3
runs/e3t-loop-s0 missed_object_rate: record 026 0.127660  re-scored 0.127660  diff 0.000000
runs/e3t-loop-s0 false_object_rate: record 026 0.221519  re-scored 0.221519  diff 0.000000
--- runs/e3t-loop-s1 ---
2026-11-03 15:43:23,899 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-03 15:43:41,278 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.2151  missed-object 0.1371  consistency 0.8562
2026-11-03 15:43:41,281 INFO    t2o.cli:   false-object rate       0.2151  (LOWER better)
2026-11-03 15:43:41,282 INFO    t2o.cli:   missed-object rate      0.1371  (LOWER better)
2026-11-03 15:43:41,282 INFO    t2o.cli:   detection-consistency   0.8562  (higher better)
2026-11-03 15:43:41,282 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-03 15:43:41,282 INFO    t2o.cli:     Fuse           false 0.1250  missed 0.1667
2026-11-03 15:43:41,282 INFO    t2o.cli:     Pole           false 0.1043  missed 0.0458
2026-11-03 15:43:41,282 INFO    t2o.cli:     Switch         false 0.4568  missed 0.3333
2026-11-03 15:43:41,282 INFO    t2o.cli:     Transformer    false 0.4262  missed 0.0278
2026-11-03 15:43:41,291 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s1\metrics.json stage 3
runs/e3t-loop-s1 missed_object_rate: record 026 0.137116  re-scored 0.137116  diff 0.000000
runs/e3t-loop-s1 false_object_rate: record 026 0.215054  re-scored 0.215054  diff 0.000000
--- runs/e3t-loop-s3 ---
2026-11-03 15:44:02,200 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-03 15:44:19,716 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.1685  missed-object 0.1135  consistency 0.8649
2026-11-03 15:44:19,719 INFO    t2o.cli:   false-object rate       0.1685  (LOWER better)
2026-11-03 15:44:19,719 INFO    t2o.cli:   missed-object rate      0.1135  (LOWER better)
2026-11-03 15:44:19,720 INFO    t2o.cli:   detection-consistency   0.8649  (higher better)
2026-11-03 15:44:19,720 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-03 15:44:19,720 INFO    t2o.cli:     Fuse           false 0.1166  missed 0.1429
2026-11-03 15:44:19,720 INFO    t2o.cli:     Pole           false 0.0813  missed 0.0392
2026-11-03 15:44:19,720 INFO    t2o.cli:     Switch         false 0.3467  missed 0.2576
2026-11-03 15:44:19,720 INFO    t2o.cli:     Transformer    false 0.3396  missed 0.0278
2026-11-03 15:44:19,735 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s3\metrics.json stage 3
runs/e3t-loop-s3 missed_object_rate: record 026 0.113475  re-scored 0.113475  diff 0.000000
runs/e3t-loop-s3 false_object_rate: record 026 0.168514  re-scored 0.168514  diff 0.000000
pooled rates match record 026 for every run
=== FINISHED 2026-11-03 15:44:22 +03:00 ===
```

### Step 4 — export on the erased frames

```
--- runs/e3b-loop-s0 ---
2026-11-03 16:13:04,510 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:13:05,917 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:13:29,704 INFO    t2o.engine.export: exported 153 translated images to runs\e3b-loop-s0\stage3\erased-translated\train (normalize=clamp)
2026-11-03 16:13:29,833 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:13:52,595 INFO    t2o.engine.export: exported 153 translated images to runs\e3b-loop-s0\stage3\erased-translated\val (normalize=clamp)
2026-11-03 16:13:52,611 INFO    t2o.cli: wrote runs\e3b-loop-s0\stage3\erased-translated\data.yaml
initialize network with normal
initialize network with normal
--- runs/e3b-loop-s1 ---
2026-11-03 16:14:18,829 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:14:18,919 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:14:41,921 INFO    t2o.engine.export: exported 153 translated images to runs\e3b-loop-s1\stage3\erased-translated\train (normalize=clamp)
2026-11-03 16:14:42,014 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:15:04,656 INFO    t2o.engine.export: exported 153 translated images to runs\e3b-loop-s1\stage3\erased-translated\val (normalize=clamp)
2026-11-03 16:15:04,667 INFO    t2o.cli: wrote runs\e3b-loop-s1\stage3\erased-translated\data.yaml
initialize network with normal
initialize network with normal
--- runs/e3b-loop-s2 ---
2026-11-03 16:15:30,074 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:15:30,152 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:15:52,978 INFO    t2o.engine.export: exported 153 translated images to runs\e3b-loop-s2\stage3\erased-translated\train (normalize=clamp)
2026-11-03 16:15:53,079 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:16:16,041 INFO    t2o.engine.export: exported 153 translated images to runs\e3b-loop-s2\stage3\erased-translated\val (normalize=clamp)
2026-11-03 16:16:16,061 INFO    t2o.cli: wrote runs\e3b-loop-s2\stage3\erased-translated\data.yaml
initialize network with normal
initialize network with normal
--- runs/e3b-loop-s3 ---
2026-11-03 16:16:40,183 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:16:40,267 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:17:01,517 INFO    t2o.engine.export: exported 153 translated images to runs\e3b-loop-s3\stage3\erased-translated\train (normalize=clamp)
2026-11-03 16:17:01,616 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:17:25,134 INFO    t2o.engine.export: exported 153 translated images to runs\e3b-loop-s3\stage3\erased-translated\val (normalize=clamp)
2026-11-03 16:17:25,147 INFO    t2o.cli: wrote runs\e3b-loop-s3\stage3\erased-translated\data.yaml
initialize network with normal
initialize network with normal
--- runs/e3b-loop-s4 ---
2026-11-03 16:17:50,390 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:17:50,466 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:18:12,259 INFO    t2o.engine.export: exported 153 translated images to runs\e3b-loop-s4\stage3\erased-translated\train (normalize=clamp)
2026-11-03 16:18:12,358 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:18:34,110 INFO    t2o.engine.export: exported 153 translated images to runs\e3b-loop-s4\stage3\erased-translated\val (normalize=clamp)
2026-11-03 16:18:34,126 INFO    t2o.cli: wrote runs\e3b-loop-s4\stage3\erased-translated\data.yaml
initialize network with normal
initialize network with normal
--- runs/e3b-loop-s5 ---
2026-11-03 16:19:01,982 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:19:02,070 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:19:25,170 INFO    t2o.engine.export: exported 153 translated images to runs\e3b-loop-s5\stage3\erased-translated\train (normalize=clamp)
2026-11-03 16:19:25,295 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:19:47,510 INFO    t2o.engine.export: exported 153 translated images to runs\e3b-loop-s5\stage3\erased-translated\val (normalize=clamp)
2026-11-03 16:19:47,525 INFO    t2o.cli: wrote runs\e3b-loop-s5\stage3\erased-translated\data.yaml
initialize network with normal
initialize network with normal
--- runs/e3t-loop-s0 ---
2026-11-03 16:20:10,809 INFO    t2o.translators.pix2pix_turbo: loading stabilityai/sd-turbo
2026-11-03 16:20:12,489 INFO    httpx: HTTP Request: GET https://huggingface.co/api/agent-harnesses "HTTP/1.1 200 OK"
2026-11-03 16:20:12,733 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/tokenizer/config.json "HTTP/1.1 404 Not Found"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-11-03 16:20:12,734 WARNING huggingface_hub.utils._http: Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-11-03 16:20:12,994 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/tokenizer/config.json "HTTP/1.1 404 Not Found"
2026-11-03 16:20:13,238 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/tokenizer/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:20:13,366 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/tokenizer%2Ftokenizer_config.json "HTTP/1.1 200 OK"
2026-11-03 16:20:13,650 INFO    httpx: HTTP Request: GET https://huggingface.co/api/models/stabilityai/sd-turbo/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
2026-11-03 16:20:13,891 INFO    httpx: HTTP Request: GET https://huggingface.co/api/models/stabilityai/sd-turbo/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
2026-11-03 16:20:14,403 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/text_encoder/config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:20:14,533 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/text_encoder%2Fconfig.json "HTTP/1.1 200 OK"
2026-11-03 16:20:14,781 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/text_encoder/config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:20:14,909 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/text_encoder%2Fconfig.json "HTTP/1.1 200 OK"
2026-11-03 16:20:15,185 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/text_encoder/model.safetensors "HTTP/1.1 302 Found"

Loading weights:   0%|          | 0/372 [00:00<?, ?it/s]
Loading weights:  27%|##7       | 101/372 [00:00<00:00, 993.60it/s]
Loading weights:  54%|#####4    | 201/372 [00:00<00:00, 913.52it/s]
Loading weights: 100%|##########| 372/372 [00:00<00:00, 1330.39it/s]
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\huggingface_hub\utils\_validators.py:205: UserWarning: The `local_dir_use_symlinks` argument is deprecated and ignored in `hf_hub_download`. Downloading to a local directory does not use symlinks anymore.
  warnings.warn(
2026-11-03 16:20:16,169 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/vae/config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:20:16,296 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/vae%2Fconfig.json "HTTP/1.1 200 OK"
2026-11-03 16:20:16,555 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/b261bac6fd2cf515557d5d0707481eafa0485ec2/vae/diffusion_pytorch_model.safetensors.index.json "HTTP/1.1 404 Not Found"
2026-11-03 16:20:16,950 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/unet/config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:20:17,107 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/unet%2Fconfig.json "HTTP/1.1 200 OK"
2026-11-03 16:20:17,359 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/b261bac6fd2cf515557d5d0707481eafa0485ec2/unet/diffusion_pytorch_model.safetensors.index.json "HTTP/1.1 404 Not Found"
2026-11-03 16:20:18,009 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/scheduler/scheduler_config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:20:18,137 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/scheduler%2Fscheduler_config.json "HTTP/1.1 200 OK"
2026-11-03 16:20:38,285 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:20:38,378 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:21:24,245 INFO    t2o.engine.export: exported 153 translated images to runs\e3t-loop-s0\stage3\erased-translated\train (normalize=clamp)
2026-11-03 16:21:24,418 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:22:08,443 INFO    t2o.engine.export: exported 153 translated images to runs\e3t-loop-s0\stage3\erased-translated\val (normalize=clamp)
2026-11-03 16:22:08,462 INFO    t2o.cli: wrote runs\e3t-loop-s0\stage3\erased-translated\data.yaml
initialize network with normal
--- runs/e3t-loop-s1 ---
2026-11-03 16:22:31,869 INFO    t2o.translators.pix2pix_turbo: loading stabilityai/sd-turbo
2026-11-03 16:22:33,387 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/tokenizer/config.json "HTTP/1.1 404 Not Found"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-11-03 16:22:33,388 WARNING huggingface_hub.utils._http: Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-11-03 16:22:33,626 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/tokenizer/config.json "HTTP/1.1 404 Not Found"
2026-11-03 16:22:33,883 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/tokenizer/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:22:34,011 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/tokenizer%2Ftokenizer_config.json "HTTP/1.1 200 OK"
2026-11-03 16:22:34,265 INFO    httpx: HTTP Request: GET https://huggingface.co/api/models/stabilityai/sd-turbo/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
2026-11-03 16:22:34,509 INFO    httpx: HTTP Request: GET https://huggingface.co/api/models/stabilityai/sd-turbo/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
2026-11-03 16:22:34,864 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/text_encoder/config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:22:34,993 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/text_encoder%2Fconfig.json "HTTP/1.1 200 OK"
2026-11-03 16:22:35,243 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/text_encoder/config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:22:35,369 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/text_encoder%2Fconfig.json "HTTP/1.1 200 OK"
2026-11-03 16:22:35,615 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/text_encoder/model.safetensors "HTTP/1.1 302 Found"

Loading weights:   0%|          | 0/372 [00:00<?, ?it/s]
Loading weights:  36%|###5      | 133/372 [00:00<00:00, 1327.59it/s]
Loading weights:  72%|#######1  | 266/372 [00:00<00:00, 1253.74it/s]
Loading weights: 100%|##########| 372/372 [00:00<00:00, 1657.33it/s]
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\huggingface_hub\utils\_validators.py:205: UserWarning: The `local_dir_use_symlinks` argument is deprecated and ignored in `hf_hub_download`. Downloading to a local directory does not use symlinks anymore.
  warnings.warn(
2026-11-03 16:22:36,409 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/vae/config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:22:36,538 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/vae%2Fconfig.json "HTTP/1.1 200 OK"
2026-11-03 16:22:36,786 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/b261bac6fd2cf515557d5d0707481eafa0485ec2/vae/diffusion_pytorch_model.safetensors.index.json "HTTP/1.1 404 Not Found"
2026-11-03 16:22:37,101 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/unet/config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:22:37,227 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/unet%2Fconfig.json "HTTP/1.1 200 OK"
2026-11-03 16:22:37,468 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/b261bac6fd2cf515557d5d0707481eafa0485ec2/unet/diffusion_pytorch_model.safetensors.index.json "HTTP/1.1 404 Not Found"
2026-11-03 16:22:37,918 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/scheduler/scheduler_config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:22:38,048 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/scheduler%2Fscheduler_config.json "HTTP/1.1 200 OK"
2026-11-03 16:22:44,653 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:22:44,731 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:23:28,858 INFO    t2o.engine.export: exported 153 translated images to runs\e3t-loop-s1\stage3\erased-translated\train (normalize=clamp)
2026-11-03 16:23:28,952 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:24:12,370 INFO    t2o.engine.export: exported 153 translated images to runs\e3t-loop-s1\stage3\erased-translated\val (normalize=clamp)
2026-11-03 16:24:12,378 INFO    t2o.cli: wrote runs\e3t-loop-s1\stage3\erased-translated\data.yaml
initialize network with normal
--- runs/e3t-loop-s3 ---
2026-11-03 16:24:35,603 INFO    t2o.translators.pix2pix_turbo: loading stabilityai/sd-turbo
2026-11-03 16:24:37,273 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/tokenizer/config.json "HTTP/1.1 404 Not Found"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-11-03 16:24:37,273 WARNING huggingface_hub.utils._http: Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-11-03 16:24:37,512 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/tokenizer/config.json "HTTP/1.1 404 Not Found"
2026-11-03 16:24:37,783 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/tokenizer/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:24:37,909 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/tokenizer%2Ftokenizer_config.json "HTTP/1.1 200 OK"
2026-11-03 16:24:38,159 INFO    httpx: HTTP Request: GET https://huggingface.co/api/models/stabilityai/sd-turbo/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
2026-11-03 16:24:38,400 INFO    httpx: HTTP Request: GET https://huggingface.co/api/models/stabilityai/sd-turbo/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
2026-11-03 16:24:38,768 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/text_encoder/config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:24:38,901 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/text_encoder%2Fconfig.json "HTTP/1.1 200 OK"
2026-11-03 16:24:39,139 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/text_encoder/config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:24:39,262 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/text_encoder%2Fconfig.json "HTTP/1.1 200 OK"
2026-11-03 16:24:39,511 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/text_encoder/model.safetensors "HTTP/1.1 302 Found"

Loading weights:   0%|          | 0/372 [00:00<?, ?it/s]
Loading weights:  35%|###5      | 132/372 [00:00<00:00, 1308.06it/s]
Loading weights:  72%|#######1  | 266/372 [00:00<00:00, 1323.03it/s]
Loading weights: 100%|##########| 372/372 [00:00<00:00, 1626.77it/s]
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\huggingface_hub\utils\_validators.py:205: UserWarning: The `local_dir_use_symlinks` argument is deprecated and ignored in `hf_hub_download`. Downloading to a local directory does not use symlinks anymore.
  warnings.warn(
2026-11-03 16:24:40,291 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/vae/config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:24:40,419 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/vae%2Fconfig.json "HTTP/1.1 200 OK"
2026-11-03 16:24:40,654 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/b261bac6fd2cf515557d5d0707481eafa0485ec2/vae/diffusion_pytorch_model.safetensors.index.json "HTTP/1.1 404 Not Found"
2026-11-03 16:24:40,985 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/unet/config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:24:41,111 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/unet%2Fconfig.json "HTTP/1.1 200 OK"
2026-11-03 16:24:41,351 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/b261bac6fd2cf515557d5d0707481eafa0485ec2/unet/diffusion_pytorch_model.safetensors.index.json "HTTP/1.1 404 Not Found"
2026-11-03 16:24:41,786 INFO    httpx: HTTP Request: HEAD https://huggingface.co/stabilityai/sd-turbo/resolve/main/scheduler/scheduler_config.json "HTTP/1.1 307 Temporary Redirect"
2026-11-03 16:24:41,914 INFO    httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/stabilityai/sd-turbo/b261bac6fd2cf515557d5d0707481eafa0485ec2/scheduler%2Fscheduler_config.json "HTTP/1.1 200 OK"
2026-11-03 16:24:48,778 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:24:48,858 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:25:34,080 INFO    t2o.engine.export: exported 153 translated images to runs\e3t-loop-s3\stage3\erased-translated\train (normalize=clamp)
2026-11-03 16:25:34,174 INFO    t2o.data.dataset: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch\val\visible\images: 153 paired samples, 153 annotated
2026-11-03 16:26:17,839 INFO    t2o.engine.export: exported 153 translated images to runs\e3t-loop-s3\stage3\erased-translated\val (normalize=clamp)
2026-11-03 16:26:17,850 INFO    t2o.cli: wrote runs\e3t-loop-s3\stage3\erased-translated\data.yaml
initialize network with normal
=== FINISHED 2026-11-03 16:26:20 +03:00 ===
```

### Step 5 — completion rate

```
--- runs/e3b-loop-s0 ---
2026-11-03 16:49:49,233 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:50:03,936 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-03 16:50:03,937 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-03 16:50:03,958 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\metrics.json stage 3
--- runs/e3b-loop-s1 ---
2026-11-03 16:50:23,199 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:50:36,546 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-03 16:50:36,547 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-03 16:50:36,566 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s1\metrics.json stage 3
--- runs/e3b-loop-s2 ---
2026-11-03 16:50:55,958 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:51:08,246 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-03 16:51:08,246 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-03 16:51:08,266 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s2\metrics.json stage 3
--- runs/e3b-loop-s3 ---
2026-11-03 16:51:27,834 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:51:41,179 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-03 16:51:41,180 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-03 16:51:41,199 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s3\metrics.json stage 3
--- runs/e3b-loop-s4 ---
2026-11-03 16:52:00,413 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:52:12,983 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-03 16:52:12,984 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-03 16:52:13,006 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s4\metrics.json stage 3
--- runs/e3b-loop-s5 ---
2026-11-03 16:52:33,280 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:52:46,472 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-03 16:52:46,472 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-03 16:52:46,491 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s5\metrics.json stage 3
--- runs/e3t-loop-s0 ---
2026-11-03 16:53:05,246 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:53:17,409 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-03 16:53:17,409 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-03 16:53:17,421 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s0\metrics.json stage 3
--- runs/e3t-loop-s1 ---
2026-11-03 16:53:37,559 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:53:49,597 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-03 16:53:49,601 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-03 16:53:49,614 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s1\metrics.json stage 3
--- runs/e3t-loop-s3 ---
2026-11-03 16:54:09,907 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-03 16:54:23,362 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-03 16:54:23,362 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-03 16:54:23,383 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s3\metrics.json stage 3
=== FINISHED 2026-11-03 16:54:25 +03:00 ===
```

### Step 6 — readout

The whole log, verbatim:

```
cell seed completion_rate missed_Fuse false_Fuse missed_Switch false_Switch missed_pooled false_pooled
---- ---- --------------- ----------- ---------- ------------- ------------ ------------- ------------
e3b     0            0.00        0.15       0.08          0.27         0.28          0.13         0.16
e3b     1            0.00        0.14       0.11          0.38         0.41          0.14         0.19
e3b     2            0.00        0.16       0.15          0.23         0.30          0.12         0.19
e3b     3            0.00        0.20       0.08          0.21         0.30          0.14         0.17
e3b     4            0.00        0.18       0.12          0.29         0.35          0.14         0.18
e3b     5            0.00        0.20       0.13          0.24         0.35          0.15         0.17


e3b completion_rate  n=6  mean=0.0000  std=0.0000 (sample, ddof=1)
e3b missed_Fuse      n=6  mean=0.1726  std=0.0229 (sample, ddof=1)
e3b false_Fuse       n=6  mean=0.1110  std=0.0284 (sample, ddof=1)
e3b missed_Switch    n=6  mean=0.2702  std=0.0602 (sample, ddof=1)
e3b false_Switch     n=6  mean=0.3324  std=0.0487 (sample, ddof=1)
e3b missed_pooled    n=6  mean=0.1359  std=0.0127 (sample, ddof=1)
e3b false_pooled     n=6  mean=0.1751  std=0.0132 (sample, ddof=1)

cell seed completion_rate missed_Fuse false_Fuse missed_Switch false_Switch missed_pooled false_pooled
---- ---- --------------- ----------- ---------- ------------- ------------ ------------- ------------
e3t     0            0.00        0.18       0.18          0.23         0.39          0.13         0.22
e3t     1            0.00        0.17       0.12          0.33         0.46          0.14         0.22
e3t     3            0.00        0.14       0.12          0.26         0.35          0.11         0.17


e3t completion_rate  n=3  mean=0.0000  std=0.0000 (sample, ddof=1)
e3t missed_Fuse      n=3  mean=0.1647  std=0.0209 (sample, ddof=1)
e3t false_Fuse       n=3  mean=0.1404  std=0.0342 (sample, ddof=1)
e3t missed_Switch    n=3  mean=0.2727  std=0.0546 (sample, ddof=1)
e3t false_Switch     n=3  mean=0.3963  std=0.0558 (sample, ddof=1)
e3t missed_pooled    n=3  mean=0.1261  std=0.0119 (sample, ddof=1)
e3t false_pooled     n=3  mean=0.2017  std=0.0289 (sample, ddof=1)
=== FINISHED 2026-11-03 16:55:00 +03:00 ===
```

Re-render from the results files without re-running: the step-6 script under `## Command`, run in
the repo root on VSIBMEOT05. It reads `judges.rtdetr` at stage 3 from each run's `metrics.json`.

## Findings

### F216 — The per-class re-score reproduces record 026 exactly, so the new per-class rates come from the same matches

ADR-03 required this run to check that the re-score leaves the pooled rates unchanged. It does.

- **Pooled rates:** on all nine runs, missed- and false-object rates before and after the
  write-back differ by `0.000000`. That is 18 comparisons against the runbook's 1e-4 stop line.
- **Arm means:** the readout's pooled means are e3b missed 0.1359 / false 0.1751 and e3t
  0.1261 / 0.2017. These equal record 026's loop-arm means quoted under F214 to four decimals.
- **The completion scorer agrees with the faithfulness scorer.** On the unerased e3b seed 0 export,
  the completion rate is the judge's own recall. It read 0.8747. One minus the stored missed-object
  rate, 1 − 0.125296, is 0.874704.
- This is the same computation on the same exports. It is a reproduction, not second evidence for
  any claim of record 026.

### F217 — Stage 1 puts back none of the erased fuses and switches: the completion rate is 0.0000 on all nine loop runs

The comparison is the same runs on the **unerased** frames, from the step-3 re-score. There the
judge finds 80–86% of the fuses (per-class missed 0.1429–0.1964) and 62–79% of the switches
(0.2121–0.3788). After erasure it finds none, on every run:

| cell | seeds | completion rate, every seed | mean | seed sd |
| --- | --- | --- | --- | --- |
| e3b (pix2pix) | 0–5 | 0.0000 | 0.0000 | 0.0000 |
| e3t (turbo) | 0, 1, 3 | 0.0000 | 0.0000 | 0.0000 |

- **The denominator is 234 boxes**: 168 fuses and 66 switches. The erase step reported 234. The
  split by class is derived here, not printed. Every per-class missed rate in step 3 is a whole
  number of boxes out of 168 (Fuse) or 66 (Switch), for example 26/168 = 0.1548 and 18/66 = 0.2727,
  and 168 + 66 = 234.
- **One box put back on one seed would read 0.0043** (1/234). So this result means no box came back,
  not that a few rounded away.
- **Both backbones behave the same.** The Telea fill with a 4 px rim leaves neither pix2pix nor
  turbo anything to complete a fuse or switch from. By ADR-03's definition, any completion stage 2
  adds is an invention, measured against a stage-1 floor of exactly zero.
- **The scorer's own denominator is not in the log.** A later label count on the server found the
  234 boxes on disk (provenance caveats).

### F218 — Plan Q3's margin rule gives the completion rate a margin of zero, and under ADR-01's wording that test cannot pass

Plan Q3 takes the completion margin from the stage-1 loop arm's seed spread. F217 makes that spread
0.0000 in both cells, so the margin would be 0.

ADR-01's pass rule, which ADR-04 is to follow, passes a lower-is-better rate when the seed-paired
95% CI lies "below +margin" (`docs/features/out-of-family-judge/decisions.md:36-37`). Stage 1's
completion is 0 on every seed. So cascade − stage 1 is the cascade's own completion rate on each
seed, which can never be negative. Its CI's upper end is then 0 or more, never below 0. **The test
fails whatever stage 2 does**, even if stage 2 also puts back nothing.

This applies the rule's wording to this run's numbers. No rule was run. The per-class rates do not
have this problem: every per-class seed sd is above zero (F219).

### F219 — Per-class seed spreads are 1.2–4.7× the pooled ones and widest on Switch, as ADR-03 expected

ADR-03 expected per-class margins to come out wider than pooled ones, because each class rests on
fewer boxes. The comparison is each per-class seed sd against the same cell's pooled sd, from the
readout:

| cell (n) | rate | Fuse sd | Switch sd | pooled sd | Fuse / pooled | Switch / pooled |
| --- | --- | --- | --- | --- | --- | --- |
| e3b (6) | missed-object | 0.0229 | 0.0602 | 0.0127 | 1.8× | 4.7× |
| e3b (6) | false-object | 0.0284 | 0.0487 | 0.0132 | 2.2× | 3.7× |
| e3t (3) | missed-object | 0.0209 | 0.0546 | 0.0119 | 1.8× | 4.6× |
| e3t (3) | false-object | 0.0342 | 0.0558 | 0.0289 | 1.2× | 1.9× |

- **The granularity is coarse.** Val has 66 switches, so one switch moves the Switch missed rate by
  0.0152. e3b's Switch missed sd, 0.0602, is about four switches. One fuse moves the Fuse missed
  rate by 0.0060 (1/168). The false-object denominators are prediction counts, which differ by
  seed, so they have no fixed box size.
- **Switch is also the worst class on level.** Mean Switch missed-object rate 0.2702 (e3b) and
  0.2727 (e3t), against Fuse 0.1726 / 0.1647 and pooled 0.1359 / 0.1261. Mean Switch false-object
  rate 0.3324 / 0.3963, against Fuse 0.1110 / 0.1404 and pooled 0.1751 / 0.2017.
- **e3t's sds rest on three seeds**, so they are unstable, as ADR-01 noted for its pooled margins.

## Provenance caveats

- **No log shows the completion scorer's denominator.** `CompletionEvaluator.compute` returns 0.0
  when it read no erased box (`src/t2o/metrics/faithfulness.py:282-284`). A missing label file also
  loads as zero boxes rather than an error (`src/t2o/data/labels.py:21`). So the logs alone cannot
  tell "0 of 234 put back" apart from "no labels read". Reading the code, `t2o erase` writes the
  labels beside the erased images (`src/t2o/data/erasure.py:202-210`), which is where the scorer
  looks them up (`src/t2o/data/pairing.py:42-46`). The 168 + 66 split also matches the erase step's
  234. **A label count on the server confirmed it on 2026-11-04**, after the run. Under
  `runs/cascade-gate/erased-custom-val-fuse-switch/val/visible/labels` there are 153 files and 234
  lines, class 1 (Fuse) 168 and class 3 (Switch) 66, and no other class. The user reported that
  count from PowerShell, and no log of it was saved. It confirms the labels on disk. That the scorer
  read exactly those files is still a reading of the code, because `evaluate_completion` does not
  log its count.
- **The eye check of the erased frames** (runbook step 1: fuses and switches gone, poles and
  transformers untouched) is in no log. Whether it was done is not recorded.
- **The readout log has no provenance block.** Its machine, git_sha, environment and exact command
  are not recorded, and neither is its start time. It ran after the completion step, whose FINISHED
  stamp is 16:54:25, and its own FINISHED stamp is 16:55:00. It only reads the `metrics.json` files
  the five logged steps wrote. The command under `## Command` comes from the runbook, not the log.
- **The readout table rounds to two decimals** (PowerShell `Format-Table`). The summary lines carry
  four. The per-seed per-class values at four decimals are in the step-3 log.
- **The write-backs changed nine `metrics.json` files.** Step 3 replaced each run's
  `judges.rtdetr.faithfulness` with a version that adds the per-class keys. F216 shows the pooled
  values are unchanged. Step 5 added `judges.rtdetr.completion` at stage 3.
- **No control arm and no paired test.** This is by design (plan Q3). Nothing here says anything
  about loop versus control.
- **The turbo exports fetched `stabilityai/sd-turbo` from the Hub unauthenticated**, resolving to
  commit `b261bac6fd2c…`. That this is the same base revision the e3t runs trained on is not checked
  here.

## Next

ADR-04 (CASCADE-GATE-04) is now unblocked. It can take the Fuse and Switch per-class margins from
F219 as plan Q3 says. It has to settle F218 before any stage-2 output is scored, since a zero margin
under ADR-01's strict rule fails every cascade. That is ADR-04's decision, not this record's.

**A prediction this record can be checked against:** any non-zero completion rate under stage 2 is
an invented fuse or switch, because stage 1 puts back none of the 234 on any seed (F217).
