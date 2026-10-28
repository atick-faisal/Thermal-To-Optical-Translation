# 033 — Faithfulness decided on val (ADR-02): 11 of 12 rate tests pass, turbo's false-object rate fails; and ModTr's rates under the RT-DETR judge

**Date:** 2026-10-28 · **Task:** [no-task] · **Machine:** VSIBMEOT05 (custom set on `cuda:0`, FLIR on `cuda:1`, in parallel) · **Wall clock:** custom 12 min 13 s (10:07:46 → 10:19:59 +03:00), FLIR 6 min 31 s (10:08:18 → 10:14:49 +03:00), readout 6 s (10:33:49 → 10:33:55 +03:00)
**git_sha:** `pre-spec-migration-193-g477c9f7` (a clean tree at `477c9f7`, no `-dirty` suffix) · **W&B:** none (no W&B line in any log) · **Log:** `logs/2026-10-28-modtr-faithfulness-custom.txt`, `logs/2026-10-28-modtr-faithfulness-flir.txt`, `logs/2026-10-28-modtr-faithfulness-readout.txt`, with `logs/2026-10-28-modtr-faithfulness-rtdetr.csv`

This record does two things. It scores the six ModTr exports for Faithfulness under the RT-DETR
judge, which is the server run the logs hold. It also applies ADR-02
(`docs/features/out-of-family-judge/decisions.md`) to record 026's stage-3 loop − control numbers,
which is the verdict that closes Faithfulness. The verdict needed no new run.

## Command

From the provenance blocks, as run. Each track ran its three seeds in one loop:

```
# Track A, cuda:0, seeds 0, 1, 3
uv run t2o faithfulness --translated runs/modtr-s{0,1,3}/stage0/translated --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --weights runs/reference-rtdetr-l/weights/best.pt --write-back --judge rtdetr --device cuda:0

# Track B, cuda:1, seeds 0, 1, 3
uv run t2o faithfulness --translated runs/modtr-flir-s{0,1,3}/stage0/translated --data dataset/processed/flir/data.yaml --weights runs/reference-flir-rtdetr-l/weights/best.pt --write-back --judge rtdetr --device cuda:1

# Readout
read judges.rtdetr.faithfulness from runs\modtr-*\metrics.json, stage 0
```

The `{0,1,3}` is shorthand here. Each log holds three separate provenance blocks, one per seed,
each with its own literal command. The readout is the runbook's PowerShell loop, and its
provenance line is the label the runbook gave it.

## Environment

The first block of the custom-set log. The other six blocks (two more custom, three FLIR, one
readout) differ only in their `captured` and `command` lines:

```
=== PROVENANCE ===
captured: 2026-10-28 10:07:46 +03:00
hostname: VSIBMEOT05
os: Microsoft Windows Server 2022 Standard 10.0.20348 (build 20348)
cpu: AMD EPYC 7763 64-Core Processor
cpu_logical_cores: 64
ram_total: 128.0 GB
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16; NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16
cuda_driver: 13.0
git_sha: pre-spec-migration-193-g477c9f7
working_dir: D:\Atick\GitHub\Thermal-To-Optical-Translation
command: uv run t2o faithfulness --translated runs/modtr-s0/stage0/translated --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --weights runs/reference-rtdetr-l/weights/best.pt --write-back --judge rtdetr --device cuda:0
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
| Judge | `rtdetr`: record 026's RT-DETR-l checkpoints, `runs/reference-rtdetr-l` (custom) and `runs/reference-flir-rtdetr-l` (FLIR). ADR-01 ties the margins to exactly these | the commands |
| Faithfulness settings | `t2o faithfulness` defaults (IoU 0.5, conf 0.25, batch 16, imgsz 640), as in record 026's step 5 | no flag overrides them in the commands |
| Scored split | val: 153 custom pairs, 1,013 FLIR pairs | `faithfulness val on 153 pairs` / `on 1013 pairs` |
| ModTr arm | records 030 and 031's exports, `runs/modtr{,-flir}-s{0,1,3}/stage0/translated` | the commands |
| ModTr summary | mean ± sample sd (ddof = 1) over seeds 0, 1, 3, computed on the Mac from the CSV. No CI, as decided on 2026-10-28. `t2o aggregate` prints the same sd convention (`src/t2o/analysis/aggregate.py:102`, `:480`) | computed for this record |
| Verdict rule | ADR-02: ADR-01's pass rule on val. A rate passes when the 95% bootstrap CI of the seed-paired stage-3 loop − control difference lies below +margin (false-object, missed-object) or above −margin (consistency) | `decisions.md`, ADR-01 and ADR-02 |
| Verdict inputs | record 026's `judges.rtdetr.faithfulness` paired lines (`026-out-of-family-rtdetr-judge.md:676-681`, `:1041-1046`, `:1174-1179`, `:1247-1252`) and ADR-01's margin table (F186) | quoted under F213 |

## Results

### ModTr, custom set (`cuda:0`)

Everything after the provenance blocks, verbatim:

```
2026-10-28 10:18:08,841 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-10-28 10:18:29,458 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.2674  missed-object 0.1773  consistency 0.7887
2026-10-28 10:18:29,458 INFO    t2o.cli:   false-object rate       0.2674  (LOWER better)
2026-10-28 10:18:29,458 INFO    t2o.cli:   missed-object rate      0.1773  (LOWER better)
2026-10-28 10:18:29,458 INFO    t2o.cli:   detection-consistency   0.7887  (higher better)
2026-10-28 10:18:29,505 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s0\metrics.json stage 0
2026-10-28 10:18:57,120 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-10-28 10:19:12,930 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.2511  missed-object 0.2033  consistency 0.7734
2026-10-28 10:19:12,930 INFO    t2o.cli:   false-object rate       0.2511  (LOWER better)
2026-10-28 10:19:12,930 INFO    t2o.cli:   missed-object rate      0.2033  (LOWER better)
2026-10-28 10:19:12,930 INFO    t2o.cli:   detection-consistency   0.7734  (higher better)
2026-10-28 10:19:12,967 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s1\metrics.json stage 0
2026-10-28 10:19:40,056 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-10-28 10:19:57,793 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.2629  missed-object 0.1584  consistency 0.8105
2026-10-28 10:19:57,793 INFO    t2o.cli:   false-object rate       0.2629  (LOWER better)
2026-10-28 10:19:57,793 INFO    t2o.cli:   missed-object rate      0.1584  (LOWER better)
2026-10-28 10:19:57,793 INFO    t2o.cli:   detection-consistency   0.8105  (higher better)
2026-10-28 10:19:57,835 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-s3\metrics.json stage 0
=== FINISHED 2026-10-28 10:19:59 +03:00 ===
```

### ModTr, FLIR (`cuda:1`)

```
2026-10-28 10:09:28,863 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir
2026-10-28 10:11:17,990 INFO    t2o.metrics.faithfulness: faithfulness val on 1013 pairs: false-object 0.4280  missed-object 0.2685  consistency 0.4152
2026-10-28 10:11:17,990 INFO    t2o.cli:   false-object rate       0.4280  (LOWER better)
2026-10-28 10:11:17,990 INFO    t2o.cli:   missed-object rate      0.2685  (LOWER better)
2026-10-28 10:11:17,998 INFO    t2o.cli:   detection-consistency   0.4152  (higher better)
2026-10-28 10:11:18,032 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s0\metrics.json stage 0
2026-10-28 10:11:46,435 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir
2026-10-28 10:13:02,827 INFO    t2o.metrics.faithfulness: faithfulness val on 1013 pairs: false-object 0.4118  missed-object 0.2794  consistency 0.4170
2026-10-28 10:13:02,845 INFO    t2o.cli:   false-object rate       0.4118  (LOWER better)
2026-10-28 10:13:02,845 INFO    t2o.cli:   missed-object rate      0.2794  (LOWER better)
2026-10-28 10:13:02,845 INFO    t2o.cli:   detection-consistency   0.4170  (higher better)
2026-10-28 10:13:02,892 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s1\metrics.json stage 0
2026-10-28 10:13:31,182 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir
2026-10-28 10:14:47,001 INFO    t2o.metrics.faithfulness: faithfulness val on 1013 pairs: false-object 0.3846  missed-object 0.3308  consistency 0.3836
2026-10-28 10:14:47,011 INFO    t2o.cli:   false-object rate       0.3846  (LOWER better)
2026-10-28 10:14:47,011 INFO    t2o.cli:   missed-object rate      0.3308  (LOWER better)
2026-10-28 10:14:47,011 INFO    t2o.cli:   detection-consistency   0.3836  (higher better)
2026-10-28 10:14:47,056 INFO    t2o.engine.loop: recorded judge rtdetr into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-s3\metrics.json stage 0
=== FINISHED 2026-10-28 10:14:49 +03:00 ===
```

### Readout

```
run           false_object_rate missed_object_rate detection_consistency
---           ----------------- ------------------ ---------------------
modtr-s0                   0.27               0.18                  0.79
modtr-flir-s0              0.43               0.27                  0.42
modtr-s1                   0.25               0.20                  0.77
modtr-flir-s1              0.41               0.28                  0.42
modtr-s3                   0.26               0.16                  0.81
modtr-flir-s3              0.38               0.33                  0.38


=== FINISHED 2026-10-28 10:33:55 +03:00 ===
```

`logs/2026-10-28-modtr-faithfulness-rtdetr.csv`, verbatim:

```
"run","false_object_rate","missed_object_rate","detection_consistency"
"modtr-s0","0.2673684210526316","0.1773049645390071","0.7886710239651417"
"modtr-flir-s0","0.42802617230098144","0.2684797768479777","0.4152240786537186"
"modtr-s1","0.2511111111111111","0.2033096926713948","0.7734204793028322"
"modtr-flir-s1","0.4117647058823529","0.2794049279404928","0.41697327944990653"
"modtr-s3","0.2629399585921325","0.15839243498817968","0.8104575163398693"
"modtr-flir-s3","0.3846318264400983","0.3307763830776383","0.3835575125158333"
```

ModTr mean ± sd over seeds 0, 1, 3, computed on the Mac from the CSV (`statistics.stdev`, ddof = 1):

| dataset | false-object | missed-object | consistency |
| --- | --- | --- | --- |
| custom | 0.2605 ± 0.0084 | 0.1797 ± 0.0226 | 0.7908 ± 0.0186 |
| FLIR | 0.4081 ± 0.0219 | 0.2929 ± 0.0333 | 0.4053 ± 0.0188 |

Re-render without re-scoring: the runbook's readout loop (`modtr-faithfulness-runbook.md`) reads
the six `metrics.json` files the scoring wrote back.

## Findings

### F213 — Faithfulness is decided on val: 11 of 12 rate tests pass, and turbo's false-object rate fails because its n = 3 interval reaches +0.0538, past its 0.0315 margin

ADR-02 applies ADR-01's rule to record 026's stage-3 loop − control differences under `rtdetr`.
The comparison is each rate's paired 95% CI against its cell's margin, which is the control arm's
seed sd (F186). The p values are record 026's exact sign-flip p against zero, which ADR-02 asks to
be reported beside each pass.

| cell (n) | rate | loop − control | 95% CI | margin | rule | p | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e3b (6) | false-object | −0.0177 | [−0.0324, +0.0014] | 0.0179 | upper < +0.0179 | .156 | pass |
| e3b (6) | missed-object | −0.0299 | [−0.0394, −0.0173] | 0.0094 | upper < +0.0094 | .031 | pass |
| e3b (6) | consistency | +0.0309 | [+0.0200, +0.0418] | 0.0075 | lower > −0.0075 | .031 | pass |
| e3t (3) | false-object | +0.0040 | [−0.0461, **+0.0538**] | 0.0315 | upper < +0.0315 | .750 | **FAIL** |
| e3t (3) | missed-object | −0.0173 | [−0.0284, −0.0024] | 0.0112 | upper < +0.0112 | .250 | pass |
| e3t (3) | consistency | +0.0269 | [+0.0240, +0.0305] | 0.0038 | lower > −0.0038 | .250 | pass |
| e3f (6) | false-object | −0.1487 | [−0.1647, −0.1356] | 0.0169 | upper < +0.0169 | .031 | pass |
| e3f (6) | missed-object | −0.3359 | [−0.4504, −0.2234] | 0.1708 | upper < +0.1708 | .031 | pass |
| e3f (6) | consistency | +0.1997 | [+0.1153, +0.2820] | 0.1222 | lower > −0.1222 | .031 | pass |
| e3md (6) | false-object | −0.0989 | [−0.1103, −0.0842] | 0.0171 | upper < +0.0171 | .031 | pass |
| e3md (6) | missed-object | −0.1306 | [−0.1397, −0.1223] | 0.0143 | upper < +0.0143 | .031 | pass |
| e3md (6) | consistency | +0.1001 | [+0.0925, +0.1070] | 0.0107 | lower > −0.0107 | .031 | pass |

- **FLIR and msrs-day pass on every rate**, and every interval lies on the loop's side of zero, not
  only inside the margin. FLIR's missed-object and consistency margins barely constrain the result
  (ADR-01). Its false-object test, with a 0.0169 margin, passes by 0.15.
- **pix2pix on the custom set (e3b) passes on every rate.** Its false-object pass is the weakest
  in the table. The interval's upper end, +0.0014, crosses zero, and p = .156, so the pass shows no
  difference either way, not a demonstrated equivalence (ADR-02).
- **turbo on the custom set (e3t) fails on false-object.** Its point estimate, +0.0040, sits well
  inside the 0.0315 margin. The fail comes from the width of the n = 3 interval, whose upper end is
  0.0223 past the margin. Under ADR-02 a fail is robust, because an honest interval at n = 3 would
  be wider still. It is reported as a fail and not outvoted by e3b.
- **turbo's other two rates pass**, on intervals entirely on the loop's side of zero. Consistency
  clears a margin of 0.0038, the tightest in the table.

### F214 — On the custom set, ModTr is worse than both loops and both controls on every faithfulness rate: false-object 0.2605 against 0.1751–0.2017

The comparison is record 026's stage-3 arm means under the same judge checkpoint
(`026-out-of-family-rtdetr-judge.md:1145-1150` for e3b, `:1218-1223` for e3t). These are point
estimates over different seed sets: e3b uses seeds 0–5, while e3t and ModTr use 0, 1 and 3.

| arm | false-object | missed-object | consistency |
| --- | --- | --- | --- |
| ModTr (n = 3) | **0.2605 ± 0.0084** | **0.1797 ± 0.0226** | **0.7908 ± 0.0186** |
| e3b loop (6) | 0.1751 | 0.1359 | 0.8384 |
| e3b control (6) | 0.1928 | 0.1659 | 0.8076 |
| e3t loop (3) | 0.2017 | 0.1261 | 0.8613 |
| e3t control (3) | 0.1977 | 0.1434 | 0.8344 |

- **ModTr invents the most objects.** Its false-object rate is 0.0854 above the e3b loop's and
  0.0588 above the e3t loop's, against seed sds of 0.0084 (ModTr) and 0.0132 / 0.0289 (the loops).
- **It also misses the most** (0.1797 against 0.1261–0.1659) and is the least consistent (0.7908
  against 0.8076–0.8613).
- **This matches its mAP ranking on the custom set.** F201 found ModTr weaker than the loop's own
  no-loop translator. Here it is weaker on all three faithfulness rates too.

### F215 — On FLIR the order flips: ModTr's false-object rate is 0.4081 against the e3f loop's 0.6171, its missed-object rate is lower too, and its consistency ties the loop's

The comparison is record 026's e3f stage-3 arm means under FLIR's RT-DETR judge
(`026-out-of-family-rtdetr-judge.md:569-574`), point estimates, with e3f over seeds 0–5 and ModTr
over 0, 1 and 3.

| arm | false-object | missed-object | consistency |
| --- | --- | --- | --- |
| ModTr (n = 3) | **0.4081 ± 0.0219** | **0.2929 ± 0.0333** | **0.4053 ± 0.0188** |
| e3f loop (6) | 0.6171 | 0.3329 | 0.4120 |
| e3f control (6) | 0.7658 | 0.6688 | 0.2123 |

- **ModTr invents far fewer objects than the loop**, 0.2090 lower, about 10× either arm's seed sd
  (0.0219 ModTr, 0.0210 loop).
- **It misses fewer**, 0.0400 lower. That is about one ModTr seed sd (0.0333), so this gap is
  small.
- **Consistency is a tie.** 0.4053 against 0.4120 is a gap of 0.0067, inside ModTr's 0.0188 sd.
- **This matches F207's reading on mAP.** On FLIR, ModTr is the best zero-shot translator measured,
  and here it is the most faithful on false-object as well. The leading reading of why FLIR reverses
  the custom set's order is still the roll confound (F119), which reaches pix2pix and not ModTr.
  This run does not test it.

## Provenance caveats

- **F213's numbers are composed from record 026**, scored on 2026-10-19 at `8fe8fea`. No server
  run here produced them. This record applies a rule to them and adds no new evidence about the
  loop.
- **The margins and the verdict come from the same val data.** Record 026 printed the paired
  differences on 2026-10-19 and ADR-01 fixed the margins on 2026-10-20 from the control spread.
  ADR-02 and `goal.md:63-64` require this to be disclosed.
- **Passes rest on an anti-conservative interval.** The percentile bootstrap at n = 3–6 is too
  narrow (ADR-02, `design.md:664`). The one fail is robust to that. The eleven passes are weaker
  than they look, most of all the three e3t passes at n = 3, which have p = .25.
- **ModTr's comparisons in F214 and F215 are unpaired point estimates, without CIs**, as decided on
  2026-10-28. The seed sets differ: e3b and e3f use 0–5, while ModTr and e3t use 0, 1 and 3.
- **ModTr is a one-shot translator, scored at its only stage (`stage0`).** The loop and control are
  scored at stage 3. ModTr sits outside the pass rule (ADR-02).
- **The FLIR judge is its epoch-1 checkpoint (F187)**, inherited from record 026. ADR-01 binds the
  margins to that checkpoint, and this run used the same one.
- **Record 026 does not print its faithfulness pair counts.** This run scored the full val splits,
  153 and 1,013 pairs, with the same default settings. That record 026 scored the same pairs is an
  assumption from those defaults, not a check.
- **The custom track's first seed started late.** Its provenance block was captured at 10:07:46,
  but its first `t2o` line is at 10:18:08, a gap of 10 min 22 s. The FLIR track's first gap was
  70 s, and every later seed's was 23–24 s. The log does not show the cause. The gap is idle time,
  not scoring time, and no number depends on it.
- **The log filenames carry no task segment** (`logs/2026-10-28-modtr-faithfulness-*.txt`), so
  they miss the `logs/<date>-<task>-<what>.txt` convention. No task owns this run (ADR-02's
  `Tasks: none`). They are cited as saved.
- **The CSV is the server's `runs\modtr-faithfulness-rtdetr.csv`**, the runbook's output path,
  copied back under `logs/` with a date prefix. Its values match the per-seed log lines to four
  decimals. The readout table rounds to two decimals because of PowerShell's `Format-Table`.

## Next

Faithfulness is now decided, per cell and per rate (F213). ADR-02 says nothing reopens it, since no
test split will arrive to re-score it.

- **The paper reports the custom set's Faithfulness as split.** pix2pix passes on all three rates.
  turbo passes on two and fails on false-object. The paper has to give the reason in ADR-02's
  terms: a +0.0040 point estimate with p = .75, failed by the width of an n = 3 interval, not by
  more invented objects.
- **ModTr's rates go in beside the loop and control**, as `goal.md:65` asks ("for every arm"). They
  tell the same story as Margin. On the custom set ModTr is the least faithful arm (F214). On FLIR
  it is the most faithful on false-object (F215), which mirrors its FLIR Margin lead (F205, F210).
- **No further Faithfulness run is suggested by this result.** More turbo seeds or a retuned margin
  are what ADR-02 rejected.
