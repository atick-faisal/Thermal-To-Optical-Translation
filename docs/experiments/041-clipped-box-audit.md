# 041 — Clipped-box audit: the build's clamp leaves 17% of custom val boxes with under half their area, yet every detector already matches 88–99% of them at IoU 0.5; plan Q3's rule says fix on every arm at both thresholds, through the drop's own false positives

**Date:** 2026-11-16 → 2026-11-17 (steps 1–2 on 11-16, the rest on 11-17) · **Task:** CLIPPED-BOX-AUDIT-05 · **Machine:** VSIBMEOT05, 2× A100 (routes on `cuda:0`, single arms on `cuda:1`, run concurrently) · **Wall clock:** 2026-11-16 20:15:52 → 2026-11-17 15:40:03 +03:00, an overnight gap between steps 2 and 3a; per step below
**git_sha:** `pre-spec-migration-310-gccfce13` (all 29 provenance blocks; clean, no `-dirty` suffix) · **W&B:** none (no W&B line in any log) · **Log:** `logs/2026-11-16-CLIPPED-BOX-AUDIT-05-build-log-search.txt`, `logs/2026-11-16-CLIPPED-BOX-AUDIT-05-polygon-check.txt`, `logs/2026-11-16-CLIPPED-BOX-AUDIT-05-label-source.txt`, `logs/2026-11-16-CLIPPED-BOX-AUDIT-05-audit.txt`, `logs/2026-11-16-CLIPPED-BOX-AUDIT-05-looseness.txt`, `logs/2026-11-16-CLIPPED-BOX-AUDIT-05-drop-trees.txt`, `logs/2026-11-16-CLIPPED-BOX-AUDIT-05-rescore-routes.txt`, `logs/2026-11-17-CLIPPED-BOX-AUDIT-05-rescore-single.txt`, `logs/2026-11-16-CLIPPED-BOX-AUDIT-05-rescore-routes-rerun.txt`; with `logs/2026-11-16-CLIPPED-BOX-AUDIT-05-audit.csv`, `logs/2026-11-16-CLIPPED-BOX-AUDIT-05-looseness-val.csv`, `logs/2026-11-16-CLIPPED-BOX-AUDIT-05-looseness-train.csv`, `logs/2026-11-16-CLIPPED-BOX-AUDIT-05-rescore-routes-00px.csv`, `-03px.csv`, `-06px.csv`, `-12px.csv`, `-24px.csv`

This run is the clipped-box audit of `docs/features/clipped-box-audit/plan.md`. It trains nothing.
The custom set's boxes were drawn on the optical image, and the sister build script moves them onto
the thermal frame. It warps each box's corners, re-boxes them, clamps the box to the frame and drops
it below 25% visibility. The clamp cuts only the axis that crosses the edge, so a thin diagonal
object such as a pole is left in a loose box. The run measures that looseness in the frozen labels
and in the registration axis's own re-boxing. It then re-scores the checkpoints behind the paper's
custom numbers on val copies with the loosest boxes removed, and applies plan Q3's rule, which was
fixed before any score.

Per-step wall clock (start → `FINISHED`, +03:00):

| Step | Log | Start | Finish | Wall clock |
| --- | --- | --- | --- | --- |
| 1. build log search | `build-log-search` | 11-16 20:15:52 | 20:16:39 | 47 s |
| 2. polygon check | `polygon-check` | 11-16 20:17:26 | 20:18:02 | 36 s |
| 3a. label source | `label-source` | 11-17 09:44:37 | 09:44:48 | 11 s |
| 3. frozen-label audit | `audit` | 11-17 10:07:11 | 10:07:36 | 25 s |
| 4. registration looseness | `looseness` | 11-17 10:07:49 | 10:10:26 | 2 min 37 s |
| 5. drop trees | `drop-trees` | 11-17 10:15:24 | 10:24:55 | 9 min 31 s |
| 6. routes, first run (`cuda:0`) | `rescore-routes` | 11-17 10:34:14 | 12:01:23 | 1 h 27 min 09 s |
| 7. single arms (`cuda:1`) | `rescore-single` | 11-17 10:40:19 | 13:48:26 | 3 h 08 min 07 s |
| 6b. routes, rerun (`cuda:0`) | `rescore-routes-rerun` | 11-17 12:57:13 | 15:40:03 | 2 h 42 min 50 s |

A three-arm route call takes 37–43 min. A single-arm call takes about 14.5 min.

## Command

```
# 1. build-log-search
search D:\Atick\data\dataset\Brazil-Aligned for 'hit the frame edge'; search PSReadLine history for build_rgbt_dataset

# 2. polygon-check
search D:\Atick\data for Label Studio JSON with polygons; count >5-field lines in D:\Atick\data\dataset\Brazil-Aligned\yolo_original

# 3a. label-source
find every FLIR1448 label under D:\Atick\data; compare D:\Atick\data\dataset\Brazil-Aligned\yolo_original with D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul (dates, names, classes, line counts)

# 3. audit
uv run python scripts/clipped_box_audit.py --original-dir D:\Atick\data\dataset\Brazil-Aligned\brazil_all_29_jul_2026_yolo --alignment-dir D:\Atick\data\dataset\Brazil-Aligned\alignment --frozen-dir D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul --out runs\clipped-box-audit\audit.csv --splits train val

# 4. looseness (two blocks)
uv run python scripts/registration_box_looseness.py --labels D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\val\infrared\labels --out runs\clipped-box-audit\looseness-val.csv
uv run python scripts/registration_box_looseness.py --labels D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\train\infrared\labels --out runs\clipped-box-audit\looseness-train.csv

# 5. drop-trees (the block's command line is a description; the runbook's step 5 loop ran the builders)
clipped_box_drop_tree.py on D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml and on runs/{e3b-loop-s0,regc-03px-s0,regc-06px-s0,regc-12px-s0,regc-24px-s0}/stage3/translated at thresholds 0.5, 0.75, audit runs\clipped-box-audit\audit.csv

# 6. rescore-routes, first run (0 px good; 3 px paired with regc-12px-s0, see caveats; 6, 12, 24 px crashed)
uv run python scripts/clipped_box_rescore.py --arm label-transfer runs\registration-custom\detectors\custom-l0-s0\weights\best.pt 0.003989 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --arm zero-shot runs\reference-yolo11s\weights\best.pt 0.013972 runs\e3b-loop-s0\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\e3b-loop-s0-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\e3b-loop-s0-0.75\data.yaml --arm adapted D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\detector\stage3\weights\best.pt 0.005033 runs\e3b-loop-s0\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\e3b-loop-s0-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\e3b-loop-s0-0.75\data.yaml --out runs\clipped-box-audit\rescore\routes-00px.csv --device cuda:0
uv run python scripts/clipped_box_rescore.py --arm label-transfer runs\registration-custom\detectors\custom-l3-s0\weights\best.pt 0.001684 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --arm zero-shot runs\reference-yolo11s\weights\best.pt 0.013972 runs\regc-12px-s0\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\regc-12px-s0-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\regc-12px-s0-0.75\data.yaml --arm adapted D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\regc-12px-s0\stage3\detector\stage3\weights\best.pt 0.005033 runs\regc-12px-s0\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\regc-12px-s0-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\regc-12px-s0-0.75\data.yaml --out runs\clipped-box-audit\rescore\routes-03px.csv --device cuda:0
uv run python scripts/clipped_box_rescore.py --arm label-transfer runs\registration-custom\detectors\custom-l6-s0\weights\best.pt 0.003033 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --arm zero-shot  0.013972 runs\\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\-0.75\data.yaml --arm adapted  0.005033 runs\\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\-0.75\data.yaml --out runs\clipped-box-audit\rescore\routes-06px.csv --device cuda:0
uv run python scripts/clipped_box_rescore.py --arm label-transfer runs\registration-custom\detectors\custom-l12-s0\weights\best.pt 0.005257 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --arm zero-shot  0.013972 runs\\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\-0.75\data.yaml --arm adapted  0.005033 runs\\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\-0.75\data.yaml --out runs\clipped-box-audit\rescore\routes-12px.csv --device cuda:0
uv run python scripts/clipped_box_rescore.py --arm label-transfer runs\registration-custom\detectors\custom-l24-s0\weights\best.pt 0.02065 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --arm zero-shot  0.013972 runs\\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\-0.75\data.yaml --arm adapted  0.005033 runs\\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\-0.75\data.yaml --out runs\clipped-box-audit\rescore\routes-24px.csv --device cuda:0

# 7. rescore-single
uv run python scripts/clipped_box_rescore.py --arm e8-A-n600-s0 runs\e8\detectors\A-n600-s0\weights\best.pt 0.0096 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --out runs\clipped-box-audit\rescore\single-e8-A-n600-s0.csv --device cuda:1
uv run python scripts/clipped_box_rescore.py --arm e8-A-n600-s1 runs\e8\detectors\A-n600-s1\weights\best.pt 0.0096 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --out runs\clipped-box-audit\rescore\single-e8-A-n600-s1.csv --device cuda:1
uv run python scripts/clipped_box_rescore.py --arm e8-A-n600-s2 runs\e8\detectors\A-n600-s2\weights\best.pt 0.0096 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --out runs\clipped-box-audit\rescore\single-e8-A-n600-s2.csv --device cuda:1
uv run python scripts/clipped_box_rescore.py --arm label-transfer-l0-s1 runs\registration-custom\detectors\custom-l0-s1\weights\best.pt 0.003989 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --out runs\clipped-box-audit\rescore\single-label-transfer-l0-s1.csv --device cuda:1
uv run python scripts/clipped_box_rescore.py --arm label-transfer-l0-s2 runs\registration-custom\detectors\custom-l0-s2\weights\best.pt 0.003989 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --out runs\clipped-box-audit\rescore\single-label-transfer-l0-s2.csv --device cuda:1
uv run python scripts/clipped_box_rescore.py --arm label-transfer-l3-s1 runs\registration-custom\detectors\custom-l3-s1\weights\best.pt 0.001684 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --out runs\clipped-box-audit\rescore\single-label-transfer-l3-s1.csv --device cuda:1
uv run python scripts/clipped_box_rescore.py --arm label-transfer-l3-s2 runs\registration-custom\detectors\custom-l3-s2\weights\best.pt 0.001684 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --out runs\clipped-box-audit\rescore\single-label-transfer-l3-s2.csv --device cuda:1
uv run python scripts/clipped_box_rescore.py --arm label-transfer-l6-s1 runs\registration-custom\detectors\custom-l6-s1\weights\best.pt 0.003033 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --out runs\clipped-box-audit\rescore\single-label-transfer-l6-s1.csv --device cuda:1
uv run python scripts/clipped_box_rescore.py --arm label-transfer-l6-s2 runs\registration-custom\detectors\custom-l6-s2\weights\best.pt 0.003033 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --out runs\clipped-box-audit\rescore\single-label-transfer-l6-s2.csv --device cuda:1
uv run python scripts/clipped_box_rescore.py --arm label-transfer-l12-s1 runs\registration-custom\detectors\custom-l12-s1\weights\best.pt 0.005257 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --out runs\clipped-box-audit\rescore\single-label-transfer-l12-s1.csv --device cuda:1
uv run python scripts/clipped_box_rescore.py --arm label-transfer-l12-s2 runs\registration-custom\detectors\custom-l12-s2\weights\best.pt 0.005257 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --out runs\clipped-box-audit\rescore\single-label-transfer-l12-s2.csv --device cuda:1
uv run python scripts/clipped_box_rescore.py --arm label-transfer-l24-s1 runs\registration-custom\detectors\custom-l24-s1\weights\best.pt 0.02065 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --out runs\clipped-box-audit\rescore\single-label-transfer-l24-s1.csv --device cuda:1
uv run python scripts/clipped_box_rescore.py --arm label-transfer-l24-s2 runs\registration-custom\detectors\custom-l24-s2\weights\best.pt 0.02065 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --out runs\clipped-box-audit\rescore\single-label-transfer-l24-s2.csv --device cuda:1

# 6b. rescore-routes-rerun
uv run python scripts/clipped_box_rescore.py --arm label-transfer runs\registration-custom\detectors\custom-l3-s0\weights\best.pt 0.001684 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --arm zero-shot runs\reference-yolo11s\weights\best.pt 0.013972 runs\regc-03px-s0\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\regc-03px-s0-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\regc-03px-s0-0.75\data.yaml --arm adapted D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\regc-03px-s0\stage3\detector\stage3\weights\best.pt 0.005033 runs\regc-03px-s0\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\regc-03px-s0-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\regc-03px-s0-0.75\data.yaml --out runs\clipped-box-audit\rescore\routes-03px.csv --device cuda:0
uv run python scripts/clipped_box_rescore.py --arm label-transfer runs\registration-custom\detectors\custom-l6-s0\weights\best.pt 0.003033 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --arm zero-shot runs\reference-yolo11s\weights\best.pt 0.013972 runs\regc-06px-s0\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\regc-06px-s0-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\regc-06px-s0-0.75\data.yaml --arm adapted D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\regc-06px-s0\stage3\detector\stage3\weights\best.pt 0.005033 runs\regc-06px-s0\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\regc-06px-s0-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\regc-06px-s0-0.75\data.yaml --out runs\clipped-box-audit\rescore\routes-06px.csv --device cuda:0
uv run python scripts/clipped_box_rescore.py --arm label-transfer runs\registration-custom\detectors\custom-l12-s0\weights\best.pt 0.005257 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --arm zero-shot runs\reference-yolo11s\weights\best.pt 0.013972 runs\regc-12px-s0\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\regc-12px-s0-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\regc-12px-s0-0.75\data.yaml --arm adapted D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\regc-12px-s0\stage3\detector\stage3\weights\best.pt 0.005033 runs\regc-12px-s0\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\regc-12px-s0-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\regc-12px-s0-0.75\data.yaml --out runs\clipped-box-audit\rescore\routes-12px.csv --device cuda:0
uv run python scripts/clipped_box_rescore.py --arm label-transfer runs\registration-custom\detectors\custom-l24-s0\weights\best.pt 0.02065 runs\clipped-box-audit\manifests\custom-original\data.yaml 0.5=runs\clipped-box-audit\manifests\custom-0.5\data.yaml 0.75=runs\clipped-box-audit\manifests\custom-0.75\data.yaml --arm zero-shot runs\reference-yolo11s\weights\best.pt 0.013972 runs\regc-24px-s0\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\regc-24px-s0-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\regc-24px-s0-0.75\data.yaml --arm adapted D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\regc-24px-s0\stage3\detector\stage3\weights\best.pt 0.005033 runs\regc-24px-s0\stage3\translated\data.yaml 0.5=runs\clipped-box-audit\trees\regc-24px-s0-0.5\data.yaml 0.75=runs\clipped-box-audit\trees\regc-24px-s0-0.75\data.yaml --out runs\clipped-box-audit\rescore\routes-24px.csv --device cuda:0
```

## Environment

From `logs/2026-11-16-CLIPPED-BOX-AUDIT-05-build-log-search.txt`, the first log. Across all 29
blocks in the nine logs, only the `captured` and `command` lines differ. This was checked by
removing those two lines from every block and counting the distinct lines that remain: each of the
other 15 lines appears 29 times.

```
=== PROVENANCE ===
captured: 2026-11-16 20:15:52 +03:00
hostname: VSIBMEOT05
os: Microsoft Windows Server 2022 Standard 10.0.20348 (build 20348)
cpu: AMD EPYC 7763 64-Core Processor
cpu_logical_cores: 64
ram_total: 128.0 GB
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16; NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16
cuda_driver: 13.0
git_sha: pre-spec-migration-310-gccfce13
working_dir: D:\Atick\GitHub\Thermal-To-Optical-Translation
command: search D:\Atick\data\dataset\Brazil-Aligned for 'hit the frame edge'; search PSReadLine history for build_rgbt_dataset
python: 3.12.7
torch: 2.12.1+cu130
cuda_torch_build: 13.0
cudnn: 92000
cuda_available: True
gpu0_torch: NVIDIA A100-SXM4-40GB, 39.7 GiB, sm 8.0
=== END PROVENANCE ===
```

## Configuration

- **Audit inputs.** Original labels `Brazil-Aligned\brazil_all_29_jul_2026_yolo` (5-class; found
  by step 3a, see F250), homographies `Brazil-Aligned\alignment`, frozen tree `yolo_rgbt_29_jul`.
  The audit used its defaults: `--min-visibility 0.25`, `--min-box-px 2.0`
  (`scripts/clipped_box_audit.py:426-427`). Audit CSV sha256 `5417D5C6…16D2C3E4`. Step 5's log
  re-reads the same digest.
- **Drop thresholds.** 0.5 is the limit for matching at mAP50. 0.75 is where mAP50-95's stricter
  IoU levels start to suffer (plan step 3). At 0.5 the drop removes 72 val boxes from 65 of 153
  files. At 0.75 it removes 184 boxes from 147 files. Each translation export drops the same
  audit lines.
- **Scorer.** `clipped_box_rescore.py` defaults `--imgsz 640 --batch 16` (`:213-214`). The `$resolve`
  helper asserted that each translation run had scored at those settings.
- **Arms.** Each route call ranks label transfer, zero-shot and adapted at one level, all seed 0:
  label transfer `custom-l{L}-s0`, the reference `yolo11s` judge on `regc-{L}px-s0` or
  `e3b-loop-s0`'s stage-3 export, and that run's stage-3 detector. Single calls hold one checkpoint
  each: E8 arm A `A-n600-s{0,1,2}`, and label transfer `custom-l{L}-s{1,2}` at every level.
- **Seed spreads (Q3's clause), mAP50 SD.** Label transfer per level, from record 040's custom CSV:
  0.003989 / 0.001684 / 0.003033 / 0.005257 / 0.020650. Zero-shot 0.013972 and adapted 0.005033:
  `e3b-loop`'s six-seed SDs, applied at every level. E8 arm A 0.0096, from record 011.
- **Q3 rule** (`scripts/clipped_box_rescore.py:17-19, 185-186`). A threshold says `fix` when, on
  mAP50, any arm's rank changes or any arm's |delta| is larger than its spread.

## Results

Every non-provenance line is quoted verbatim for steps 1–5. For the three re-score logs,
ultralytics' progress bars and per-pass banners are left out. What remains is every `__main__` and
`t2o.metrics.task` summary line, every traceback, the step-7 CSV dump the log printed, and the five
route CSVs.

### 1. Build log search

```
--- saved report() lines ---
--- build commands in history ---
uv run python .\experiments\Alignment\build_rgbt_dataset.py --root "D:\Atick\data\dataset\Brazil-Aligned"
uv run python .\experiments\Alignment\build_rgbt_dataset.py --root "D:\Atick\data\dataset\Brazil-Aligned"
uv run python .\experiments\Alignment\build_rgbt_dataset.py --root "D:\Atick\data\dataset\Brazil-Aligned" --overwrite
uv run python .\experiments\Alignment\build_rgbt_dataset.py --root "D:\Atick\data\dataset\Brazil-Aligned" --overwrite
python .\build_rgbt_dataset.py --root "D:\Atick\data\dataset\Brazil-Aligned" --original-dir "yolo_original_reannotated" --output-dir yolo_rgbt_reannotated --dry-run
python .\build_rgbt_dataset.py --root "D:\Atick\data\dataset\Brazil-Aligned" --original-dir "yolo_original_reannotated" --output-dir yolo_rgbt_reannotated 
python .\build_rgbt_dataset.py --root "D:\Atick\data\dataset\Brazil-Aligned" --original-dir "yolo_original_reannotated" --output-dir yolo_rgbt_reannotated --drop-empty
python .\build_rgbt_dataset.py --root "D:\Atick\data\dataset\Brazil-Aligned" --original-dir "brazil_all_29_jul_2026_yolo" --output-dir yolo_rgbt_29_jul
    Write-Provenance -Command "search $ALIGNED for 'hit the frame edge'; search PSReadLine history for build_rgbt_dataset"`
    Select-String -Path (Get-PSReadLineOption).HistorySavePath -Pattern 'build_rgbt_dataset' |`
--- frozen tree data.yaml ---
path: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
train: train/visible/images
val: val/visible/images
nc: 5
names:
- Connector
- Fuse
- Pole
- Switch
- Transformer
# channels: 6
rgbt:
  visible_token: visible
  infrared_token: infrared
  clahe:
    visible:
      enabled: true
      clip_limit: 4.0
info:
  year: 2026
  version: '1.0'
  contributor: Label Studio
=== FINISHED 2026-11-16 20:16:39 +03:00 ===
```

### 2. Polygon check

```
--- JSON files naming a Label Studio box or polygon type ---
--- yolo_original label lines with more than 5 fields (segmentation polygons) ---
total lines: 2929
lines with >5 fields: 0
=== FINISHED 2026-11-16 20:18:02 +03:00 ===
```

### 3a. Label source

```
--- 1. every FLIR1448 label file (the 5-class source reads: 1 2 1 1; the 4-class export: 0 1 0 0) ---
D:\Atick\data\dataset\Brazil-Aligned\brazil_all_29_jul_2026_yolo\train\labels\00e6087b-FLIR1448_embedded.txt | 2026-07-29 19:58 | classes: 4 1 2 1 1
D:\Atick\data\dataset\Brazil-Aligned\yolo_original\train\labels\00e6087b-FLIR1448_embedded.txt | 2026-02-11 16:06 | classes: 3 0 1 0 0
D:\Atick\data\dataset\Brazil-Aligned\yolo_original_reannotated\train\labels\00e6087b-FLIR1448_embedded.txt | 2026-07-28 17:35 | classes: 4 1 2 1 1
D:\Atick\data\dataset\Brazil-Aligned\yolo_original_reannotated\val\labels\00e6087b-FLIR1448_embedded.txt | 2026-07-28 17:35 | classes: 4 1 2 1 1
D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt\train\infrared\labels\FLIR1448.txt | 2026-07-26 16:22 | classes: 0 1 0 0
D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt\train\visible\labels\FLIR1448.txt | 2026-07-28 17:35 | classes: 4 1 2 1 1
D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt\val\visible\labels\FLIR1448.txt | 2026-07-28 17:35 | classes: 4 1 2 1 1
D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\train\infrared\labels\FLIR1448.txt | 2026-07-29 20:21 | classes: 1 2 1 1
D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\train\visible\labels\FLIR1448.txt | 2026-07-29 20:21 | classes: 1 2 1 1
D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul_inner\train\infrared\labels\FLIR1448.txt | 2026-10-23 13:05 | classes: 1 2 1 1
D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul_inner\train\visible\labels\FLIR1448.txt | 2026-10-23 13:05 | classes: 1 2 1 1
D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_reannotated\train\infrared\labels\FLIR1448.txt | 2026-07-28 19:16 | classes: 1 2 1 1
D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_reannotated\train\visible\labels\FLIR1448.txt | 2026-07-28 19:16 | classes: 1 2 1 1
D:\Atick\data\dataset\brazil_all_29_jul_2026\labels\00e6087b-FLIR1448_embedded.txt | 2026-07-29 19:42 | classes: 4 1 2 1 1
D:\Atick\data\dataset\brazil_yolo_dataset\train\labels\00e6087b-FLIR1448_embedded.txt | 2026-07-28 17:35 | classes: 4 1 2 1 1
D:\Atick\data\dataset\brazil_yolo_dataset\val\labels\00e6087b-FLIR1448_embedded.txt | 2026-07-28 17:35 | classes: 4 1 2 1 1
D:\Atick\data\dataset\brazil_yolo_dataset_cropped\train\labels\00e6087b-FLIR1448_embedded.txt | 2026-04-29 11:43 | classes: 3 0 1 0 0
D:\Atick\data\dataset\brazil_yolo_dataset_cropped_destripe\train\labels\00e6087b-FLIR1448_embedded.txt | 2026-04-29 11:43 | classes: 3 0 1 0 0
D:\Atick\data\dataset\brazil_yolo_dataset_cropped_destripe_hr\train\labels\00e6087b-FLIR1448_embedded.txt | 2026-04-29 11:43 | classes: 3 0 1 0 0
D:\Atick\data\dataset\brazil_yolo_dataset_cropped_destripe_hr_2\train\labels\00e6087b-FLIR1448_embedded.txt | 2026-04-29 11:43 | classes: 3 0 1 0 0
--- 2. label file dates (oldest .. newest) ---
D:\Atick\data\dataset\Brazil-Aligned\yolo_original\train\labels: 648 files, 2026-02-11 16:06 .. 2026-02-11 16:06
D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\train\infrared\labels: 600 files, 2026-07-29 20:21 .. 2026-07-29 20:22
D:\Atick\data\dataset\Brazil-Aligned\yolo_original\val\labels: 162 files, 2026-02-11 16:06 .. 2026-02-11 16:06
D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\val\infrared\labels: 153 files, 2026-07-29 20:22 .. 2026-07-29 20:22
--- 3. names in the original root yaml (what the build read) ---
# data.yaml
path: D:\Atick\Pole-Segmentation\yolo_dataset
train: train/images
val: val/images
nc: 4
names:
- Fuse
- Pole
- Switch
- Transformer
info:
  year: 2026
  version: '1.0'
  contributor: Label Studio
--- 4. class histograms and grown files per split ---
train original classes: 0=812, 1=816, 2=463, 3=211
train frozen   classes: 1=741, 2=596, 3=397, 4=134
train frozen files with more lines than their original: 15
  FLIR0364 : original 3, frozen 4
  FLIR0370 : original 5, frozen 6
  FLIR0387 : original 3, frozen 4
  FLIR0440 : original 2, frozen 3
  FLIR1172 : original 1, frozen 4
  FLIR1260 : original 9, frozen 12
  FLIR1264 : original 1, frozen 4
  FLIR1342 : original 3, frozen 12
  FLIR1402 : original 4, frozen 10
  FLIR1424 : original 4, frozen 6
  FLIR1458 : original 1, frozen 4
  FLIR1704 : original 1, frozen 4
  FLIR1710 : original 3, frozen 4
  FLIR1722 : original 2, frozen 4
  FLIR1812 : original 4, frozen 7
Get-ChildItem: 
Line |
   7 |      foreach ($f in Get-ChildItem $dir -File -Filter *.txt) {
     |                     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     | Cannot find path 'D:\Atick\GitHub\Thermal-To-Optical-Translation\System.Collections.Hashtable\val\labels' because it does not exist.
Get-ChildItem: 
Line |
   7 |      foreach ($f in Get-ChildItem $dir -File -Filter *.txt) {
     |                     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     | Cannot find path 'D:\Atick\GitHub\Thermal-To-Optical-Translation\System.Collections.Hashtable\val\infrared\labels' because it does not exist.
val original classes: 
val frozen   classes: 
val frozen files with more lines than their original: 0
=== FINISHED 2026-11-17 09:44:48 +03:00 ===
```

### 3. Frozen-label audit

```
2026-11-17 10:07:22,073 WARNING __main__: FLIR0282_1 (train): no alignment file, skipping as the build did
2026-11-17 10:07:22,796 WARNING __main__: FLIR0384_1 (train): no alignment file, skipping as the build did
2026-11-17 10:07:23,144 WARNING __main__: FLIR0278_1 (train): no alignment file, skipping as the build did
2026-11-17 10:07:23,838 WARNING __main__: FLIR0276_1 (train): no alignment file, skipping as the build did
2026-11-17 10:07:25,053 WARNING __main__: FLIR0354_1 (train): no alignment file, skipping as the build did
2026-11-17 10:07:25,377 WARNING __main__: FLIR0370_1 (train): no alignment file, skipping as the build did
2026-11-17 10:07:26,031 WARNING __main__: FLIR0350_1 (train): no alignment file, skipping as the build did
2026-11-17 10:07:28,854 WARNING __main__: FLIR0348_1 (train): no alignment file, skipping as the build did
2026-11-17 10:07:29,885 WARNING __main__: FLIR0388_1 (train): no alignment file, skipping as the build did
2026-11-17 10:07:30,692 WARNING __main__: FLIR0318_1 (train): no alignment file, skipping as the build did
2026-11-17 10:07:31,268 WARNING __main__: FLIR0360_1 (train): no alignment file, skipping as the build did
2026-11-17 10:07:31,778 WARNING __main__: FLIR0328_1 (train): no alignment file, skipping as the build did
2026-11-17 10:07:31,850 WARNING __main__: train: 12 images skipped for a missing alignment file
2026-11-17 10:07:34,168 WARNING __main__: FLIR0390_1 (val): no alignment file, skipping as the build did
2026-11-17 10:07:36,617 WARNING __main__: val: 1 images skipped for a missing alignment file
2026-11-17 10:07:36,688 INFO    __main__: train class 1: 843 boxes | 141 clipped | 102 dropped | kept under (0.5, 0.75): [8, 23]
2026-11-17 10:07:36,688 INFO    __main__: train class 2: 725 boxes | 722 clipped | 129 dropped | kept under (0.5, 0.75): [203, 592]
2026-11-17 10:07:36,688 INFO    __main__: train class 3: 465 boxes | 105 clipped | 68 dropped | kept under (0.5, 0.75): [6, 17]
2026-11-17 10:07:36,688 INFO    __main__: train class 4: 198 boxes | 155 clipped | 64 dropped | kept under (0.5, 0.75): [48, 76]
2026-11-17 10:07:36,688 INFO    __main__: val class 1: 191 boxes | 30 clipped | 23 dropped | kept under (0.5, 0.75): [4, 5]
2026-11-17 10:07:36,688 INFO    __main__: val class 2: 187 boxes | 187 clipped | 34 dropped | kept under (0.5, 0.75): [56, 152]
2026-11-17 10:07:36,688 INFO    __main__: val class 3: 83 boxes | 23 clipped | 17 dropped | kept under (0.5, 0.75): [3, 5]
2026-11-17 10:07:36,688 INFO    __main__: val class 4: 58 boxes | 50 clipped | 22 dropped | kept under (0.5, 0.75): [9, 22]
2026-11-17 10:07:36,688 INFO    __main__: wrote 2750 box rows to runs\clipped-box-audit\audit.csv
5417D5C6C9C6F64C6EDFB7A1CBCB5A78A25E69C3662DDAB372691B3D16D2C3E4
=== FINISHED 2026-11-17 10:07:36 +03:00 ===
```

### 4. Registration-axis looseness

```
2026-11-17 10:08:01,612 INFO    __main__: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\val\infrared\labels: 423 boxes at 640x480
2026-11-17 10:08:20,785 INFO    __main__:   0.0 px diagonal  mean IoU 1.0000 | clamped 68/423 (0.161) | dropped 0
2026-11-17 10:08:20,785 INFO    __main__:   0.0 px rectangle mean IoU 1.0000 | clamped 68/423 (0.161) | dropped 0
2026-11-17 10:08:20,785 INFO    __main__:   3.0 px diagonal  mean IoU 0.9643 | clamped 142/423 (0.336) | dropped 0
2026-11-17 10:08:20,785 INFO    __main__:   3.0 px rectangle mean IoU 1.0000 | clamped 142/423 (0.336) | dropped 0
2026-11-17 10:08:20,785 INFO    __main__:   6.0 px diagonal  mean IoU 0.9304 | clamped 142/423 (0.336) | dropped 0
2026-11-17 10:08:20,785 INFO    __main__:   6.0 px rectangle mean IoU 1.0000 | clamped 142/423 (0.336) | dropped 0
2026-11-17 10:08:20,785 INFO    __main__:  12.0 px diagonal  mean IoU 0.8671 | clamped 143/423 (0.338) | dropped 0
2026-11-17 10:08:20,785 INFO    __main__:  12.0 px rectangle mean IoU 0.9999 | clamped 143/423 (0.338) | dropped 0
2026-11-17 10:08:20,785 INFO    __main__:  24.0 px diagonal  mean IoU 0.7553 | clamped 144/423 (0.340) | dropped 0
2026-11-17 10:08:20,785 INFO    __main__:  24.0 px rectangle mean IoU 0.9997 | clamped 144/423 (0.340) | dropped 0
2026-11-17 10:08:20,785 INFO    __main__: wrote 10 rows to runs\clipped-box-audit\looseness-val.csv
2026-11-17 10:08:34,468 INFO    __main__: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\train\infrared\labels: 1868 boxes at 640x480
2026-11-17 10:10:26,206 INFO    __main__:   0.0 px diagonal  mean IoU 1.0000 | clamped 258/1868 (0.138) | dropped 0
2026-11-17 10:10:26,206 INFO    __main__:   0.0 px rectangle mean IoU 1.0000 | clamped 258/1868 (0.138) | dropped 0
2026-11-17 10:10:26,206 INFO    __main__:   3.0 px diagonal  mean IoU 0.9666 | clamped 595/1868 (0.319) | dropped 0
2026-11-17 10:10:26,206 INFO    __main__:   3.0 px rectangle mean IoU 1.0000 | clamped 595/1868 (0.319) | dropped 0
2026-11-17 10:10:26,206 INFO    __main__:   6.0 px diagonal  mean IoU 0.9346 | clamped 597/1868 (0.320) | dropped 0
2026-11-17 10:10:26,206 INFO    __main__:   6.0 px rectangle mean IoU 1.0000 | clamped 597/1868 (0.320) | dropped 0
2026-11-17 10:10:26,206 INFO    __main__:  12.0 px diagonal  mean IoU 0.8746 | clamped 600/1868 (0.321) | dropped 0
2026-11-17 10:10:26,206 INFO    __main__:  12.0 px rectangle mean IoU 0.9999 | clamped 600/1868 (0.321) | dropped 0
2026-11-17 10:10:26,206 INFO    __main__:  24.0 px diagonal  mean IoU 0.7675 | clamped 607/1868 (0.325) | dropped 0
2026-11-17 10:10:26,212 INFO    __main__:  24.0 px rectangle mean IoU 0.9998 | clamped 607/1868 (0.325) | dropped 0
2026-11-17 10:10:26,212 INFO    __main__: wrote 10 rows to runs\clipped-box-audit\looseness-train.csv
=== FINISHED 2026-11-17 10:10:26 +03:00 ===
```

### 5. Drop trees

```
audit sha256: 5417D5C6C9C6F64C6EDFB7A1CBCB5A78A25E69C3662DDAB372691B3D16D2C3E4
runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 10:16:52,949 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-17 10:17:38,917 INFO    __main__: drop tree runs\clipped-box-audit\trees\custom-0.5\data.yaml: 72 val box(es) below 0.50 removed from 65 of 153 file(s)
runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 10:18:11,684 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\translated
2026-11-17 10:18:34,213 INFO    __main__: drop tree runs\clipped-box-audit\trees\e3b-loop-s0-0.5\data.yaml: 72 val box(es) below 0.50 removed from 65 of 153 file(s)
2026-11-17 10:18:59,316 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\regc-03px-s0\stage3\translated
2026-11-17 10:19:26,178 INFO    __main__: drop tree runs\clipped-box-audit\trees\regc-03px-s0-0.5\data.yaml: 72 val box(es) below 0.50 removed from 65 of 153 file(s)
2026-11-17 10:19:50,602 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\regc-06px-s0\stage3\translated
2026-11-17 10:20:18,859 INFO    __main__: drop tree runs\clipped-box-audit\trees\regc-06px-s0-0.5\data.yaml: 72 val box(es) below 0.50 removed from 65 of 153 file(s)
2026-11-17 10:20:42,775 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\regc-12px-s0\stage3\translated
2026-11-17 10:21:08,960 INFO    __main__: drop tree runs\clipped-box-audit\trees\regc-12px-s0-0.5\data.yaml: 72 val box(es) below 0.50 removed from 65 of 153 file(s)
2026-11-17 10:21:34,442 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\regc-24px-s0\stage3\translated
2026-11-17 10:22:01,803 INFO    __main__: drop tree runs\clipped-box-audit\trees\regc-24px-s0-0.5\data.yaml: 72 val box(es) below 0.50 removed from 65 of 153 file(s)
2026-11-17 10:22:25,663 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-17 10:22:33,180 INFO    __main__: drop tree runs\clipped-box-audit\trees\custom-0.75\data.yaml: 184 val box(es) below 0.75 removed from 147 of 153 file(s)
runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 10:23:04,324 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\stage3\translated
2026-11-17 10:23:07,537 INFO    __main__: drop tree runs\clipped-box-audit\trees\e3b-loop-s0-0.75\data.yaml: 184 val box(es) below 0.75 removed from 147 of 153 file(s)
2026-11-17 10:23:30,618 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\regc-03px-s0\stage3\translated
2026-11-17 10:23:33,057 INFO    __main__: drop tree runs\clipped-box-audit\trees\regc-03px-s0-0.75\data.yaml: 184 val box(es) below 0.75 removed from 147 of 153 file(s)
2026-11-17 10:23:56,682 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\regc-06px-s0\stage3\translated
2026-11-17 10:23:59,274 INFO    __main__: drop tree runs\clipped-box-audit\trees\regc-06px-s0-0.75\data.yaml: 184 val box(es) below 0.75 removed from 147 of 153 file(s)
2026-11-17 10:24:23,793 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\regc-12px-s0\stage3\translated
2026-11-17 10:24:27,208 INFO    __main__: drop tree runs\clipped-box-audit\trees\regc-12px-s0-0.75\data.yaml: 184 val box(es) below 0.75 removed from 147 of 153 file(s)
2026-11-17 10:24:49,890 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\regc-24px-s0\stage3\translated
2026-11-17 10:24:52,970 INFO    __main__: drop tree runs\clipped-box-audit\trees\regc-24px-s0-0.75\data.yaml: 184 val box(es) below 0.75 removed from 147 of 153 file(s)
=== FINISHED 2026-11-17 10:24:55 +03:00 ===
```

### 6. Routes, first run (summary lines and tracebacks)

```
2026-11-17 10:34:36,792 INFO    __main__: label-transfer on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 10:36:57,287 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8899  R 0.8868  mAP50 0.9297  mAP50-95 0.6592
2026-11-17 10:39:16,296 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 10:39:16,296 INFO    __main__: label-transfer on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 10:41:41,716 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8182  R 0.8560  mAP50 0.8504  mAP50-95 0.6039
2026-11-17 10:44:07,955 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 10:44:07,955 INFO    __main__: label-transfer on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 10:46:34,986 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5207  R 0.9034  mAP50 0.5963  mAP50-95 0.3880
2026-11-17 10:49:00,460 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 10:49:00,461 INFO    __main__: zero-shot on original tree: runs\e3b-loop-s0\stage3\translated\data.yaml
2026-11-17 10:51:29,078 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8552  R 0.7834  mAP50 0.8548  mAP50-95 0.5684
2026-11-17 10:53:59,801 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\e3b-loop-s0\stage3\translated\data.yaml
2026-11-17 10:53:59,801 INFO    __main__: zero-shot on 0.5 tree: runs\clipped-box-audit\trees\e3b-loop-s0-0.5\data.yaml
2026-11-17 10:56:27,374 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7502  R 0.7745  mAP50 0.7791  mAP50-95 0.5226
2026-11-17 10:58:54,627 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\e3b-loop-s0-0.5\data.yaml
2026-11-17 10:58:54,627 INFO    __main__: zero-shot on 0.75 tree: runs\clipped-box-audit\trees\e3b-loop-s0-0.75\data.yaml
2026-11-17 11:01:16,017 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.4898  R 0.8107  mAP50 0.6183  mAP50-95 0.3769
2026-11-17 11:03:37,403 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\e3b-loop-s0-0.75\data.yaml
2026-11-17 11:03:37,403 INFO    __main__: adapted on original tree: runs\e3b-loop-s0\stage3\translated\data.yaml
2026-11-17 11:05:55,413 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9261  R 0.8496  mAP50 0.9145  mAP50-95 0.6283
2026-11-17 11:08:14,301 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\e3b-loop-s0\stage3\translated\data.yaml
2026-11-17 11:08:14,306 INFO    __main__: adapted on 0.5 tree: runs\clipped-box-audit\trees\e3b-loop-s0-0.5\data.yaml
2026-11-17 11:10:32,253 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7892  R 0.8796  mAP50 0.8331  mAP50-95 0.5765
2026-11-17 11:12:49,203 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\e3b-loop-s0-0.5\data.yaml
2026-11-17 11:12:49,208 INFO    __main__: adapted on 0.75 tree: runs\clipped-box-audit\trees\e3b-loop-s0-0.75\data.yaml
2026-11-17 11:15:06,401 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5379  R 0.8950  mAP50 0.5597  mAP50-95 0.3673
2026-11-17 11:17:24,106 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\e3b-loop-s0-0.75\data.yaml
2026-11-17 11:17:24,108 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 11:17:24,109 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 11:17:24,111 INFO    __main__: wrote 6 row(s) to runs\clipped-box-audit\rescore\routes-00px.csv
2026-11-17 11:17:51,756 INFO    __main__: label-transfer on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 11:20:09,924 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8912  R 0.8703  mAP50 0.9284  mAP50-95 0.6550
2026-11-17 11:22:30,122 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 11:22:30,126 INFO    __main__: label-transfer on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 11:24:47,955 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8164  R 0.8467  mAP50 0.8635  mAP50-95 0.6093
2026-11-17 11:27:08,485 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 11:27:08,485 INFO    __main__: label-transfer on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 11:29:28,037 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5740  R 0.8549  mAP50 0.5935  mAP50-95 0.3916
2026-11-17 11:31:49,148 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 11:31:49,148 INFO    __main__: zero-shot on original tree: runs\regc-12px-s0\stage3\translated\data.yaml
2026-11-17 11:34:09,691 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8567  R 0.7183  mAP50 0.7967  mAP50-95 0.5108
2026-11-17 11:36:32,851 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\regc-12px-s0\stage3\translated\data.yaml
2026-11-17 11:36:32,851 INFO    __main__: zero-shot on 0.5 tree: runs\clipped-box-audit\trees\regc-12px-s0-0.5\data.yaml
2026-11-17 11:38:54,864 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7341  R 0.7529  mAP50 0.7341  mAP50-95 0.4731
2026-11-17 11:41:16,216 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-12px-s0-0.5\data.yaml
2026-11-17 11:41:16,227 INFO    __main__: zero-shot on 0.75 tree: runs\clipped-box-audit\trees\regc-12px-s0-0.75\data.yaml
2026-11-17 11:43:37,686 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.4964  R 0.7597  mAP50 0.5301  mAP50-95 0.3079
2026-11-17 11:46:00,580 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-12px-s0-0.75\data.yaml
2026-11-17 11:46:00,581 INFO    __main__: adapted on original tree: runs\regc-12px-s0\stage3\translated\data.yaml
2026-11-17 11:48:22,075 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9131  R 0.8223  mAP50 0.8844  mAP50-95 0.5845
2026-11-17 11:50:41,419 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\regc-12px-s0\stage3\translated\data.yaml
2026-11-17 11:50:41,419 INFO    __main__: adapted on 0.5 tree: runs\clipped-box-audit\trees\regc-12px-s0-0.5\data.yaml
2026-11-17 11:53:01,530 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7761  R 0.8454  mAP50 0.8172  mAP50-95 0.5376
2026-11-17 11:55:22,117 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-12px-s0-0.5\data.yaml
2026-11-17 11:55:22,117 INFO    __main__: adapted on 0.75 tree: runs\clipped-box-audit\trees\regc-12px-s0-0.75\data.yaml
2026-11-17 11:57:41,185 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5337  R 0.8547  mAP50 0.5622  mAP50-95 0.3420
2026-11-17 11:59:59,880 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-12px-s0-0.75\data.yaml
2026-11-17 11:59:59,885 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 11:59:59,885 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 11:59:59,885 INFO    __main__: wrote 6 row(s) to runs\clipped-box-audit\rescore\routes-03px.csv
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\Users\atick\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1027, in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\atick\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1013, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'runs\\config.yaml'
Traceback (most recent call last):
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\scripts\clipped_box_rescore.py", line 240, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\scripts\clipped_box_rescore.py", line 223, in main
    arms = [parse_arm(values) for values in args.arm]
            ^^^^^^^^^^^^^^^^^
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\scripts\clipped_box_rescore.py", line 106, in parse_arm
    arm = Arm(name, Path(weights), float(spread), Path(original), drops)
                                   ^^^^^^^^^^^^^
ValueError: could not convert string to float: 'runs\\\\stage3\\translated\\data.yaml'
Copy-Item: 
Line |
  23 |          Copy-Item $csv "logs\$D-$TASK-rescore-routes-$($level.ToStrin …
     |          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     | Cannot find path 'D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\clipped-box-audit\rescore\routes-06px.csv' because it does not exist.
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\Users\atick\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1027, in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\atick\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1013, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'runs\\config.yaml'
Traceback (most recent call last):
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\scripts\clipped_box_rescore.py", line 240, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\scripts\clipped_box_rescore.py", line 223, in main
    arms = [parse_arm(values) for values in args.arm]
            ^^^^^^^^^^^^^^^^^
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\scripts\clipped_box_rescore.py", line 106, in parse_arm
    arm = Arm(name, Path(weights), float(spread), Path(original), drops)
                                   ^^^^^^^^^^^^^
ValueError: could not convert string to float: 'runs\\\\stage3\\translated\\data.yaml'
Copy-Item: 
Line |
  23 |          Copy-Item $csv "logs\$D-$TASK-rescore-routes-$($level.ToStrin …
     |          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     | Cannot find path 'D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\clipped-box-audit\rescore\routes-12px.csv' because it does not exist.
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\Users\atick\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1027, in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\atick\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1013, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'runs\\config.yaml'
Traceback (most recent call last):
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\scripts\clipped_box_rescore.py", line 240, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\scripts\clipped_box_rescore.py", line 223, in main
    arms = [parse_arm(values) for values in args.arm]
            ^^^^^^^^^^^^^^^^^
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\scripts\clipped_box_rescore.py", line 106, in parse_arm
    arm = Arm(name, Path(weights), float(spread), Path(original), drops)
                                   ^^^^^^^^^^^^^
ValueError: could not convert string to float: 'runs\\\\stage3\\translated\\data.yaml'
Copy-Item: 
Line |
  23 |          Copy-Item $csv "logs\$D-$TASK-rescore-routes-$($level.ToStrin …
     |          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     | Cannot find path 'D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\clipped-box-audit\rescore\routes-24px.csv' because it does not exist.
=== FINISHED 2026-11-17 12:01:23 +03:00 ===
```

### 7. Single arms (summary lines, then the CSV dump the log printed)

```
2026-11-17 10:40:44,015 INFO    __main__: e8-A-n600-s0 on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 10:43:09,630 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9165  R 0.8282  mAP50 0.9195  mAP50-95 0.6438
2026-11-17 10:45:39,547 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 10:45:39,547 INFO    __main__: e8-A-n600-s0 on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 10:48:05,735 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8059  R 0.8295  mAP50 0.8447  mAP50-95 0.5909
2026-11-17 10:50:31,323 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 10:50:31,327 INFO    __main__: e8-A-n600-s0 on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 10:53:03,837 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5350  R 0.8568  mAP50 0.6046  mAP50-95 0.3891
2026-11-17 10:55:32,379 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 10:55:32,390 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 10:55:32,390 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 10:55:32,390 INFO    __main__: wrote 2 row(s) to runs\clipped-box-audit\rescore\single-e8-A-n600-s0.csv
2026-11-17 10:55:59,114 INFO    __main__: e8-A-n600-s1 on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 10:58:27,774 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9627  R 0.8482  mAP50 0.9384  mAP50-95 0.6598
2026-11-17 11:00:50,890 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 11:00:50,890 INFO    __main__: e8-A-n600-s1 on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 11:03:13,291 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8362  R 0.8784  mAP50 0.8750  mAP50-95 0.6160
2026-11-17 11:05:30,810 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 11:05:30,811 INFO    __main__: e8-A-n600-s1 on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 11:07:49,054 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5777  R 0.8808  mAP50 0.6176  mAP50-95 0.4014
2026-11-17 11:10:07,354 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 11:10:07,354 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 11:10:07,354 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 11:10:07,354 INFO    __main__: wrote 2 row(s) to runs\clipped-box-audit\rescore\single-e8-A-n600-s1.csv
2026-11-17 11:10:34,065 INFO    __main__: e8-A-n600-s2 on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 11:12:53,792 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9027  R 0.8894  mAP50 0.9343  mAP50-95 0.6570
2026-11-17 11:15:10,817 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 11:15:10,827 INFO    __main__: e8-A-n600-s2 on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 11:17:27,456 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7967  R 0.8815  mAP50 0.8483  mAP50-95 0.5939
2026-11-17 11:19:44,219 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 11:19:44,219 INFO    __main__: e8-A-n600-s2 on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 11:22:03,577 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5484  R 0.8799  mAP50 0.5997  mAP50-95 0.3974
2026-11-17 11:24:21,437 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 11:24:21,437 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 11:24:21,437 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 11:24:21,437 INFO    __main__: wrote 2 row(s) to runs\clipped-box-audit\rescore\single-e8-A-n600-s2.csv
2026-11-17 11:24:47,697 INFO    __main__: label-transfer-l0-s1 on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 11:27:09,895 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9225  R 0.8625  mAP50 0.9280  mAP50-95 0.6511
2026-11-17 11:29:29,308 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 11:29:29,313 INFO    __main__: label-transfer-l0-s1 on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 11:31:49,419 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7813  R 0.8632  mAP50 0.8448  mAP50-95 0.5930
2026-11-17 11:34:09,283 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 11:34:09,286 INFO    __main__: label-transfer-l0-s1 on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 11:36:32,661 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5384  R 0.8659  mAP50 0.5999  mAP50-95 0.3941
2026-11-17 11:38:53,482 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 11:38:53,485 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 11:38:53,485 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 11:38:53,490 INFO    __main__: wrote 2 row(s) to runs\clipped-box-audit\rescore\single-label-transfer-l0-s1.csv
2026-11-17 11:39:20,568 INFO    __main__: label-transfer-l0-s2 on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 11:41:43,555 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9199  R 0.8730  mAP50 0.9356  mAP50-95 0.6694
2026-11-17 11:44:05,051 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 11:44:05,051 INFO    __main__: label-transfer-l0-s2 on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 11:46:28,356 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7794  R 0.8876  mAP50 0.8425  mAP50-95 0.6038
2026-11-17 11:48:47,820 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 11:48:47,831 INFO    __main__: label-transfer-l0-s2 on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 11:51:08,744 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5400  R 0.8932  mAP50 0.5798  mAP50-95 0.3888
2026-11-17 11:53:29,263 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 11:53:29,263 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 11:53:29,263 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 11:53:29,263 INFO    __main__: wrote 2 row(s) to runs\clipped-box-audit\rescore\single-label-transfer-l0-s2.csv
2026-11-17 11:53:54,993 INFO    __main__: label-transfer-l3-s1 on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 11:56:19,139 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9375  R 0.8619  mAP50 0.9317  mAP50-95 0.6633
2026-11-17 11:58:37,496 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 11:58:37,496 INFO    __main__: label-transfer-l3-s1 on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 12:00:56,019 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8064  R 0.8863  mAP50 0.8522  mAP50-95 0.6044
2026-11-17 12:03:13,759 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 12:03:13,764 INFO    __main__: label-transfer-l3-s1 on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 12:05:26,362 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5678  R 0.8829  mAP50 0.6356  mAP50-95 0.4249
2026-11-17 12:07:29,908 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 12:07:29,908 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 12:07:29,908 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 12:07:29,908 INFO    __main__: wrote 2 row(s) to runs\clipped-box-audit\rescore\single-label-transfer-l3-s1.csv
2026-11-17 12:07:52,649 INFO    __main__: label-transfer-l3-s2 on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 12:10:08,837 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9450  R 0.8300  mAP50 0.9309  mAP50-95 0.6594
2026-11-17 12:12:26,703 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 12:12:26,707 INFO    __main__: label-transfer-l3-s2 on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 12:14:43,323 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8091  R 0.8534  mAP50 0.8548  mAP50-95 0.6114
2026-11-17 12:17:01,355 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 12:17:01,356 INFO    __main__: label-transfer-l3-s2 on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 12:19:20,328 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5702  R 0.8752  mAP50 0.6266  mAP50-95 0.4175
2026-11-17 12:21:38,412 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 12:21:38,412 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 12:21:38,412 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 12:21:38,421 INFO    __main__: wrote 2 row(s) to runs\clipped-box-audit\rescore\single-label-transfer-l3-s2.csv
2026-11-17 12:22:05,244 INFO    __main__: label-transfer-l6-s1 on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 12:24:25,944 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9184  R 0.8583  mAP50 0.9266  mAP50-95 0.6419
2026-11-17 12:26:44,912 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 12:26:44,912 INFO    __main__: label-transfer-l6-s1 on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 12:29:03,938 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7901  R 0.8885  mAP50 0.8633  mAP50-95 0.5965
2026-11-17 12:31:22,362 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 12:31:22,362 INFO    __main__: label-transfer-l6-s1 on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 12:33:37,868 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5525  R 0.8964  mAP50 0.5820  mAP50-95 0.3778
2026-11-17 12:35:45,031 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 12:35:45,034 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 12:35:45,034 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 12:35:45,034 INFO    __main__: wrote 2 row(s) to runs\clipped-box-audit\rescore\single-label-transfer-l6-s1.csv
2026-11-17 12:36:07,219 INFO    __main__: label-transfer-l6-s2 on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 12:38:09,650 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9096  R 0.8993  mAP50 0.9316  mAP50-95 0.6558
2026-11-17 12:40:26,614 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 12:40:26,620 INFO    __main__: label-transfer-l6-s2 on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 12:42:49,673 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7715  R 0.9159  mAP50 0.8796  mAP50-95 0.6245
2026-11-17 12:45:08,366 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 12:45:08,366 INFO    __main__: label-transfer-l6-s2 on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 12:47:23,926 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5427  R 0.9122  mAP50 0.6430  mAP50-95 0.4011
2026-11-17 12:49:42,254 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 12:49:42,254 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 12:49:42,254 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 12:49:42,254 INFO    __main__: wrote 2 row(s) to runs\clipped-box-audit\rescore\single-label-transfer-l6-s2.csv
2026-11-17 12:50:09,300 INFO    __main__: label-transfer-l12-s1 on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 12:52:29,436 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8640  R 0.8956  mAP50 0.9218  mAP50-95 0.6157
2026-11-17 12:54:45,437 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 12:54:45,437 INFO    __main__: label-transfer-l12-s1 on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 12:57:05,182 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7481  R 0.8783  mAP50 0.8615  mAP50-95 0.5769
2026-11-17 12:59:27,234 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 12:59:27,234 INFO    __main__: label-transfer-l12-s1 on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 13:01:53,336 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.4947  R 0.9034  mAP50 0.5675  mAP50-95 0.3498
2026-11-17 13:04:14,796 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 13:04:14,796 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 13:04:14,796 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 13:04:14,800 INFO    __main__: wrote 2 row(s) to runs\clipped-box-audit\rescore\single-label-transfer-l12-s1.csv
2026-11-17 13:04:41,097 INFO    __main__: label-transfer-l12-s2 on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 13:07:05,617 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9028  R 0.8524  mAP50 0.9139  mAP50-95 0.6036
2026-11-17 13:09:28,533 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 13:09:28,535 INFO    __main__: label-transfer-l12-s2 on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 13:11:51,577 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7741  R 0.8726  mAP50 0.8400  mAP50-95 0.5493
2026-11-17 13:14:11,984 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 13:14:11,990 INFO    __main__: label-transfer-l12-s2 on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 13:16:31,369 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5464  R 0.8843  mAP50 0.5731  mAP50-95 0.3558
2026-11-17 13:18:50,758 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 13:18:50,758 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 13:18:50,758 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 13:18:50,761 INFO    __main__: wrote 2 row(s) to runs\clipped-box-audit\rescore\single-label-transfer-l12-s2.csv
2026-11-17 13:19:18,135 INFO    __main__: label-transfer-l24-s1 on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 13:21:43,618 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8113  R 0.7663  mAP50 0.8345  mAP50-95 0.4856
2026-11-17 13:24:02,997 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 13:24:02,997 INFO    __main__: label-transfer-l24-s1 on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 13:26:25,286 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.6819  R 0.7834  mAP50 0.7577  mAP50-95 0.4371
2026-11-17 13:28:47,937 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 13:28:47,937 INFO    __main__: label-transfer-l24-s1 on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 13:31:11,612 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.4628  R 0.8055  mAP50 0.5035  mAP50-95 0.2767
2026-11-17 13:33:33,808 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 13:33:33,808 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 13:33:33,808 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 13:33:33,812 INFO    __main__: wrote 2 row(s) to runs\clipped-box-audit\rescore\single-label-transfer-l24-s1.csv
2026-11-17 13:34:01,319 INFO    __main__: label-transfer-l24-s2 on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 13:36:24,603 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8387  R 0.8159  mAP50 0.8725  mAP50-95 0.4976
2026-11-17 13:38:47,867 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 13:38:47,867 INFO    __main__: label-transfer-l24-s2 on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 13:41:13,072 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7058  R 0.8378  mAP50 0.8175  mAP50-95 0.4719
2026-11-17 13:43:37,515 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 13:43:37,515 INFO    __main__: label-transfer-l24-s2 on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 13:46:00,664 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.4795  R 0.8670  mAP50 0.5900  mAP50-95 0.3316
2026-11-17 13:48:24,191 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 13:48:24,193 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 13:48:24,193 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 13:48:24,193 INFO    __main__: wrote 2 row(s) to runs\clipped-box-audit\rescore\single-label-transfer-l24-s2.csv
--- single-e8-A-n600-s0.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
e8-A-n600-s0,0.5,0.0096,0.9195190178936467,0.8447350242946159,-0.07478399359903076,0.6437800861030566,0.5909325825236446,-0.052847503579412036,72,68,1,1,False,True,fix
e8-A-n600-s0,0.75,0.0096,0.9195190178936467,0.6045945936458602,-0.31492442424778644,0.6437800861030566,0.38905250131400837,-0.2547275847890482,184,180,1,1,False,True,fix
--- single-e8-A-n600-s1.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
e8-A-n600-s1,0.5,0.0096,0.9384163292132042,0.8750389639920618,-0.06337736522114235,0.6597988875579486,0.6159783796820907,-0.043820507875857895,72,68,1,1,False,True,fix
e8-A-n600-s1,0.75,0.0096,0.9384163292132042,0.6176386455094077,-0.32077768370379645,0.6597988875579486,0.4014131282431346,-0.258385759314814,184,178,1,1,False,True,fix
--- single-e8-A-n600-s2.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
e8-A-n600-s2,0.5,0.0096,0.9343088325774178,0.8483315824330541,-0.08597725014436375,0.6569665377117747,0.593941967883942,-0.06302456982783267,72,69,1,1,False,True,fix
e8-A-n600-s2,0.75,0.0096,0.9343088325774178,0.5996786259468926,-0.33463020663052523,0.6569665377117747,0.3974328858676836,-0.2595336518440911,184,179,1,1,False,True,fix
--- single-label-transfer-l0-s1.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
label-transfer-l0-s1,0.5,0.003989,0.9280269144806557,0.8447656872556729,-0.08326122722498275,0.6510529219745989,0.5929647503721572,-0.05808817160244162,72,69,1,1,False,True,fix
label-transfer-l0-s1,0.75,0.003989,0.9280269144806557,0.5998915369116712,-0.3281353775689845,0.6510529219745989,0.39410888787315645,-0.2569440341014424,184,179,1,1,False,True,fix
--- single-label-transfer-l0-s2.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
label-transfer-l0-s2,0.5,0.003989,0.9356200274791998,0.8424896704745217,-0.09313035700467809,0.6693596666667514,0.6037706589662936,-0.06558900770045784,72,70,1,1,False,True,fix
label-transfer-l0-s2,0.75,0.003989,0.9356200274791998,0.5798461729147895,-0.35577385456441035,0.6693596666667514,0.38881436870766706,-0.28054529795908434,184,181,1,1,False,True,fix
--- single-label-transfer-l12-s1.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
label-transfer-l12-s1,0.5,0.005257,0.9217532694650716,0.861528470696751,-0.06022479876832065,0.6156733404618077,0.5769148463965331,-0.03875849406527454,72,68,1,1,False,True,fix
label-transfer-l12-s1,0.75,0.005257,0.9217532694650716,0.5674599269162214,-0.35429334254885025,0.6156733404618077,0.3498039411850929,-0.26586939927671477,184,179,1,1,False,True,fix
--- single-label-transfer-l12-s2.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
label-transfer-l12-s2,0.5,0.005257,0.9139099498074756,0.8400236516183345,-0.07388629818914116,0.6036044736025923,0.5493007387921868,-0.05430373481040551,72,68,1,1,False,True,fix
label-transfer-l12-s2,0.75,0.005257,0.9139099498074756,0.573137066076469,-0.34077288373100667,0.6036044736025923,0.35581202270515,-0.24779245089744228,184,178,1,1,False,True,fix
--- single-label-transfer-l24-s1.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
label-transfer-l24-s1,0.5,0.02065,0.8345465304914281,0.7577127261605091,-0.07683380433091902,0.4856266349254428,0.43709291763400027,-0.04853371729144251,72,67,1,1,False,True,fix
label-transfer-l24-s1,0.75,0.02065,0.8345465304914281,0.503507759798173,-0.33103877069325516,0.4856266349254428,0.2767195294767974,-0.20890710544864538,184,177,1,1,False,True,fix
--- single-label-transfer-l24-s2.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
label-transfer-l24-s2,0.5,0.02065,0.872491024199681,0.8174992740728915,-0.054991750126789585,0.49758839968078084,0.4719476159929069,-0.025640783687873925,72,69,1,1,False,True,fix
label-transfer-l24-s2,0.75,0.02065,0.872491024199681,0.5899886507710272,-0.2825023734286538,0.49758839968078084,0.3316325728224533,-0.16595582685832755,184,179,1,1,False,True,fix
--- single-label-transfer-l3-s1.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
label-transfer-l3-s1,0.5,0.001684,0.9316507400608023,0.8522459730516357,-0.07940476700916654,0.6633360397900566,0.6043924300808261,-0.058943609709230516,72,67,1,1,False,True,fix
label-transfer-l3-s1,0.75,0.001684,0.9316507400608023,0.63555857439708,-0.2960921656637223,0.6633360397900566,0.42493984289717124,-0.23839619689288538,184,178,1,1,False,True,fix
--- single-label-transfer-l3-s2.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
label-transfer-l3-s2,0.5,0.001684,0.9309241975743054,0.8547524974924923,-0.07617170008181318,0.6594036681474783,0.6113598718406565,-0.04804379630682176,72,71,1,1,False,True,fix
label-transfer-l3-s2,0.75,0.001684,0.9309241975743054,0.6266084712346707,-0.3043157263396348,0.6594036681474783,0.4174647643641721,-0.24193890378330618,184,182,1,1,False,True,fix
--- single-label-transfer-l6-s1.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
label-transfer-l6-s1,0.5,0.003033,0.9266477308332345,0.8632932624271041,-0.06335446840613035,0.6418505711362534,0.596528086228069,-0.04532248490818447,72,70,1,1,False,True,fix
label-transfer-l6-s1,0.75,0.003033,0.9266477308332345,0.5819993543069866,-0.3446483765262479,0.6418505711362534,0.37784920361575436,-0.2640013675204991,184,181,1,1,False,True,fix
--- single-label-transfer-l6-s2.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
label-transfer-l6-s2,0.5,0.003033,0.9316064421986574,0.8795869850507674,-0.052019457147890025,0.6558167538961104,0.6244752271005749,-0.03134152679553548,72,67,1,1,False,True,fix
label-transfer-l6-s2,0.75,0.003033,0.9316064421986574,0.6429545154452704,-0.288651926753387,0.6558167538961104,0.4010863042686378,-0.2547304496274726,184,178,1,1,False,True,fix
=== FINISHED 2026-11-17 13:48:26 +03:00 ===
```

### 6b. Routes, rerun (summary lines)

```
2026-11-17 12:57:38,240 INFO    __main__: label-transfer on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 12:59:59,523 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8912  R 0.8703  mAP50 0.9284  mAP50-95 0.6550
2026-11-17 13:02:26,156 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 13:02:26,156 INFO    __main__: label-transfer on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 13:04:46,269 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8164  R 0.8467  mAP50 0.8635  mAP50-95 0.6093
2026-11-17 13:07:09,772 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 13:07:09,772 INFO    __main__: label-transfer on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 13:09:31,781 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5740  R 0.8549  mAP50 0.5935  mAP50-95 0.3916
2026-11-17 13:11:53,109 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 13:11:53,111 INFO    __main__: zero-shot on original tree: runs\regc-03px-s0\stage3\translated\data.yaml
2026-11-17 13:14:13,994 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8131  R 0.7106  mAP50 0.7682  mAP50-95 0.4739
2026-11-17 13:16:34,245 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\regc-03px-s0\stage3\translated\data.yaml
2026-11-17 13:16:34,245 INFO    __main__: zero-shot on 0.5 tree: runs\clipped-box-audit\trees\regc-03px-s0-0.5\data.yaml
2026-11-17 13:18:54,414 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7292  R 0.7394  mAP50 0.6945  mAP50-95 0.4245
2026-11-17 13:21:16,986 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-03px-s0-0.5\data.yaml
2026-11-17 13:21:16,986 INFO    __main__: zero-shot on 0.75 tree: runs\clipped-box-audit\trees\regc-03px-s0-0.75\data.yaml
2026-11-17 13:23:37,485 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5284  R 0.8172  mAP50 0.5895  mAP50-95 0.3451
2026-11-17 13:25:59,734 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-03px-s0-0.75\data.yaml
2026-11-17 13:25:59,734 INFO    __main__: adapted on original tree: runs\regc-03px-s0\stage3\translated\data.yaml
2026-11-17 13:28:22,768 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9038  R 0.8706  mAP50 0.9242  mAP50-95 0.6410
2026-11-17 13:30:47,236 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\regc-03px-s0\stage3\translated\data.yaml
2026-11-17 13:30:47,236 INFO    __main__: adapted on 0.5 tree: runs\clipped-box-audit\trees\regc-03px-s0-0.5\data.yaml
2026-11-17 13:33:09,890 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7676  R 0.8946  mAP50 0.8594  mAP50-95 0.5975
2026-11-17 13:35:31,709 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-03px-s0-0.5\data.yaml
2026-11-17 13:35:31,709 INFO    __main__: adapted on 0.75 tree: runs\clipped-box-audit\trees\regc-03px-s0-0.75\data.yaml
2026-11-17 13:37:54,901 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5383  R 0.9090  mAP50 0.5867  mAP50-95 0.3743
2026-11-17 13:40:19,092 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-03px-s0-0.75\data.yaml
2026-11-17 13:40:19,092 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 13:40:19,097 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 13:40:19,097 INFO    __main__: wrote 6 row(s) to runs\clipped-box-audit\rescore\routes-03px.csv
2026-11-17 13:40:46,630 INFO    __main__: label-transfer on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 13:43:11,771 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9095  R 0.8671  mAP50 0.9322  mAP50-95 0.6359
2026-11-17 13:45:34,297 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 13:45:34,297 INFO    __main__: label-transfer on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 13:47:57,343 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8197  R 0.8610  mAP50 0.8614  mAP50-95 0.5852
2026-11-17 13:50:11,744 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 13:50:11,744 INFO    __main__: label-transfer on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 13:52:22,891 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5376  R 0.9007  mAP50 0.6562  mAP50-95 0.4019
2026-11-17 13:54:43,397 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 13:54:43,402 INFO    __main__: zero-shot on original tree: runs\regc-06px-s0\stage3\translated\data.yaml
2026-11-17 13:57:05,721 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8459  R 0.7655  mAP50 0.8256  mAP50-95 0.5546
2026-11-17 13:59:26,462 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\regc-06px-s0\stage3\translated\data.yaml
2026-11-17 13:59:26,464 INFO    __main__: zero-shot on 0.5 tree: runs\clipped-box-audit\trees\regc-06px-s0-0.5\data.yaml
2026-11-17 14:01:47,255 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7391  R 0.7787  mAP50 0.7705  mAP50-95 0.5292
2026-11-17 14:04:08,018 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-06px-s0-0.5\data.yaml
2026-11-17 14:04:08,018 INFO    __main__: zero-shot on 0.75 tree: runs\clipped-box-audit\trees\regc-06px-s0-0.75\data.yaml
2026-11-17 14:06:30,265 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5548  R 0.7723  mAP50 0.5589  mAP50-95 0.3612
2026-11-17 14:08:51,418 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-06px-s0-0.75\data.yaml
2026-11-17 14:08:51,418 INFO    __main__: adapted on original tree: runs\regc-06px-s0\stage3\translated\data.yaml
2026-11-17 14:11:14,310 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9173  R 0.8371  mAP50 0.9053  mAP50-95 0.6297
2026-11-17 14:13:36,909 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\regc-06px-s0\stage3\translated\data.yaml
2026-11-17 14:13:36,911 INFO    __main__: adapted on 0.5 tree: runs\clipped-box-audit\trees\regc-06px-s0-0.5\data.yaml
2026-11-17 14:16:03,350 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7804  R 0.8590  mAP50 0.8319  mAP50-95 0.5825
2026-11-17 14:18:30,162 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-06px-s0-0.5\data.yaml
2026-11-17 14:18:30,162 INFO    __main__: adapted on 0.75 tree: runs\clipped-box-audit\trees\regc-06px-s0-0.75\data.yaml
2026-11-17 14:20:54,109 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5415  R 0.8605  mAP50 0.5351  mAP50-95 0.3508
2026-11-17 14:23:18,515 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-06px-s0-0.75\data.yaml
2026-11-17 14:23:18,515 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 14:23:18,515 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 14:23:18,520 INFO    __main__: wrote 6 row(s) to runs\clipped-box-audit\rescore\routes-06px.csv
2026-11-17 14:23:46,341 INFO    __main__: label-transfer on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 14:26:12,190 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8722  R 0.8779  mAP50 0.9118  mAP50-95 0.5940
2026-11-17 14:28:36,330 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 14:28:36,330 INFO    __main__: label-transfer on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 14:30:59,425 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7787  R 0.8501  mAP50 0.8387  mAP50-95 0.5448
2026-11-17 14:33:22,977 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 14:33:22,977 INFO    __main__: label-transfer on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 14:35:41,455 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5464  R 0.8586  mAP50 0.5560  mAP50-95 0.3285
2026-11-17 14:38:07,447 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 14:38:07,448 INFO    __main__: zero-shot on original tree: runs\regc-12px-s0\stage3\translated\data.yaml
2026-11-17 14:40:33,896 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8567  R 0.7183  mAP50 0.7967  mAP50-95 0.5108
2026-11-17 14:42:59,388 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\regc-12px-s0\stage3\translated\data.yaml
2026-11-17 14:42:59,388 INFO    __main__: zero-shot on 0.5 tree: runs\clipped-box-audit\trees\regc-12px-s0-0.5\data.yaml
2026-11-17 14:45:20,249 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7341  R 0.7529  mAP50 0.7341  mAP50-95 0.4731
2026-11-17 14:47:43,281 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-12px-s0-0.5\data.yaml
2026-11-17 14:47:43,286 INFO    __main__: zero-shot on 0.75 tree: runs\clipped-box-audit\trees\regc-12px-s0-0.75\data.yaml
2026-11-17 14:50:08,285 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.4964  R 0.7597  mAP50 0.5301  mAP50-95 0.3079
2026-11-17 14:52:32,627 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-12px-s0-0.75\data.yaml
2026-11-17 14:52:32,627 INFO    __main__: adapted on original tree: runs\regc-12px-s0\stage3\translated\data.yaml
2026-11-17 14:54:54,008 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9131  R 0.8223  mAP50 0.8844  mAP50-95 0.5845
2026-11-17 14:57:16,265 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\regc-12px-s0\stage3\translated\data.yaml
2026-11-17 14:57:16,265 INFO    __main__: adapted on 0.5 tree: runs\clipped-box-audit\trees\regc-12px-s0-0.5\data.yaml
2026-11-17 14:59:31,195 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7761  R 0.8454  mAP50 0.8172  mAP50-95 0.5376
2026-11-17 15:01:17,787 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-12px-s0-0.5\data.yaml
2026-11-17 15:01:17,787 INFO    __main__: adapted on 0.75 tree: runs\clipped-box-audit\trees\regc-12px-s0-0.75\data.yaml
2026-11-17 15:03:33,956 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5337  R 0.8547  mAP50 0.5622  mAP50-95 0.3420
2026-11-17 15:05:38,765 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-12px-s0-0.75\data.yaml
2026-11-17 15:05:38,765 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 15:05:38,765 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 15:05:38,768 INFO    __main__: wrote 6 row(s) to runs\clipped-box-audit\rescore\routes-12px.csv
2026-11-17 15:06:01,522 INFO    __main__: label-transfer on original tree: runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 15:07:57,746 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.8668  R 0.7887  mAP50 0.8676  mAP50-95 0.4941
2026-11-17 15:09:51,624 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-original\data.yaml
2026-11-17 15:09:51,624 INFO    __main__: label-transfer on 0.5 tree: runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 15:11:43,496 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7169  R 0.8141  mAP50 0.8180  mAP50-95 0.4675
2026-11-17 15:13:37,234 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.5\data.yaml
2026-11-17 15:13:37,234 INFO    __main__: label-transfer on 0.75 tree: runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 15:15:34,422 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.4896  R 0.8510  mAP50 0.5460  mAP50-95 0.2890
2026-11-17 15:17:31,242 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\manifests\custom-0.75\data.yaml
2026-11-17 15:17:31,242 INFO    __main__: zero-shot on original tree: runs\regc-24px-s0\stage3\translated\data.yaml
2026-11-17 15:19:22,895 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7852  R 0.6971  mAP50 0.7383  mAP50-95 0.4249
2026-11-17 15:21:18,386 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\regc-24px-s0\stage3\translated\data.yaml
2026-11-17 15:21:18,386 INFO    __main__: zero-shot on 0.5 tree: runs\clipped-box-audit\trees\regc-24px-s0-0.5\data.yaml
2026-11-17 15:23:10,559 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.6637  R 0.7240  mAP50 0.6747  mAP50-95 0.3917
2026-11-17 15:25:04,243 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-24px-s0-0.5\data.yaml
2026-11-17 15:25:04,243 INFO    __main__: zero-shot on 0.75 tree: runs\clipped-box-audit\trees\regc-24px-s0-0.75\data.yaml
2026-11-17 15:26:56,248 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.4367  R 0.7406  mAP50 0.4250  mAP50-95 0.2266
2026-11-17 15:28:47,255 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-24px-s0-0.75\data.yaml
2026-11-17 15:28:47,255 INFO    __main__: adapted on original tree: runs\regc-24px-s0\stage3\translated\data.yaml
2026-11-17 15:30:38,769 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.9180  R 0.7675  mAP50 0.8377  mAP50-95 0.4633
2026-11-17 15:32:29,942 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\regc-24px-s0\stage3\translated\data.yaml
2026-11-17 15:32:29,942 INFO    __main__: adapted on 0.5 tree: runs\clipped-box-audit\trees\regc-24px-s0-0.5\data.yaml
2026-11-17 15:34:23,895 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7852  R 0.7937  mAP50 0.7818  mAP50-95 0.4250
2026-11-17 15:36:15,627 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-24px-s0-0.5\data.yaml
2026-11-17 15:36:15,628 INFO    __main__: adapted on 0.75 tree: runs\clipped-box-audit\trees\regc-24px-s0-0.75\data.yaml
2026-11-17 15:38:08,717 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.5446  R 0.8139  mAP50 0.5375  mAP50-95 0.2683
2026-11-17 15:40:02,016 INFO    t2o.metrics.task: captured per-image stats for 153 images from runs\clipped-box-audit\trees\regc-24px-s0-0.75\data.yaml
2026-11-17 15:40:02,018 INFO    __main__: threshold 0.50: Q3 says fix
2026-11-17 15:40:02,018 INFO    __main__: threshold 0.75: Q3 says fix
2026-11-17 15:40:02,019 INFO    __main__: wrote 6 row(s) to runs\clipped-box-audit\rescore\routes-24px.csv
=== FINISHED 2026-11-17 15:40:03 +03:00 ===
```

### Route CSVs (`logs/2026-11-16-CLIPPED-BOX-AUDIT-05-rescore-routes-*px.csv`)

```
--- 2026-11-16-CLIPPED-BOX-AUDIT-05-rescore-routes-00px.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
label-transfer,0.5,0.003989,0.9297053272744761,0.8503724235242125,-0.07933290375026358,0.6591637546587517,0.6039415183987253,-0.055222236260026314,72,68,1,1,False,True,fix
zero-shot,0.5,0.013972,0.8547938638426396,0.7790581019893799,-0.0757357618532597,0.5684192728599392,0.5226205787988316,-0.04579869406110759,72,67,3,3,False,True,fix
adapted,0.5,0.005033,0.914490176548812,0.8330566847832456,-0.08143349176556647,0.6282668099648829,0.5764783264975383,-0.05178848346734466,72,66,2,2,False,True,fix
label-transfer,0.75,0.003989,0.9297053272744761,0.5962941764384155,-0.33341115083606054,0.6591637546587517,0.3879580767030002,-0.27120567795575146,184,178,1,2,True,True,fix
zero-shot,0.75,0.013972,0.8547938638426396,0.6182783877244213,-0.23651547611821822,0.5684192728599392,0.3769032833412802,-0.191515989518659,184,177,3,1,True,True,fix
adapted,0.75,0.005033,0.914490176548812,0.559688638562504,-0.3548015379863081,0.6282668099648829,0.36734337513881626,-0.26092343482606667,184,176,2,3,True,True,fix
--- 2026-11-16-CLIPPED-BOX-AUDIT-05-rescore-routes-03px.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
label-transfer,0.5,0.001684,0.9284387702383913,0.8634885178048595,-0.06495025243353181,0.6550008478692717,0.6092687413791091,-0.04573210649016257,72,67,1,1,False,True,fix
zero-shot,0.5,0.013972,0.7682027018651895,0.6945293138198314,-0.07367338804535806,0.4739144440025112,0.42454723351868645,-0.04936721048382475,72,65,3,3,False,True,fix
adapted,0.5,0.005033,0.9242175310943375,0.8593580789171271,-0.06485945217721034,0.6409691095363214,0.5974893088854502,-0.0434798006508712,72,65,2,2,False,True,fix
label-transfer,0.75,0.001684,0.9284387702383913,0.5934945050951639,-0.3349442651432274,0.6550008478692717,0.39163016454136157,-0.26337068332791014,184,178,1,1,False,True,fix
zero-shot,0.75,0.013972,0.7682027018651895,0.5895252558714902,-0.17867744599369928,0.4739144440025112,0.3451246002651356,-0.12878984373737562,184,173,3,2,True,True,fix
adapted,0.75,0.005033,0.9242175310943375,0.5867194759296204,-0.3374980551647171,0.6409691095363214,0.374264410901866,-0.26670469863445534,184,176,2,3,True,True,fix
--- 2026-11-16-CLIPPED-BOX-AUDIT-05-rescore-routes-06px.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
label-transfer,0.5,0.003033,0.9321531386727582,0.861419203519632,-0.0707339351531262,0.6358743776365348,0.5851904664668072,-0.05068391116972759,72,69,1,1,False,True,fix
zero-shot,0.5,0.013972,0.8255608005733688,0.7705038233803725,-0.05505697719299629,0.5545570407611121,0.5291920654097542,-0.025364975351357955,72,68,3,3,False,True,fix
adapted,0.5,0.005033,0.90525461490126,0.8319020329786624,-0.07335258192259764,0.6296850241277931,0.5824690096993865,-0.047216014428406616,72,67,2,2,False,True,fix
label-transfer,0.75,0.003033,0.9321531386727582,0.6562021485889975,-0.27595099008376067,0.6358743776365348,0.40187962514140685,-0.2339947524951279,184,180,1,1,False,True,fix
zero-shot,0.75,0.013972,0.8255608005733688,0.5589472646842877,-0.2666135358890811,0.5545570407611121,0.3612144436170358,-0.1933425971440763,184,177,3,2,True,True,fix
adapted,0.75,0.005033,0.90525461490126,0.5351475869555181,-0.3701070279457419,0.6296850241277931,0.35082671349506633,-0.2788583106327268,184,177,2,3,True,True,fix
--- 2026-11-16-CLIPPED-BOX-AUDIT-05-rescore-routes-12px.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
label-transfer,0.5,0.005257,0.9117673925491945,0.8387412232579956,-0.07302616929119887,0.5940020760803556,0.5448290797192366,-0.04917299636111905,72,68,1,1,False,True,fix
zero-shot,0.5,0.013972,0.7966884589448613,0.7340503607637463,-0.06263809818111499,0.5107959903486861,0.473050220918088,-0.03774576943059804,72,63,3,3,False,True,fix
adapted,0.5,0.005033,0.8843788744784368,0.8171997489357277,-0.0671791255427091,0.5845120781967197,0.5375676695026856,-0.046944408694034045,72,63,2,2,False,True,fix
label-transfer,0.75,0.005257,0.9117673925491945,0.5559764665313329,-0.35579092601786155,0.5940020760803556,0.328495684656627,-0.2655063914237286,184,178,1,2,True,True,fix
zero-shot,0.75,0.013972,0.7966884589448613,0.5301225072893593,-0.266565951655502,0.5107959903486861,0.30790053910876664,-0.20289545123991942,184,173,3,3,False,True,fix
adapted,0.75,0.005033,0.8843788744784368,0.5622394875789327,-0.32213938689950405,0.5845120781967197,0.3419834986567539,-0.24252857953996576,184,172,2,1,True,True,fix
--- 2026-11-16-CLIPPED-BOX-AUDIT-05-rescore-routes-24px.csv
arm,threshold,spread,map50_original,map50_drop,delta_map50,map50_95_original,map50_95_drop,delta_map50_95,boxes_dropped,lost_matches,rank_original,rank_drop,rank_changed,exceeds_spread,verdict
label-transfer,0.5,0.02065,0.867638855354869,0.8179800783015485,-0.049658777053320446,0.49409606205292655,0.4674937526459969,-0.026602309406929647,72,68,1,1,False,True,fix
zero-shot,0.5,0.013972,0.7383118575539191,0.6746504628773804,-0.06366139467653875,0.42492402593449885,0.39174450082932244,-0.03317952510517641,72,63,3,3,False,True,fix
adapted,0.5,0.005033,0.837654104521311,0.7818280660567128,-0.055826038464598104,0.46330035165172545,0.424952669126498,-0.03834768252522747,72,63,2,2,False,True,fix
label-transfer,0.75,0.02065,0.867638855354869,0.5460038690188719,-0.3216349863359971,0.49409606205292655,0.28904964503709696,-0.2050464170158296,184,179,1,1,False,True,fix
zero-shot,0.75,0.013972,0.7383118575539191,0.4250128977992967,-0.31329895975462246,0.42492402593449885,0.22663571895487947,-0.19828830697961938,184,171,3,3,False,True,fix
adapted,0.75,0.005033,0.837654104521311,0.537548587569263,-0.3001055169520479,0.46330035165172545,0.2683050711885443,-0.19499528046318115,184,170,2,2,False,True,fix
```

Re-render without re-scoring: the CSVs above are the results files.
`scripts/clipped_box_rescore.py` has no report-only mode.

## Findings

### F245 — The build's clamp leaves 72 of custom val's 423 boxes (17.0%) with under half their warped area and 184 (43.5%) under three-quarters, and poles carry it: 56 and 152 of the 153 kept

The plan's limit for comparison: for a thin diagonal object clipped on one side, the retained-area
fraction equals the best IoU any detector can reach against a tight box. A box kept below 0.5 is
therefore out of reach at mAP50 for a detector that boxes the visible object tightly.

The audit recomputed every kept box and matched the frozen labels to six decimals. The script raises
`AuditFidelityError` before writing a CSV otherwise (`scripts/clipped_box_audit.py:364-378`), and it
wrote 2,750 rows. Per-class, from step 3's log:

- **Pole** (class 2) hits the frame edge on all 187 val boxes and on 722 of 725 train boxes. Of the
  153 val poles kept, 56 are under 0.5 and 152 are under 0.75. Train: 203 and 592 of 596 kept.
- **Transformer** (class 4): 9 and 22 of 36 kept val boxes. **Fuse** (1): 4 and 5 of 168.
  **Switch** (3): 3 and 5 of 66.
- **Train overall:** 265 of 1,868 kept boxes (14.2%) under 0.5 and 708 (37.9%) under 0.75.
- **At build time** the 25% rule already dropped 96 of 519 val boxes and 363 of 2,231 train boxes.

The kept counts (423 val, 1,868 train) match step 4's independent count of the frozen label files.

### F246 — The detectors already match the loose boxes: 63–71 of the 72 val boxes under 0.5 and 170–182 of the 184 under 0.75 were true positives at IoU 0.5, on every one of the 28 checkpoint-tree rows

This is `lost_matches` against `boxes_dropped`. `lost_matches` is the fall in true positives at
IoU 0.5 from the original tree to the drop tree, per checkpoint, from the same `model.val()`. It is
near-exact (`clipped_box_rescore.py:10-15`). Every checkpoint scored here was trained on this
tree's train boxes, which carry the same clamp (records 004, 011, 036, 037). So the ceiling
F245 states for a tight detector does not bind them. At most 1–9 of the 72 sub-0.5 boxes went
unmatched on the original tree.

The precision and recall lines show where the drop's mAP loss comes from. Across the 56 valid
drop-tree passes (the first log's bad 3 px block excluded):

- precision falls by 0.0717–0.3882;
- recall moves by −0.0308 to +0.1066;
- in every pass, precision falls by at least 0.0409 more than recall moves.

E8 arm A seed 0 is typical (single log lines 37, 71, 105): precision 0.9165 → 0.8059 → 0.5350,
recall 0.8282 → 0.8295 → 0.8568.

Removing a matched box turns its prediction into a false positive. That is the bias plan Q2 named for
the drop method ("dropping biases it down through false positives on the dropped objects"). The
other bias, missed matches caused by the loose labels, is bounded here by those 1–9 unmatched boxes.

### F247 — Plan Q3's pre-registered rule says `fix` on all 36 threshold verdicts (18 calls × 2 thresholds), every arm through the spread clause, but the deltas it fired on are F246's removed true positives

At 0.5, |ΔmAP50| runs from 0.0497 (label transfer, 24 px, seed 0) to 0.0931 (label transfer, 0 px,
seed 2). The largest spread passed in is 0.0207 (label transfer at 24 px). At 0.75, ΔmAP50 runs
from −0.1787 (zero-shot, 3 px) to −0.3701 (adapted, 6 px). Rank changes add to the verdict only at
0.75 (F248). The rule cannot tell a label effect from the drop's own bias. Per F246, the drop's bias
is most of each delta. The verdict is recorded here as the rule returned it.

### F248 — At the mAP50 limit (0.5 tree) the paper's route order holds at every level, label transfer > adapted > zero-shot on both trees; at 0.75 it reorders at 0, 3, 6 and 12 px, where 43.5% of val boxes are gone

Each route CSV gives `rank_original`, `rank_drop` and `rank_changed`. At 0.5, all 15 rows read
`False`. At 0.75 the order changes at four of five levels:

- **0 px:** zero-shot first (0.6183) over label transfer (0.5963).
- **3 and 6 px:** adapted falls from 2 to 3.
- **12 px:** adapted first (0.5622) over label transfer (0.5560).
- **24 px:** unchanged.

The original-tree scores reproduce record 040. Zero-shot `map50_original` reads 0.8548 / 0.7682 /
0.8256 / 0.7967 / 0.7383 at 0 / 3 / 6 / 12 / 24 px, and adapted at 0 px reads 0.9145. E8 arm A's
three seeds average 0.9307 against record 011's 0.9300 ± 0.0096.

The adapted-to-label-transfer gap on the 0.5 tree is smallest at 3 px: 0.8594 against 0.8635,
seed 0 only.

### F249 — The registration axis's own re-boxing loosens train boxes by worst-case mean IoU 0.9666 / 0.9346 / 0.8746 / 0.7675 at 3 / 6 / 12 / 24 px, and by best-case at most 0.0002; this bounds the label looseness behind F224's slope but does not apportion it

The comparison is each box `misalign_label_lines` writes against the tight box of the same object
warped by `Misalignment.homography()`. Two bounding shapes are used:

- **Thin diagonal segment** (worst case). Train mean IoU: 0.9666 / 0.9346 / 0.8746 / 0.7675. Val:
  0.9643 / 0.9304 / 0.8671 / 0.7553.
- **Box-filling rectangle** (best case). Mean IoU ≥ 0.9998 on train and ≥ 0.9997 on val at every
  level.

The train clamp rate steps from 13.8% at 0 px to 31.9% at 3 px, then stays at 31.9–32.5% up to
24 px. No box is dropped.

Against F224's label-transfer curve: F224 is flat to 6 px (−0.0010), where worst-case looseness is
already 0.9346. It falls −0.0729 at 24 px, where worst-case looseness reaches 0.7675. Each real
object sits somewhere in the band, at a place this run cannot know.

### F250 — The frozen custom tree was built from `brazil_all_29_jul_2026_yolo` at the default `--min-visibility`, not from the sister's default `yolo_original`, which is an older 4-class export; no polygon source was found

- **The build command.** PSReadLine history holds `python .\build_rgbt_dataset.py --root
  "D:\Atick\data\dataset\Brazil-Aligned" --original-dir "brazil_all_29_jul_2026_yolo" --output-dir
  yolo_rgbt_29_jul`, with no `--min-visibility` flag.
- **The source folder.** That folder's FLIR1448 label reads `4 1 2 1 1` and was written
  2026-07-29 19:58, 23 minutes before the frozen tree. `yolo_original` reads `3 0 1 0 0`, was
  written 2026-02-11, and names 4 classes with no `Connector`.
- **The audit's match.** The audit at `--min-visibility 0.25` reproduced every kept frozen box from
  `brazil_all_29_jul_2026_yolo` (F245).
- **The drift of `yolo_original`.** Against it, 15 frozen train files hold more boxes than their
  original, and the build only ever drops boxes.
- **No build summary.** No saved `report()` line exists anywhere under `Brazil-Aligned`.
- **No polygons.** No JSON under `D:\Atick\data` names `rectanglelabels` or `polygonlabels`, and
  0 of 2,929 `yolo_original` lines have more than five fields. That line count was taken on the old
  export, not on the build's source.

## Provenance caveats

- **Log dates.** Seven of the nine logs carry 2026-11-16 in their names, but only steps 1 and 2 ran
  that day. The window's `$D` was set on 11-16 and reused on 11-17. Only `rescore-single`, from a
  window opened on 11-17, carries the date it ran. The provenance timestamps are the record's dates.
- **First routes run.** Its 0 px block is good. Its 3 px block scored `custom-l3-s0` beside
  `regc-12px-s0`'s export and checkpoints. The `[ordered]` table was indexed by position
  (`docs/features/clipped-box-audit/pitfalls.md`). That block's zero-shot reads 0.7967, the 12 px
  value. Its CSV was deleted by runbook step 6a. The 3 px CSV here is the rerun's (zero-shot
  original 0.7682). The 6, 12 and 24 px blocks crashed before scoring
  (`ValueError: could not convert string to float`). The rerun re-scored 3, 6, 12 and 24 px. The bad
  3 px block's lines are still in the first log, quoted under Results.
- **Step 3a's val half failed.** The PowerShell path became `System.Collections.Hashtable\val\labels`,
  so its val histograms and drift count are empty. Its command line names `yolo_original` because it
  ran before `$ORIG` was repointed. It compared the old export, which is its purpose.
- **The first audit run is missing.** It ran against `yolo_original` and, per the runbook's step-3a
  note, failed with `743 label files disagree`. It was overwritten by the passing run under the same
  filename, so it is known only from the runbook.
- **The polygon line count** ran on `yolo_original`, not on `brazil_all_29_jul_2026_yolo` (F250).
- **Images without homographies.** Thirteen images (12 train, 1 val) have no alignment file, and
  the audit skipped them "as the build did". The fidelity pass is consistent with that, but the
  build's own skip was not observed.
- **Seed coverage of the route calls.** They are seed 0 only. Zero-shot and adapted use `e3b-loop`'s
  six-seed spreads at every level, the only translation spreads measured. Label-transfer spreads were
  typed into the runbook by hand from record 040's CSV.
- **No per-class AP.** Ultralytics printed only the `all` row, so the 0.75 tree's effect on the Pole
  class (152 of 153 removed) is not measured. F248's 0.75 reorder is over a val set with almost no
  poles.
- **`lost_matches` is near-exact.** Removing a box can, rarely, let a prediction re-match another
  (`clipped_box_rescore.py:13-15`).
- **Concurrency.** Steps 6 and 7 ran at the same time on the two cards. No timing claim depends on
  either running alone.

## Next

Plan step 4 applies the Q3 verdict. The rule says `fix`, and the plan opens a corrected-tree feature
"only if Q3's rule says so". F246 and F247 say the drop-tree delta cannot measure label looseness on
detectors trained on the same loose boxes. Whatever measurement a fix feature gates on, it cannot be
this delta.

A falsifiable prediction for any val tree whose boxes are tightened, scored with these same
checkpoints: mAP50 goes **down**, not up. 63–71 of the 72 sub-0.5 boxes are matched today by
predictions as loose as the labels. A tight box would reject them at IoU 0.5. A tighter val set
therefore means anything only beside detectors retrained on tightened train boxes.
