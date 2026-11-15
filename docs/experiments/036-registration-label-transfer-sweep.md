# 036 — Registration-error axis, label-transfer half: custom holds to 6 px and drops 0.073 mAP50 at 24 px; FLIR drops 0.222 on its 3-class primary, and its three seeds came out identical

**Date:** 2026-11-11 · **Task:** REGISTRATION-AXIS-06 · **Machine:** VSIBMEOT05, 2× A100 (custom `cuda:0`, FLIR `cuda:1`) · **Wall clock:** custom 3 h 39 min (2026-11-11 09:34:31 → 13:13:57 +03:00, 13.5–16.0 min per cell); FLIR 24 h 08 min (2026-11-11 19:30:55 → 2026-11-12 19:38:41, 84.6–102.3 min per cell). Before that, the first attempt on 2026-11-10 hung for 15 h 27 min without finishing a cell
**git_sha:** `pre-spec-migration-256-gb541e76` on both sweeps and on the crashed attempt (clean tree, no `-dirty` suffix) · **W&B:** none (no W&B line in any log) · **Log:** `logs/2026-11-11-REGISTRATION-AXIS-06-sweep-custom.txt` (condensed), `logs/2026-11-11-REGISTRATION-AXIS-06-sweep-flir.txt` (condensed), `logs/2026-11-10-REGISTRATION-AXIS-06-sweep-custom.csv`, `logs/2026-11-10-REGISTRATION-AXIS-06-sweep-flir.csv`, `logs/2026-11-10-REGISTRATION-AXIS-06-sweep-custom.txt` (the crashed attempt), `logs/2026-11-11-REGISTRATION-AXIS-06-commit-diagnosis.txt`

This is the label-transfer half of goal C2's registration-error axis (legacy C4), as set out in
`docs/features/registration-axis/plan.md`. Each level is a known corner-field misalignment
(0 / 3 / 6 / 12 / 24 px). It is pushed into the **train** boxes only, and each level gets its own
label tree. A thermal detector is trained on that tree and scored on the untouched val split. The
translation half of the same campaign is record 037. The runbook asked for one record per half.
This record holds the two sweeps, their CSVs, the crashed first attempt and the commit diagnosis
behind the crash.

## Command

From the provenance blocks, as run:

```
uv run python scripts/registration_sweep.py --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --dataset custom --out runs/registration-custom --workers 8 --device cuda:0
uv run python scripts/registration_sweep.py --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --dataset custom --out runs/registration-custom --workers 4 --device cuda:0
uv run python scripts/registration_sweep.py --data dataset/processed/flir/data.yaml --dataset flir --out runs/registration-flir --init-weights yolo11s.pt --epochs 100 --primary-classes bicycle car person --workers 4 --device cuda:1
```

The first line is the crashed 2026-11-10 attempt at `--workers 8`. The restart lowered every step
to `--workers 4`. `--seeds` and `--levels` were left at their defaults, which are 0 1 2 and
0 3 6 12 24 (`scripts/registration_sweep.py:263`).

## Environment

The block from the custom sweep's log. All three sweep logs carry the same block, apart from the
`captured`, `command` and `git_sha` lines (checked by hashing each block with those three lines
removed). The same holds for every translation log in record 037.

```
=== PROVENANCE ===
captured: 2026-11-11 09:34:31 +03:00
hostname: VSIBMEOT05
os: Microsoft Windows Server 2022 Standard 10.0.20348 (build 20348)
cpu: AMD EPYC 7763 64-Core Processor
cpu_logical_cores: 64
ram_total: 128.0 GB
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16; NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16
cuda_driver: 13.0
git_sha: pre-spec-migration-256-gb541e76
working_dir: D:\Atick\GitHub\Thermal-To-Optical-Translation
command: uv run python scripts/registration_sweep.py --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --dataset custom --out runs/registration-custom --workers 4 --device cuda:0
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
| Levels | 0 / 3 / 6 / 12 / 24 px, measured on each set's native frame (640×480) | 0 / 3 / 6 / 12 / 24 px on 640×512 |
| Seeds | 0, 1, 2 | 0, 1, 2 |
| Detector | `yolo11n.pt` (COCO), 50 epochs: E8 arm A's recipe | `yolo11s.pt` (COCO), 100 epochs: the FLIR judge's recipe (record 013) |
| Train corpus | all 600 train images, thermal side, boxes moved by the level | all 4,129 train images, thermal side, de-rolled boxes moved by the level |
| Val | 153 images, untouched | 1,013 images, untouched de-rolled thermal labels |
| Headline metric | all-class mAP50. Connector's AP50 is empty in every row | `primary_map50` over bicycle, car and person (ADR-06 of `modtr-baseline`), with all-class alongside |
| Calibration digest | none | `210142740cdba163` (`calibration/flir.json`) on every row |
| Level digests | 0 `c2d6dbd0ff938465`, 3 `47b6bcb1ce5a355d`, 6 `586d0f9d8a5d8ef8`, 12 `4e5407a052bee5a9`, 24 `2ab7dd48f99a15e6` | 0 `45aab9344c36a4f0`, 3 `f157052d96339273`, 6 `9ed047b01e276e51`, 12 `4893f777ee6899db`, 24 `ff3dba9b2deab47f` |
| Boxes dropped by the warp | 0 at every level | 0, except 2 at 24 px |
| Batch, workers | 16, 4 | 16, 4 |

One tree per level is shared by all three seeds. At fraction 1.0 the budget manifest ignores the
seed, so the seed reaches only `model.train(seed=…)` (`scripts/registration_sweep.py:188-201`,
`src/t2o/engine/detector_stage.py:157-170`). Cells ran seed-major: every level at seed 0 first.

## Results

The sweeps' per-cell lines, verbatim from the two logs:

```
2026-11-11 09:36:00,148 INFO    __main__: custom at 640x480: 15 cell(s), 0 already in D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\registration-tidy.csv, 15 to run
2026-11-11 09:36:00,148 INFO    __main__: [1/15] 0 px  seed=0
2026-11-11 09:51:03,077 INFO    __main__: [1/15] 0 px  seed=0  mAP50 0.9297  (15.0 min)
2026-11-11 09:51:03,077 INFO    __main__: [2/15] 3 px  seed=0
2026-11-11 10:07:03,545 INFO    __main__: [2/15] 3 px  seed=0  mAP50 0.9284  (16.0 min)
2026-11-11 10:07:03,545 INFO    __main__: [3/15] 6 px  seed=0
2026-11-11 10:22:16,931 INFO    __main__: [3/15] 6 px  seed=0  mAP50 0.9322  (15.2 min)
2026-11-11 10:22:16,931 INFO    __main__: [4/15] 12 px  seed=0
2026-11-11 10:37:29,114 INFO    __main__: [4/15] 12 px  seed=0  mAP50 0.9118  (15.2 min)
2026-11-11 10:37:29,114 INFO    __main__: [5/15] 24 px  seed=0
2026-11-11 10:52:16,759 INFO    __main__: [5/15] 24 px  seed=0  mAP50 0.8676  (14.8 min)
2026-11-11 10:52:16,761 INFO    __main__: [6/15] 0 px  seed=1
2026-11-11 11:07:23,059 INFO    __main__: [6/15] 0 px  seed=1  mAP50 0.9280  (15.1 min)
2026-11-11 11:07:23,064 INFO    __main__: [7/15] 3 px  seed=1
2026-11-11 11:21:28,113 INFO    __main__: [7/15] 3 px  seed=1  mAP50 0.9317  (14.1 min)
2026-11-11 11:21:28,113 INFO    __main__: [8/15] 6 px  seed=1
2026-11-11 11:35:28,639 INFO    __main__: [8/15] 6 px  seed=1  mAP50 0.9266  (14.0 min)
2026-11-11 11:35:28,639 INFO    __main__: [9/15] 12 px  seed=1
2026-11-11 11:49:23,323 INFO    __main__: [9/15] 12 px  seed=1  mAP50 0.9218  (13.9 min)
2026-11-11 11:49:23,328 INFO    __main__: [10/15] 24 px  seed=1
2026-11-11 12:03:23,053 INFO    __main__: [10/15] 24 px  seed=1  mAP50 0.8345  (14.0 min)
2026-11-11 12:03:23,053 INFO    __main__: [11/15] 0 px  seed=2
2026-11-11 12:18:00,091 INFO    __main__: [11/15] 0 px  seed=2  mAP50 0.9356  (14.6 min)
2026-11-11 12:18:00,097 INFO    __main__: [12/15] 3 px  seed=2
2026-11-11 12:32:32,004 INFO    __main__: [12/15] 3 px  seed=2  mAP50 0.9309  (14.5 min)
2026-11-11 12:32:32,005 INFO    __main__: [13/15] 6 px  seed=2
2026-11-11 12:46:28,224 INFO    __main__: [13/15] 6 px  seed=2  mAP50 0.9316  (13.9 min)
2026-11-11 12:46:28,224 INFO    __main__: [14/15] 12 px  seed=2
2026-11-11 13:00:27,244 INFO    __main__: [14/15] 12 px  seed=2  mAP50 0.9139  (14.0 min)
2026-11-11 13:00:27,244 INFO    __main__: [15/15] 24 px  seed=2
2026-11-11 13:13:54,392 INFO    __main__: [15/15] 24 px  seed=2  mAP50 0.8725  (13.5 min)
2026-11-11 13:13:54,392 INFO    __main__: sweep complete: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\registration-tidy.csv
```

```
2026-11-11 19:32:11,073 INFO    __main__: flir at 640x512: 15 cell(s), 0 already in D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\registration-tidy.csv, 15 to run
2026-11-11 19:32:11,073 INFO    __main__: [1/15] 0 px  seed=0
2026-11-11 21:10:20,482 INFO    __main__: [1/15] 0 px  seed=0  mAP50 0.6069  (98.2 min)
2026-11-11 21:10:20,482 INFO    __main__: [2/15] 3 px  seed=0
2026-11-11 22:34:56,685 INFO    __main__: [2/15] 3 px  seed=0  mAP50 0.6065  (84.6 min)
2026-11-11 22:34:56,686 INFO    __main__: [3/15] 6 px  seed=0
2026-11-12 00:12:50,199 INFO    __main__: [3/15] 6 px  seed=0  mAP50 0.5923  (97.9 min)
2026-11-12 00:12:50,199 INFO    __main__: [4/15] 12 px  seed=0
2026-11-12 01:54:42,570 INFO    __main__: [4/15] 12 px  seed=0  mAP50 0.5582  (101.9 min)
2026-11-12 01:54:42,570 INFO    __main__: [5/15] 24 px  seed=0
2026-11-12 03:37:02,551 INFO    __main__: [5/15] 24 px  seed=0  mAP50 0.4161  (102.3 min)
2026-11-12 03:37:02,551 INFO    __main__: [6/15] 0 px  seed=1
2026-11-12 05:16:16,040 INFO    __main__: [6/15] 0 px  seed=1  mAP50 0.6069  (99.2 min)
2026-11-12 05:16:16,041 INFO    __main__: [7/15] 3 px  seed=1
2026-11-12 06:51:34,069 INFO    __main__: [7/15] 3 px  seed=1  mAP50 0.6065  (95.3 min)
2026-11-12 06:51:34,069 INFO    __main__: [8/15] 6 px  seed=1
2026-11-12 08:30:32,333 INFO    __main__: [8/15] 6 px  seed=1  mAP50 0.5923  (99.0 min)
2026-11-12 08:30:32,333 INFO    __main__: [9/15] 12 px  seed=1
2026-11-12 10:08:51,586 INFO    __main__: [9/15] 12 px  seed=1  mAP50 0.5582  (98.3 min)
2026-11-12 10:08:51,586 INFO    __main__: [10/15] 24 px  seed=1
2026-11-12 11:48:26,899 INFO    __main__: [10/15] 24 px  seed=1  mAP50 0.4161  (99.6 min)
2026-11-12 11:48:26,899 INFO    __main__: [11/15] 0 px  seed=2
2026-11-12 13:20:30,055 INFO    __main__: [11/15] 0 px  seed=2  mAP50 0.6069  (92.1 min)
2026-11-12 13:20:30,055 INFO    __main__: [12/15] 3 px  seed=2
2026-11-12 14:53:29,853 INFO    __main__: [12/15] 3 px  seed=2  mAP50 0.6065  (93.0 min)
2026-11-12 14:53:29,853 INFO    __main__: [13/15] 6 px  seed=2
2026-11-12 16:29:02,920 INFO    __main__: [13/15] 6 px  seed=2  mAP50 0.5923  (95.6 min)
2026-11-12 16:29:02,920 INFO    __main__: [14/15] 12 px  seed=2
2026-11-12 18:03:47,234 INFO    __main__: [14/15] 12 px  seed=2  mAP50 0.5582  (94.7 min)
2026-11-12 18:03:47,234 INFO    __main__: [15/15] 24 px  seed=2
2026-11-12 19:38:39,114 INFO    __main__: [15/15] 24 px  seed=2  mAP50 0.4161  (94.9 min)
2026-11-12 19:38:39,114 INFO    __main__: sweep complete: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\registration-tidy.csv
```

The two results files, verbatim. They are `runs/registration-{custom,flir}/registration-tidy.csv`
on the server, pasted back under the names above:

```
dataset,misalignment_px,misalignment_digest,seed,init_weights,epochs,precision,recall,map50,primary_map50,map50_95,boxes_dropped,calibration_digest,weights,data,ap50_Connector,ap50_Fuse,ap50_Pole,ap50_Switch,ap50_Transformer
custom,0.0,c2d6dbd0ff938465,0,yolo11n.pt,50,0.8899083578237317,0.886784830596264,0.9297053272744761,,0.6591637546587517,0,,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\detectors\custom-l0-s0\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\manifests\l0\data.yaml,,0.9473763136525651,0.9787621743892466,0.8448806920012522,0.9478021290548404
custom,3.0,47b6bcb1ce5a355d,0,yolo11n.pt,50,0.8911763674442563,0.8702535722649352,0.9284387702383913,,0.6550008478692717,0,,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\detectors\custom-l3-s0\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\manifests\l3\data.yaml,,0.9458222912033719,0.9694298874424402,0.8700313045623113,0.9284715977454417
custom,6.0,586d0f9d8a5d8ef8,0,yolo11n.pt,50,0.9094795772181226,0.8670858601786426,0.9321531386727582,,0.6358743776365348,0,,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\detectors\custom-l6-s0\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\manifests\l6\data.yaml,,0.9542520522258394,0.969928354034083,0.8325163975153594,0.971915750915751
custom,12.0,4e5407a052bee5a9,0,yolo11n.pt,50,0.8721624488488546,0.8779435347012428,0.9117673925491945,,0.5940020760803556,0,,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\detectors\custom-l12-s0\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\manifests\l12\data.yaml,,0.9302102483157515,0.9671248075537116,0.8099198248796466,0.939814689447668
custom,24.0,2ab7dd48f99a15e6,0,yolo11n.pt,50,0.8667890605833801,0.7886693003242256,0.867638855354869,,0.49409606205292655,0,,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\detectors\custom-l24-s0\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\manifests\l24\data.yaml,,0.8563825299026255,0.9695595809337217,0.6948820795203974,0.9497312310627312
custom,0.0,c2d6dbd0ff938465,1,yolo11n.pt,50,0.9224769133504078,0.8624851938182709,0.9280269144806557,,0.6510529219745989,0,,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\detectors\custom-l0-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\manifests\l0\data.yaml,,0.9447651147669895,0.979562638299611,0.818953487077746,0.9688264177782758
custom,3.0,47b6bcb1ce5a355d,1,yolo11n.pt,50,0.93751014773871,0.8619469333504343,0.9316507400608023,,0.6633360397900566,0,,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\detectors\custom-l3-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\manifests\l3\data.yaml,,0.9368076701192055,0.9753231555374975,0.8563103821496119,0.9581617524368947
custom,6.0,586d0f9d8a5d8ef8,1,yolo11n.pt,50,0.9184245374091531,0.858298624198632,0.9266477308332345,,0.6418505711362534,0,,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\detectors\custom-l6-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\manifests\l6\data.yaml,,0.9292211826634666,0.9758026487545661,0.8561150965418574,0.9454519953730479
custom,12.0,4e5407a052bee5a9,1,yolo11n.pt,50,0.8639517700815496,0.8956202242889659,0.9217532694650716,,0.6156733404618077,0,,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\detectors\custom-l12-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\manifests\l12\data.yaml,,0.946072760699801,0.9713674305838736,0.8140084214364889,0.9555644651401229
custom,24.0,2ab7dd48f99a15e6,1,yolo11n.pt,50,0.8112801984439247,0.7662620245804115,0.8345465304914281,,0.4856266349254428,0,,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\detectors\custom-l24-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\manifests\l24\data.yaml,,0.7990208489543753,0.9681579576163909,0.6815417043800698,0.8894656110148765
custom,0.0,c2d6dbd0ff938465,2,yolo11n.pt,50,0.9199266416294143,0.8729664303478842,0.9356200274791998,,0.6693596666667514,0,,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\detectors\custom-l0-s2\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\manifests\l0\data.yaml,,0.9308509929385598,0.9874260649702123,0.8524462952512701,0.9717567567567568
custom,3.0,47b6bcb1ce5a355d,2,yolo11n.pt,50,0.9449798249036488,0.8299821924477312,0.9309241975743054,,0.6594036681474783,0,,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\detectors\custom-l3-s2\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\manifests\l3\data.yaml,,0.9394013542452057,0.9762956776426914,0.86068523551822,0.9473145228911053
custom,6.0,586d0f9d8a5d8ef8,2,yolo11n.pt,50,0.9095747581818047,0.8993392193569392,0.9316064421986574,,0.6558167538961104,0,,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\detectors\custom-l6-s2\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\manifests\l6\data.yaml,,0.9429465091185755,0.9613143536853964,0.8542792008604527,0.9678857051302049
custom,12.0,4e5407a052bee5a9,2,yolo11n.pt,50,0.902799041009223,0.8523614200224344,0.9139099498074756,,0.6036044736025923,0,,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\detectors\custom-l12-s2\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\manifests\l12\data.yaml,,0.9188849232779445,0.9652618180461943,0.8112395307695708,0.9602535271361929
custom,24.0,2ab7dd48f99a15e6,2,yolo11n.pt,50,0.8387390212308812,0.8158865310896102,0.872491024199681,,0.49758839968078084,0,,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\detectors\custom-l24-s2\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-custom\manifests\l24\data.yaml,,0.8670211573888542,0.9598526530228617,0.7222833092667883,0.9408069771202199
```

```
dataset,misalignment_px,misalignment_digest,seed,init_weights,epochs,precision,recall,map50,primary_map50,map50_95,boxes_dropped,calibration_digest,weights,data,ap50_bicycle,ap50_car,ap50_dog,ap50_person
flir,0.0,45aab9344c36a4f0,0,yolo11s.pt,100,0.7398870213969431,0.5469940938411355,0.6069062760283741,0.7763747281441065,0.32361326971896487,0,210142740cdba163,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\detectors\flir-l0-s0\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\manifests\l0\data.yaml,0.5976639840632614,0.9046412426902474,0.09850091968117718,0.8268189576788104
flir,3.0000000000000004,f157052d96339273,0,yolo11s.pt,100,0.6593196719286358,0.5508951065016773,0.6065233136646615,0.7764249130457589,0.31855206477333226,0,210142740cdba163,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\detectors\flir-l3-s0\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\manifests\l3\data.yaml,0.6103798389248716,0.9016112893485733,0.09681851552136951,0.8172836108638319
flir,6.000000000000001,9ed047b01e276e51,0,yolo11s.pt,100,0.6265507265263084,0.5721661315171929,0.5923014035926589,0.7631310097367415,0.3056648123632387,0,210142740cdba163,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\detectors\flir-l6-s0\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\manifests\l6\data.yaml,0.5754713136618177,0.8949370019502966,0.07981258516041125,0.8189847135981103
flir,12.000000000000002,4893f777ee6899db,0,yolo11s.pt,100,0.7337824020202094,0.5184158249762957,0.5581982047197362,0.7185023681977434,0.24718194326498408,0,210142740cdba163,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\detectors\flir-l12-s0\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\manifests\l12\data.yaml,0.5352039310014082,0.846970876314553,0.07728571428571429,0.7733322972772692
flir,24.000000000000004,ff3dba9b2deab47f,0,yolo11s.pt,100,0.7325603692312457,0.39821217585447355,0.41609603619769997,0.5547109084960952,0.15635808181941427,2,210142740cdba163,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\detectors\flir-l24-s0\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\manifests\l24\data.yaml,0.30172665438192814,0.7178435582906038,0.00025141930251419307,0.6445625128157537
flir,0.0,45aab9344c36a4f0,1,yolo11s.pt,100,0.7398870213969431,0.5469940938411355,0.6069062760283741,0.7763747281441065,0.32361326971896487,0,210142740cdba163,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\detectors\flir-l0-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\manifests\l0\data.yaml,0.5976639840632614,0.9046412426902474,0.09850091968117718,0.8268189576788104
flir,3.0000000000000004,f157052d96339273,1,yolo11s.pt,100,0.6593196719286358,0.5508951065016773,0.6065233136646615,0.7764249130457589,0.31855206477333226,0,210142740cdba163,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\detectors\flir-l3-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\manifests\l3\data.yaml,0.6103798389248716,0.9016112893485733,0.09681851552136951,0.8172836108638319
flir,6.000000000000001,9ed047b01e276e51,1,yolo11s.pt,100,0.6265507265263084,0.5721661315171929,0.5923014035926589,0.7631310097367415,0.3056648123632387,0,210142740cdba163,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\detectors\flir-l6-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\manifests\l6\data.yaml,0.5754713136618177,0.8949370019502966,0.07981258516041125,0.8189847135981103
flir,12.000000000000002,4893f777ee6899db,1,yolo11s.pt,100,0.7337824020202094,0.5184158249762957,0.5581982047197362,0.7185023681977434,0.24718194326498408,0,210142740cdba163,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\detectors\flir-l12-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\manifests\l12\data.yaml,0.5352039310014082,0.846970876314553,0.07728571428571429,0.7733322972772692
flir,24.000000000000004,ff3dba9b2deab47f,1,yolo11s.pt,100,0.7325603692312457,0.39821217585447355,0.41609603619769997,0.5547109084960952,0.15635808181941427,2,210142740cdba163,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\detectors\flir-l24-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\manifests\l24\data.yaml,0.30172665438192814,0.7178435582906038,0.00025141930251419307,0.6445625128157537
flir,0.0,45aab9344c36a4f0,2,yolo11s.pt,100,0.7398870213969431,0.5469940938411355,0.6069062760283741,0.7763747281441065,0.32361326971896487,0,210142740cdba163,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\detectors\flir-l0-s2\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\manifests\l0\data.yaml,0.5976639840632614,0.9046412426902474,0.09850091968117718,0.8268189576788104
flir,3.0000000000000004,f157052d96339273,2,yolo11s.pt,100,0.6593196719286358,0.5508951065016773,0.6065233136646615,0.7764249130457589,0.31855206477333226,0,210142740cdba163,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\detectors\flir-l3-s2\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\manifests\l3\data.yaml,0.6103798389248716,0.9016112893485733,0.09681851552136951,0.8172836108638319
flir,6.000000000000001,9ed047b01e276e51,2,yolo11s.pt,100,0.6265507265263084,0.5721661315171929,0.5923014035926589,0.7631310097367415,0.3056648123632387,0,210142740cdba163,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\detectors\flir-l6-s2\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\manifests\l6\data.yaml,0.5754713136618177,0.8949370019502966,0.07981258516041125,0.8189847135981103
flir,12.000000000000002,4893f777ee6899db,2,yolo11s.pt,100,0.7337824020202094,0.5184158249762957,0.5581982047197362,0.7185023681977434,0.24718194326498408,0,210142740cdba163,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\detectors\flir-l12-s2\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\manifests\l12\data.yaml,0.5352039310014082,0.846970876314553,0.07728571428571429,0.7733322972772692
flir,24.000000000000004,ff3dba9b2deab47f,2,yolo11s.pt,100,0.7325603692312457,0.39821217585447355,0.41609603619769997,0.5547109084960952,0.15635808181941427,2,210142740cdba163,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\detectors\flir-l24-s2\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\registration-flir\manifests\l24\data.yaml,0.30172665438192814,0.7178435582906038,0.00025141930251419307,0.6445625128157537
```

Seed means computed from those CSVs (sd is the sample sd over n = 3):

| Level | custom mAP50 | Δ vs 0 px | custom mAP50-95 | custom Switch / Fuse AP50 | FLIR 3-class | Δ vs 0 px | FLIR all-class | FLIR mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9311 ± 0.0040 | — | 0.6599 | 0.8388 / 0.9410 | 0.7764 | — | 0.6069 | 0.3236 |
| 3 | 0.9303 ± 0.0017 | −0.0008 | 0.6592 | 0.8623 / 0.9407 | 0.7764 | +0.0001 | 0.6065 | 0.3186 |
| 6 | 0.9301 ± 0.0030 | −0.0010 | 0.6445 | 0.8476 / 0.9421 | 0.7631 | −0.0132 | 0.5923 | 0.3057 |
| 12 | 0.9158 ± 0.0053 | −0.0153 | 0.6044 | 0.8117 / 0.9317 | 0.7185 | −0.0579 | 0.5582 | 0.2472 |
| 24 | 0.8582 ± 0.0206 | −0.0729 | 0.4924 | 0.6996 / 0.8408 | 0.5547 | −0.2217 | 0.4161 | 0.1564 |

FLIR's sd is 0.0000 at every level (F227).

No renderer exists for this axis yet. REGISTRATION-AXIS-07 builds the degradation report from the
two CSVs.

### The crash behind the restart

The custom sweep's 2026-11-10 attempt died in its first cell, while a data-loader worker was
importing torch. Its last lines:

```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\Users\atick\AppData\Local\Programs\Python\Python312\Lib\multiprocessing\spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\atick\AppData\Local\Programs\Python\Python312\Lib\multiprocessing\spawn.py", line 131, in _main
    prepare(preparation_data)
  File "C:\Users\atick\AppData\Local\Programs\Python\Python312\Lib\multiprocessing\spawn.py", line 246, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
  File "C:\Users\atick\AppData\Local\Programs\Python\Python312\Lib\multiprocessing\spawn.py", line 297, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen runpy>", line 287, in run_path
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\scripts\registration_sweep.py", line 53, in <module>
    from t2o.data.budget import write_budget_manifest
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\src\t2o\data\__init__.py", line 5, in <module>
    from t2o.data.budget import BudgetError, write_budget_manifest
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\src\t2o\data\budget.py", line 37, in <module>
    from t2o.data.dataset import IMAGE_SUFFIXES, annotated_subset
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\src\t2o\data\dataset.py", line 27, in <module>
    import torch
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torch\__init__.py", line 287, in <module>
    _load_dll_libraries()
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torch\__init__.py", line 270, in _load_dll_libraries
    raise err
OSError: [WinError 1455] The paging file is too small for this operation to complete. Error loading "D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\torch\lib\cublas64_13.dll" or one of its dependencies.
=== FINISHED 2026-11-11 09:30:14 +03:00 ===
```

The diagnosis was taken on 2026-11-11 while both jobs were still hung. It is verbatim, except
that the query truncated the `Cmd` column itself:

```
ProcessId ParentProcessId CreationDate        CommitGB Cmd
--------- --------------- ------------        -------- ---
    32228           30976 03/11/2026 15:34:59    20.61 
    84556           59664 04/11/2026 10:47:47    20.45 
    93036           31748 04/11/2026 10:47:40    17.50 
    35872           30976 03/11/2026 14:35:40    17.45 
    36700           30976 03/11/2026 12:11:51    16.52 
    45900           73684 04/11/2026 15:00:28     8.01 
    63240           20296 10/11/2026 18:04:23     4.52 "C:\Users\atick\AppData\Local\Programs\Python\Python312\python.exe" "D:\Atick\GitHub\Thermal-To-Optical-Translation\.ven
    66672           61268 10/11/2026 18:03:49     3.99 "C:\Users\atick\AppData\Local\Programs\Python\Python312\python.exe" scripts/registration_sweep.py --data D:\Atick\data\d
    44148           73684 04/11/2026 13:57:55     2.47 
    65500           73684 04/11/2026 13:23:10     2.32 
    19408           31748 04/11/2026 13:51:52     2.16 
    41996           63240 10/11/2026 18:22:23     2.05 "C:\Users\atick\AppData\Local\Programs\Python\Python312\python.exe" "-c" "from multiprocessing.spawn import spawn_main; 
    52780           63240 10/11/2026 18:22:23     2.05 "C:\Users\atick\AppData\Local\Programs\Python\Python312\python.exe" "-c" "from multiprocessing.spawn import spawn_main; 
    61480           63240 10/11/2026 18:22:23     2.04 "C:\Users\atick\AppData\Local\Programs\Python\Python312\python.exe" "-c" "from multiprocessing.spawn import spawn_main; 
    40364           63240 10/11/2026 18:22:23     2.04 "C:\Users\atick\AppData\Local\Programs\Python\Python312\python.exe" "-c" "from multiprocessing.spawn import spawn_main; 
    55444           63240 10/11/2026 18:22:23     2.04 "C:\Users\atick\AppData\Local\Programs\Python\Python312\python.exe" "-c" "from multiprocessing.spawn import spawn_main; 
    22800           63240 10/11/2026 18:22:24     2.04 "C:\Users\atick\AppData\Local\Programs\Python\Python312\python.exe" "-c" "from multiprocessing.spawn import spawn_main; 
    38748           63240 10/11/2026 18:22:24     1.99 "C:\Users\atick\AppData\Local\Programs\Python\Python312\python.exe" "-c" "from multiprocessing.spawn import spawn_main; 
    83780           63240 10/11/2026 18:22:24     1.98 "C:\Users\atick\AppData\Local\Programs\Python\Python312\python.exe" "-c" "from multiprocessing.spawn import spawn_main; 
    75268           30976 03/11/2026 11:20:32     1.77 
    93456           73684 04/11/2026 13:47:21     1.59 
    23240           59664 04/11/2026 13:51:34     1.43 
    92888           30976 03/11/2026 11:35:44     1.38 
    50528           66672 10/11/2026 18:06:19     1.07 "C:\Users\atick\AppData\Local\Programs\Python\Python312\python.exe" "-c" "from multiprocessing.spawn import spawn_main; 
    19676           30976 03/11/2026 15:22:55     1.03 
    59680           59664 04/11/2026 13:50:12     0.97 
    63272           30976 03/11/2026 14:48:57     0.97 
    31296           31748 04/11/2026 11:42:05     0.82 
    51736           59664 04/11/2026 10:45:50     0.82 
    10264           31748 04/11/2026 09:20:30     0.79 
    80004           30976 03/11/2026 09:52:00     0.26 
    53216           30976 03/11/2026 15:34:57     0.21 
    81840           30976 03/11/2026 15:02:19     0.21 
    54616           30976 03/11/2026 15:34:59     0.21 
    35304           69664 28/10/2026 14:35:41     0.19 
    56792           30976 03/11/2026 15:34:58     0.17 
    74396           30976 28/10/2026 13:54:57     0.17 
    13276           30976 05/11/2026 10:03:14     0.16 
    13372           30976 03/11/2026 11:06:20     0.16 
    78516           59664 04/11/2026 10:46:29     0.14 
    79736           69664 28/10/2026 11:39:40     0.07 
    78292           73684 04/11/2026 15:00:22     0.07 
    27216           30976 03/11/2026 15:02:14     0.07 
    72856           69664 28/10/2026 14:35:23     0.07 
    59440           30976 03/11/2026 09:47:08     0.07 
    90216           69664 28/10/2026 11:39:40     0.07 
    65316           30976 03/11/2026 09:47:01     0.07 
    91152            9444 28/09/2026 20:24:29     0.04 "C:\Users\atick\AppData\Local\Programs\Python\Python312\python.exe" "C:\Users\atick\.local\bin\nvitop.exe"
    31356           30976 28/10/2026 12:39:00     0.01 
    84452           31748 04/11/2026 09:20:29     0.01 
    33460           59664 04/11/2026 10:45:50     0.01 
    69040           69664 28/10/2026 11:39:40     0.01 
    29780           73684 04/11/2026 10:46:07     0.01 
    20296           71600 10/11/2026 18:04:23     0.00 "D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Scripts\python.exe" "D:\Atick\GitHub\Thermal-To-Optical-Translatio
    61268           55232 10/11/2026 18:03:49     0.00 "D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Scripts\python.exe" scripts/registration_sweep.py --data D:\Atick\
     9444           43744 28/09/2026 20:24:28     0.00 "C:\Users\atick\AppData\Roaming\uv\tools\nvitop\Scripts\python.exe" "C:\Users\atick\.local\bin\nvitop.exe"


python total commit GB: 147.4

Timestamp                 CounterSamples
---------                 --------------
11/11/2026 09:15:58       \\vsibmeot05\memory\commit limit :
                          204531109888
                         
                          \\vsibmeot05\memory\committed bytes :
                          203646001152
                         
                         


Caption   : C:\pagefile.sys
Name      : C:\pagefile.sys
PeakUsage : 28903
```

## Findings

### F224 — Custom label transfer is flat to 6 px and then falls: 0.9311 at 0 px, 0.9158 at 12 px, 0.8582 at 24 px; mAP50-95 and Switch drop first and furthest

Against its own 0 px point (n = 3 each): 3 px −0.0008 and 6 px −0.0010 are inside the seed
spread (sd 0.0017–0.0040, and the ranges overlap). 12 px is −0.0153, with all three 12 px seeds
(0.9118–0.9218) below all three 0 px seeds (0.9280–0.9356). 24 px is −0.0729, and the seed sd
widens to 0.0206. The localisation-sensitive mAP50-95 moves earlier and further: −0.0153 at 6 px,
−0.0554 at 12 px and −0.1674 at 24 px. By class, Switch loses most, 0.8388 → 0.6996 (−0.1392),
then Fuse 0.9410 → 0.8408 (−0.1002). Pole and Transformer lose 0.0160 and 0.0361. Small boxes
suffer most from misplaced labels, which is what a box-shift error should do.

### F225 — The custom 0 px point reproduces E8 arm A at N = 600: 0.9311 ± 0.0040 against 0.9300 ± 0.0096, the same recipe and seeds 0–2

The 0 px tree is the custom set's thermal labels unmoved (digest `c2d6dbd0ff938465`, 0 boxes
dropped). It was trained with E8 arm A's recipe (`yolo11n.pt`, 50 epochs, all 600 images, seeds
0–2). It lands 0.0011 above record 011's arm A, inside both spreads. This confirms only arm A's
side of F82. Nothing here re-measures C or D. It does confirm that the sweep's zero point is the
same baseline E8 measured, so the curve and E8 share an anchor.

### F226 — FLIR label transfer degrades about 3× faster than custom by 24 px: the 3-class primary falls 0.7764 → 0.7185 (12 px) → 0.5547 (24 px), −0.2217 against custom's −0.0729

The FLIR primary is flat to 3 px (+0.0001), then −0.0132 at 6 px, −0.0579 at 12 px and −0.2217
at 24 px, a 28.6% relative loss against custom's 7.8%. All-class follows, 0.6069 → 0.4161
(−0.1908). Every class falls: car 0.9046 → 0.7178, person 0.8268 → 0.6446, bicycle
0.5977 → 0.3017, dog 0.0985 → 0.0003. The two sweeps are not recipe-matched: `yolo11s`, 100
epochs and 4,129 images on FLIR, against `yolo11n`, 50 epochs and 600 images on custom. A pixel
level also means something different relative to each set's objects. So "3× faster" compares the
two curves as each campaign will report them. It does not isolate a dataset property. FLIR's
side is n = 1 in effect (F227).

### F227 — FLIR's seed does nothing: all three seeds give byte-identical metrics at every level, and identical validation lines from the first epoch, while custom's seeds diverge from the first epoch under the same code

In the FLIR CSV, seeds 0, 1 and 2 agree to every printed digit, in every metric and per-class
column, at all five levels. The cause is in training, not in scoring. In the log, ultralytics'
argument line shows `seed=0`, `seed=1` and `seed=2` for the three cells. Yet each level's first
two per-epoch validation lines are identical across the three seeds: at 0 px, 0.436 and then 0.39
mAP50 for all three. The custom sweep runs the same `_cell` and `train_detector` path
(`scripts/registration_sweep.py:175-201`, `src/t2o/engine/detector_stage.py:157-170`). Its seeds
diverge at once, with a first validation mAP50 of 0.137, 0.136 and 0.155 for seeds 0–2 at 0 px.
The visible difference between the two invocations is `--init-weights yolo11s.pt --epochs 100`,
against the defaults. Record 022 trained `yolo11s.pt` for 100 epochs at seeds 1–3 on msrs-day through the same
`train_detector`, and its seeds differ (V 0.7877 / 0.7994 / 0.7922). So `yolo11s` does not ignore
its seed everywhere. **The cause is not
established.** Two consequences follow. FLIR's label-transfer half is one measurement per level,
and its 0.0000 sd is not a noise estimate. REGISTRATION-AXIS-07 plans to plot it "with its seed
spread", and REGISTRATION-AXIS-04's C4 bracket takes its T row from these 0 px cells. And about
16 of the sweep's 24 GPU-hours, seeds 1 and 2, bought no new information.

### F228 — The 2026-11-10 crash was the host commit limit, not GPU memory, and both jobs hung for 15 h instead of exiting

At 09:15:58 the next morning, committed bytes stood at 203,646,001,152 against a limit of
204,531,109,888 (99.6%). Python processes held 147.4 GB of that. The eleven processes whose
command lines name this repo held 25.81 GB. Eight of them were spawned data-loader workers of the
FLIR translation run, at 1.98–2.05 GB each. Of the remaining 121.6 GB of Python commit, all but 0.04 GB
sat in processes started between 28 Oct and 5 Nov, whose command lines the query could not read. The
five largest of these held 92.5 GB, and the log does not name their owner. The jobs therefore
took about an eighth of the commit, and an `OSError: [WinError 1455]` in a spawned worker's
`import torch` did not end the run. The custom sweep's last timestamped line is at 18:06:04 and
the FLIR run's at 18:22:23. Both FINISHED stamps came at 09:30, written only when the hung
processes were killed by hand. Neither error was raised on the main thread. The sweep's came
from a spawned worker's `import torch`, and the FLIR run's from a multiprocessing queue feeder
thread (`multiprocessing\queues.py`, `_feed`). In both cases the main process went on waiting,
so a 1455 in a log means a job is stuck, not slow.

## Provenance caveats

- **Both sweep logs were condensed on the server, and the files keep the raw names.** The runbook
  asked for a `-condensed.txt` suffix. The condensed files only dropped progress-bar redraws below
  100%. Custom: raw `D:\Atick\GitHub\Thermal-To-Optical-Translation\logs\2026-11-11-REGISTRATION-AXIS-06-sweep-custom.txt`,
  6,447,250 bytes, SHA256 `8FBF438CE6E909B8EE28BD6A12A1AFB1DD31BE6E2996C946F7B853EA5D1044CC`,
  38,220 raw lines. FLIR: raw `D:\Atick\GitHub\Thermal-To-Optical-Translation\logs\2026-11-11-REGISTRATION-AXIS-06-sweep-flir.txt`,
  83,042,602 bytes, SHA256 `66627E2228E932DDBB45B4B6854B0E54A235ACD24A45299C86C5D497E7DAA4B5`,
  446,911 raw lines. The drop rule for both: lines matching `^ESC\[K.*\s\d{1,2}% `.
- **The two CSVs are dated 2026-11-10 but come from the 2026-11-11 sweeps.** The crashed attempt
  wrote no row, and each CSV's 15 rows match its 2026-11-11 log's per-cell lines.
- **Neither 0 px is truly zero.** The custom set's boxes were drawn on visible and moved to thermal
  by the sister repo's registration, so every custom cell, val included, is scored against boxes
  that already carry that error. FLIR's thermal labels were de-rolled by `calibration/flir.json`,
  which leaves about 2.25 px of median per-pair scatter (record 012).
- **FLIR is effectively one seed per level** (F227). Custom has three.
- **The FLIR label-transfer sweep and the FLIR translation runs don't share a training corpus.**
  The sweep trains on all 4,129 train images. Record 037's translators train on the 600-pair
  subset.
- **The two datasets' recipes differ** (`yolo11s` / 100 epochs against `yolo11n` / 50). This is
  by plan, so each half matches its own earlier references (record 011, record 013). It is also
  why F226 is not a dataset comparison.
- **The diagnosis log ends after its first block.** The runbook's step 2 appends an "after
  stopping" block, but none is in the file. Committed bytes after the stop are not recorded.
- **`--workers` changed between the attempts (8 → 4).** No sweep cell finished at 8, so every
  reported cell ran at 4.

## Next

- **REGISTRATION-AXIS-07**, the degradation report. It should show FLIR's label-transfer half as
  n = 1 per level, not as a seed band (F227). Custom's band is real.
- **Before any FLIR detector campaign with more than one seed**, find out why FLIR's seed is
  inert. A cheap test: two 3-epoch `yolo11s.pt` runs on FLIR at seeds 0 and 1, compared at their
  first validation line. Prediction: they match, as they did here. If they do, every multi-seed
  FLIR detector arm built on this path, the C4 bracket's T row included, is one draw.
- **The C4 bracket** (REGISTRATION-AXIS-04, `scripts/gate_table.py --label-transfer-csv`) can read
  the 0 px FLIR rows now. Its T row is one checkpoint in effect.
