# 025 — msrs-day annotation budget sweep: a thermal-trained detector needs ≈173 labelled images to match the loop on the passive classes

**Date:** 2026-10-15 · **Task:** MSRS-BUDGET-SWEEP-02 · **Machine:** VSIBMEOT05, supplied by the human after the run (the ultralytics banner reads `CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)`) · **Wall clock:** not recorded for the whole pass; the last segment ran 8 cells in 2 h 33 min 41 s (18:21:30 → 20:55:11)
**git_sha:** not recorded · **W&B:** none (the sweep script does not log to W&B; neither log mentions it) · **Log:** `logs/e8-tidy.csv` (all 22 cells), `logs/2026-10-15-MSRS-BUDGET-SWEEP-02-sweep-resume.txt` (cells 15–22), `logs/2026-10-15-MSRS-BUDGET-SWEEP-02-sweep.txt` (the stopped attempt)

## Command

```
not recorded
```

## Environment

```
not recorded
```

## Configuration

Pass 1 of the plan's two invocations (`docs/features/msrs-budget-sweep/plan.md` Q2–Q5): arms A0, C,
D and A; no B.

| Axis | Value | Where it is evidenced |
| --- | --- | --- |
| Budgets `N` | 25, 50, 100, 200, 536 | the CSV's `n_annotated` |
| Seeds | 0, 1, 2 for A, C and D; A0 once at `seed = -1` | the CSV |
| A's detector | COCO `yolo11s.pt`, 100 epochs, batch 16, imgsz 640, workers 16, `deterministic=True`, `device=0` | the trainer-args line in the resume log |
| Judge for A0 / C / D | `runs\reference-msrs-day-yolo11s-s1\weights\best.pt` (V at seed 1, record 021) | the CSV's `weights` |
| C / D exports | `runs\e3md-control-s{0,1,2}` / `runs\e3md-loop-s{0,1,2}`, stage 3 (record 024) | the CSV's `data` |
| Primary classes | car_stop, color_cone — `primary_map50` is their mean, checked by hand on A, N=536, s0: (0.6615 + 0.4417) / 2 = 0.5516 | the CSV |
| Val split | msrs-day val, 179 images, 976 instances | `all 179 976` in the resume log |

A's thermal boxes are msrs-day's visible boxes mirrored onto thermal (`mirror_thermal_labels.py`),
the same labels T trained on in record 022.

**Cost, segment measured.** A full-split cell (N = 536) took 24.0–24.9 min:

- ultralytics' 100 epochs, 0.260–0.269 h (15.6–16.1 min);
- setup, the label scan and ultralytics' own final val, about 6 min;
- the sweep's re-score, 2.1 min (20:03:34 → 20:05:41 on s0). The plan estimated about a minute.

N = 100 cells took 12.7–14.2 min, and N = 200 cells 17.5–18.1 min.

## Results

The sweep's own lines from the resume log, verbatim:

```
2026-10-15 18:21:30,153 INFO    __main__: 22 cell(s): 14 already in D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\e8-tidy.csv, 8 to run
2026-10-15 18:21:30,153 INFO    __main__: [1/8] arm A  N=100  seed=1
2026-10-15 18:34:10,746 INFO    __main__: [1/8] arm A  N=100  seed=1  mAP50 0.4645  primary 0.2912  (12.7 min)
2026-10-15 18:34:10,753 INFO    __main__: [2/8] arm A  N=100  seed=2
2026-10-15 18:48:23,294 INFO    __main__: [2/8] arm A  N=100  seed=2  mAP50 0.4224  primary 0.2249  (14.2 min)
2026-10-15 18:48:23,294 INFO    __main__: [3/8] arm A  N=200  seed=0
2026-10-15 19:06:27,345 INFO    __main__: [3/8] arm A  N=200  seed=0  mAP50 0.5475  primary 0.4237  (18.1 min)
2026-10-15 19:06:27,345 INFO    __main__: [4/8] arm A  N=200  seed=1
2026-10-15 19:24:12,131 INFO    __main__: [4/8] arm A  N=200  seed=1  mAP50 0.5162  primary 0.3583  (17.7 min)
2026-10-15 19:24:12,131 INFO    __main__: [5/8] arm A  N=200  seed=2
2026-10-15 19:41:42,289 INFO    __main__: [5/8] arm A  N=200  seed=2  mAP50 0.5450  primary 0.4165  (17.5 min)
2026-10-15 19:41:42,289 INFO    __main__: [6/8] arm A  N=536  seed=0
2026-10-15 20:05:41,596 INFO    __main__: [6/8] arm A  N=536  seed=0  mAP50 0.6518  primary 0.5516  (24.0 min)
2026-10-15 20:05:41,596 INFO    __main__: [7/8] arm A  N=536  seed=1
2026-10-15 20:30:15,010 INFO    __main__: [7/8] arm A  N=536  seed=1  mAP50 0.6477  primary 0.5312  (24.6 min)
2026-10-15 20:30:15,010 INFO    __main__: [8/8] arm A  N=536  seed=2
2026-10-15 20:55:10,984 INFO    __main__: [8/8] arm A  N=536  seed=2  mAP50 0.6583  primary 0.5651  (24.9 min)
2026-10-15 20:55:10,984 INFO    __main__: sweep complete: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\e8-tidy.csv
```

The stopped attempt, a resume itself, launched at 18:12:19 and stopped by hand at epoch 54 of
A, N=100, s1. Its two `__main__` lines and its last four lines, verbatim:

```
2026-10-15 18:12:19,965 INFO    __main__: 22 cell(s): 14 already in D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\e8-tidy.csv, 8 to run
2026-10-15 18:12:19,965 INFO    __main__: [1/8] arm A  N=100  seed=1
...
                   all        179        976      0.607      0.416      0.425      0.214
Traceback (most recent call last):
  File "D:\Atick\GitHub\Thermal-To-Optical-Translation\scripts\annotation_sweep.py", line 472, in <module>
```

`logs/e8-tidy.csv`, verbatim. It is the only source for cells 1–14:

```
arm,n_annotated,seed,detector,val_pixels,epochs,precision,recall,map50,primary_map50,map50_95,weights,data,ap50_car,ap50_person,ap50_bike,ap50_car_stop,ap50_color_cone,n_train_car,n_train_person,n_train_bike,n_train_car_stop,n_train_color_cone
A0,0,-1,reference,thermal,0,0.43040061137834895,0.1735614857531459,0.14119303757660745,0.06974576943834,0.05891456396917519,runs\reference-msrs-day-yolo11s-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\manifests\A0\data.yaml,0.08697406985206486,0.3862096037675238,0.0932899753867686,0.09692591715989549,0.04256562171678451,,,,,
C,0,0,reference,translated-l0,0,0.6852846769166938,0.3373907240295922,0.3667951897810747,0.18785275653822403,0.18763947049731172,runs\reference-msrs-day-yolo11s-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-control-s0\stage3\translated\data.yaml,0.7500850435633528,0.4018277114987356,0.3063576807668369,0.25565888575283335,0.12004662732361468,,,,,
C,0,1,reference,translated-l0,0,0.6407499690563908,0.35178376939637224,0.3628242175865114,0.18971832571226463,0.181609591141407,runs\reference-msrs-day-yolo11s-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-control-s1\stage3\translated\data.yaml,0.743590875595834,0.3884676562734322,0.3026259046387616,0.2352344716826401,0.14420217974188915,,,,,
C,0,2,reference,translated-l0,0,0.6358422977473112,0.3241654654613989,0.3479641454274108,0.18868243511093172,0.17500751378572488,runs\reference-msrs-day-yolo11s-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-control-s2\stage3\translated\data.yaml,0.6810772773077718,0.40709826565985374,0.2742803139475647,0.2739633071136406,0.10340156310822286,,,,,
D,536,0,reference,translated-lpos,0,0.7309778622841313,0.4861913677442728,0.5199685067137534,0.36323563848999973,0.28258482800111323,runs\reference-msrs-day-yolo11s-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-loop-s0\stage3\translated\data.yaml,0.8441781146117584,0.5677738943333694,0.46141924764363995,0.4683714311098354,0.2580998458701641,,,,,
D,536,1,reference,translated-lpos,0,0.712489786437356,0.4643637342021414,0.49612904897372234,0.3649133725622231,0.25518706273611264,runs\reference-msrs-day-yolo11s-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-loop-s1\stage3\translated\data.yaml,0.8141101151937209,0.5602041643219362,0.37650422022850843,0.4811224163003541,0.24870432882409205,,,,,
D,536,2,reference,translated-lpos,0,0.7757531510688324,0.47868416419271254,0.5150432717440109,0.36647800024263083,0.26710390264489825,runs\reference-msrs-day-yolo11s-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3md-loop-s2\stage3\translated\data.yaml,0.8174675016349495,0.5567939705042624,0.46799888609558066,0.45678636978433346,0.2761696307009282,,,,,
A,25,0,finetuned,thermal,100,0.5009316631069103,0.2914455147420614,0.29287323786180164,0.12089275611843653,0.14218744776668196,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\detectors\A-n25-s0\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\manifests\A-n25-s0\data.yaml,0.5606375721435262,0.5577455419846029,0.10419756294400623,0.1235772936750402,0.11820821856183286,36,70,16,35,8
A,25,1,finetuned,thermal,100,0.5909436652669162,0.29369766426927757,0.27284816940058765,0.042880426529883706,0.12305361486599933,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\detectors\A-n25-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\manifests\A-n25-s1\data.yaml,0.5243520575177789,0.55009910479849,0.2040288316269017,0.08508677230415349,0.0006740807556139288,23,51,25,19,5
A,25,2,finetuned,thermal,100,0.5587109063629834,0.29750722937158397,0.297987218237043,0.11110589095526298,0.14729912949872997,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\detectors\A-n25-s2\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\manifests\A-n25-s2\data.yaml,0.5830495223170924,0.5297604992871144,0.15491428767048215,0.11914668764382469,0.10306509426670127,34,53,10,15,10
A,50,0,finetuned,thermal,100,0.6027958067729406,0.36877743740444435,0.380155429446245,0.2085610305734636,0.1937871049285215,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\detectors\A-n50-s0\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\manifests\A-n50-s0\data.yaml,0.6972759297547728,0.5786646483286286,0.20771450800089647,0.23816052181696543,0.17896153932996176,67,153,29,60,27
A,50,1,finetuned,thermal,100,0.63019789617376,0.3489898597378267,0.3630394280996997,0.17958321238221087,0.17377959479268465,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\detectors\A-n50-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\manifests\A-n50-s1\data.yaml,0.6197392874547301,0.5907428569897004,0.24554857128964624,0.24953158408417608,0.10963484068024568,55,125,38,50,18
A,50,2,finetuned,thermal,100,0.6890208546035422,0.3240690398577107,0.36005186264066297,0.19569827665122988,0.17817006108284889,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\detectors\A-n50-s2\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\manifests\A-n50-s2\data.yaml,0.6475686343718174,0.5558381102147132,0.2054560153143244,0.1929699381578031,0.19842661514465665,58,111,18,40,35
A,100,0,finetuned,thermal,100,0.6710812592590589,0.41400284144730365,0.433002594067982,0.2984101914925978,0.2261070652726817,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\detectors\A-n100-s0\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\manifests\A-n100-s0\data.yaml,0.752666115202217,0.555894738497156,0.25963173365534165,0.35811604877663983,0.23870433420855575,129,264,55,116,50
A,100,1,finetuned,thermal,100,0.7114429092958497,0.42471826549114666,0.46446792349393534,0.2912345905462715,0.2355706354399144,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\detectors\A-n100-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\manifests\A-n100-s1\data.yaml,0.7805141793568907,0.5875894384285505,0.3717668185916927,0.3322710972751891,0.25019808381735387,117,256,64,92,57
A,100,2,finetuned,thermal,100,0.5684838770532759,0.43271074623322053,0.4224290323557563,0.2248907691560353,0.22213931214566698,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\detectors\A-n100-s2\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\manifests\A-n100-s2\data.yaml,0.7655723243108953,0.5915221860259431,0.3052691131298728,0.22722175394745403,0.22255978436461654,128,241,53,67,57
A,200,0,finetuned,thermal,100,0.70894914635146,0.5144708435052406,0.5474804679261751,0.4236794497457665,0.3004730568087245,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\detectors\A-n200-s0\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\manifests\A-n200-s0\data.yaml,0.8354469698202821,0.603937531713509,0.4506589386055513,0.5305881021590553,0.3167707973324777,241,529,129,186,87
A,200,1,finetuned,thermal,100,0.7613309214827397,0.482984402137799,0.5161511676092742,0.3582923055835842,0.2884322868186968,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\detectors\A-n200-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\manifests\A-n200-s1\data.yaml,0.8293564431072075,0.6022805208512434,0.4325342629207519,0.4214069395598071,0.2951776716073613,232,519,112,151,102
A,200,2,finetuned,thermal,100,0.7843583850346223,0.5011597559246247,0.5449894624279967,0.41654802176037,0.2950892295275677,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\detectors\A-n200-s2\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\manifests\A-n200-s2\data.yaml,0.8469187834205595,0.6057102974313617,0.4392221877673224,0.469146426769723,0.363949616751017,263,497,116,149,111
A,536,0,finetuned,thermal,100,0.798060634011083,0.585205130087651,0.6517684141460582,0.5515771493219104,0.3767968056763796,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\detectors\A-n536-s0\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\manifests\A-n536-s0\data.yaml,0.9048155355351397,0.6688596231295447,0.5820126134217862,0.6614916643209101,0.44166263432291075,661,1351,330,404,262
A,536,1,finetuned,thermal,100,0.7977953945140381,0.5887339212599995,0.647655855972233,0.5311746675166027,0.3783542176546504,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\detectors\A-n536-s1\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\manifests\A-n536-s1\data.yaml,0.9129345635899653,0.6809380595013251,0.5820573217366689,0.6099591426800408,0.4523901923531647,661,1351,330,404,262
A,536,2,finetuned,thermal,100,0.7947241094094715,0.6180200983703898,0.658297804248657,0.5651426950925502,0.3835134861569258,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\detectors\A-n536-s2\weights\best.pt,D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\msrs-budget-sweep\manifests\A-n536-s2\data.yaml,0.9110388200441133,0.6814833904930945,0.5686814205209767,0.6710445968361861,0.45924079334891443,661,1351,330,404,262
```

No report command exists. Every derived number below recomputes from this CSV with
`statistics.mean` / `statistics.stdev` (sample sd, n − 1). The crossover is the method of record 011:
linear interpolation of A's per-budget means between the two bracketing budgets.

## Findings

### F177 — The sweep reproduces all three of its anchors: A0 is J at seed 1 to every printed digit, C and D are record 024's stage-3 rows to six decimals, and A at N = 536 lands on T

- **A0 = J, seed 1.** Record 021's fourth eval line (J s1) reads P 0.4304, R 0.1736, mAP50 0.1412,
  mAP50-95 0.0589. A0 reads 0.43040, 0.17356, 0.14119, 0.05891. Its 2-class score is 0.0697,
  against J's three-seed mean of 0.0526: the sweep's judge is the s1 checkpoint, not the mean.
  The plan's "A0 near 0.0526" expected the mean.
- **C and D = record 024, seeds 0–2.**
  - C: 2-class 0.18785 / 0.18972 / 0.18868 against record 024's control stage-3 rows 0.187853 /
    0.189718 / 0.188682.
  - D: 0.36324 / 0.36491 / 0.36648 against record 024's loop rows 0.363236 / 0.364913 / 0.366478.
  - All-class matches the same way.
- **A at N = 536 ≈ T.**
  - 2-class 0.5493 ± 0.0171 against T's 0.5628 ± 0.0218 (record 021).
  - All-class 0.6526 ± 0.0054 against T's 0.6476 ± 0.0067.
  - These are three new training runs, not a re-read of T's checkpoints.

Nothing drifted between the campaign, the gate and the sweep. Like F86 and F87 for E8, this is a
validity check and never counted as second evidence for the numbers it reproduces.

### F178 — On the two passive classes, a thermal-trained detector needs ≈173 labelled thermal images (bracket 100–200) to match the loop's zero-label 0.3649; ±1 sd of A's own spread puts it at [143, 203]

2-class (car_stop, color_cone) mAP50, arm A at three seeds, against D's flat line (n = 3):

| N | A mean ± sd | A seeds | A seeds above every D seed |
| --- | --- | --- | --- |
| 25 | 0.0916 ± 0.0425 | 0.1209 / 0.0429 / 0.1111 | 0/3 |
| 50 | 0.1946 ± 0.0145 | 0.2086 / 0.1796 / 0.1957 | 0/3 |
| 100 | 0.2715 ± 0.0405 | 0.2984 / 0.2912 / 0.2249 | 0/3 |
| 200 | 0.3995 ± 0.0359 | 0.4237 / 0.3583 / 0.4165 | 2/3 |
| 536 | 0.5493 ± 0.0171 | 0.5516 / 0.5312 / 0.5651 | 3/3 |

D is 0.3649 ± 0.0016. A's mean crosses it at **N ≈ 173**, 32% of the 536-image split.

**Quote the bracket, not the point.** The sweep measures at neither end of the interpolation.

- **The flat line's spread barely moves the crossover.** D ± 1 sd, read through A's mean curve,
  gives [172, 174].
- **A's own spread is what sets the width.** D read through A ± 1 sd gives **[143, 203]**.
- **This is the reverse of the custom set** (F88). There, the flat line's translator-seed spread
  set the width.

All-class mAP50 agrees: D 0.5104 ± 0.0126 is crossed at N ≈ 173. D ± 1 sd gives [160, 186], and
A ± 1 sd gives [153, 192].

**Against the custom set** (record 011, F81): there the loop was worth ≈214 of 600 images (36%),
all-class. That comparison crosses datasets and detectors: `yolo11n` at 50 epochs there,
`yolo11s` at 100 epochs here. It is context, not a replication.

### F179 — Annotation-free translation without the loop (C) is worth ≈48 labelled thermal images on the passive classes, interval [43, 59]

- **2-class.** C is 0.1888 ± 0.0009 (n = 3). A's mean crosses it at **N ≈ 48.6**, between the
  N = 25 mean (0.0916) and the N = 50 mean (0.1946). Read through A ± 1 sd: **[43, 59]**. At N = 50,
  2/3 A seeds beat every C seed; at N = 100, 3/3 do.
- **All-class.** C is 0.3592 ± 0.0099, crossed at N ≈ 47.3. Read through A ± 1 sd: [44, 52].
- **The loop's coupling term is what turns ≈48 into ≈173.** Without the loop, translation's
  label value is 9% of the split; with it, 32%.
- **Against the custom set** (F81): annotation-free translation was worth ≈150 of 600 there
  (25%), all-class. The same cross-dataset caveat as F178 applies.

### F180 — Every draw held both passive classes, down to N = 25; the leanest color_cone draw is the seed that scored near zero on it

Per-class training box counts (`n_train_*`), car_stop / color_cone, seeds 0 / 1 / 2:

| N | car_stop boxes | color_cone boxes | AP50 car_stop | AP50 color_cone |
| --- | --- | --- | --- | --- |
| 25 | 35 / 19 / 15 | 8 / 5 / 10 | 0.1236 / 0.0851 / 0.1191 | 0.1182 / **0.0007** / 0.1031 |
| 50 | 60 / 50 / 40 | 27 / 18 / 35 | 0.2382 / 0.2495 / 0.1930 | 0.1790 / 0.1096 / 0.1984 |
| 100 | 116 / 92 / 67 | 50 / 57 / 57 | 0.3581 / 0.3323 / 0.2272 | 0.2387 / 0.2502 / 0.2226 |
| 200 | 186 / 151 / 149 | 87 / 102 / 111 | 0.5306 / 0.4214 / 0.4691 | 0.3168 / 0.2952 / 0.3639 |
| 536 | 404 / 404 / 404 | 262 / 262 / 262 | 0.6615 / 0.6100 / 0.6710 | 0.4417 / 0.4524 / 0.4592 |

- **No draw missed a passive class.** The plan Q6 worry, a small draw of contiguous frames with
  no passive object, did not happen at any budget swept. Plain seeded random stays the right
  selector at N ≥ 25.
- **The near-zero cell is real data, not a scoring fault.** N = 25, s1 holds 5 color_cone boxes
  against 8 and 10 for the other seeds, and scores 0.0007 against 0.1182 and 0.1031. That one
  cell is most of N = 25's 2-class sd (0.0425).
- **One cell is an observation, not a dose-response claim.** At N ≥ 50 the counts and the APs
  do not rank together.

## Provenance caveats

- **No provenance block.**
  - `git_sha`, the exact command and the environment are `not recorded`. The hostname comes
    from the human, not from the log.
  - The checkout held MSRS-BUDGET-SWEEP-02's code, `89e1f90` or later: the CSV carries its
    `n_train_*` columns. That is a lower bound, not a SHA.
  - The configuration table is assembled from the trainer-args line and the CSV, not from a
    command line.
- **Cells 1–14 have no console log.** These are A0; C and D at s0–2; A at N = 25 and 50; and
  A, N = 100, s0. The invocation that ran them wrote its log under the filename a later resume
  reused, and that resume overwrote it. These cells are evidenced by the CSV row alone: no
  per-cell timing, no training curve, no ultralytics per-class table. Their training curves
  should still sit on the server under `runs\msrs-budget-sweep\detectors\`.
- **The stopped attempt.**
  - Its log is the `| Tee-Object` form, with every progress redraw on its own line and
    box-drawing characters garbled.
  - Its traceback stops after the first frame. Nothing in the log says why the run stopped;
    the human reports stopping it by hand.
  - It wrote no CSV row.
  - The resume retrained A, N=100, s1 from epoch 1 (resume log line 60). That row is a complete
    run, not a continuation.
- **Pass 2 was skipped, by the human's decision on 2026-10-18.** C and D stand at n = 3, not the
  plan's six (Q4). Every crossover above reads against three translator seeds. The six-seed
  values are not measured here; Next derives them from record 024's rows.
- **Interpolation spans budgets 2× apart.** ≈48 sits inside 25–50, and ≈173 inside 100–200.
  Neither N was measured.
- **Arm A trains and is scored on mirrored labels.** These are msrs-day's visible boxes copied
  onto thermal, with a median residual misalignment of 4.63 px (record 024's caveats), against
  car_stop boxes of roughly 4×7 px. T shares the error, so the sweep is consistent with itself.
  But "labelled thermal images" here means thermal images with transferred labels.
- **The log filenames.** `logs/e8-tidy.csv` misses the `logs/<date>-<task>-<what>.txt`
  convention. The resume log is dated by launch; it was copied to the Mac on 2026-10-18.

## Next

**Pass 2 was skipped as a pure reproduction. Its outcome, if anyone runs it, is predicted here.**

- **Why it adds nothing new.** F177 shows C and D re-score record 024's exports to six decimals.
  Seeds 3–5 would reproduce record 024's stage-3 rows for those seeds.
- **Predicted D at n = 6:** 2-class 0.3592 ± 0.0182, crossover N ≈ 169 (D ± 1 sd: [154, 183]).
- **Predicted C at n = 6:** 0.1920 ± 0.0102, crossover N ≈ 49 ([47, 55]).
- **All-class:** D 0.5087, crossed at ≈171; C 0.3655, crossed at ≈49.
- **A miss beyond rounding would mean something drifted** between campaign and sweep, and would
  be a finding in its own right.

**The width that matters is A's (F178), unlike the custom set's (F88).** If the crossover's
interval needs narrowing, the lever is more A seeds at N = 100 and 200, not more translator seeds.
Six seeds there would also allow an exact sign-flip test (floor p = .031), which plan Q4 declined
at three.
