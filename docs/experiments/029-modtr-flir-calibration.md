# 029 — ModTr FLIR calibration run: the trigger fires, so FLIR's budget becomes 339 epochs

**Date:** 2026-10-25 (verdict read 2026-10-26) · **Task:** MODTR-BASELINE-08 · **Machine:** VSIBMEOT05 (`cuda:1`) · **Wall clock:** 12 h 47 min 24 s (2026-10-25 13:17:04 → 2026-10-26 02:04:28 +03:00)
**git_sha:** `pre-spec-migration-184-gddeeafe` (`git describe --always --dirty`, so a clean tree at `ddeeafe`) · **W&B:** none (`runtime.wandb: false` in `experiments/modtr_flir.yaml:87`; no W&B line in the log) · **Log:** `logs/2026-10-25-modtr-baseline-08-flir-calibration-run.txt`, with `logs/2026-10-25-modtr-baseline-08-flir-inner-split.txt`, `logs/2026-10-25-modtr-baseline-08-flir-calibration-pilot.txt` and `logs/2026-10-25-modtr-baseline-08-flir-calibration-verdict.txt`

## Command

From each log's provenance block, as run, in order:

```
uv run python scripts/modtr_calibration.py split --data dataset/processed/flir/data.yaml --out D:\Atick\data\dataset\FLIR-aligned-inner --config experiments/modtr_flir.yaml
uv run t2o loop --config experiments/modtr_flir.yaml --data D:\Atick\data\dataset\FLIR-aligned-inner\data.yaml --max-train-images 480 --val-loss-images 120 --seed 0 --device cuda:1 --monitor-map --no-detector --epochs 2 --name modtr-flir-calibration-pilot
uv run t2o loop --config experiments/modtr_flir.yaml --data D:\Atick\data\dataset\FLIR-aligned-inner\data.yaml --max-train-images 480 --val-loss-images 120 --seed 0 --device cuda:1 --monitor-map --no-detector --name modtr-flir-calibration
uv run python scripts/modtr_calibration.py verdict --run runs/modtr-flir-calibration
```

The record is the third and fourth commands. The split is their input, and the 2-epoch pilot only
proved the run starts, so neither gets a record of its own.

## Environment

The calibration run's block, verbatim:

```
=== PROVENANCE ===
captured: 2026-10-25 13:17:04 +03:00
hostname: VSIBMEOT05
os: Microsoft Windows Server 2022 Standard 10.0.20348 (build 20348)
cpu: AMD EPYC 7763 64-Core Processor
cpu_logical_cores: 64
ram_total: 128.0 GB
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16; NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16
cuda_driver: 13.0
git_sha: pre-spec-migration-184-gddeeafe
working_dir: D:\Atick\GitHub\Thermal-To-Optical-Translation
command: uv run t2o loop --config experiments/modtr_flir.yaml --data D:\Atick\data\dataset\FLIR-aligned-inner\data.yaml --max-train-images 480 --val-loss-images 120 --seed 0 --device cuda:1 --monitor-map --no-detector --name modtr-flir-calibration
python: 3.12.7
torch: 2.12.1+cu130
cuda_torch_build: 13.0
cudnn: 92000
cuda_available: True
gpu0_torch: NVIDIA A100-SXM4-40GB, 39.7 GiB, sm 8.0
=== END PROVENANCE ===
```

The split, pilot and verdict blocks are the same apart from `captured:` (12:51:25 and 12:55:53
on 2026-10-25; 11:29:43 on 2026-10-26) and their command lines. All four carry the same
`git_sha`. Unlike record 028's, the Python probe ran: the `python`, `torch`, `cudnn` lines are
present and no `error:` line is. The ultralytics banner gives `Ultralytics 8.4.117
Python-3.12.7 torch-2.12.1+cu130 CUDA:1`.

## Configuration

modtr-baseline ADR-04 and ADR-05, as `experiments/modtr_flir.yaml` sets them at `ddeeafe`.

| Axis | Value | Where it is evidenced |
| --- | --- | --- |
| Translator | ModTr (ResNet34 U-Net), detection-only objective | modtr-baseline ADR-01; `modtr_flir.yaml:43-47, :71-73` |
| Budget | 1 stage × 400 epochs, `task_weight` 1.0, `grad_scale` 1.0, last checkpoint | `stage 0 done: task_weight=1.0 epochs=400`; `modtr_flir.yaml:32, :50-51` |
| Optimiser | AdamW, lr 1e-4, batch 8, per-step cosine (ADR-04) | `modtr_flir.yaml:33-34` |
| Seed | 0 | the command |
| Inner split | 480 / 120 of the campaign's 600-image FLIR subset (cap 600, subset seed 0), fraction 0.20, split seed 0; validation never read | the split log's `inner split` line (ADR-05) |
| Training data | the inner split's 480 train pairs; `--max-train-images 480` overrides the config's 600, which the inner split cannot fill | `480 paired samples, 480 annotated`; the command |
| Val loss | the inner split's 120 val pairs; `--val-loss-images 120` overrides the config's 153 | the command |
| Monitor | the inner split's 120 val pairs, translated and scored every epoch | `exported 120 translated images`; every `all 120 949` row |
| In-loop detector | `runs/inloop-flir-yolo11n/weights/best.pt` (`yolo11n`, trained on FLIR visible, record 013) | `modtr_flir.yaml:58`; `eval best.pt` on every epoch |
| Detector stage | off (`--no-detector`) | the command |
| Trigger | 10-epoch trailing mean of inner mAP@50 in points; fires when its final value is > 2.0 below its maximum; budget = the best window's last epoch + 1 | ADR-04; `src/t2o/analysis/epoch_budget.py:61-78` |
| Loss rule | mean `loss_det` over the last 5 epochs below the mean over the first 5 | ADR-03 |
| Device | `cuda:1`; the runbook ran the custom-set campaign on `cuda:0` in parallel | the command; `modtr-server-runbook.md:7-8` |
| `config_hash` | not in the log | — |

## Results

The verdict log from the end of its provenance block to its finish stamp, verbatim:

```
2026-10-26 11:30:08,295 INFO    __main__: budget: 339 epochs (trigger fired) -- trailing mean peaks at 65.28 points ending epoch 338, ends at 62.71
2026-10-26 11:30:08,296 INFO    __main__: loss: PASS -- loss_det mean 5.6907 over the first 5 epochs, 1.5023 over the last 5
exit code: 0
=== FINISHED 2026-10-26 11:30:10 +03:00 ===
```

The split log's result line, verbatim:

```
2026-10-25 12:52:56,301 INFO    t2o.data.inner_split: inner split D:\Atick\data\dataset\FLIR-aligned-inner\data.yaml: 480 train / 120 val of D:\Atick\GitHub\Thermal-To-Optical-Translation\dataset\processed\flir\data.yaml's 600 training images (fraction 0.20, seed 0; cap 600, subset seed 0)
```

The run log is 9,078 lines: one block per epoch, 400 times. Its opening through epoch 0 and its
last two epochs follow. ANSI colour escapes, blank lines, and the per-batch `Scanning ...` and
`Class Images Instances Box(P ...` progress lines are stripped; everything else is as the log
holds it.

```
2026-10-25 13:17:39,681 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\data\dataset\FLIR-aligned-inner
2026-10-25 13:17:39,726 INFO    t2o.data.dataset: D:\Atick\data\dataset\FLIR-aligned-inner\train\visible\images: 480 paired samples, 480 annotated
2026-10-25 13:17:40,315 INFO    t2o.data.manifest: data.yaml: 4 classes ['bicycle', 'car', 'dog', 'person'], root D:\Atick\data\dataset\FLIR-aligned-inner
2026-10-25 13:17:41,313 INFO    t2o.data.dataset: D:\Atick\data\dataset\FLIR-aligned-inner\train\visible\images: 480 paired samples, 480 annotated
2026-10-25 13:17:41,382 INFO    t2o.data.dataset: D:\Atick\data\dataset\FLIR-aligned-inner\val\visible\images: 120 paired samples, 120 annotated
2026-10-25 13:18:41,659 INFO    t2o.engine.export: exported 120 translated images to runs\modtr-flir-calibration\stage0\monitor\val (normalize=clamp)
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11n summary (fused): 101 layers, 2,582,932 parameters, 0 gradients, 6.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 39.16.4 MB/s, size: 631.3 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: New cache created: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-calibration\stage0\monitor\val\labels.cache
                   all        120        949      0.273     0.0862     0.0852     0.0327
Speed: 1.2ms preprocess, 2.8ms inference, 0.0ms loss, 1.3ms postprocess per image
2026-10-25 13:19:35,263 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.2728  R 0.0862  mAP50 0.0852  mAP50-95 0.0327
2026-10-25 13:19:35,274 INFO    t2o.engine.trainer: epoch 0 took 113.9 s
[... epoch 0's memory line, epochs 1–396, and epoch 397 up to its memory line: elided here, in the log ...]
2026-10-26 02:00:28,464 INFO    t2o.engine.trainer: epoch 397 peak memory: 5.70 GiB allocated, 6.24 GiB reserved of 39.70 GiB
2026-10-26 02:01:25,785 INFO    t2o.engine.export: exported 120 translated images to runs\modtr-flir-calibration\stage0\monitor\val (normalize=clamp)
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11n summary (fused): 101 layers, 2,582,932 parameters, 0 gradients, 6.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 29.95.6 MB/s, size: 508.2 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: New cache created: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-calibration\stage0\monitor\val\labels.cache
                   all        120        949      0.779      0.535      0.635      0.329
Speed: 1.5ms preprocess, 1.5ms inference, 0.0ms loss, 1.3ms postprocess per image
2026-10-26 02:02:24,598 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7792  R 0.5352  mAP50 0.6348  mAP50-95 0.3290
2026-10-26 02:02:24,601 INFO    t2o.engine.trainer: epoch 398 took 115.4 s
2026-10-26 02:02:24,601 INFO    t2o.engine.trainer: epoch 398 peak memory: 5.72 GiB allocated, 6.25 GiB reserved of 39.70 GiB
2026-10-26 02:03:24,565 INFO    t2o.engine.export: exported 120 translated images to runs\modtr-flir-calibration\stage0\monitor\val (normalize=clamp)
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:1 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11n summary (fused): 101 layers, 2,582,932 parameters, 0 gradients, 6.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 46.83.1 MB/s, size: 567.6 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: New cache created: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-flir-calibration\stage0\monitor\val\labels.cache
                   all        120        949      0.762      0.546      0.634      0.327
Speed: 1.8ms preprocess, 1.6ms inference, 0.0ms loss, 1.8ms postprocess per image
2026-10-26 02:04:24,927 INFO    t2o.metrics.task: eval best.pt on data.yaml: P 0.7617  R 0.5456  mAP50 0.6335  mAP50-95 0.3266
2026-10-26 02:04:24,927 INFO    t2o.engine.trainer: epoch 399 took 119.6 s
2026-10-26 02:04:24,927 INFO    t2o.engine.trainer: epoch 399 peak memory: 5.72 GiB allocated, 6.26 GiB reserved of 39.70 GiB
2026-10-26 02:04:25,527 INFO    t2o.engine.loop: stage 0 done: task_weight=1.0 epochs=400
2026-10-26 02:04:25,534 INFO    t2o.cli: stage 0 (task_weight=1.0): 400 epochs
=== FINISHED 2026-10-26 02:04:28 +03:00 ===
```

Derived from all 400 `eval best.pt ... mAP50` lines in the run log. The table samples every 25th
epoch plus epoch 0, epoch 9 (the first full window) and epoch 338 (the peak window's last
epoch). The trailing mean at epoch e is the mean of epochs e−9 … e.

| epoch | mAP@50 (points) | 10-epoch trailing mean |
| --- | --- | --- |
| 0 | 8.52 | — |
| 9 | 56.84 | 43.79 |
| 24 | 56.72 | 57.92 |
| 49 | 61.99 | 60.07 |
| 74 | 60.97 | 62.37 |
| 99 | 59.58 | 63.07 |
| 124 | 63.40 | 62.66 |
| 149 | 59.46 | 60.34 |
| 174 | 60.67 | 60.30 |
| 199 | 68.56 | 61.65 |
| 224 | 62.14 | 60.15 |
| 249 | 63.66 | 64.21 |
| 274 | 62.60 | 62.56 |
| 299 | 63.03 | 63.07 |
| 324 | 62.19 | 63.84 |
| 338 | 67.43 | 65.28 |
| 349 | 66.47 | 64.58 |
| 374 | 63.45 | 63.15 |
| 399 | 63.35 | 62.71 |

From the same lines, and the same computation over record 028's custom-set run log
(`logs/2026-10-23-modtr-baseline-07-callibration-run.txt`) for comparison:

| Statistic | FLIR (this run) | Custom set (record 028) |
| --- | --- | --- |
| Trailing mean first within 2 points of its peak | window ending epoch 56 | window ending epoch 64 |
| Trailing mean from that window on: median / sd / range | 62.90 / 1.48 / 59.54–65.28 | 93.72 / 0.58 / 92.15–94.52 |
| Raw per-epoch sd over the same epochs | 2.22 | 0.69 |
| Largest drop of the trailing mean below its running maximum, up to the peak | 4.45 (63.98 → 59.54, window ending 184) | 0.73 |
| End-points among epochs 100–399 at which the trigger would have fired | 126 of 300 (first at 129) | 0 of 300 |
| Trailing mAP@50-95: peak / final / drop | 32.99 (ending 338) / 31.52 / 1.47 | 70.30 (ending 358) / 69.78 / 0.52 |
| Epoch `took`: median | 115.3 s (range 84.5–129.8) | 59.2 s |
| `exported` → `eval` line (the monitor's val pass): median | 56.7 s (range 38.7–71.4), 49% of an epoch | 9.7 s, 16% of an epoch |
| `took` minus that interval: median | 58.3 s (range 44.8–70.6) | 49.4 s |
| Peak memory | 5.73 GiB allocated, 6.27 GiB reserved of 39.70 | 5.36 / 5.90 |

The single best epoch is 252 at 69.56. 319 of the 400 epochs score ≥ 60.00, 42 score ≥ 65.00.
Per-epoch `took` times sum to 12.70 h. After the peak window the trailing mean falls to its
lowest, 62.63, in the window ending epoch 396.

Re-render the verdict without re-running, on the server:
`uv run python scripts/modtr_calibration.py verdict --run runs/modtr-flir-calibration`. Per-epoch
`loss_det` and `val_map50` are in `runs/modtr-flir-calibration/stage0/monitor.jsonl` there.

## Findings

### F196 — ModTr's calibration trigger fires on FLIR: the trailing inner mAP@50 ends 2.57 points below its peak against ADR-04's 2.0, so the FLIR campaign's budget is 339 epochs

The verdict reads a 10-epoch trailing mean that peaks at 65.28 points in the window ending epoch
338 and ends at 62.71. That is a 2.57-point drop, 0.57 over the 2.0-point trigger ADR-04 fixed
before either run. Re-deriving the curve from the log's own four-decimal `mAP50` lines,
independently of `monitor.jsonl`, gives the same peak window and 65.279 / 62.714. By ADR-04's
rule the FLIR campaign trains for 339 epochs, the peak window's last epoch plus one. The custom
set's run did not fire (0.42 points, F193), so ModTr now carries two budgets: 400 epochs on the
custom set, 339 on FLIR. ADR-05 foresaw this and requires it to be disclosed per dataset.

### F197 — The FLIR firing sits inside the curve's own noise, not on a clear decline: the trailing mean swung 4.45 points before its peak, and the trigger would have fired at 126 of 300 earlier end-points

The 2.57-point drop is not large against how much FLIR's trailing mean moves on its plateau.
From the window ending epoch 56, when it first comes within 2 points of its peak, the trailing
mean has an sd of 1.48 and spans 59.54–65.28. Before reaching its peak it had already fallen
4.45 points below a running maximum (63.98 → 59.54, window ending 184) and recovered. Had the
run stopped at any epoch from 100 to 399, the same rule would have fired at 126 of those 300
end-points, the first at epoch 129. The final 62.71 sits 0.19 below the plateau's median of
62.90, so the run ended at an ordinary level, not a low one. The comparison is record 028's
custom-set curve under the same computation: plateau sd 0.58, largest pre-peak drawdown 0.73,
and the trigger would have fired at 0 of 300 end-points. FLIR's curve is about 2.5× noisier
(1.48 vs 0.58 on the trailing mean, 2.22 vs 0.69 per epoch), and the 2.0 threshold sits inside
that noise on FLIR but well outside it on the custom set. The trailing mAP@50-95 peaks in the
same window (ending 338) and falls 1.47 points, but it scores the same predictions, so it is not
independent evidence of over-training. Read plainly: the rule fired as written, and 339 is
where a noisy curve happened to peak, not a measured onset of over-fitting. ADR-04 says changing
the rule after its result exists needs a superseding ADR.

### F198 — ADR-03's smoke rule passes on FLIR: mean `loss_det` falls from 5.6907 over the first 5 epochs to 1.5023 over the last 5, −74%

The pass rule is "last-5 mean below first-5 mean", and the run clears it by a factor of 3.8. The
custom set's run fell −80%, by a factor of 5.1 (F194). The monitor agrees from the other side:
inner mAP@50 rises 8.52 → 63.35 from epoch 0 to 399. So the detection-only objective trains
ModTr against FLIR's in-loop `yolo11n` too, at `grad_scale: 1.0`.

### F199 — A FLIR calibration epoch costs 1.95× the custom set's, and the monitor's val pass is the whole difference: 56.7 s of a 115.3 s epoch against 9.7 s of 59.2 s

The run took 12 h 47 min from provenance to finish stamp, and its 400 epoch times sum to 12.70
h. The median epoch is 115.3 s, against 59.2 s for the custom set's calibration at the same
480 / 120 split sizes (F195). The interval from each epoch's `exported` line to its `eval` line,
which holds the in-loop detector's val pass on the 120 translated images, has a median of 56.7
s on FLIR, 49% of the epoch, against 9.7 s (16%) on the custom set. What is left of an epoch,
training plus the export, is 58.3 s on FLIR against 49.4 s. The ultralytics progress bars show
the val batches themselves finishing in about 1–1.3 s, so most of the 56.7 s is spent elsewhere
in that interval; the log does not say where. Peak memory is 6.27 GiB reserved, against 5.90 on
the custom set. The campaign does not run the monitor, so its per-epoch cost should sit near
the 58.3 s, not the 115.3 s.

## Provenance caveats

- **The in-loop detector has most likely seen the monitor's images.** FLIR's in-loop `yolo11n`
  was trained on FLIR visible (record 013), and the inner split is drawn from FLIR's training
  images. ADR-05 names this as likely, not confirmed. If so, the curve's level is inflated, as
  in record 028. The trigger reads a drop from the curve's own peak, so F196 is less exposed
  than the level.
- **One seed.** Every number is seed 0, as ADR-04 and ADR-05 fixed. F197's noise figures are
  within-run, across epochs. How the curve and its peak epoch vary across seeds is not measured.
- **F197's "126 of 300" is a post-hoc reading.** It replays the trigger at end-points the run
  was never meant to stop at. It describes the curve's noise; it is not a second test of the
  rule.
- **The custom-set campaign shared the server.** The runbook ran it on `cuda:0` in parallel with
  this run on `cuda:1` (`modtr-server-runbook.md:7-8`). This log does not say whether it was
  running at any given epoch. CPU, host RAM and disk contention would show in F199's times, not
  in the mAP curve. The pilot's two epochs ran at 115.2 and 115.3 s, the same as the run.
- **The monitor's cost is not separable from the export.** F199's 56.7 s is from log
  timestamps, not a timer. The export's own time is not logged, so the 58.3 s still holds the
  export of 120 images.
- **The pilot is not a reproducibility check.** Its epochs 0–1 read 9.01 / 17.71 against the
  run's 8.52 / 24.35 at the same seed. But `--epochs 2` shortens the cosine schedule's `T_max`,
  as in record 028.
- **ADR-04's schedule cost now applies.** The calibration ran a 400-epoch cosine. A 339-epoch
  campaign has a different `T_max`, so this run speaks to a schedule the campaign does not
  exactly repeat. ADR-04 lists this cost.
- **The run's `config_hash` is not in the log.** It is in
  `runs/modtr-flir-calibration/config.yaml` on the server.
- **Slow image access.** ultralytics warns "Slow image access detected" on every monitor val
  pass. It may account for part of F199's 56.7 s; the log does not show how much.

## Next

ADR-05's condition is met once this record lands: the result is recorded, and by ADR-04's rule
FLIR's ModTr budget is 339 epochs. `experiments/modtr_flir.yaml:32` still says 400, so it needs
a task to change it before the FLIR campaign launches. F197 bears on whether the rule should
stand on a curve this noisy. Keeping it as written is what ADR-04 pre-registered; changing it
needs a superseding ADR before the campaign, not after. A falsifiable prediction from F199: one
FLIR campaign seed at 339 epochs on 600 images trains in at most about 6.9 h on one card
(58.3 s × 600 / 480 × 339), because the campaign does not pay for the monitor.
