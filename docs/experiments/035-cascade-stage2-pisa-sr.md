# 035 — Cascade gate, stage 2: zero-shot PiSA-SR fails gate ADR-04, 2 of 14 tests pass; it invents nothing where fuses and switches were erased, but every fuse, switch and pooled rate gets worse

**Date:** 2026-11-08 · **Task:** CASCADE-GATE-07 · **Machine:** VSIBMEOT05, 1× A100 (`cuda:1`) · **Wall clock:** 2 h 11 min end to end (09:22:29 → 11:33:15 +03:00), 34 min of it idle between the completion step and the readout. Per step: weights 53 s, smoke 48 min 38 s (46 min of it a cold-cache fetch of SD-2.1-base), refine 28 min 14 s, faithfulness 5 min 56 s, completion 5 min 19 s, readout 30 s
**git_sha:** `pre-spec-migration-229-g223a8b9` (a clean tree at `223a8b9`, no `-dirty` suffix; every step's `gate:` line confirms ADR-04 accepted at that HEAD) · **W&B:** none (no W&B line in any log) · **Log:** `logs/2026-11-08-cascade-gate-07-weights.txt`, `logs/2026-11-08-cascade-gate-07-smoke.txt`, `logs/2026-11-08-cascade-gate-07-refine.txt`, `logs/2026-11-08-cascade-gate-07-faithfulness.txt`, `logs/2026-11-08-cascade-gate-07-completion.txt`, `logs/2026-11-08-cascade-gate-07-readout.txt`

This run applies the cascade gate, ADR-04 in `docs/features/cascade-gate/decisions.md`. Stage 2 is
PiSA-SR's default setting, zero-shot and frozen (ADR-01). It refines the same nine stage-3 loop
exports that record 034 scored: e3b (pix2pix) seeds 0–5 and e3t (turbo) seeds 0, 1, 3. Each one is
refined twice, once on real val and once on record 034's erased val, where only fuse and switch
boxes were inpainted out (ADR-03). The `rtdetr` judge then scores the refined frames. The readout
compares cascade − stage 1, paired by seed, on ADR-04's seven gated tests per cell. ADR-04 was
committed on 2026-11-04, before any stage-2 frame existed. The cascade enters the IEEE TIM paper
only if all fourteen tests pass.

## Command

From the provenance blocks, as run. Steps 3–5 each ran a loop, and their provenance `command:` line
is the runbook's label for that loop, with `<cell>`, `<s>` and `<input>` left in:

```
# 1. fetch the PiSA-SR LoRA checkpoint
uv run --group scripts gdown --folder https://drive.google.com/drive/folders/1oLetijWNd59xwJE5oU-eXylQBifxWdss -O runs/cascade-gate/weights

# 2. smoke (ADR-02): refine e3b seed 0's val export into a throwaway folder, then check sizes and change
uv run t2o refine --config experiments/cascade_pisa_sr.yaml --source runs/e3b-loop-s0/stage3/translated --checkpoint runs/cascade-gate/weights/pisa_sr.pkl --run-dir runs/cascade-gate --name smoke-e3b-s0 --device cuda:1

# 3. refine every run, on val and on erased val (18 refines)
uv run t2o refine --config experiments/cascade_pisa_sr.yaml --source runs/<cell>-loop-s<s>/stage3/<input> --checkpoint runs/cascade-gate/weights/pisa_sr.pkl --run-dir runs/<cell>-loop-s<s>/stage3 --name cascade-<input> --device cuda:1, for e3b s0-5 and e3t s0,1,3, input in translated, erased-translated

# 4. faithfulness of the refined val frames, written back
uv run t2o faithfulness --translated runs/<cell>-loop-s<s>/stage3/cascade-translated --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --weights runs/reference-rtdetr-l/weights/best.pt --write-back --judge rtdetr-cascade --device cuda:1, for e3b s0-5 and e3t s0,1,3

# 5. completion rate of the refined erased frames, written back
uv run t2o completion --translated runs/<cell>-loop-s<s>/stage3/cascade-erased-translated --data runs/cascade-gate/erased-custom-val-fuse-switch/data.yaml --weights runs/reference-rtdetr-l/weights/best.pt --write-back --judge rtdetr-cascade --device cuda:1, for e3b s0-5 and e3t s0,1,3

# 6. readout: ADR-04's fourteen tests
uv run python runs/cascade-gate/stage2-readout.py
```

The smoke check script and the readout script were both printed into their logs before they ran.
Both appear verbatim under `## Results`, so this record does not depend on the untracked runbook
(`RUNBOOK-cascade-gate-stage2.md`).

## Environment

The weights log's block. The other five blocks differ from it only in their `captured` and
`command` lines (checked by hashing each block with those two lines removed).

```
=== PROVENANCE ===
captured: 2026-11-08 09:22:29 +03:00
hostname: VSIBMEOT05
os: Microsoft Windows Server 2022 Standard 10.0.20348 (build 20348)
cpu: AMD EPYC 7763 64-Core Processor
cpu_logical_cores: 64
ram_total: 128.0 GB
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16; NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16
cuda_driver: 13.0
git_sha: pre-spec-migration-229-g223a8b9
working_dir: D:\Atick\GitHub\Thermal-To-Optical-Translation
command: uv run --group scripts gdown --folder https://drive.google.com/drive/folders/1oLetijWNd59xwJE5oU-eXylQBifxWdss -O runs/cascade-gate/weights
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
| Runs | the custom set's loop arm at stage 3: e3b (pix2pix) seeds 0–5, e3t (turbo) seeds 0, 1, 3, the same nine as record 034 | `--- runs/... ---` lines in each log |
| Stage-1 inputs | each run's existing `stage3/translated` (val) and record 034's `stage3/erased-translated` (erased val). Both were made before this run and are unchanged by it | the refine commands |
| Stage 2 | PiSA-SR's default setting (ADR-01): both LoRAs active at 1.0, one step at `t = 1`, empty prompt, AdaIN colour match to the stage-1 input, padding instead of resizing, VAE posterior mode, `dtype: float32` | `experiments/cascade_pisa_sr.yaml`; ADR-01 |
| SD-2.1 base | `Manojb/stable-diffusion-2-1-base` at revision `0094d483a120f3f33dafbd187ea4aa60d10de75c` | `t2o.translators.pisa_sr: loading ...` lines |
| LoRA checkpoint | `runs/cascade-gate/weights/pisa_sr.pkl`, 32,865,445 bytes, sha256 `13502f7bde5e4e585f84f24c3d904d9c848e068bf89f984046fe87da81a391f6`. It loads as 6 adapters, 1,032 tensors, on all 19 loads | the weights log; `merged 6 PiSA-SR adapters (1032 tensors)` |
| Refine config hash | one per refine, 18 distinct. The smoke and the main e3b seed 0 val refine share `e01670e12d6ad8fd` though their run dir and name differ | `t2o.cli: refine config hash ...` lines |
| Judge | `rtdetr`: record 026's custom-set RT-DETR-l checkpoint, `runs/reference-rtdetr-l/weights/best.pt`. Cascade scores are written under `judges.rtdetr-cascade`, beside stage 1's `judges.rtdetr` | the commands; `recorded judge rtdetr-cascade ...` lines |
| Scorer settings | `t2o faithfulness` / `t2o completion` defaults: IoU 0.5, conf 0.25, imgsz 640, batch 16, as record 034 | no flag overrides them in the commands |
| Erased tree | record 034's `runs/cascade-gate/erased-custom-val-fuse-switch`: Fuse and Switch only, Telea, 4 px, radius 3.0, 234 boxes on 153 frames | the completion command's `--data` |
| Readout statistics | `bootstrap_ci` (10,000 resamples, seed 0, 95% percentile) and the exact `sign_flip_p_value` from `t2o.analysis.aggregate`, on the seed-paired differences | the readout script; `src/t2o/analysis/aggregate.py:302-335` |
| Margins | ADR-04's: Fuse and Switch from record 034's stage-1 seed sd, pooled from out-of-family-judge ADR-01; completion at zero tolerance | the readout script's `MARGINS` |

## Results

### Step 1 — weights

Everything after the provenance block, verbatim, except gdown's 65 progress-bar lines:

```
gate: ADR-04 accepted at HEAD 223a8b9
warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
         If the cache and target directories are on different filesystems, hardlinking may not be supported.
         If this is intentional, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` to suppress this warning.
Installed 4 packages in 678ms
Retrieving folder contents
Retrieving folder contents completed
Building directory structure
Building directory structure completed
Downloading...
From (original): https://drive.google.com/uc?id=1PWFM5zUd6lNMOyKX64anvUXhKEOCrf8L
From (redirected): https://drive.google.com/uc?id=1PWFM5zUd6lNMOyKX64anvUXhKEOCrf8L&confirm=t&uuid=448a2073-9921-4e0a-a479-4a50b4bae266
To: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\weights\pisa_sr.pkl
Processing file 1PWFM5zUd6lNMOyKX64anvUXhKEOCrf8L pisa_sr.pkl

Download completed
D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\weights\pisa_sr.pkl  32,865,445 bytes
sha256: 13502f7bde5e4e585f84f24c3d904d9c848e068bf89f984046fe87da81a391f6  runs/cascade-gate/weights/pisa_sr.pkl
=== FINISHED 2026-11-08 09:23:22 +03:00 ===
```

### Step 2 — smoke

Everything after the provenance block, except the `httpx` request lines, the `Loading weights`
progress bar and the 153 per-frame `refined <frame>.png` lines. Those per-frame lines are identical
to the main run's e3b seed 0 val refine (F222).

```
gate: ADR-04 accepted at HEAD 223a8b9
2026-11-08 09:26:09,203 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-11-08 09:26:10,803 WARNING huggingface_hub.utils._http: Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\huggingface_hub\file_download.py:141: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\atick\.cache\huggingface\hub\models--Manojb--stable-diffusion-2-1-base. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)

D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\huggingface_hub\utils\_validators.py:205: UserWarning: The `local_dir_use_symlinks` argument is deprecated and ignored in `hf_hub_download`. Downloading to a local directory does not use symlinks anymore.
  warnings.warn(
D:\Atick\GitHub\Thermal-To-Optical-Translation\.venv\Lib\site-packages\peft\tuners\tuners_utils.py:305: UserWarning: Already found a `peft_config` attribute in the model. This will lead to having multiple adapters in the model. Make sure to know what you are doing!
  warnings.warn(
2026-11-08 10:12:20,933 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:12:24,261 INFO    t2o.cli: refine config hash e01670e12d6ad8fd
2026-11-08 10:13:15,270 INFO    t2o.engine.refine: refined 153 frames into runs\cascade-gate\smoke-e3b-s0; mean |change| 0.0335 over the split
2026-11-08 10:13:15,272 INFO    t2o.cli: wrote runs\cascade-gate\smoke-e3b-s0\data.yaml
"""ADR-02 smoke: every refined frame keeps its stage-1 size, and the mean change is not zero."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

stage1, refined = (Path(arg) / 'val' / 'images' for arg in sys.argv[1:3])
names = sorted(p.name for p in stage1.iterdir())
if names != sorted(p.name for p in refined.iterdir()):
    sys.exit('smoke FAILED: the refined frames do not match the stage-1 frames by name')
changes = []
for name in names:
    with Image.open(stage1 / name) as before, Image.open(refined / name) as after:
        if before.size != after.size:
            sys.exit(f'smoke FAILED: {name} is {after.size} refined, {before.size} in stage 1')
        diff = np.asarray(after.convert('RGB'), float) - np.asarray(before.convert('RGB'), float)
        changes.append(float(np.abs(diff).mean()) / 255.0)
mean = sum(changes) / len(changes)
print(f'smoke: {len(names)} frames, every size matches its stage-1 frame')
print(f'smoke: mean |change| {mean:.4f} (per frame min {min(changes):.4f}, max {max(changes):.4f})')
if mean == 0.0:
    sys.exit('smoke FAILED: mean change is zero, so stage 2 is an identity (ADR-02)')
print('smoke: PASS')
smoke: 153 frames, every size matches its stage-1 frame
smoke: mean |change| 0.0335 (per frame min 0.0188, max 0.0812)
smoke: PASS
=== FINISHED 2026-11-08 10:13:26 +03:00 ===
```

### Step 3 — refine

An excerpt: the run separators and the `t2o` lines, without the 2,754 per-frame
`refined <frame>.png` lines, the `httpx` lines and the Hugging Face warnings. It is rebuilt with
`grep -E '^(---|gate:|=== FINISHED)|t2o\.' logs/2026-11-08-cascade-gate-07-refine.txt | grep -vE 'refined [^ ]+\.png'`.

```
gate: ADR-04 accepted at HEAD 223a8b9
--- runs/e3b-loop-s0/stage3/translated ---
2026-11-08 10:17:01,685 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:17:17,652 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:17:20,718 INFO    t2o.cli: refine config hash e01670e12d6ad8fd
2026-11-08 10:18:04,280 INFO    t2o.engine.refine: refined 153 frames into runs\e3b-loop-s0\stage3\cascade-translated; mean |change| 0.0335 over the split
2026-11-08 10:18:04,282 INFO    t2o.cli: wrote runs\e3b-loop-s0\stage3\cascade-translated\data.yaml
--- runs/e3b-loop-s0/stage3/erased-translated ---
2026-11-08 10:18:30,708 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:18:48,706 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:18:52,475 INFO    t2o.cli: refine config hash 195364ff01b7d2ca
2026-11-08 10:19:46,779 INFO    t2o.engine.refine: refined 153 frames into runs\e3b-loop-s0\stage3\cascade-erased-translated; mean |change| 0.0328 over the split
2026-11-08 10:19:46,782 INFO    t2o.cli: wrote runs\e3b-loop-s0\stage3\cascade-erased-translated\data.yaml
--- runs/e3b-loop-s1/stage3/translated ---
2026-11-08 10:20:10,241 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:20:26,820 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:20:30,417 INFO    t2o.cli: refine config hash b4da38d183babc65
2026-11-08 10:21:17,531 INFO    t2o.engine.refine: refined 153 frames into runs\e3b-loop-s1\stage3\cascade-translated; mean |change| 0.0338 over the split
2026-11-08 10:21:17,532 INFO    t2o.cli: wrote runs\e3b-loop-s1\stage3\cascade-translated\data.yaml
--- runs/e3b-loop-s1/stage3/erased-translated ---
2026-11-08 10:21:45,072 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:22:02,337 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:22:06,145 INFO    t2o.cli: refine config hash 3c1e69cd25d981cd
2026-11-08 10:23:00,325 INFO    t2o.engine.refine: refined 153 frames into runs\e3b-loop-s1\stage3\cascade-erased-translated; mean |change| 0.0331 over the split
2026-11-08 10:23:00,328 INFO    t2o.cli: wrote runs\e3b-loop-s1\stage3\cascade-erased-translated\data.yaml
--- runs/e3b-loop-s2/stage3/translated ---
2026-11-08 10:23:25,467 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:23:41,851 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:23:45,257 INFO    t2o.cli: refine config hash dbbef719eb843fb3
2026-11-08 10:24:32,408 INFO    t2o.engine.refine: refined 153 frames into runs\e3b-loop-s2\stage3\cascade-translated; mean |change| 0.0350 over the split
2026-11-08 10:24:32,411 INFO    t2o.cli: wrote runs\e3b-loop-s2\stage3\cascade-translated\data.yaml
--- runs/e3b-loop-s2/stage3/erased-translated ---
2026-11-08 10:24:57,612 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:25:13,287 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:25:16,452 INFO    t2o.cli: refine config hash 19592025f8f21052
2026-11-08 10:26:03,461 INFO    t2o.engine.refine: refined 153 frames into runs\e3b-loop-s2\stage3\cascade-erased-translated; mean |change| 0.0342 over the split
2026-11-08 10:26:03,464 INFO    t2o.cli: wrote runs\e3b-loop-s2\stage3\cascade-erased-translated\data.yaml
--- runs/e3b-loop-s3/stage3/translated ---
2026-11-08 10:26:30,513 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:26:48,085 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:26:51,865 INFO    t2o.cli: refine config hash 8040a450896b9f84
2026-11-08 10:27:44,913 INFO    t2o.engine.refine: refined 153 frames into runs\e3b-loop-s3\stage3\cascade-translated; mean |change| 0.0329 over the split
2026-11-08 10:27:44,916 INFO    t2o.cli: wrote runs\e3b-loop-s3\stage3\cascade-translated\data.yaml
--- runs/e3b-loop-s3/stage3/erased-translated ---
2026-11-08 10:28:09,806 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:28:25,957 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:28:29,503 INFO    t2o.cli: refine config hash 028b2efd4eecf861
2026-11-08 10:29:16,434 INFO    t2o.engine.refine: refined 153 frames into runs\e3b-loop-s3\stage3\cascade-erased-translated; mean |change| 0.0321 over the split
2026-11-08 10:29:16,436 INFO    t2o.cli: wrote runs\e3b-loop-s3\stage3\cascade-erased-translated\data.yaml
--- runs/e3b-loop-s4/stage3/translated ---
2026-11-08 10:29:42,093 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:29:57,870 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:30:01,126 INFO    t2o.cli: refine config hash 104d86a1cd832ee0
2026-11-08 10:30:48,280 INFO    t2o.engine.refine: refined 153 frames into runs\e3b-loop-s4\stage3\cascade-translated; mean |change| 0.0335 over the split
2026-11-08 10:30:48,281 INFO    t2o.cli: wrote runs\e3b-loop-s4\stage3\cascade-translated\data.yaml
--- runs/e3b-loop-s4/stage3/erased-translated ---
2026-11-08 10:31:14,351 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:31:29,852 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:31:32,964 INFO    t2o.cli: refine config hash 37e4944b211cbf5f
2026-11-08 10:32:19,869 INFO    t2o.engine.refine: refined 153 frames into runs\e3b-loop-s4\stage3\cascade-erased-translated; mean |change| 0.0327 over the split
2026-11-08 10:32:19,875 INFO    t2o.cli: wrote runs\e3b-loop-s4\stage3\cascade-erased-translated\data.yaml
--- runs/e3b-loop-s5/stage3/translated ---
2026-11-08 10:32:45,442 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:33:00,491 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:33:03,878 INFO    t2o.cli: refine config hash 144cabb253ff460c
2026-11-08 10:33:50,108 INFO    t2o.engine.refine: refined 153 frames into runs\e3b-loop-s5\stage3\cascade-translated; mean |change| 0.0329 over the split
2026-11-08 10:33:50,111 INFO    t2o.cli: wrote runs\e3b-loop-s5\stage3\cascade-translated\data.yaml
--- runs/e3b-loop-s5/stage3/erased-translated ---
2026-11-08 10:34:15,263 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:34:31,749 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:34:35,235 INFO    t2o.cli: refine config hash c1a413b0a1f92cff
2026-11-08 10:35:27,368 INFO    t2o.engine.refine: refined 153 frames into runs\e3b-loop-s5\stage3\cascade-erased-translated; mean |change| 0.0322 over the split
2026-11-08 10:35:27,373 INFO    t2o.cli: wrote runs\e3b-loop-s5\stage3\cascade-erased-translated\data.yaml
--- runs/e3t-loop-s0/stage3/translated ---
2026-11-08 10:35:50,539 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:36:07,219 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:36:10,904 INFO    t2o.cli: refine config hash dfff697f427e4c29
2026-11-08 10:37:02,729 INFO    t2o.engine.refine: refined 153 frames into runs\e3t-loop-s0\stage3\cascade-translated; mean |change| 0.0351 over the split
2026-11-08 10:37:02,733 INFO    t2o.cli: wrote runs\e3t-loop-s0\stage3\cascade-translated\data.yaml
--- runs/e3t-loop-s0/stage3/erased-translated ---
2026-11-08 10:37:26,701 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:37:43,096 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:37:46,363 INFO    t2o.cli: refine config hash 18f347749f311b94
2026-11-08 10:38:34,713 INFO    t2o.engine.refine: refined 153 frames into runs\e3t-loop-s0\stage3\cascade-erased-translated; mean |change| 0.0345 over the split
2026-11-08 10:38:34,720 INFO    t2o.cli: wrote runs\e3t-loop-s0\stage3\cascade-erased-translated\data.yaml
--- runs/e3t-loop-s1/stage3/translated ---
2026-11-08 10:38:58,266 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:39:14,998 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:39:18,097 INFO    t2o.cli: refine config hash 374f03b60fe661ea
2026-11-08 10:40:08,840 INFO    t2o.engine.refine: refined 153 frames into runs\e3t-loop-s1\stage3\cascade-translated; mean |change| 0.0341 over the split
2026-11-08 10:40:08,844 INFO    t2o.cli: wrote runs\e3t-loop-s1\stage3\cascade-translated\data.yaml
--- runs/e3t-loop-s1/stage3/erased-translated ---
2026-11-08 10:40:31,556 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:40:47,747 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:40:51,123 INFO    t2o.cli: refine config hash 13c6da1dc492cd8c
2026-11-08 10:41:38,994 INFO    t2o.engine.refine: refined 153 frames into runs\e3t-loop-s1\stage3\cascade-erased-translated; mean |change| 0.0334 over the split
2026-11-08 10:41:38,996 INFO    t2o.cli: wrote runs\e3t-loop-s1\stage3\cascade-erased-translated\data.yaml
--- runs/e3t-loop-s3/stage3/translated ---
2026-11-08 10:42:04,929 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:42:20,422 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:42:23,691 INFO    t2o.cli: refine config hash b004156212228f09
2026-11-08 10:43:11,606 INFO    t2o.engine.refine: refined 153 frames into runs\e3t-loop-s3\stage3\cascade-translated; mean |change| 0.0357 over the split
2026-11-08 10:43:11,608 INFO    t2o.cli: wrote runs\e3t-loop-s3\stage3\cascade-translated\data.yaml
--- runs/e3t-loop-s3/stage3/erased-translated ---
2026-11-08 10:43:37,721 INFO    t2o.translators.pisa_sr: loading Manojb/stable-diffusion-2-1-base at 0094d483a120f3f33dafbd187ea4aa60d10de75c, LoRA checkpoint runs\cascade-gate\weights\pisa_sr.pkl
2026-11-08 10:43:53,259 INFO    t2o.translators.pisa_sr: merged 6 PiSA-SR adapters (1032 tensors)
2026-11-08 10:43:56,545 INFO    t2o.cli: refine config hash ca15d88331dc7415
2026-11-08 10:44:43,993 INFO    t2o.engine.refine: refined 153 frames into runs\e3t-loop-s3\stage3\cascade-erased-translated; mean |change| 0.0350 over the split
2026-11-08 10:44:43,995 INFO    t2o.cli: wrote runs\e3t-loop-s3\stage3\cascade-erased-translated\data.yaml
=== FINISHED 2026-11-08 10:44:46 +03:00 ===
```

### Step 4 — faithfulness on refined val

Everything after the provenance block, verbatim:

```
gate: ADR-04 accepted at HEAD 223a8b9
--- runs/e3b-loop-s0 ---
2026-11-08 10:46:10,194 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-08 10:46:32,560 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.2566  missed-object 0.1986  consistency 0.7691
2026-11-08 10:46:32,562 INFO    t2o.cli:   false-object rate       0.2566  (LOWER better)
2026-11-08 10:46:32,563 INFO    t2o.cli:   missed-object rate      0.1986  (LOWER better)
2026-11-08 10:46:32,563 INFO    t2o.cli:   detection-consistency   0.7691  (higher better)
2026-11-08 10:46:32,563 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-08 10:46:32,564 INFO    t2o.cli:     Fuse           false 0.2025  missed 0.2500
2026-11-08 10:46:32,564 INFO    t2o.cli:     Pole           false 0.1696  missed 0.0719
2026-11-08 10:46:32,564 INFO    t2o.cli:     Switch         false 0.4507  missed 0.4091
2026-11-08 10:46:32,564 INFO    t2o.cli:     Transformer    false 0.4286  missed 0.1111
2026-11-08 10:46:32,621 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\metrics.json stage 3
--- runs/e3b-loop-s1 ---
2026-11-08 10:46:53,629 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-08 10:47:09,846 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.3154  missed-object 0.2199  consistency 0.7582
2026-11-08 10:47:09,848 INFO    t2o.cli:   false-object rate       0.3154  (LOWER better)
2026-11-08 10:47:09,848 INFO    t2o.cli:   missed-object rate      0.2199  (LOWER better)
2026-11-08 10:47:09,848 INFO    t2o.cli:   detection-consistency   0.7582  (higher better)
2026-11-08 10:47:09,848 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-08 10:47:09,848 INFO    t2o.cli:     Fuse           false 0.2313  missed 0.2679
2026-11-08 10:47:09,848 INFO    t2o.cli:     Pole           false 0.2278  missed 0.0915
2026-11-08 10:47:09,849 INFO    t2o.cli:     Switch         false 0.5870  missed 0.4242
2026-11-08 10:47:09,849 INFO    t2o.cli:     Transformer    false 0.4000  missed 0.1667
2026-11-08 10:47:09,904 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s1\metrics.json stage 3
--- runs/e3b-loop-s2 ---
2026-11-08 10:47:31,247 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-08 10:47:50,338 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.2528  missed-object 0.2033  consistency 0.7734
2026-11-08 10:47:50,340 INFO    t2o.cli:   false-object rate       0.2528  (LOWER better)
2026-11-08 10:47:50,341 INFO    t2o.cli:   missed-object rate      0.2033  (LOWER better)
2026-11-08 10:47:50,341 INFO    t2o.cli:   detection-consistency   0.7734  (higher better)
2026-11-08 10:47:50,341 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-08 10:47:50,341 INFO    t2o.cli:     Fuse           false 0.1389  missed 0.2619
2026-11-08 10:47:50,341 INFO    t2o.cli:     Pole           false 0.1860  missed 0.0850
2026-11-08 10:47:50,341 INFO    t2o.cli:     Switch         false 0.4545  missed 0.3636
2026-11-08 10:47:50,341 INFO    t2o.cli:     Transformer    false 0.4655  missed 0.1389
2026-11-08 10:47:50,390 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s2\metrics.json stage 3
--- runs/e3b-loop-s3 ---
2026-11-08 10:48:10,258 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-08 10:48:26,526 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.2727  missed-object 0.2246  consistency 0.7364
2026-11-08 10:48:26,529 INFO    t2o.cli:   false-object rate       0.2727  (LOWER better)
2026-11-08 10:48:26,529 INFO    t2o.cli:   missed-object rate      0.2246  (LOWER better)
2026-11-08 10:48:26,529 INFO    t2o.cli:   detection-consistency   0.7364  (higher better)
2026-11-08 10:48:26,530 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-08 10:48:26,530 INFO    t2o.cli:     Fuse           false 0.2434  missed 0.3155
2026-11-08 10:48:26,530 INFO    t2o.cli:     Pole           false 0.2011  missed 0.0915
2026-11-08 10:48:26,530 INFO    t2o.cli:     Switch         false 0.4545  missed 0.3636
2026-11-08 10:48:26,530 INFO    t2o.cli:     Transformer    false 0.3333  missed 0.1111
2026-11-08 10:48:26,596 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s3\metrics.json stage 3
--- runs/e3b-loop-s4 ---
2026-11-08 10:48:48,170 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-08 10:49:04,373 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.2625  missed-object 0.1962  consistency 0.7734
2026-11-08 10:49:04,375 INFO    t2o.cli:   false-object rate       0.2625  (LOWER better)
2026-11-08 10:49:04,375 INFO    t2o.cli:   missed-object rate      0.1962  (LOWER better)
2026-11-08 10:49:04,375 INFO    t2o.cli:   detection-consistency   0.7734  (higher better)
2026-11-08 10:49:04,375 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-08 10:49:04,375 INFO    t2o.cli:     Fuse           false 0.1575  missed 0.2679
2026-11-08 10:49:04,375 INFO    t2o.cli:     Pole           false 0.1864  missed 0.0588
2026-11-08 10:49:04,376 INFO    t2o.cli:     Switch         false 0.5287  missed 0.3788
2026-11-08 10:49:04,376 INFO    t2o.cli:     Transformer    false 0.3725  missed 0.1111
2026-11-08 10:49:04,442 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s4\metrics.json stage 3
--- runs/e3b-loop-s5 ---
2026-11-08 10:49:25,913 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-08 10:49:41,815 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.2431  missed-object 0.2199  consistency 0.7364
2026-11-08 10:49:41,817 INFO    t2o.cli:   false-object rate       0.2431  (LOWER better)
2026-11-08 10:49:41,817 INFO    t2o.cli:   missed-object rate      0.2199  (LOWER better)
2026-11-08 10:49:41,817 INFO    t2o.cli:   detection-consistency   0.7364  (higher better)
2026-11-08 10:49:41,817 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-08 10:49:41,817 INFO    t2o.cli:     Fuse           false 0.1399  missed 0.2679
2026-11-08 10:49:41,817 INFO    t2o.cli:     Pole           false 0.2069  missed 0.0980
2026-11-08 10:49:41,817 INFO    t2o.cli:     Switch         false 0.5000  missed 0.4394
2026-11-08 10:49:41,817 INFO    t2o.cli:     Transformer    false 0.2889  missed 0.1111
2026-11-08 10:49:41,878 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s5\metrics.json stage 3
--- runs/e3t-loop-s0 ---
2026-11-08 10:50:03,515 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-08 10:50:21,332 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.3194  missed-object 0.1939  consistency 0.8039
2026-11-08 10:50:21,335 INFO    t2o.cli:   false-object rate       0.3194  (LOWER better)
2026-11-08 10:50:21,335 INFO    t2o.cli:   missed-object rate      0.1939  (LOWER better)
2026-11-08 10:50:21,335 INFO    t2o.cli:   detection-consistency   0.8039  (higher better)
2026-11-08 10:50:21,335 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-08 10:50:21,336 INFO    t2o.cli:     Fuse           false 0.2390  missed 0.2798
2026-11-08 10:50:21,336 INFO    t2o.cli:     Pole           false 0.2487  missed 0.0719
2026-11-08 10:50:21,336 INFO    t2o.cli:     Switch         false 0.5222  missed 0.3485
2026-11-08 10:50:21,336 INFO    t2o.cli:     Transformer    false 0.4444  missed 0.0278
2026-11-08 10:50:21,408 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s0\metrics.json stage 3
--- runs/e3t-loop-s1 ---
2026-11-08 10:50:41,916 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-08 10:50:59,194 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.2764  missed-object 0.1891  consistency 0.8039
2026-11-08 10:50:59,196 INFO    t2o.cli:   false-object rate       0.2764  (LOWER better)
2026-11-08 10:50:59,196 INFO    t2o.cli:   missed-object rate      0.1891  (LOWER better)
2026-11-08 10:50:59,196 INFO    t2o.cli:   detection-consistency   0.8039  (higher better)
2026-11-08 10:50:59,196 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-08 10:50:59,196 INFO    t2o.cli:     Fuse           false 0.1667  missed 0.2560
2026-11-08 10:50:59,196 INFO    t2o.cli:     Pole           false 0.1886  missed 0.0719
2026-11-08 10:50:59,197 INFO    t2o.cli:     Switch         false 0.5287  missed 0.3788
2026-11-08 10:50:59,197 INFO    t2o.cli:     Transformer    false 0.4355  missed 0.0278
2026-11-08 10:50:59,235 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s1\metrics.json stage 3
--- runs/e3t-loop-s3 ---
2026-11-08 10:51:20,503 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul
2026-11-08 10:51:37,309 INFO    t2o.metrics.faithfulness: faithfulness val on 153 pairs: false-object 0.2963  missed-object 0.1915  consistency 0.7974
2026-11-08 10:51:37,311 INFO    t2o.cli:   false-object rate       0.2963  (LOWER better)
2026-11-08 10:51:37,311 INFO    t2o.cli:   missed-object rate      0.1915  (LOWER better)
2026-11-08 10:51:37,311 INFO    t2o.cli:   detection-consistency   0.7974  (higher better)
2026-11-08 10:51:37,311 INFO    t2o.cli:     Connector      false 0.0000  missed 0.0000
2026-11-08 10:51:37,312 INFO    t2o.cli:     Fuse           false 0.2395  missed 0.2440
2026-11-08 10:51:37,312 INFO    t2o.cli:     Pole           false 0.2011  missed 0.0654
2026-11-08 10:51:37,312 INFO    t2o.cli:     Switch         false 0.5309  missed 0.4242
2026-11-08 10:51:37,312 INFO    t2o.cli:     Transformer    false 0.4237  missed 0.0556
2026-11-08 10:51:37,347 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s3\metrics.json stage 3
=== FINISHED 2026-11-08 10:51:39 +03:00 ===
```

### Step 5 — completion on refined erased val

Everything after the provenance block, verbatim:

```
gate: ADR-04 accepted at HEAD 223a8b9
--- runs/e3b-loop-s0 ---
2026-11-08 10:53:57,152 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-08 10:54:10,154 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-08 10:54:10,154 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-08 10:54:10,198 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s0\metrics.json stage 3
--- runs/e3b-loop-s1 ---
2026-11-08 10:54:32,130 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-08 10:54:44,519 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-08 10:54:44,520 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-08 10:54:44,579 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s1\metrics.json stage 3
--- runs/e3b-loop-s2 ---
2026-11-08 10:55:05,711 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-08 10:55:18,048 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-08 10:55:18,049 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-08 10:55:18,089 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s2\metrics.json stage 3
--- runs/e3b-loop-s3 ---
2026-11-08 10:55:40,638 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-08 10:55:54,515 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-08 10:55:54,516 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-08 10:55:54,580 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s3\metrics.json stage 3
--- runs/e3b-loop-s4 ---
2026-11-08 10:56:16,176 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-08 10:56:30,392 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-08 10:56:30,393 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-08 10:56:30,442 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s4\metrics.json stage 3
--- runs/e3b-loop-s5 ---
2026-11-08 10:56:52,353 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-08 10:57:06,081 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-08 10:57:06,082 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-08 10:57:06,149 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3b-loop-s5\metrics.json stage 3
--- runs/e3t-loop-s0 ---
2026-11-08 10:57:28,942 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-08 10:57:42,701 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-08 10:57:42,702 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-08 10:57:42,776 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s0\metrics.json stage 3
--- runs/e3t-loop-s1 ---
2026-11-08 10:58:03,674 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-08 10:58:14,821 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-08 10:58:14,823 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-08 10:58:14,881 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s1\metrics.json stage 3
--- runs/e3t-loop-s3 ---
2026-11-08 10:58:36,734 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\cascade-gate\erased-custom-val-fuse-switch
2026-11-08 10:58:49,789 INFO    t2o.metrics.faithfulness: completion val on 153 erased frames: completion rate 0.0000
2026-11-08 10:58:49,790 INFO    t2o.cli:   completion rate         0.0000  (LOWER better)
2026-11-08 10:58:49,830 INFO    t2o.engine.loop: recorded judge rtdetr-cascade into D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\e3t-loop-s3\metrics.json stage 3
=== FINISHED 2026-11-08 10:58:51 +03:00 ===
```

### Step 6 — readout

Everything after the provenance block, verbatim. The script is printed first, then its output:

```
gate: ADR-04 accepted at HEAD 223a8b9
"""Cascade gate readout (cascade-gate ADR-04): cascade - stage 1 at stage 3, paired by seed."""
import json
from pathlib import Path

from t2o.analysis.aggregate import bootstrap_ci, sign_flip_p_value

CELLS = {'e3b': (0, 1, 2, 3, 4, 5), 'e3t': (0, 1, 3)}
STAGE1, CASCADE = 'rtdetr', 'rtdetr-cascade'
# ADR-04: the Fuse and Switch margins come from record 034; the pooled ones are
# out-of-family-judge ADR-01's. A rate passes when its CI's upper end is below +margin.
MARGINS = {
    'e3b': {
        'missed_object_rate_by_class.Fuse': 0.0229,
        'false_object_rate_by_class.Fuse': 0.0284,
        'missed_object_rate_by_class.Switch': 0.0602,
        'false_object_rate_by_class.Switch': 0.0487,
        'false_object_rate': 0.0179,
        'missed_object_rate': 0.0094,
    },
    'e3t': {
        'missed_object_rate_by_class.Fuse': 0.0209,
        'false_object_rate_by_class.Fuse': 0.0342,
        'missed_object_rate_by_class.Switch': 0.0546,
        'false_object_rate_by_class.Switch': 0.0558,
        'false_object_rate': 0.0315,
        'missed_object_rate': 0.0112,
    },
}


def judges(cell, seed):
    stages = json.loads(Path(f'runs/{cell}-loop-s{seed}/metrics.json').read_text())
    return next(stage for stage in stages if int(stage['stage']) == 3)['judges']


def value(scores, path):
    for key in path.split('.'):
        scores = scores[key]
    return float(scores)


def compare(cell, path):
    """Print the per-seed pairs and the interval; return the cascade values and the CI's top."""
    seeds = CELLS[cell]
    stage1 = [value(judges(cell, seed)[STAGE1], path) for seed in seeds]
    cascade = [value(judges(cell, seed)[CASCADE], path) for seed in seeds]
    diffs = [after - before for before, after in zip(stage1, cascade)]
    low, high = bootstrap_ci(diffs)
    print(f'{cell} {path}')
    for seed, before, after, diff in zip(seeds, stage1, cascade, diffs):
        print(f'    seed {seed}: stage1 {before:.4f}  cascade {after:.4f}  diff {diff:+.4f}')
    print(
        f'    n={len(diffs)}  mean diff {sum(diffs) / len(diffs):+.4f}  '
        f'95% CI [{low:+.4f}, {high:+.4f}]  sign-flip p {sign_flip_p_value(diffs):.4f}'
    )
    return cascade, high


verdicts = []
for cell in CELLS:
    cascade, _ = compare(cell, 'completion.completion_rate')
    passed = all(rate == 0.0 for rate in cascade)
    verdicts.append(passed)
    print(f'    -> {"PASS" if passed else "FAIL"}: zero tolerance, 0.0000 on every seed')
    for rate, margin in MARGINS[cell].items():
        _, high = compare(cell, f'faithfulness.{rate}')
        passed = high < margin
        verdicts.append(passed)
        print(f'    -> {"PASS" if passed else "FAIL"}: CI upper {high:+.4f} vs margin +{margin:.4f}')
    compare(cell, 'faithfulness.detection_consistency')
    print('    -> reported, not gated')

print(
    f'GATE: {"PASS" if all(verdicts) else "FAIL"} -- {sum(verdicts)} of {len(verdicts)} '
    'tests pass; the cascade enters the paper only if all pass (ADR-04)'
)
e3b completion.completion_rate
    seed 0: stage1 0.0000  cascade 0.0000  diff +0.0000
    seed 1: stage1 0.0000  cascade 0.0000  diff +0.0000
    seed 2: stage1 0.0000  cascade 0.0000  diff +0.0000
    seed 3: stage1 0.0000  cascade 0.0000  diff +0.0000
    seed 4: stage1 0.0000  cascade 0.0000  diff +0.0000
    seed 5: stage1 0.0000  cascade 0.0000  diff +0.0000
    n=6  mean diff +0.0000  95% CI [+0.0000, +0.0000]  sign-flip p 1.0000
    -> PASS: zero tolerance, 0.0000 on every seed
e3b faithfulness.missed_object_rate_by_class.Fuse
    seed 0: stage1 0.1548  cascade 0.2500  diff +0.0952
    seed 1: stage1 0.1429  cascade 0.2679  diff +0.1250
    seed 2: stage1 0.1607  cascade 0.2619  diff +0.1012
    seed 3: stage1 0.1964  cascade 0.3155  diff +0.1190
    seed 4: stage1 0.1845  cascade 0.2679  diff +0.0833
    seed 5: stage1 0.1964  cascade 0.2679  diff +0.0714
    n=6  mean diff +0.0992  95% CI [+0.0843, +0.1141]  sign-flip p 0.0312
    -> FAIL: CI upper +0.1141 vs margin +0.0229
e3b faithfulness.false_object_rate_by_class.Fuse
    seed 0: stage1 0.0839  cascade 0.2025  diff +0.1187
    seed 1: stage1 0.1056  cascade 0.2313  diff +0.1257
    seed 2: stage1 0.1506  cascade 0.1389  diff -0.0117
    seed 3: stage1 0.0753  cascade 0.2434  diff +0.1681
    seed 4: stage1 0.1218  cascade 0.1575  diff +0.0357
    seed 5: stage1 0.1290  cascade 0.1399  diff +0.0108
    n=6  mean diff +0.0745  95% CI [+0.0228, +0.1292]  sign-flip p 0.0938
    -> FAIL: CI upper +0.1292 vs margin +0.0284
e3b faithfulness.missed_object_rate_by_class.Switch
    seed 0: stage1 0.2727  cascade 0.4091  diff +0.1364
    seed 1: stage1 0.3788  cascade 0.4242  diff +0.0455
    seed 2: stage1 0.2273  cascade 0.3636  diff +0.1364
    seed 3: stage1 0.2121  cascade 0.3636  diff +0.1515
    seed 4: stage1 0.2879  cascade 0.3788  diff +0.0909
    seed 5: stage1 0.2424  cascade 0.4394  diff +0.1970
    n=6  mean diff +0.1263  95% CI [+0.0859, +0.1616]  sign-flip p 0.0312
    -> FAIL: CI upper +0.1616 vs margin +0.0602
e3b faithfulness.false_object_rate_by_class.Switch
    seed 0: stage1 0.2836  cascade 0.4507  diff +0.1671
    seed 1: stage1 0.4143  cascade 0.5870  diff +0.1727
    seed 2: stage1 0.3014  cascade 0.4545  diff +0.1532
    seed 3: stage1 0.2973  cascade 0.4545  diff +0.1572
    seed 4: stage1 0.3472  cascade 0.5287  diff +0.1815
    seed 5: stage1 0.3506  cascade 0.5000  diff +0.1494
    n=6  mean diff +0.1635  95% CI [+0.1549, +0.1728]  sign-flip p 0.0312
    -> FAIL: CI upper +0.1728 vs margin +0.0487
e3b faithfulness.false_object_rate
    seed 0: stage1 0.1553  cascade 0.2566  diff +0.1013
    seed 1: stage1 0.1865  cascade 0.3154  diff +0.1288
    seed 2: stage1 0.1905  cascade 0.2528  diff +0.0623
    seed 3: stage1 0.1651  cascade 0.2727  diff +0.1076
    seed 4: stage1 0.1787  cascade 0.2625  diff +0.0837
    seed 5: stage1 0.1743  cascade 0.2431  diff +0.0688
    n=6  mean diff +0.0921  95% CI [+0.0745, +0.1107]  sign-flip p 0.0312
    -> FAIL: CI upper +0.1107 vs margin +0.0179
e3b faithfulness.missed_object_rate
    seed 0: stage1 0.1253  cascade 0.1986  diff +0.0733
    seed 1: stage1 0.1442  cascade 0.2199  diff +0.0757
    seed 2: stage1 0.1158  cascade 0.2033  diff +0.0875
    seed 3: stage1 0.1395  cascade 0.2246  diff +0.0851
    seed 4: stage1 0.1418  cascade 0.1962  diff +0.0544
    seed 5: stage1 0.1489  cascade 0.2199  diff +0.0709
    n=6  mean diff +0.0745  95% CI [+0.0654, +0.0823]  sign-flip p 0.0312
    -> FAIL: CI upper +0.0823 vs margin +0.0094
e3b faithfulness.detection_consistency
    seed 0: stage1 0.8431  cascade 0.7691  diff -0.0741
    seed 1: stage1 0.8257  cascade 0.7582  diff -0.0675
    seed 2: stage1 0.8649  cascade 0.7734  diff -0.0915
    seed 3: stage1 0.8301  cascade 0.7364  diff -0.0937
    seed 4: stage1 0.8366  cascade 0.7734  diff -0.0632
    seed 5: stage1 0.8301  cascade 0.7364  diff -0.0937
    n=6  mean diff -0.0806  95% CI [-0.0901, -0.0704]  sign-flip p 0.0312
    -> reported, not gated
e3t completion.completion_rate
    seed 0: stage1 0.0000  cascade 0.0000  diff +0.0000
    seed 1: stage1 0.0000  cascade 0.0000  diff +0.0000
    seed 3: stage1 0.0000  cascade 0.0000  diff +0.0000
    n=3  mean diff +0.0000  95% CI [+0.0000, +0.0000]  sign-flip p 1.0000
    -> PASS: zero tolerance, 0.0000 on every seed
e3t faithfulness.missed_object_rate_by_class.Fuse
    seed 0: stage1 0.1845  cascade 0.2798  diff +0.0952
    seed 1: stage1 0.1667  cascade 0.2560  diff +0.0893
    seed 3: stage1 0.1429  cascade 0.2440  diff +0.1012
    n=3  mean diff +0.0952  95% CI [+0.0893, +0.1012]  sign-flip p 0.2500
    -> FAIL: CI upper +0.1012 vs margin +0.0209
e3t faithfulness.false_object_rate_by_class.Fuse
    seed 0: stage1 0.1796  cascade 0.2390  diff +0.0594
    seed 1: stage1 0.1250  cascade 0.1667  diff +0.0417
    seed 3: stage1 0.1166  cascade 0.2395  diff +0.1230
    n=3  mean diff +0.0747  95% CI [+0.0417, +0.1230]  sign-flip p 0.2500
    -> FAIL: CI upper +0.1230 vs margin +0.0342
e3t faithfulness.missed_object_rate_by_class.Switch
    seed 0: stage1 0.2273  cascade 0.3485  diff +0.1212
    seed 1: stage1 0.3333  cascade 0.3788  diff +0.0455
    seed 3: stage1 0.2576  cascade 0.4242  diff +0.1667
    n=3  mean diff +0.1111  95% CI [+0.0455, +0.1667]  sign-flip p 0.2500
    -> FAIL: CI upper +0.1667 vs margin +0.0546
e3t faithfulness.false_object_rate_by_class.Switch
    seed 0: stage1 0.3855  cascade 0.5222  diff +0.1367
    seed 1: stage1 0.4568  cascade 0.5287  diff +0.0719
    seed 3: stage1 0.3467  cascade 0.5309  diff +0.1842
    n=3  mean diff +0.1309  95% CI [+0.0719, +0.1842]  sign-flip p 0.2500
    -> FAIL: CI upper +0.1842 vs margin +0.0558
e3t faithfulness.false_object_rate
    seed 0: stage1 0.2215  cascade 0.3194  diff +0.0978
    seed 1: stage1 0.2151  cascade 0.2764  diff +0.0613
    seed 3: stage1 0.1685  cascade 0.2963  diff +0.1278
    n=3  mean diff +0.0956  95% CI [+0.0613, +0.1278]  sign-flip p 0.2500
    -> FAIL: CI upper +0.1278 vs margin +0.0315
e3t faithfulness.missed_object_rate
    seed 0: stage1 0.1277  cascade 0.1939  diff +0.0662
    seed 1: stage1 0.1371  cascade 0.1891  diff +0.0520
    seed 3: stage1 0.1135  cascade 0.1915  diff +0.0780
    n=3  mean diff +0.0654  95% CI [+0.0520, +0.0780]  sign-flip p 0.2500
    -> FAIL: CI upper +0.0780 vs margin +0.0112
e3t faithfulness.detection_consistency
    seed 0: stage1 0.8627  cascade 0.8039  diff -0.0588
    seed 1: stage1 0.8562  cascade 0.8039  diff -0.0523
    seed 3: stage1 0.8649  cascade 0.7974  diff -0.0675
    n=3  mean diff -0.0595  95% CI [-0.0675, -0.0523]  sign-flip p 0.2500
    -> reported, not gated
GATE: FAIL -- 2 of 14 tests pass; the cascade enters the paper only if all pass (ADR-04)
=== FINISHED 2026-11-08 11:33:15 +03:00 ===
```

Re-render from the results files without re-running: `uv run python runs/cascade-gate/stage2-readout.py`
in the repo root on VSIBMEOT05. It reads `judges.rtdetr` and `judges.rtdetr-cascade` at stage 3
from each run's `metrics.json`.

## Findings

### F220 — The cascade fails gate ADR-04: 2 of 14 tests pass, so it stays out of the January paper

The comparison is cascade − stage 1 on the same seed, against ADR-04's margins. The two completion
tests pass. All twelve rate tests fail:

| cell (n) | test | mean diff | 95% CI | margin | sign-flip p | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| e3b (6) | completion rate (erased val) | +0.0000 | [+0.0000, +0.0000] | 0, every seed | 1.0000 | PASS |
| e3b (6) | missed Fuse | +0.0992 | [+0.0843, +0.1141] | 0.0229 | 0.0312 | FAIL |
| e3b (6) | false Fuse | +0.0745 | [+0.0228, +0.1292] | 0.0284 | 0.0938 | FAIL |
| e3b (6) | missed Switch | +0.1263 | [+0.0859, +0.1616] | 0.0602 | 0.0312 | FAIL |
| e3b (6) | false Switch | +0.1635 | [+0.1549, +0.1728] | 0.0487 | 0.0312 | FAIL |
| e3b (6) | false pooled | +0.0921 | [+0.0745, +0.1107] | 0.0179 | 0.0312 | FAIL |
| e3b (6) | missed pooled | +0.0745 | [+0.0654, +0.0823] | 0.0094 | 0.0312 | FAIL |
| e3t (3) | completion rate (erased val) | +0.0000 | [+0.0000, +0.0000] | 0, every seed | 1.0000 | PASS |
| e3t (3) | missed Fuse | +0.0952 | [+0.0893, +0.1012] | 0.0209 | 0.2500 | FAIL |
| e3t (3) | false Fuse | +0.0747 | [+0.0417, +0.1230] | 0.0342 | 0.2500 | FAIL |
| e3t (3) | missed Switch | +0.1111 | [+0.0455, +0.1667] | 0.0546 | 0.2500 | FAIL |
| e3t (3) | false Switch | +0.1309 | [+0.0719, +0.1842] | 0.0558 | 0.2500 | FAIL |
| e3t (3) | false pooled | +0.0956 | [+0.0613, +0.1278] | 0.0315 | 0.2500 | FAIL |
| e3t (3) | missed pooled | +0.0654 | [+0.0520, +0.0780] | 0.0112 | 0.2500 | FAIL |

- **The fails are not close.** Every failing CI lies wholly above zero. Every mean difference is at
  least twice its margin; the nearest are the Switch missed rates, at 2.1× (e3b) and 2.0× (e3t).
  The CI upper ends are 2.7–8.8× their margins. ADR-04 reads a fail as robust, and here no choice
  of interval method could turn one into a pass.
- **The p-values sit at their floors.** `sign_flip_p_value` cannot go below 2/2ⁿ
  (`src/t2o/analysis/aggregate.py:308-309`): 0.0312 at n = 6 and 0.2500 at n = 3. Every e3b rate
  but false Fuse reaches 0.0312, and every e3t rate reaches 0.2500. So on those rates, every seed
  moved the same way. e3t's p cannot show significance at three seeds, whatever the effect.
- **The two passes are real zeros.** On erased val, the judge found none of the 234 erased fuses
  and switches on any of the nine refined runs: 2,106 box checks, all empty, the same as stage 1
  (F217). Stage 2 drew no fuse or switch where the thermal frame holds no trace of one.
- Under ADR-04, any single fail keeps the cascade in paper 2. Twelve fail.

### F221 — Zero-shot PiSA-SR does not restore fuses or switches: on every seed it misses more of them, about 1 in 10 more fuses and 1 in 8 more switches

The hypothesis (ADR-03) was that stage 2 would rebuild the faint or deformed fuses and switches
stage 1 draws, so the judge would miss fewer. The comparison is stage 1 on the same seeds, real val.
Means over the seeds, from the readout:

| cell | rate | stage 1 | cascade | change |
| --- | --- | --- | --- | --- |
| e3b | missed Fuse | 0.1726 | 0.2718 | +0.0992 |
| e3b | missed Switch | 0.2702 | 0.3965 | +0.1263 |
| e3b | false Fuse | 0.1110 | 0.1856 | +0.0745 |
| e3b | false Switch | 0.3324 | 0.4959 | +0.1635 |
| e3b | missed pooled | 0.1359 | 0.2104 | +0.0745 |
| e3b | false pooled | 0.1751 | 0.2672 | +0.0921 |
| e3b | consistency | 0.8384 | 0.7578 | −0.0806 |
| e3t | missed Fuse | 0.1647 | 0.2599 | +0.0952 |
| e3t | missed Switch | 0.2727 | 0.3838 | +0.1111 |
| e3t | false Fuse | 0.1404 | 0.2151 | +0.0747 |
| e3t | false Switch | 0.3963 | 0.5273 | +0.1309 |
| e3t | missed pooled | 0.1261 | 0.1915 | +0.0654 |
| e3t | false pooled | 0.2017 | 0.2974 | +0.0956 |
| e3t | consistency | 0.8613 | 0.8017 | −0.0595 |

- **The win goes the other way on every seed.** None of the 18 seed pairs misses fewer fuses or
  fewer switches. The smallest rise is +0.0455 on Switch (e3b seed 1 and e3t seed 1), which is 3 of
  val's 66 switches. On average the cascade misses about 16–17 more of the 168 fuses and 7–8 more
  of the 66 switches per seed. Those counts are derived here (rate × box count), not printed.
- **The red flag on real frames rises too.** Unmatched fuse and switch predictions rise in both
  cells. Of the 54 gated seed pairs on real val, 53 move the wrong way. The one exception is e3b
  seed 2's false Fuse rate, −0.0117.
- **Pooled, the judge misses about half as much again** (+55% e3b, +52% e3t), and
  detection-consistency with real visible frames drops on all nine seeds.
- **It does this with a small pixel change.** The mean absolute change from the stage-1 input is
  0.0329–0.0357 of the pixel range per val split, and 0.0141–0.1312 per frame across all 18
  refines. ADR-01 had left PiSA-SR's ×1 behaviour on these inputs as unknown.
- **A reading, not tested:** missed and false rates rising together is what boxes drawn shifted or
  deformed below IoU 0.5 would produce, rather than objects drawn from nothing. That would fit F220's
  zero completion. This run cannot separate the two, because it logs rates, not matches.
- **What the frames look like fits a texture loss, not a shape repair.** By the user's eye check
  (provenance caveats), stage 2 keeps stage 1's distorted shapes and flattens each component into
  one texture and colour, with sharp edges. That removes detail the judge may use, and it does not
  fix the deformation ADR-03 hoped stage 2 would fix. It is a qualitative report, not a measurement,
  and it does not show which of the two causes the higher rates.

### F222 — ADR-02's smoke passes, the checkpoint strict-loads under torch 2.12.1, and the refine is deterministic across invocations

- **Smoke.** All 153 refined frames keep their stage-1 size, and the mean change is 0.0335 (per
  frame 0.0188–0.0812), so stage 2 is not an identity. That is the PASS ADR-02 asked for.
- **The checkpoint loads.** ADR-01 called loading `pisa_sr.pkl` (saved with torch 2.0 and peft 0.9)
  under the current stack "believed, not verified". On all 19 loads, the strict loader logged
  `merged 6 PiSA-SR adapters (1032 tensors)` under torch 2.12.1+cu130 and raised nothing.
  CASCADE-GATE-05's loader raises on an unused checkpoint key or an unfilled LoRA parameter.
- **Determinism.** The comparison is the smoke against the main run's e3b seed 0 val refine, two
  separate invocations on the same source. They print the same config hash, `e01670e12d6ad8fd`,
  though their run dir and name differ, and the same mean change on all 153 frames, to four
  decimals. So stage 2 adds no run-to-run noise at that resolution, and the cascade's seed spread
  is stage 1's.

### F223 — A warm PiSA-SR refine costs 0.28–0.35 s per 640×480 frame on one A100 in float32, about turbo's stage-1 export speed and twice pix2pix's

- **Per refine:** 44–54 s for a 153-frame split, from the config-hash line to the split summary,
  plus 18–22 s from the load line to the config hash. The 18 refines took 28 min 14 s, including
  `uv` start-up.
- **Against stage 1:** in record 034's export log, measured the same way per 153-frame split
  (dataset line to `exported` line), pix2pix took 21–24 s and turbo 43–46 s.
- **The first load is the expensive one.** On the cold cache, the smoke's load line (09:26:09) to
  its adapter merge (10:12:20) took 46 min 11 s, mostly unauthenticated Hub downloads of the SD-2.1
  text encoder, VAE and UNet. A warm load takes 15–18 s to the merge.
- **Scoring is cheap:** faithfulness on nine runs took 5 min 56 s, completion 5 min 19 s.

## Provenance caveats

- **Stage-1 scores were read, not re-made.** The readout's stage-1 values are `judges.rtdetr`, as
  record 034 wrote them. Their arm means match record 034's readout to four decimals (e3b missed
  pooled 0.1359, false pooled 0.1751; e3t 0.1261 / 0.2017; e3b missed Fuse 0.1726). So this run's
  write-backs did not disturb them.
- **The write-backs added keys to nine `metrics.json` files:** `judges.rtdetr-cascade.faithfulness`
  (step 4) and `judges.rtdetr-cascade.completion` (step 5), at stage 3. Nothing under
  `judges.rtdetr` was touched, according to the runbook's design and the matching means above.
- **The completion scorer's denominator is again not in the log.** Record 034 confirmed 234 labels
  on disk in the same erased tree, and this run reads that same tree. That the scorer read exactly
  those files is still a reading of the code.
- **The eye check of the refined frames is in no log.** The user looked at refined frames on the
  server and reported it on 2026-11-08, after the run. Components stay as distorted as in stage 1,
  but noise is suppressed and edges are sharp. The frames read like abstract paintings: clean and
  sharp, with little or no detail. A fuse's shape can be made out, but its whole area is mostly one
  texture and one colour. How many frames were looked at, and which ones, is not recorded.
- **The smoke ran on all 153 frames, not "a handful"** as ADR-02 worded it. The runbook says why:
  `t2o refine` takes a whole split.
- **Determinism (F222) is checked at four decimals of a per-frame mean**, not by a pixel-exact
  diff of the two output folders.
- **The SD-2.1 base came from a third-party mirror**, unauthenticated, pinned by revision. Whether it
  matches StabilityAI's original weights is believed, not verified (ADR-01). The peft and diffusers
  versions are not in the provenance block.
- **F223's comparison spans two runs five days apart**, at different git SHAs, both on `cuda:1`.
  Each times a 153-frame split from a log line, so the numbers include file reading and writing.
- **e3t has three seeds.** Its sign-flip p cannot go below 0.25. Its fails rest on the CI and on
  every seed moving the same way.

## Next

The gate has answered: under ADR-04, the zero-shot cascade does not enter the January paper. It
stays paper-2 work, as `docs/goal.md`'s Non-Goals already frame it. ADR-01 says that changing the
model or its λ setting after a stage-2 output is scored needs a superseding ADR, and the paper would
then have to disclose both versions. So a second zero-shot try (PiSA-SR at a lower λ_sem, or
OSEDiff) is a decision to take openly, not a retune. ADR-01 already rejected the least generative
setting as the post-hoc choice the gate exists to prevent.

**A prediction this record can be checked against:** if F221's reading is right, and stage 2
shifts or deforms the fuses and switches stage 1 drew rather than inventing new ones, then most of
the cascade's extra unmatched fuse and switch predictions overlap an unmatched ground-truth box of
the same class at an IoU between 0.1 and 0.5. A match-level dump of the `rtdetr-cascade` scores on
the existing `cascade-translated` folders would test this without any new refine.
