# Thermal→Visible Translation with Detection-in-the-Loop

**Implementation plan.** Companion to [RESEARCH_FINDINGS.md](RESEARCH_FINDINGS.md), which
defines *what* we are proving and *why*. This document defines *how* the code gets built.
Task-level breakdown lives in [TASKS.md](TASKS.md).

---

## 1. Objective and scope

The research question, unchanged from `RESEARCH_FINDINGS.md` §1:

> Does closing a training-time detection-consistency loop around a diffusion-based
> thermal→visible translator improve downstream power-line component detection, without
> hallucinating or erasing safety-relevant components?

The repo is an instrument for producing one defensible results table and defending it. It
is not a product. No inference service, no deployment tooling, no UI, no labeling tools.

Contribution stack (`RESEARCH_FINDINGS.md` §1):

| # | Contribution | Role |
| --- | --- | --- |
| C1 | Closed-loop detection-consistency feedback for diffusion IR→VIS translation | Headline method |
| C2 | Faithfulness / hallucination metric for safety-critical translation | Co-contribution |
| C3 | Paired thermal-visible power-component detection benchmark protocol | Domain contribution |
| C4 | Analysis of *when* translation beats direct thermal detection | Defends the premise |

---

## 2. Three facts that shape everything below

**1. The custom dataset is ~850 annotated pairs.** `RESEARCH_FINDINGS.md` §6 assumes
LBBDM-f4 can be trained on it. At 850 pairs that is not credible — diffusion from scratch
will memorise. This drives the backbone revision in §4.

**2. `Clean-SeAFusion` is the direct ancestor.** Our own repo at `../Clean-SeAFusion`
already implements detection-in-the-loop correctly: frozen detector, differentiable
`v8DetectionLoss`, staged λ ramp, and the non-obvious W&B/ultralytics integration fixes.
Large parts port over rather than being rebuilt. See §6.

**3. Training data and detector weights live only on the server.** Locally we have 9 pairs,
and because the remote is public they are **not committed** — so a fresh clone has no
images at all. Local development is therefore validated exclusively against a *synthetic*
smoke fixture generated at test time, and that has to be a first-class design constraint
rather than an afterthought. See §9.

---

## 3. Environment and constraints

| | |
| --- | --- |
| Dev machine | macOS, CPU/MPS only. Code is written and smoke-tested here. |
| Train machine | Native Windows Server 2022, 2×A100 40GB, **no WSL2** |
| Transport | git push from dev → git pull on server. No shared filesystem. |
| Config | Pydantic v2 models + YAML on disk, hashed for provenance |
| W&B | Self-hosted; base URL already in the server environment |

Native Windows consequences, accommodated by design rather than fought:

- **No DDP.** NCCL does not exist on Windows. The second A100 is throughput, not scale —
  one experiment per GPU via `CUDA_VISIBLE_DEVICES`. Every method must be trainable on a
  single 40GB card.
- **Assume no triton, no xformers, no `torch.compile`.** Mitigations in order of value:
  PyTorch native `scaled_dot_product_attention`, gradient checkpointing, bf16.
- **Spawn, not fork.** Dataset classes importable at module level, everything picklable,
  entry points guarded. This is the most common source of silent hangs on this platform.
- **No bash launchers.** Invoke Python directly through our own config layer.
- **No symlinks.** Absolute config paths instead.
- `pycocotools` may need a prebuilt wheel.

---

## 4. Backbone strategy — the central revision

`RESEARCH_FINDINGS.md` §6 makes LBBDM-f4-from-scratch the Phase 2 backbone, then spends §4
and E5 working around the consequence: a multi-step sampler cannot be back-propagated
through on 40GB, so detection loss must reach it via ReFL/DRaFT approximations. At 850
pairs, training that model from scratch also will not produce a translator worth attaching
a loop to.

**`pix2pix-turbo` becomes the primary diffusion backbone.** Verified in the local checkout
at `../img2img-turbo` (`src/pix2pix_turbo.py:186`, commit `463b2d3`):

```python
self.sched = make_1step_sched()          # __init__
encoded_control = self.vae.encode(c_t).latent_dist.sample() * scaling_factor
model_pred  = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc).sample
x_denoised  = self.sched.step(model_pred, self.timesteps, encoded_control).prev_sample
output_image = self.vae.decode(x_denoised / scaling_factor).sample.clamp(-1, 1)
```

One UNet evaluation. No sampling loop. `self.timesteps` is a fixed `tensor([999])`. This
resolves four separate problems at once:

| Problem in RESEARCH_FINDINGS | Resolution |
| --- | --- |
| 850 pairs can't train diffusion from scratch | SD-turbo is pretrained; only LoRA adapters train (rank 8 UNet / 4 VAE) |
| "Full-sampler gradient backprop will not fit" (§2) | The whole generator is one forward pass — detection loss back-props through *all* of it, exactly |
| E5 tractability ladder (ReFL → DRaFT-K → LCM) | Becomes an *experiment*, not a dependency. We start distilled. |
| Reward hacking (§6 guardrails) | LoRA-only updates are the lever DRaFT identifies as most effective — built in |

LBBDM is **not** dropped. It becomes the multi-step arm of E5. That comparison — exact
gradients through a distilled one-step model vs. truncated gradients through a multi-step
one — is a more interesting methodological finding than the original framing, and the
honest version of E5 rather than a workaround.

### Backbone ladder

**pix2pix (Phase 1 gate) → pix2pix-turbo (Phase 2a primary) → LBBDM-f4 + ReFL (Phase 2b).**

CUT moves to E2 as the *unpaired* baseline. Our data is paired and CUT has no paired mode —
the author explicitly declined to add one ([issue #141](https://github.com/taesungp/contrastive-unpaired-translation/issues/141):
*"this repo does not aim to tackle the aligned setting"*).

UNSB drops to optional: single-GPU batch-1 only, a three-arg `netG(x, time_idx, z)`
signature, two zipped dataloaders per step, and a multi-step sampler buried inside
`forward()` behind a `phase == 'test'` branch. Poor cost/benefit for one baseline row.

---

## 5. Repository architecture

Five layers. The boundary that matters most: `core` never contains method-specific code.

```
src/t2o/
  config/        pydantic schemas, YAML loading, config hashing, snapshotting
  data/          manifest, pairing, dataset, labels, adapters per public dataset
  metrics/       fidelity (PSNR/SSIM/LPIPS/FID/KID), task (mAP), faithfulness (C2)
  translators/   uniform wrapper per backbone over vendored third_party/
  detection/     FrozenDetector (in-loop) + evaluation detector, strictly separated
  coupling/      detection-consistency loss and its schedule
  engine/        trainer, loop, export, checkpointing
  tracking.py    W&B RunTracker
  cli.py         argparse subcommands
experiments/     experiment config YAMLs — tracked. An experiment IS a config file.
runs/            run outputs (checkpoints, exports, metrics.json) — ignored
third_party/     vendored at pinned commits, never edited in place
tests/
```

Splitting `experiments/` (configs, tracked) from `runs/` (outputs, ignored) departs
deliberately from `RESEARCH_FINDINGS.md` §3's "experiments/ configs and results only".
Configs must reach the server via git; results must not come back through it.

### Invariants

1. **One evaluation path.** Every method computes every metric through the same code.
   Non-negotiable — it is what makes the comparison table defensible.
2. **Frozen data contract.** Splits decided once, hashed, version-controlled. No method
   sees its own split logic.
3. **Backbones interchangeable.** A translator is anything that can `fit()` and
   `translate(batch) -> Tensor`. Swapping one for another is a config change.
4. **Third-party code vendored at pinned commits, never edited in place.** All adaptation
   lives in wrappers we own.
5. **The loop is a first-class component**, switchable off cleanly — because switching it
   off *is* the central ablation.
6. **An experiment is a config file.** Results carry their config hash.
7. **Three detector roles, never conflated.** The *in-loop* detector guides training and
   receives generator gradients. The *evaluation* detector is fine-tuned on translated
   exports and never receives them. The *reference* detector is never trained at all — it
   only scores translations zero-shot, and is what §12's gate arm is measured with. Encoded
   structurally as three sub-sections of `DetectorConfig` (`config/schema.py`), so no two
   roles can share a weights file by accident.

---

## 6. What ports from Clean-SeAFusion

Read and verified during planning. Near-verbatim ports, not inspiration.

| Source (`../Clean-SeAFusion/`) | Target | Change |
| --- | --- | --- |
| `src/seafusion/models/detector.py` — `FrozenDetector` | `detection/frozen.py` | None. `train()` override, `_normalize_args` checkpoint repair, stride validation all still apply. |
| `src/seafusion/losses/task.py` — `DetectionTaskLoss` | `coupling/detection_loss.py` | Drop YCbCr recombination; the translator emits RGB directly. **Keep the batch-size division** (`components / batch_size`) that keeps λ invariant to batch size. |
| `src/seafusion/data/dataset.py` | `data/dataset.py` | Rename to translation semantics; keep the bbox crop/flip math, the class-id range precheck, and the `v8DetectionLoss`-shaped collate exactly. |
| `src/seafusion/data/{pairing,manifest,labels}.py` | `data/` | None. Filename-based pairing (never sorted-index); `data.yaml` as single source of truth. |
| `src/seafusion/engine/detector_stage.py` | `engine/detector_stage.py` | Keep `wandb_integration_disabled()` and the `_resolve_weights()` `last.pt` fallback. |
| `src/seafusion/engine/fusion_trainer.py` | `engine/trainer.py` | Keep bf16/fp16 GradScaler logic, resumable checkpoints, `config_hash()`, warn-on-config-drift resume. |
| `src/seafusion/engine/export.py` | `engine/export.py` | Translated images + copied labels + generated `data.yaml` → feeds the evaluation detector. |
| `src/seafusion/tracking.py` — `RunTracker` | `tracking.py` | None. Rank-0 only, never raises, context manager. |
| `src/seafusion/imaging.py` — `Normalize` | `imaging.py` | None. Per-image (never batch-wise) clamp/stretch on export. |
| `../RGBT-Fusion-Detection/pyproject.toml` cpu/gpu extras | `pyproject.toml` | Port immediately — it is what lets one lockfile serve Mac-CPU dev and Windows-CUDA training. |
| `../RGBT-Fusion-Detection/src/rgbt/config.py` — `Hyperparams` | `config/detector.py` | Re-express as a pydantic model. |

Two integration fixes in `detector_stage.py` that are worth their own mention, because they
cost real debugging time to find:

- **`wandb_integration_disabled()`** — ultralytics' W&B integration adopts an already-open
  run and calls `wb.run.finish()` at training end, killing the outer run. Must be patched
  at `ultralytics.utils.callbacks.wb.callbacks`, not on `model.callbacks`.
- **`_resolve_weights()`** — falls back to `last.pt` because ultralytics never writes
  `best.pt` if fitness was ever NaN.

**Must be built from scratch:** all fidelity metrics. No PSNR/SSIM/LPIPS/FID/KID exists in
any of our repos, and `torchmetrics`/`piq`/`lpips` appear in zero pyprojects. mAP is fine
(ultralytics `DetMetrics`).

**Do not port `../Experiment-Logging-WB/main.py`** — it hardcodes a W&B API key on line 20.
That credential should be rotated.

---

## 7. Vendoring strategy: networks and losses, not training frameworks

The upstream repos each ship a full training framework (`BaseModel` + `opt` namespaces +
`BaseDataset` + bash launchers). Wrapping four of those is where this kind of project
drowns — and the bash/`opt`/symlink machinery is exactly what does not survive native
Windows.

Vendor **model definitions and loss functions only**; drive all of them with the one
trainer ported from Clean-SeAFusion.

| Backbone | Pin | Vendor | Ignore |
| --- | --- | --- | --- |
| pix2pix / CycleGAN | `2a7afba` (2025-08-06) | `models/networks.py` — `define_G`, `define_D`, `GANLoss` | `BaseModel`, `options/`, `data/`, `train.py`, `scripts/*.sh` |
| CUT / FastCUT | `b3ac297` (2023-09-05) | generator + `models/patchnce.py::PatchNCELoss` + `PatchSampleF` | everything else |
| pix2pix-turbo | `463b2d3` (local checkout is current) | `src/model.py` — `my_vae_encoder_fwd`, `my_vae_decoder_fwd` | `train_pix2pix_turbo.py` — reimplement its loss assembly. Also `src/pix2pix_turbo.py::Pix2Pix_Turbo`: **reimplemented, not vendored** (M2a). It starts with `sys.path.append("src/")` + `from model import`, so it does not import outside upstream's cwd, and 100 of its 229 lines are checkpoint downloads for tasks we do not use. Same split M1 made between `networks.py` and `Pix2PixModel`. |
| LBBDM-f4 | `02c3b13` (2024-08-01) | `BrownianBridgeModel`, `LatentBrownianBridgeModel`, `model/VQGAN/` | `runners/`, `main.py`, `configs/` |

This makes invariant 3 real, and it defuses the single biggest practical risk in the
project:

> **The dependency maze is a property of the frameworks, not the models.** img2img-turbo's
> open issues (#97, #119, #139, #145) are all one problem — `diffusers==0.25.1` needs
> `huggingface_hub.cached_download`, removed upstream; `transformers==4.35.2` collides with
> `peft>=0.14`. The known-good combination users report is a 2023-era stack that will not
> coexist with modern torch. But `Pix2Pix_Turbo` is a ~230-line class using only stable
> APIs (`AutoencoderKL`, `UNet2DConditionModel`, `CLIPTextModel`, `AutoTokenizer`,
> `peft.LoraConfig`). Vendoring that one file and running it against *current* diffusers
> sidesteps the entire maze.
>
> Same story elsewhere: CUT's Pillow-10 `Image.BICUBIC` crash and its missing
> `torch.load(weights_only=)` both live in `data/base_dataset.py` and `models/base_model.py`
> — files we never import.

Two patches we *do* inherit and must make deliberately:

- `Pix2Pix_Turbo.__init__` hardcodes `.cuda()` (lines 33, 40–43, 158–162), and
  `src/model.py::make_1step_sched` does `set_timesteps(1, device="cuda")`. **Our wrapper
  owns device placement.**
- `LatentBrownianBridgeModel.decode()` is `@torch.no_grad()`. A grad-enabled copy is
  required for the ReFL path. The VQ quantizer already does straight-through
  `z_q = z + (z_q - z).detach()`, so gradients do flow once the decorator is gone.

### Extra dependencies

`diffusers`, `transformers`, `peft`, `lpips`, `torchmetrics` (FID/KID), `pycocotools`.

**Not** needed: `accelerate` (our trainer replaces it) and `vision_aided_loss` (the CLIP
discriminator is optional — start without it; the detection loss is the point).

### Licensing to note in the paper

- **sd-turbo**: Stability AI Non-Commercial Research Community License. Fine for an
  academic paper; constrains downstream use.
- **CUT**: bundles NVIDIA StyleGAN2 code (non-commercial) via `models/stylegan_networks.py`,
  which `models/networks.py` imports at module level even on the `resnet_9blocks` path.
- **Ultralytics**: AGPL-3.0, already accepted in `RESEARCH_FINDINGS.md` §4. Note it is
  network copyleft.

---

## 8. Coupling design

The translator emits RGB directly, so the coupling term is simpler than SeAFusion's:

```
loss = λ_l2·L2 + λ_lpips·LPIPS + λ_gan·GAN + λ_det·DetectionTaskLoss(rgb_pred, batch)
```

The first three terms are what `img2img-turbo/src/train_pix2pix_turbo.py:176-200` already
computes; the fourth is the ported `DetectionTaskLoss`. Insertion point is a single added
term at line 179's `loss = loss_l2 + loss_lpips`.

Two things come free:

- `net_lpips` is already in that loop → the **fidelity floor** guardrail (early-stop when
  LPIPS rises past a threshold) needs no new machinery.
- LoRA rank is already a knob → the anti-reward-hacking lever is a config field.

### λ_det schedule

Follows `../Clean-SeAFusion/src/seafusion/engine/loop.py`: staged `task_weights: [0,1,2,3]`,
translator **and** detector warm-started across stages. Note that original SeAFusion
re-instantiates its generator from scratch every stage (`train.py:203`), so its "loop" only
ever accumulates progress in the task network — making the alternation pointless in one
direction. Clean-SeAFusion already documents and fixes this. Do not reintroduce it.

**`λ_det = 0` must be a clean no-op path**: at weight 0 the frozen detector is never even
constructed. That is what makes E3 a genuine control.

### Ultralytics must be pinned `>=8.4.108,<8.5`

The `Detect` head output format **changed between 8.3 and 8.4**. 8.4 returns a dict
(`{"boxes", "scores", "feats"}`) from `forward_head`; 8.3 returns a list of concatenated
feature maps that `v8DetectionLoss` reshapes itself. Code written against one silently does
not work against the other.

This is also an independent reason to keep the YOLOv11-RGBT fork out of the loop: it is a
hard fork of ultralytics **8.3.75** with the whole tree vendored, unupgradable without
redoing the fork.

### The differentiable detection loss contract

Verified against ultralytics source. `v8DetectionLoss` detaches **only** the
TaskAlignedAssigner inputs (`loss.py:430-431`) — label assignment is a non-differentiable
discrete choice. The loss terms themselves consume non-detached `pred_scores` and
`pred_distri`, so gradients flow through the whole backbone/neck/head into the image.

Required `batch` keys are exactly three: `batch_idx` `(N,)`, `cls` `(N,1)`, `bboxes`
`(N,4)` **normalised cxcywh**. Image size is derived from `preds["feats"][0].shape[2:]`, so
the generated image never needs to be threaded into the batch dict.

Six things that bite, all already handled in the Clean-SeAFusion port:

1. Keep the detector in `eval()` — BN uses running statistics, so the task gradient does
   not shift with batch composition. The loss still works: `Detect.forward` returns
   `(y, preds)` outside training and `parse_output` unwraps exactly that.
2. De-parallelize before constructing the loss (`unwrap_model`; named `de_parallel` before
   ~8.4.112).
3. `model.args` must expose `.box`/`.cls`/`.dfl` — normalise it.
4. Divide by batch size; the loss is returned pre-multiplied.
5. Input must be divisible by the coarsest stride (32).
6. **Do not route generated images through `preprocess_batch`** — it does `.float()/255` on
   a uint8 dataloader tensor, which is a fresh graph root. Feed float32 `[0,1]` directly.

### Two guardrails from the reward-tuning literature

- **Saturating reward, not pure maximisation.** ReFL uses `relu(-r + 2)`; AlignProp uses
  `|r - target|`. Both stop rewarding a sample once it is good enough, which blunts reward
  hacking directly. The detector *will* find adversarial textures otherwise.
- **Aggressive constant downscale on the reward gradient.** ReFL `grad_scale=1e-3`,
  AlignProp `loss_coeff=0.01`. The `[0,1,2,3]` ramp is calibrated to SeAFusion's
  *segmentation* loss scale, not a detection loss — recalibrate empirically in Phase 0.
  **This recalibration was skipped, and E3 paid for it (TASKS.md M1.2 step 7).** At
  `grad_scale: 1.0e-2` the detection term was measured at 0.9 / 1.7 / **2.3%** of the
  objective across the ramp — roughly 19× smaller than the LPIPS term it competed against,
  which was itself 44% of the objective, with GAN at 52%. (Not the objective's *smallest*
  term: `loss_l2` measured 1.0%.) A null measured at that dose says nothing about coupling.

  **Calibrated value for pix2pix: `grad_scale: 0.15`**, achieving 10.0 / 16.1 / 19.8% of the
  objective over a full 100-epoch stage (M1.2 step 8). At that dose E3 came back positive.
  The guardrail this downscale was providing transfers to `reward_target` plus the per-stage
  LPIPS readout — and the readout has since caught something: the calibrated dose costs
  +0.0097 stage-3 LPIPS, where at `1.0e-2` fidelity was neutral within ±0.016.

  **The calibration is per backbone, not a global constant.** 0.15 is a property of *this*
  objective's composition with *this* generator. sd-turbo starts pretrained, so its
  `loss_det` sits at a different magnitude from epoch 0 and the same `grad_scale` can easily
  land back near 2% — reproducing step 7's null for step 7's reason at another ~72
  GPU-hours. Every new backbone re-runs the 25-epoch `scripts/loss_share.py` probe before its
  campaign. This is the single most expensive mistake available in this project and it has
  already been made once.

### Worth stealing from DetFusion: object-aware content loss

DetFusion itself is unportable (mmdetection 0.2.14 era, torch 1.1/1.3, Linux-only, CUDA
extensions via `bash compile.sh`). But its `DetcropPixelLoss`
(`mmdet/models/losses/fusion_loss.py:64-128`) is ~20 lines and directly serves C2:
**inside ground-truth boxes match the per-pixel max of the source modalities; outside match
their mean.** Reimplement rather than port. It gives a spatially-aware fidelity term that
specifically protects the safety-relevant components.

---

## 9. Data

### Contract

One internal representation; adapters normalise every dataset into it. Adding a dataset
never touches training code.

- Layout `{split}/{visible,infrared}/{images,labels}`, YOLO txt labels, filename-paired.
- Pairing is **path-segment substitution**, never sorted-index, with eager validation.
- `data.yaml` is the single source of truth. Keep the `rgbt:` token block (it drives
  pairing); drop `channels: 6` — our detector is stock 3-channel.
- Splits decided once, hashed, version-controlled.

### The custom dataset

~850 annotated pairs, of which **600 train / 153 val (753) are all any experiment in this repo
touches** — the remaining ~100 are held out as an unseen test set, used only at reporting time.
That isolation is structural, not procedural: `DatasetManifest` reads only
`path`/`train`/`val`/`nc`/`names`/`rgbt`, so a `test:` key in `data.yaml` is invisible to every
code path here. Frozen on the server as `yolo_rgbt_29_jul` (`combined_hash 7ede3433adc9c0b8`);
see TASKS.md M0.9 for why that record cannot reach git. **Report "753 train+val of 853", not
"850 pairs".** Scale arguments below that say "850" are order-of-magnitude and hold at 753.

**4 annotated classes** (Fuse, Pole, Switch, Transformer) inside a
manifest that declares `nc: 5`. The fifth, `Connector` (index 0), is a Label Studio artifact —
created in the labelling project and never used — so it has zero instances in both splits and
is absent from every per-class AP table. Kept rather than renumbered, since it is index 0 and
dropping it would rewrite the class id in every label file for no measurable gain; see
TASKS.md M1.1 for why every consequence is benign. **Report 4 classes in the paper, not 5.**
Registered 640×480 FLIR
pairs; thermal is single-channel. Labels are shared across modalities.

A 9-pair sample sits at `dataset/yolo_rgbt/` on the dev machine with `train == val` — it is
a **local smoke fixture, not a split**.

**`dataset/` is never tracked in git, no exceptions.** The remote is public and these are
unpublished research pairs. The blanket ignore has a consequence that has to be designed
around rather than discovered: **a fresh clone — including the server's — has no images at
all.**

### Smoke-fixture discipline

This is the answer to the code-here / train-there friction, and it is load-bearing.

Because no image can be committed, the smoke suite is built on **synthetic pairs generated
at test time** via `tmp_path_factory` (already the house convention, §13): a handful of
random 640×480 arrays written as JPEGs plus hand-written YOLO txt labels, laid out in the
same `{split}/{visible,infrared}/{images,labels}` structure with a matching `data.yaml`.
One session-scoped fixture builds it; every test consumes it.

This is strictly better than depending on committed images, and not only for the privacy
reason:

- Tests that need a *specific* pathology — an out-of-range class id, an unpairable
  filename, a degenerate box below `_MIN_BOX_SIDE`, a genuinely disjoint `train`/`val` —
  can construct it exactly, instead of hoping the 9 real pairs happen to contain it.
- The suite runs identically on a bare clone, on the server, and in CI.

Any test that wants the *real* pairs (visual spot-checks, a sanity run against genuine
thermal statistics) must `skipif` the directory is absent, and must never be the only
coverage of a code path.

Every component must run end-to-end on the synthetic fixture, on CPU, in seconds, as a
pytest. `experiments/smoke.yaml` mirrors every real config at tiny scale. Nothing is pushed
to the server without the smoke suite passing.

The suite needs a tiny CPU stand-in translator implementing the same interface, so the
data → coupling → export → eval path stays locally testable in seconds. **That stand-in is
part of the design, not a test fixture afterthought.** (The original reason was that
`Pix2Pix_Turbo` hardcodes `.cuda()` and cannot be imported on the Mac at all. M2a's wrapper
owns device placement, so the turbo backbone *does* now run on CPU — but a 1.3B-parameter
model is not what the data-path tests should be paying for.)

### Public datasets

| Access | Datasets |
| --- | --- |
| Trivially scriptable (`git clone --depth 1` / `curl`) | MSRS, CPLID, HIT-UAV, FLIR-aligned (HuggingFace mirror `UserNae3/FLIR_aligned` — avoids the Teledyne registration form) |
| Google Drive, needs `gdown` + a human first time | LLVIP, M3FD, TTPLA |
| Manual/browser | InsPLAD (Mendeley), Yetgin & Gerek (Mendeley) |

~~For the Drive-hosted three: fetch once on the Mac, re-host, then the server script is a
plain `curl`.~~ **Superseded (2026-08-23, TASKS.md M0.9):** `scripts/fetch_datasets.py` ran
directly on the server against all three, so there is no intermediate artifact and no
re-hosting decision to make. The registry is the delivery mechanism on both machines.

**Correction to `RESEARCH_FINDINGS.md` §5:** Yetgin & Gerek is **4,000 IR + 4,000 VL at
128×128, unpaired/unregistered different scenes**, with binary presence/absence labels
(wire masks are a separate deposit) — not "400 IR + 400 VL with wire masks". At 128×128 and
unpaired it is unusable as translation data. Cite-as-motivation only (which the doc already
concludes) but fix the numbers in the paper.

Still unverified: whether MSRS's `detection/` folder has box annotations usable for mAP;
InsPLAD's annotation format.

---

## 10. Phases

Each phase produces a usable result even if the next fails.

### Phase 0 — Instrument

Harness only, no research claims. Port the Clean-SeAFusion pieces; build config / data /
metrics / tracking; stand up the CPU stand-in translator and the smoke suite; write the
fidelity metrics that don't exist yet; measure actual VRAM on the server before committing
to batch size and resolution; recalibrate the λ_det scale for a detection loss.

Establishes the E1 reference bracket using the detector weights already on the server.

### Phase 1 — GAN loop (the go/no-go)

**pix2pix**, not CUT. pix2pix is the cheapest seam of all backbones — `translate` is
literally `netG(x)`, and the detection loss slots into `backward_G()` where `fake_B` is
already un-detached.

**Gate: if translated mAP does not beat raw-thermal mAP on at least one class, stop and
re-frame before escalating to diffusion.** Cheap by construction — hours, single card.

Both sides of that comparison are §12's **zero-shot** arm — one unadapted visible-trained
detector, run on raw thermal (E1's 0.1887 floor) and on translated images. The adapted arm
answers a different question and lands near 0.9 either way, so reading the gate off it would
pass it for the wrong reason.

**Result: the gate passed** (TASKS.md M1), and was re-confirmed in M1.2 under an
independently-trained judge that supplied no gradient to anything: pix2pix at λ_det=0 scores
**0.7851** zero-shot mAP50 against the thermal floor's **0.1552**, +0.630, improving all four
classes. λ_det=3 reaches 0.8470, 90% of the real-visible ceiling.

The λ_det gain itself was **not** established at the time — re-scoring under the honest judge
showed the apparent monotone ramp was partly self-grading, and put the run-to-run noise floor at
0.059 mAP50, about the size of the effect being claimed. **E3 settled it** (§11, §16): at a
calibrated dose the coupled arm beats its own control by +0.0512 mAP50 at stage 3, p = 0.031.

### Phase 2a — One-step diffusion loop (primary)

pix2pix-turbo behind the translator interface. **FLIR-aligned pretrain → custom fine-tune**
(revised from LLVIP, 2026-08-23, confirmed with the user: the adapter is already written and
verified at 4129 train pairs, and FLIR-aligned is the same camera family as the custom 640×480
data where LLVIP is 1024×1280 street scenes; LLVIP stays available for an E9 corpus ablation).
Exact end-to-end detection-loss backprop, LoRA-scaled, λ_det warmed up from near-zero — at a
`grad_scale` calibrated for *this* backbone, per §8.

Data prep fits naturally: thermal → `train_A`, visible → `train_B`, plus a
`train_prompts.json` with a constant caption. **Watch the normalisation asymmetry in
`PairedDataset` — input arrives in [0,1], target in [-1,1].**

VRAM: the documented paired recipe is 512² at `train_batch_size=2`. The maintainer reports
A6000/48GB for the *unpaired* CycleGAN-turbo variant (multiple generators + discriminators)
and calls batch 8 "too high". Paired is far lighter, but this is exactly why Phase 0
measures on the actual card first.

Guardrails against collapse and reward hacking: LoRA scaling, λ_det warmup from near-zero,
fidelity floor on LPIPS, independent evaluation detector.

### Phase 2b — Multi-step diffusion comparison arm (lower priority)

LBBDM-f4 + ReFL. The seam is precisely located:
`BrownianBridgeModel.predict_x0_from_objective`
(`model/BrownianBridge/BrownianBridgeModel.py:148-160`) is already exposed differentiably
as `log_dict["x0_recon"]` out of `p_losses` — that *is* the ReFL x̂₀, free, at a random t
per batch. For the latent variant it is a latent, so it needs the grad-enabled `decode()`
from §7.

For the truncated-gradient arm, copy **AlignProp's** pattern rather than ReFL's:
`sd_pipeline.py:206-225` runs one unified loop with every UNet call wrapped in
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`, truncating with a per-step
`if i < backprop_timestep: noise_pred = noise_pred.detach()`. Cleaner than ReFL's
`no_grad`-prefix split, and it makes K a single config value.

DRaFT has **no official code**; `trl`'s `AlignPropTrainer` is the maintained equivalent.
Reference hyperparameters: K ∈ {1,5,10,30,50} with **smaller K better**, LoRA rank 8
(small) / 32 (large), lr 2e-4–4e-4. Start at K ∈ [1, 10].

Cost flags justifying the lower priority: BBDM's pretrained weights are **Baidu-only** and
effectively unavailable (issues #16, #41, #59); it depends on `pytorch_lightning` purely as
a base class for `VQModel`; its templates carry known unfixed config bugs (issue #47),
including `UNetParams.image_size` needing the *latent* size not the image size; and
`VQModel.init_from_ckpt` uses `load_state_dict(..., strict=False)`, so a wrong checkpoint
loads silently with random weights. VQGAN f4 weights come from LDM and *are* freely
downloadable. Reference point: 24GB @ batch 8 @ 256² f4.

### Phase 3 — Defend

Full baseline suite, low-annotation sweep (E8), cross-dataset generalisation (E9),
faithfulness stress tests (E10).

### Phase 4 — Harden

Multi-seed runs, significance testing, complete ablation grid.

---

## 11. Experiment matrix

E1–E10 from `RESEARCH_FINDINGS.md` §7 survive. Changes:

| Exp | Status |
| --- | --- |
| E1 reference bracket | Unchanged. Detector on {raw thermal, real visible} × {detector trained on thermal, on visible}. The server's existing weights cover most of this. |
| E2 backbone comparison | `{pix2pix, pix2pix-turbo, LBBDM-f4}` paired at λ_det=0; `{CUT}` unpaired. UNSB optional. |
| E3 core ablation | `{pix2pix, pix2pix-turbo} × {λ_det=0, λ_det>0} × seeds`. The turbo arm is the strong one, pix2pix the cheap control. **Most important experiment in the project.** Decided on §12's **zero-shot** task arm — the adapted arm saturates and cannot separate the conditions (M1). Design settled in M1.2: the λ_det=0 arm is `task_weights: [0,0,0,0]`, so both arms run 400 warm-started epochs through identical machinery and λ_det is the only difference; stage 0 is λ=0 in *both*, making the paired stage-0 difference a free within-experiment null control. **Six seeds**, because an exact sign-flip permutation test on paired runs cannot reach p < 0.05 below n=6 (2/2⁶ = 0.031) whatever the effect size. Needs an independently-trained reference detector: scoring a λ_det>0 arm with the same checkpoint that supplied its training gradient is not separable from reward hacking. **The pix2pix cell is done and it is POSITIVE** (TASKS.md M1.2 step 8): twelve runs at the calibrated `grad_scale: 0.15`, stage-3 zero-shot mAP50 **+0.0512, p = 0.031, CI [+0.025, +0.081]** — p exactly at the design's 2/2⁶ floor, so all six seeds agreed. Corroborated by a monotone dose-response (+0.028 → +0.036 → +0.051) that was absent at 1/15th the dose, and by raw detection loss falling ~30% where it previously did not move. §16's causality criterion is **satisfied for pix2pix**, with two caveats: the stage-0 null drew wide (−0.0397, resolved by the within-arm trajectory contrast of +0.0909 — a post-hoc sensitivity analysis, not the endpoint), and fidelity is no longer neutral (+0.0097 LPIPS, CI excluding zero at p = 0.125). The earlier campaign at `grad_scale: 1.0e-2` came back null (+0.0070, p = 0.66) and is reported beside this one: it was **dose-limited** at 2.3% of the objective (step 7), and the pair of campaigns is itself the dose argument. |
| **E4 coupling mechanism** | **Scope reduced.** Was `{cascaded, bilevel (TarDAL), meta-feature (MetaFusion)}`. Both comparison arms are unportable — see below. Becomes `{cascaded, bilevel-reimplemented}`, meta-feature deferred. |
| **E5 gradient tractability** | **Reframed.** Was "which approximation makes backprop fit". Now: *exact* full-generator gradients through a one-step distilled model vs. *truncated* ReFL/K gradients through multi-step LBBDM. A cleaner and more publishable question. |
| E6 schedule | Unchanged. Warmup vs none; joint vs alternating; λ_det sweep. |
| E7 detector identity | Directly supported: in-loop detector is the existing optical `.pt`; the evaluation detector is retrained per-run on exported translations. |
| E8 low-annotation | At 850 pairs this is likely the **headline**, not the fallback. Build the annotation-fraction sweep into the data layer from the start. |
| E9 cross-dataset | Unchanged. |
| E10 faithfulness | Unchanged. False-object and missed-object rates vs λ_det. |

### Why E4's comparison arms must be reimplemented, not ported

**TarDAL's detection-in-the-loop edge is severed in the released code.**
`scripts/train_fd.py:171` calls `self.fuse.eval(...)`, but `Fuse.eval` is decorated
`@torch.no_grad()` (`pipeline/fuse.py:112-116`). The fused tensor arrives with
`requires_grad=False`, the subsequent `fus.detach_()` at `:173` is a no-op, and the
detection loss backprops **only into YOLOv5, never into the generator**. Neither of the two
commits that file has received touches this. If we implement "bilevel" as a comparison arm
we must use `Fuse.forward` (which keeps the graph) — and say so in the paper, or a reviewer
who knows the codebase will assume we reproduced the bug.

**MetaFusion's released repo is inference-only.** Its entire contents are `README.md`,
`environment.yml`, `models/metafusion_net.py`, `test.py`, `utils/dataloader.py`, and a
weights file. There is no MFE module, no detector, no training script, and **no license
file at all**. The meta-learning scaffolding *is* present (`MetaModule.update_params` is a
graph-preserving functional SGD step — the canonical MAML inner loop), but the code that
calls it is not shipped. Deferring the meta-feature arm is the honest call; revisit only if
E4 shows gradient conflict the cascaded and bilevel arms cannot resolve.

---

## 12. Metrics

**Fidelity:** PSNR, SSIM, LPIPS, FID, KID — reported, but explicitly argued as
insufficient. PSNR/SSIM reward blur; FID/KID use ImageNet backbones insensitive to
domain-specific structure and are unreliable on small sets.

Scored on the **exported images**, not the translator's float output, so fidelity and the
task metric describe the same artifact. Their job in this project is not to carry an argument
alone but to sit beside the task metric as the reward-hacking check: detection rising while
LPIPS/FID fall is the signature §8's guardrails exist to catch, and neither number diagnoses
it by itself.

**Task:** mAP@50, mAP@50:95, per-class AP — reported in **two arms**, which answer different
questions and must never be conflated:

- **Zero-shot** (`detector.reference`): a fixed, visible-trained detector, never fine-tuned on
  anything this project produced, run straight at the translated images. This is the arm the
  Phase 1 gate and E3 are decided on, because it is the only one directly comparable to E1's
  raw-thermal floor (0.1887 mAP@50) and the only one that does not presuppose thermal-domain
  annotations — the very thing E8 exists to avoid depending on.
- **Adapted** (`detector.evaluation`): the evaluation detector fine-tuned on each stage's
  translated export, then measured. Still the right number for E7 and for "how good can a
  detector get on these images", but it **saturates**: M0.10's E1 bracket put a
  same-domain-trained detector above 0.9 mAP@50 on raw thermal too, and M1's two runs landed
  every arm in 0.8984–0.9199 — a band narrower than the noise between two runs of the *same*
  configuration. It cannot separate methods on this dataset.

**Faithfulness (C2):** false-object rate, missed-object rate, detection-consistency between
translated and real-visible, and an adapted Hallucination Index.

**Rigor:** ≥3 seeds, mean ± std, paired t-test or bootstrap CIs on mAP.

---

## 13. House style

Match the mature repos (`../Clean-SeAFusion`, `../RGBT-Fusion-Detection`), not the older
`PYTHON_CODING_GUIDELINES.md`:

- `src/` layout; Python ≥3.12; `from __future__ import annotations` in every module.
- Fully annotated. `@dataclass(frozen=True, slots=True)` for value objects; `TypedDict` for
  batch dicts; `Protocol` for pluggable hooks; `StrEnum` for choice-typed config.
- pyright `standard`; ruff line-length 100.
- Module docstrings explaining *why*, citing upstream `file:line` for every deviation.
- **An inline comment on every config field naming its failure mode.**
- `logging.getLogger(__name__)` with **%-style lazy formatting — never f-strings, never
  `print`**. `basicConfig` only in `cli.py`.
- Custom exception subclasses; fail fast at startup with messages saying what was tried.
- argparse CLI with an explicit flag→config-path override table.
- pytest, `tmp_path_factory` synthetic datasets, a `slow` marker for CPU end-to-end runs.
- gitmoji + conventional commits.

**Deliberate deviation:** pydantic replaces the hand-rolled YAML→dataclass `_build`/
`_coerce` loader in `../Clean-SeAFusion/src/seafusion/config.py`. Keep that file's two best
behaviours — **unknown-key rejection** (pydantic `extra="forbid"`) and **`snapshot()`** of
the resolved config into the run dir.

**`config_hash()` covers the experiment, not the invocation.** The `runtime` section
(device, run name, run dir, W&B flags) is excluded wholly, so the same experiment run on
two GPUs under two names carries one hash — which is what lets invariant 6 mean anything.
`seed` therefore lives under `train`, not `runtime`: it is scientific, and a
silently-changed seed on resume is precisely the drift M0.8's warn-on-drift check exists to
catch. Clean-SeAFusion hashed everything (`engine/fusion_trainer.py:60`) and consequently
warned on a renamed run.

---

## 14. Dev-on-Mac / train-on-server workflow

- Remote: `github.com/atick-faisal/Thermal-To-Optical-Translation` (public). Note that the
  research design in this file and in `RESEARCH_FINDINGS.md` is therefore public.
- Port the `../RGBT-Fusion-Detection` cpu/gpu extras so `uv sync --extra cpu` works here
  and `uv sync --extra gpu` works on the server, from one lockfile:
  ```toml
  [project.optional-dependencies]
  cpu = ["torch>=...", "torchvision>=..."]
  gpu = ["torch>=...", "torchvision>=..."]
  [tool.uv]
  conflicts = [[{ extra = "cpu" }, { extra = "gpu" }]]
  ```
- Pre-push gate: `ruff`, `pyright`, and the smoke suite. Nothing else is verifiable locally.
- ~~Attempt one `uv sync` with all dependencies first — the conflict may not exist.~~
  **Resolved in M0.1: the conflict does not exist.** One `uv lock` resolved torch 2.13
  (`+cpu` and `+cu130`), diffusers 0.39, transformers 5.15, peft 0.20, ultralytics 8.4.117,
  torchmetrics 1.9 and pycocotools together. §7's thesis holds — the maze belongs to
  img2img-turbo's pinned framework, not to the model code we vendor. Note that
  uv dependency groups resolve into a *single* lockfile and venv.
- W&B self-hosted; base URL already in the server environment. Key from env, never
  committed.

---

## 15. Risks

| Risk | Signal | Mitigation |
| --- | --- | --- |
| ~~Phase 1 fails — translation never beats raw thermal~~ | E1 vs E3 at λ_det=0 | **Retired.** Measured: 0.7751 vs 0.1887 zero-shot mAP50, all four classes improved (TASKS.md M1). |
| 850 pairs too few even for LoRA fine-tuning | Turbo overfits during Phase 2a | FLIR-aligned pretrain is already in the plan (§10); escalate to heavier augmentation and lower LoRA rank. |
| **Reward hacking — mAP rises, images degrade** | **Fired, at low amplitude.** E3's calibrated campaign moves stage-3 mAP50 +0.0512 *and* LPIPS +0.0097 (M1.2 step 8 finding 9) | LPIPS alone cannot separate a fidelity trade from hallucination — `t2o faithfulness` (M1.2 step 9) counts invented and erased objects on the finished exports and decides it. If false objects rise with λ, `reward_target` becomes live and the turbo campaign waits. |
| **λ_det miscalibrated for a new backbone** | Detection share outside 20–30% on `scripts/loss_share.py` | §8's per-backbone probe, 25 epochs, before any campaign. Skipping it once already cost 72 GPU-hours and an uninterpretable null. |
| Gradient conflict — training unstable | Loss oscillation, collapse | Escalate cascaded → bilevel (E4). Meta-feature only if both fail. |
| VRAM tighter than expected | Phase 0 OOM | SDPA + gradient checkpointing + bf16; reduce batch, then resolution. |
| DataLoader hangs on Windows spawn | Phase 0, silent stalls | Module-level dataset classes, guarded entry points, low `num_workers` until stable. |
| sd-turbo download blocked on the server | Phase 2a setup | Fetch on the Mac, commit-adjacent cache or re-host. |
| Direct thermal detection wins at full annotation | E8 at 100% | Expected outcome — pivot to E8's low-label regime as headline. |

---

## 16. Acceptance criteria for drafting

Unchanged from `RESEARCH_FINDINGS.md` §10. Begin drafting when all five hold: margin
(≥ +2–4 mAP@50 over the strongest baseline), consistency (≥2–3 datasets), causality (E3
shows the loop drives the gain, seed-stable), stability (≥3 seeds, significance-tested, no
collapse), and faithfulness (hallucination rates low and reported).

**Status after M1.2 step 8 (2026-08-23): causality is SATISFIED for the pix2pix backbone.**
E3's twelve-run campaign at the calibrated `grad_scale: 0.15` puts stage-3 zero-shot mAP50 at
**+0.0512, p = 0.031, CI [+0.025, +0.081]** — the exact sign-flip floor at n=6, meaning all six
seeds moved the same way. Three things carry the claim beyond the p-value, which sits at a floor
it cannot go below: a **monotone dose-response** (+0.028 → +0.036 → +0.051) that was absent at
1/15th the dose; raw detection loss falling ~30%, outside the measured loss-space noise floor,
where at `1.0e-2` it did not move at all; and an **independent judge** (M1.2 step 1's `yolo11s`)
that supplied no gradient to anything.

*Stability* was met in full (six seeds, exact sign-flip, no collapse). Two caveats travel with
the result and must be reported:

1. **The stage-0 null drew wide** — −0.0397, against a stage-3 effect of +0.0512, i.e. 1.3× by
   magnitude where the pre-registered rule asks for "clearly larger". Stage 0 is provably
   λ-inert in both arms, so this is an unlucky draw rather than a confound. It is resolved by
   the within-arm trajectory — control gains +0.0396 over the 400-epoch budget, loop gains
   +0.1305, a difference-of-differences of **+0.0909** — which the stage-0 offset cannot touch.
   That test was **added after seeing the data** and is a sensitivity analysis, not the endpoint.
2. **Fidelity is no longer neutral**: +0.0097 LPIPS at stage 3 (CI [+.002, +.017], sign-flip
   p = 0.125). Both arms improve; the coupled arm improves about a third less. Detection up
   with fidelity down is §8's reward-hacking signature, and `t2o faithfulness` (TASKS.md M1.2
   step 9) exists to say which it is. **Until that runs, C1's gain is established and its cost
   is characterised only by LPIPS.**

The superseded campaign at `grad_scale: 1.0e-2` (+0.0070, p = 0.66) stays in the record: step 7
measured its dose at 2.3% of the objective, so it never tested coupling at a dose capable of
refuting it. The two campaigns together are the dose argument, and neither is publishable alone.

**Margin and consistency remain untouched.** E3 compares the method to its own ablation, not to
baselines; nothing here says how the method fares against E2's backbones, and the result stands
on one dataset. Those are the next two criteria, after the turbo arm replicates C1.

**Fallback framing:** if the loop helps only in low-annotation regimes, that remains a
strong honest Q1 story — pivot to data-efficiency and operator interpretability. Given 850
pairs, treat this as the *likely* outcome rather than the fallback.

**But the fallback is not a substitute for causality, and cannot be reached by lowering
`annotation_fraction` in E3.** That knob gates only the batch's `cls`/`bboxes`, i.e. only the
λ_det>0 arm's own supervision — the λ_det=0 control never reads annotations at all — so
reducing it makes the two arms *more* alike, not less. E8 is a question about the translator's
data efficiency; it answers something different from C1.
