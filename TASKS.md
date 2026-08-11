# Tasks

Working checklist for [PLAN.md](PLAN.md). Per [AGENTS.md](AGENTS.md): implement one step →
verify with tests → mark done → gitmoji commit → stop.

Phase 0 and Phase 1 are broken down to actionable items. Phases 2–4 stay coarse until the
Phase 1 go/no-go gate resolves — detailing experiments that may never run is waste.

**Every item must be verifiable on the synthetic smoke fixture, on CPU, in seconds.** If an
item cannot be, say so explicitly in its checklist entry and note what server-side check
replaces it.

**`dataset/` is never tracked** — the remote is public and the pairs are unpublished. A
fresh clone therefore has no images, so tests generate synthetic pairs with
`tmp_path_factory` (PLAN.md §9). Any test wanting the real local 9 pairs must `skipif` the
directory is absent and must never be the sole coverage of a code path.

---

## M0.1 — Repo hygiene and packaging ✅

- [x] Rewrite `.gitignore`. It was template carry-over and fought the design:
  - [x] Remove bare `experiments` (`:171`) and bare `test` (`:168`) — they match at any
        depth and would silently swallow tracked directories
  - [x] Remove `PLAN.md` / `TASKS.md` (`:237-238`) — both are tracked in
        `../Clean-SeAFusion` and `../RGBT-Fusion-Detection`, and AGENTS.md tells every
        session to read them. Ignoring them means they never reach the server.
  - [x] Prune unrelated leftovers: frontend rules, `backend/storage`, `app/models`, certs,
        `lib/`, `node_modules`
  - [x] Add `runs/`, `third_party/**/checkpoints`, `*.pt` outside the fixture
  - [x] Ignore `dataset/` wholesale — **no exceptions, not even the 9-pair fixture**. The
        remote is public and the pairs are unpublished research data. Also `*.cache` for
        ultralytics' derived `labels.cache`.
- [x] Verify with `git check-ignore` on: `experiments/`, `tests/`, `runs/`, `PLAN.md`,
      `TASKS.md`, `src/t2o/`, `dataset/`, `*.cache`, `third_party/**/checkpoints/`
      — 17-case matrix, all pass
- [x] Configure a git remote and push. The dev→server workflow depends on it.
      → `github.com/atick-faisal/Thermal-To-Optical-Translation` (public)
- [x] Rename the package `thermal_to_optical_translation` → **`t2o`**, matching PLAN.md §5
- [x] Port cpu/gpu extras + `[tool.uv] conflicts` + PyTorch index pins from
      `../RGBT-Fusion-Detection/pyproject.toml`. Build backend switched `uv_build` →
      `hatchling` to match both sibling repos.
- [x] Pin `ultralytics>=8.4.108,<8.5` — the `Detect` head output format changed between
      8.3 and 8.4 and code written against one does not work against the other
- [x] Add deps: `pydantic`, `pyyaml`, `diffusers`, `transformers`, `peft`, `lpips`,
      `torchmetrics[image]`, `pycocotools`, `pillow`, `numpy`
- [x] Fill in `pyproject.toml` `description`; replace the `main()` hello-world entry point
      with `src/t2o/cli.py` (`--version` only; subcommands land in M0.8)
- [x] Add `[tool.ruff]` (line-length 100, select `E,F,W,I,UP,B,SIM,C4,RUF`),
      `[tool.pyright]` (`typeCheckingMode = "standard"`, `venvPath = "."`),
      `[tool.pytest.ini_options]` (`testpaths`, `markers = ["slow: ..."]`)
- [x] Add `.pre-commit-config.yaml` for `prek`: ruff check, ruff format, pyright, fast tests
- [x] Write `README.md` (was empty)
- [x] Verify: `uv sync --extra cpu`, `ruff check`, `ruff format --check`, `pyright`,
      `pytest -m "not slow"`, `t2o --version` all clean

**Resolved:** the PLAN.md §7 dependency maze **did not materialise**. One `uv lock` resolved
the whole stack for both extras — `torch 2.13.0+cpu` / `2.13.0+cu130`, `diffusers 0.39.0`,
`transformers 5.15.0`, `peft 0.20.0`, `ultralytics 8.4.117`, `torchmetrics 1.9.0`,
`pycocotools 2.0.11` (macOS arm64 wheel, no compiler needed). All import cleanly on CPU.
This confirms §7's thesis: the maze is a property of img2img-turbo's pinned *framework*,
not of the model code we intend to vendor.

**Note:** `transformers` resolved to **5.x**, well past the `>=4.45` floor and two majors
past the `4.35.2` upstream img2img-turbo pins. `diffusers 0.39.0` accepts it and imports
fine, but the vendored `Pix2Pix_Turbo` uses `CLIPTextModel`/`AutoTokenizer` — verify those
call sites against the 5.x API during M2a rather than assuming.

**Deviation:** `[tool.ruff] extend-exclude = ["*.md"]`. ruff formats fenced Python inside
markdown, and PLAN.md quotes upstream source verbatim with `file:line` citations —
reformatting those silently breaks the citations.

**Not verifiable locally:** `uv sync --extra gpu`. `uv lock` resolves the CUDA path, which
is as far as the Mac can go; the actual install is the first M0.10 check.

## M0.2 — Config layer ✅

- [x] `config/` pydantic v2 models with `extra="forbid"` (unknown-key rejection) and
      frozen models. Replaces the hand-rolled `_build`/`_coerce` in
      `../Clean-SeAFusion/src/seafusion/config.py` — ~70 lines of reflection deleted.
      Three modules: `base.py` (`ConfigBase`, `ConfigError`, `deep_merge`,
      `as_config_error`), `schema.py` (sections + root `Config`), `__init__.py`
- [x] **Inline comment on every field naming its failure mode** — house convention
- [x] `config_hash()` — sha256 over sorted-JSON of the resolved config, first 16 chars
- [x] `snapshot()` — write the resolved config into the run dir
- [x] `ConfigError` subclass; per-field error locations
- [x] Port `imaging.py` `Normalize` from Clean-SeAFusion — `export.normalize` needs it now,
      and M0.8 needs `to_uint8` anyway
- [x] Tests (`tests/test_config.py`, 40 cases): unknown key rejected; bad value rejected
      with a useful message; hash stable across reorderings; hash changes when a value
      changes. No fixture data needed — this is the one milestone independent of M0.3's
      synthetic dataset
- [x] Verify: `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"` clean

**Sections:** `data` (manifest + the E8 `annotation_fraction`/`annotation_seed` hook),
`train`, `loss` (λ_l2/λ_lpips/λ_gan), `coupling` (λ_det ramp, `grad_scale`,
`reward_target`), `detector`, `translator`, `export`, `runtime`.

**Decision — `config_hash()` excludes `runtime` wholly.** Device, run name, run dir and
W&B flags are invocation details, not experiment identity: the same experiment on two GPUs
under two names must carry the same hash. `seed` therefore moved out of `runtime` (where
Clean-SeAFusion put it) into `train` — a seed *is* scientific, and a silently-changed seed
on resume is exactly the drift the check exists to catch. Diverges from
`../Clean-SeAFusion/src/seafusion/engine/fusion_trainer.py:60`, which hashed everything and
so warned on a renamed run.

**Decision — `detector` splits into `in_loop` and `evaluation` sub-sections.** PLAN.md
invariant 7 ("two detectors, never conflated") encoded structurally rather than by
convention: there is no single `detector.weights` for the two to accidentally share.

**Decision — `translator` is a pydantic discriminated union on `backbone`,** currently of
one member (`stub`). `Backbone` gains `PIX2PIX` at M1, `PIX2PIX_TURBO` at M2a, `LBBDM` at
M2b, each adding one model plus one union entry and touching nothing else — which is what
makes invariant 3 true rather than aspirational. Consequence worth knowing: omitting
`translator` entirely keeps the default, but a *partial* `translator:` block must still
name its `backbone`. Deliberate — which backbone is running is never implicit.

**Deferred:** experiment inheritance (a `base:` include key). `experiments/smoke.yaml` will
duplicate fields, but with zero experiment YAMLs on disk this is speculative; revisit at
M0.8 with ≥3 files to compare.

## M0.3 — Data layer ✅

- [x] Port `data/manifest.py` from Clean-SeAFusion — `data.yaml` as single source of truth,
      root resolution, `nc` vs `names` validation, `rgbt:` token block extraction
- [x] Port `data/pairing.py` — **path-segment** substitution, never sorted-index; eager
      `validate_pairs`
- [x] Port `data/labels.py` — YOLO txt loading
- [x] Port `data/dataset.py` → translation semantics. Keep verbatim: bbox crop/flip math,
      the `_MIN_BOX_SIDE` drop rule, the class-id range precheck (an out-of-range id
      otherwise surfaces as an opaque CUDA device-side assert deep into a run), and the
      `v8DetectionLoss`-shaped collate. Renamed `FusionSample`/`FusionBatch`/
      `FusionPairDataset`/`collate_fusion_batch` → `Translation*`/`collate_translation_batch`;
      field names (`visible`, `infrared`, `cls`, `bboxes`, `name`) kept as-is since they
      already match the `rgbt:` token names
- [x] Convert the local `dataset/yolo_rgbt/data.yaml`: drop `channels: 6`, keep the `rgbt:`
      block. Untracked, so documented the change in `manifest.py`'s module docstring
      instead of relying on the file reaching anyone else
- [x] **`tests/conftest.py` synthetic dataset builder** (`tmp_path_factory`, session-scoped)
      — random 640×480 JPEGs + hand-written YOLO labels in the
      `{split}/{visible,infrared}/{images,labels}` layout with a generated `data.yaml`.
      **This replaces the committed fixture entirely** (PLAN.md §9): `dataset/` is never
      tracked, so a fresh clone has no images. Train/val use disjoint, non-alphabetically-
      ordered stems — the real local sample has `train == val`, which hides bugs
- [x] Annotation-fraction subsampling hook (deterministic, seeded) — E8 needs it and
      retrofitting later is painful
- [x] Tests: pairing rejects ambiguous paths; class-id precheck fires; crop/flip bbox math;
      collate shapes match the `v8DetectionLoss` contract. Build each pathology explicitly
      in the synthetic fixture rather than hoping real data contains one

**Decision — `annotation_fraction` zeroes labels, not images.** Every image still trains
the translator's reconstruction losses (L2/LPIPS/GAN), since the visible target is always
present; the fraction only controls which images additionally supply detection supervision
via a seeded shuffle-and-slice (`TranslationPairDataset._annotated_subset`). Dataset length
is unaffected by the fraction — this keeps the hook orthogonal to everything else in the
class.

**Deviation from Clean-SeAFusion's own test fixture:** its `tests/conftest.py` reuses the
same filenames for `train` and `val` (only pixel content differs). Ours does not — disjoint
stems are what let a test actually catch a train/val conflation bug, which is exactly the
blind spot `TASKS.md` flagged in the real local 9-pair sample.

**Verify:** `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"`
(86 passed, includes a `skipif`-guarded sanity check against the real local dataset).

## M0.4 — Detection layer ✅

- [x] Port `detection/frozen.py` — `FrozenDetector` verbatim. Keep the `train()` override
      (an enclosing trainer's `model.train()` would otherwise flip BN back), the
      `_normalize_args` checkpoint repair, and stride-divisibility validation
- [x] `detection/evaluation.py` — the independently-trained evaluation detector. **Never
      shares weights or gradients with the in-loop detector** (invariant 7)
- [x] Port `engine/detector_stage.py` — keep `wandb_integration_disabled()` (ultralytics
      otherwise adopts the open run and calls `wb.run.finish()`, killing it) and the
      `_resolve_weights()` `last.pt` fallback (no `best.pt` is written if fitness was ever
      NaN)
- [x] Tests: frozen detector stays in eval after `.train()`; no grads accumulate on its
      params; grads *do* reach the input image; stride validation rejects bad sizes

**Verified against the installed `ultralytics 8.4.117`, not assumed:** `unwrap_model`
exists (renamed from `de_parallel` at ~8.4.112, and this project's floor is 8.4.108, so this
was a real thing to check); a freshly-built `DetectionModel` in eval mode returns
`(y, preds)` with `preds = {"boxes", "scores", "feats"}`; loading a manually-saved
checkpoint via `YOLO(path).model.args` really does come back as a plain `dict`, confirming
`_normalize_args`'s dict-repair branch is live rather than dead code.

**Decision — `train_detector` reads `config.detector.evaluation.*`, not flat
`config.detector.*`.** Clean-SeAFusion's config is flat; t2o's `DetectorConfig` was already
split into `in_loop`/`evaluation` at M0.2 to encode invariant 7 structurally.
`train_detector` always trains the *evaluation* detector — the in-loop one is never
trained, by construction. Also reads `seed` from `config.train.seed`, not
`config.runtime.seed` (the M0.2 seed-placement decision).

**Decision — `detection/evaluation.py` is new, not a port.** Clean-SeAFusion has no
equivalent: it alternates a single detector between frozen and trainable roles across
stages. t2o splits the roles into two structurally distinct types instead, so
`EvaluationDetector` is a minimal frozen dataclass (`weights: Path` + a `.load()`
convenience) with deliberately no base class or code in common with `FrozenDetector` —
checked directly in `tests/test_evaluation_detector.py` (disjoint MRO).

**Not yet exercised:** an actual end-to-end `train_detector(...)` call against ultralytics'
real training loop. That belongs with M0.8's full engine smoke test (already `slow`-marked
territory). `DetectionTaskLoss`/`v8DetectionLoss` wiring is M0.7; reusing the metrics-
extraction pattern for a standalone `model.val()` path is M0.5.

**Verify:** `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"`
(104 passed).

## M0.5 — Metrics (all from scratch — none exist in any of our repos)

Proceeds as **three separate steps/commits** (fidelity → task → faithfulness) rather than
one, per AGENTS.md's one-step → verify → commit → stop working order — the three families
are independent enough that bundling them into one commit would just be batching.

- [x] `metrics/fidelity.py` — PSNR, SSIM, LPIPS, FID, KID, all via `torchmetrics.image`
      (not a raw-`lpips` + torchmetrics mix — verified `LearnedPerceptualImagePatchSimilarity`/
      `FrechetInceptionDistance`/`KernelInceptionDistance` all accept `[0, 1]` float input
      directly via `normalize=True`, matching `TranslationSample`'s native convention with
      no manual uint8 detour). `FidelityEvaluator` wraps all five behind one
      `update`/`compute`/`reset` — the one evaluation path for fidelity.
- [ ] `metrics/task.py` — mAP@50, mAP@50:95, per-class AP via ultralytics `DetMetrics`,
      with the defensive `results_dict`-then-`box`-attribute extraction from
      `../Clean-SeAFusion/src/seafusion/engine/detector_stage.py::_extract_metrics`
- [ ] `metrics/faithfulness.py` (C2) — false-object rate, missed-object rate,
      detection-consistency translated vs real-visible. **Hallucination Index deferred**
      (see note below).
- [ ] **One evaluation path** (invariant 1): every method computes every metric through
      this code. No per-method variants.
- [x] Tests (fidelity): identical images → PSNR ∞ / SSIM 1 / LPIPS 0; known-shifted images
      give expected PSNR/SSIM ordering; FID/KID finite and non-negative on unrelated
      pools; `MetricsConfig` shape and hash participation. Split along a network boundary
      rather than a metric boundary — `FidelityEvaluator.__init__` always builds the
      pretrained AlexNet/Inception backbones, so every test that constructs it is `slow`;
      the fast suite covers only `MetricsConfig` and the KID subset-size clamp (a pure
      function). The one-time weight download succeeded in the dev sandbox, so this needed
      no "verify on server" carve-out.
- [ ] Tests (task/faithfulness): mAP on a known synthetic prediction set; faithfulness
      metrics on synthetic detections with known answers.

**Decision — Hallucination Index deferred.** The MICCAI 2024 metric it's adapted from
(arXiv:2407.12780) is a Hellinger distance between the *distribution* of a generative
model's reconstructions (mean/variance/noise-power-spectrum across many repeated samples
per input) and a zero-hallucination reference — it assumes a stochastic model sampled N
times per input. Our translators (pix2pix, pix2pix-turbo one-step, LBBDM without repeated
sampling) are deterministic per checkpoint, so the literal metric doesn't apply. Revisit
once a stochastic backbone (diffusion, Phase 2) makes repeated sampling meaningful; until
then `faithfulness.py` ships the other three C2 metrics only.

**Decision — `MetricsConfig` added to `config/schema.py`.** `lpips_net` and
`kid_subset_size` are experiment identity (they change the reported number), not
implementation detail, so they're config fields and participate in `config_hash()` like
everything else in `RESEARCH_FINDINGS.md`'s reported table. Faithfulness's IoU/confidence
thresholds will extend the same section in the next step rather than starting a new one.

## M0.6 — Translator interface and CPU stand-in

- [ ] Define the `Translator` protocol: `fit()` and `translate(batch) -> Tensor`
- [ ] `translators/stub.py` — tiny CPU stand-in (e.g. a 2-layer conv) implementing it.
      **Load-bearing, not a test fixture**: `Pix2Pix_Turbo` hardcodes `.cuda()` and cannot
      be imported on the Mac, so this is the only way the full path stays locally testable
- [ ] Tests: stand-in round-trips through the full data → coupling → export → eval path

## M0.7 — Coupling

- [ ] Port `coupling/detection_loss.py` from `../Clean-SeAFusion/src/seafusion/losses/task.py`.
      Drop the YCbCr recombination (our translator emits RGB directly). **Keep the
      batch-size division** — ultralytics returns the loss pre-multiplied by batch size, and
      not undoing it makes λ silently scale with batch size
- [ ] Feed float32 `[0,1]` directly; **never** route generated images through
      `preprocess_batch` (it does `.float()/255` on a uint8 dataloader tensor, creating a
      fresh graph root)
- [ ] Saturating reward form (ReFL-style hinge / AlignProp-style `|r - target|`) rather
      than unbounded minimisation
- [ ] `coupling/schedule.py` — staged λ_det ramp, `λ_det = 0` a clean no-op that never
      constructs the detector
- [ ] Recalibrate the λ_det scale for a *detection* loss — the `[0,1,2,3]` ramp is
      calibrated to SeAFusion's segmentation loss. Reference downscales: ReFL `1e-3`,
      AlignProp `0.01`
- [ ] Tests: λ=0 constructs no detector (assert it); loss is differentiable w.r.t. the
      generated image; λ invariant to batch size

## M0.8 — Engine and tracking

- [ ] Port `tracking.py` — `RunTracker`, rank-0 only, never raises, context manager
- [ ] Point W&B at the self-hosted base URL from env; key from env, never committed
- [ ] Port `engine/trainer.py` — bf16/fp16 GradScaler logic, autocast restricted to CUDA,
      fully resumable checkpoints (model + optimizer + scheduler + scaler + epoch +
      `config_hash` + config), warn-on-config-drift resume
- [ ] Port `engine/loop.py` — staged alternating loop. **Warm-start the translator across
      stages**; original SeAFusion re-instantiates its generator every stage
      (`train.py:203`), making the loop one-directional
- [ ] Port `engine/export.py` + `imaging.py` `Normalize` enum — per-image (never
      batch-wise) clamp/stretch
- [ ] `cli.py` — argparse subcommands with an explicit flag→config-path override table
- [ ] `experiments/smoke.yaml` mirroring every real config at tiny scale
- [ ] Tests (`slow` marker): full end-to-end on the fixture with the stand-in translator →
      `metrics.json`; resume produces identical state; same config+seed twice → identical
      hash and metrics

## M0.9 — Dataset acquisition

- [ ] `scripts/fetch_datasets.py` for the trivially-scriptable set: MSRS, CPLID, HIT-UAV,
      FLIR-aligned (HuggingFace mirror `UserNae3/FLIR_aligned` — avoids the Teledyne
      registration form)
- [ ] Fetch LLVIP, M3FD, TTPLA once on the Mac via `gdown`, re-host, then make the server
      path a plain `curl`
- [ ] Adapters normalising each into the internal representation
- [ ] Verify: MSRS `detection/` folder — does it have box annotations usable for mAP?
- [ ] Verify: InsPLAD annotation format (not stated in its README)
- [ ] Freeze and hash the splits; commit the manifest

## M0.10 — Server bring-up (cannot be verified locally)

- [ ] `uv sync --extra gpu` on the server
- [ ] Confirm dataloader does not hang under Windows spawn — start `num_workers=0`, raise
      slowly
- [ ] **Measure actual VRAM** at candidate batch sizes and resolutions before committing to
      any long run
- [ ] E1 reference bracket: detector on {raw thermal, real visible} × {detector trained on
      thermal, on visible}, using the existing `.pt` weights
- [ ] Confirm W&B self-hosted logging works end to end

---

## M1 — Phase 1: GAN loop (the go/no-go gate)

- [ ] Vendor `pytorch-CycleGAN-and-pix2pix` at `2a7afba` — **`models/networks.py` only**
      (`define_G`, `define_D`, `GANLoss`). Not `BaseModel`, `options/`, `data/`, `train.py`
- [ ] `translators/pix2pix.py` wrapper implementing the Translator protocol.
      `translate` is `netG(x)`
- [ ] Wire the detection loss where `fake_B` is already un-detached
- [ ] Smoke-test the full loop on the fixture
- [ ] Train on the custom dataset, λ_det = 0 → baseline translation quality
- [ ] Train with λ_det > 0 → the loop arm
- [ ] Evaluate both through the single evaluation path

**GATE:** if translated mAP does not beat raw-thermal mAP on at least one class, **stop**.
Re-frame around the low-annotation regime before escalating to diffusion.

- [ ] Record the gate decision and evidence in this file before proceeding

---

## M2 — Phase 2: Diffusion loop

Detail this section once the M1 gate passes.

### M2a — pix2pix-turbo (primary)

- [ ] Vendor `src/pix2pix_turbo.py` + `src/model.py` helpers at `463b2d3`; modernise
      against *current* diffusers rather than inheriting the 2023-era pinned stack
- [ ] Wrapper owns device placement (upstream hardcodes `.cuda()`)
- [ ] Data prep: thermal → `train_A`, visible → `train_B`, constant-caption
      `train_prompts.json`. Watch the normalisation asymmetry — input [0,1], target [-1,1]
- [ ] LLVIP pretrain → custom fine-tune
- [ ] Exact end-to-end detection-loss backprop, LoRA-scaled, λ_det warmed from near-zero
- [ ] Fidelity floor on LPIPS (`net_lpips` is already in the loss assembly)

### M2b — LBBDM-f4 + ReFL (comparison arm, lower priority)

- [ ] Vendor at `02c3b13`; grad-enabled copy of `LatentBrownianBridgeModel.decode()`
- [ ] Fix the known template config bugs (issue #47), esp. `UNetParams.image_size` needing
      the *latent* size
- [ ] VQGAN f4 weights from LDM; verify the checkpoint actually loaded
      (`init_from_ckpt` uses `strict=False` and will silently keep random weights)
- [ ] ReFL coupling via `predict_x0_from_objective` / `log_dict["x0_recon"]`
- [ ] Truncated-gradient arm using AlignProp's per-step detach inside a checkpointed loop

---

## M3 — Phase 3: Defend

- [ ] Full baseline suite (`RESEARCH_FINDINGS.md` §8)
- [ ] E8 low-annotation sweep — likely the headline at 850 pairs
- [ ] E9 cross-dataset generalisation
- [ ] E10 faithfulness stress tests
- [ ] E4 coupling comparison: cascaded vs bilevel-**reimplemented** (TarDAL's released code
      severs the generator gradient — see PLAN.md §11)

## M4 — Phase 4: Harden

- [ ] ≥3 seeds on every headline result
- [ ] Significance testing (paired t-test or bootstrap CIs on mAP)
- [ ] Complete ablation grid
- [ ] Check the five acceptance criteria (`RESEARCH_FINDINGS.md` §10)

---

## Corrections to fold into the paper

- [ ] `RESEARCH_FINDINGS.md` §5 — Yetgin & Gerek is **4,000 IR + 4,000 VL at 128×128,
      unpaired**, with binary presence/absence labels; wire masks are a separate Mendeley
      deposit. Not "400 IR + 400 VL with wire masks".
- [ ] Note in the methods section that TarDAL's public implementation does not actually
      backprop detection loss into its generator, and that our bilevel arm fixes this.
- [ ] Licensing: sd-turbo is Stability AI Non-Commercial; CUT bundles NVIDIA StyleGAN2
      (non-commercial); ultralytics is AGPL-3.0.
