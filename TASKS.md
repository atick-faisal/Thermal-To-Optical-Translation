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

## M0.2 — Config layer

- [ ] `config/` pydantic v2 models with `extra="forbid"` (unknown-key rejection) and
      frozen models. Replaces the hand-rolled `_build`/`_coerce` in
      `../Clean-SeAFusion/src/seafusion/config.py`
- [ ] **Inline comment on every field naming its failure mode** — house convention
- [ ] `config_hash()` — sha256 over sorted-JSON of the resolved config, first 16 chars
- [ ] `snapshot()` — write the resolved config into the run dir
- [ ] `ConfigError` subclass; per-field error locations
- [ ] Tests: unknown key rejected; bad value rejected with a useful message; hash stable
      across reorderings; hash changes when a value changes

## M0.3 — Data layer

- [ ] Port `data/manifest.py` from Clean-SeAFusion — `data.yaml` as single source of truth,
      root resolution, `nc` vs `names` validation, `rgbt:` token block extraction
- [ ] Port `data/pairing.py` — **path-segment** substitution, never sorted-index; eager
      `validate_pairs`
- [ ] Port `data/labels.py` — YOLO txt loading
- [ ] Port `data/dataset.py` → translation semantics. Keep verbatim: bbox crop/flip math,
      the `_MIN_BOX_SIDE` drop rule, the class-id range precheck (an out-of-range id
      otherwise surfaces as an opaque CUDA device-side assert deep into a run), and the
      `v8DetectionLoss`-shaped collate
- [ ] Convert the local `dataset/yolo_rgbt/data.yaml`: drop `channels: 6`, keep the `rgbt:`
      block. Untracked, so document the change in the manifest module rather than relying
      on the file reaching anyone else
- [ ] **`tests/conftest.py` synthetic dataset builder** (`tmp_path_factory`, session-scoped)
      — random 640×480 JPEGs + hand-written YOLO labels in the
      `{split}/{visible,infrared}/{images,labels}` layout with a generated `data.yaml`.
      **This replaces the committed fixture entirely** (PLAN.md §9): `dataset/` is never
      tracked, so a fresh clone has no images. Parameterise it to emit disjoint
      `train`/`val` — the real local sample has `train == val`, which hides bugs
- [ ] Annotation-fraction subsampling hook (deterministic, seeded) — E8 needs it and
      retrofitting later is painful
- [ ] Tests: pairing rejects ambiguous paths; class-id precheck fires; crop/flip bbox math;
      collate shapes match the `v8DetectionLoss` contract. Build each pathology explicitly
      in the synthetic fixture rather than hoping real data contains one

## M0.4 — Detection layer

- [ ] Port `detection/frozen.py` — `FrozenDetector` verbatim. Keep the `train()` override
      (an enclosing trainer's `model.train()` would otherwise flip BN back), the
      `_normalize_args` checkpoint repair, and stride-divisibility validation
- [ ] `detection/evaluation.py` — the independently-trained evaluation detector. **Never
      shares weights or gradients with the in-loop detector** (invariant 7)
- [ ] Port `engine/detector_stage.py` — keep `wandb_integration_disabled()` (ultralytics
      otherwise adopts the open run and calls `wb.run.finish()`, killing it) and the
      `_resolve_weights()` `last.pt` fallback (no `best.pt` is written if fitness was ever
      NaN)
- [ ] Tests: frozen detector stays in eval after `.train()`; no grads accumulate on its
      params; grads *do* reach the input image; stride validation rejects bad sizes

## M0.5 — Metrics (all from scratch — none exist in any of our repos)

- [ ] `metrics/fidelity.py` — PSNR, SSIM, LPIPS, FID, KID via `torchmetrics` + `lpips`
- [ ] `metrics/task.py` — mAP@50, mAP@50:95, per-class AP via ultralytics `DetMetrics`,
      with the defensive `results_dict`-then-`box`-attribute extraction from
      `../Clean-SeAFusion/src/seafusion/engine/detector_stage.py::_extract_metrics`
- [ ] `metrics/faithfulness.py` (C2) — false-object rate, missed-object rate,
      detection-consistency translated vs real-visible, adapted Hallucination Index
- [ ] **One evaluation path** (invariant 1): every method computes every metric through
      this code. No per-method variants.
- [ ] Tests: identical images → PSNR ∞ / SSIM 1 / LPIPS 0; known-shifted images give
      expected ordering; faithfulness metrics on synthetic detections with known answers

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
