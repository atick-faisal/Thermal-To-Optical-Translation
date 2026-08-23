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
- [x] `metrics/task.py` — mAP@50, mAP@50:95, per-class AP via ultralytics `DetMetrics`,
      with the defensive `results_dict`-then-`box`-attribute extraction from
      `../Clean-SeAFusion/src/seafusion/engine/detector_stage.py::_extract_metrics`.
      `_extract_metrics` **relocated** here from `engine/detector_stage.py` rather than
      duplicated — verified `Model.train()` and `Model.val()` both return the same
      `DetMetrics` type, so `train_detector` and the new `evaluate_detector` genuinely
      share one function (PLAN.md invariant 1), not two copies that happen to agree.
- [x] `metrics/faithfulness.py` (C2) — false-object rate, missed-object rate,
      detection-consistency translated vs real-visible, all built on one greedy
      confidence-ordered single-IoU-threshold matcher (`_greedy_match`). **Hallucination
      Index deferred** (see note below). **Detection-consistency's formula is a judgement
      call** (see note below) — RESEARCH_FINDINGS.md §9 names the metric without a formula.
- [x] **One evaluation path** (invariant 1): fidelity, task, and faithfulness each have
      exactly one canonical implementation (`FidelityEvaluator`, `evaluate_detector`,
      `FaithfulnessEvaluator`); nothing downstream should grow a per-method variant of any
      of the three. Holding this as future translator/coupling/engine code lands is an
      ongoing discipline, not a one-time box to check, but the three implementations
      themselves are singular as of this milestone.
- [x] Tests (fidelity): identical images → PSNR ∞ / SSIM 1 / LPIPS 0; known-shifted images
      give expected PSNR/SSIM ordering; FID/KID finite and non-negative on unrelated
      pools; `MetricsConfig` shape and hash participation. Split along a network boundary
      rather than a metric boundary — `FidelityEvaluator.__init__` always builds the
      pretrained AlexNet/Inception backbones, so every test that constructs it is `slow`;
      the fast suite covers only `MetricsConfig` and the KID subset-size clamp (a pure
      function). The one-time weight download succeeded in the dev sandbox, so this needed
      no "verify on server" carve-out.
- [x] Tests (task): `_extract_metrics` tests moved verbatim from `test_detector_stage.py`
      (same function, new home); new per-class AP extraction tests, including that a class
      with zero ground-truth instances is omitted rather than reported as a misleading
      0.0; one `slow`-marked end-to-end `evaluate_detector()` call against the synthetic
      fixture, exercising ultralytics' real `model.val()`.
- [x] Tests (faithfulness): synthetic detections with known answers throughout — box
      coordinates chosen so the correct match/no-match outcome is obvious by construction
      (identical box+class matches; disjoint boxes don't; same location but different
      class doesn't; higher-confidence prediction claims a contested match first; a
      hallucinated detection raises false-object rate; an erased component raises
      missed-object rate; disagreement with the real-visible detector lowers
      detection-consistency; all three empty-denominator cases default to their
      no-failure value rather than 0/0).

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
everything else in `RESEARCH_FINDINGS.md`'s reported table. `iou_threshold` and
`conf_threshold` (faithfulness) extended the same section rather than starting a new one.

**Decision — detection-consistency's formula.** RESEARCH_FINDINGS.md §9 names the metric
without a formula. Defined as: of everything the detector finds on the real visible image,
what fraction it also finds (same class, same place) on the translated image. Deliberately
distinct from missed-object rate: that compares against hand-labelled ground truth
(annotation-quality dependent, and only defined where labels exist); this compares the
translator against the detector's own behaviour on the untranslated photo, needs no labels
at all, and so stays well-defined at any `annotation_fraction` (E8). Revisit if a reviewer
or a closer reading of prior work suggests a different formula before this is reported.

**Decision — matching is greedy, confidence-ordered, single-IoU-threshold.** Same principle
COCOeval/ultralytics use per class per image, minus the multi-threshold AP integration that
follows it there (a different question, already answered by `metrics/task.py`). Chosen for
simplicity and because it is the standard operationalisation of "false positive"/"false
negative" in the detection literature, not because the alternatives (Hungarian assignment,
multi-threshold) were found lacking — revisit only if reviewers ask for one of those.

## M0.6 — Translator interface and CPU stand-in ✅

- [x] Define the `Translator` protocol: `fit()` and `translate(batch) -> Tensor`
      (`translators/protocol.py`, `@runtime_checkable` so `isinstance()` is testable)
- [x] `translators/stub.py` — tiny CPU stand-in (2-layer conv, `Sigmoid`-bounded to
      `[0,1]`) implementing it. **Load-bearing, not a test fixture**: `Pix2Pix_Turbo`
      hardcodes `.cuda()` and cannot be imported on the Mac, so this is the only way the
      full path stays locally testable
- [x] Tests (`tests/test_stub_translator.py`): protocol conformance; config
      (`hidden_channels`) reaches the module; `translate()` shape/dtype/`[0,1]` range;
      gradient connectivity from `translate()` output back to the translator's own
      parameters (what M0.7's coupling loss will depend on); `fit()` returns a finite loss
      and drives it down on a fixed batch; one `slow` round trip through the real
      `dataset_root` fixture → `translate()` → `FidelityEvaluator`

**Decision — round trip scoped to data → translate → fidelity-eval, not the full
data → coupling → export → eval path.** `coupling/` (M0.7) and `engine/export.py` (M0.8)
don't exist yet, so the wider round trip named in the original checklist item isn't
buildable yet. The coupling and export legs will be exercised for real once those
milestones land; this narrows what M0.6 verifies rather than leaving the item unaddressed.

**Decision — each translator owns its own optimizer(s) internally; `fit()` is a complete
step.** Keeps the `Translator` protocol backbone-agnostic: a GAN's D-then-G steps and a
one-step diffusion model's single step both fit behind one `fit(batch) -> dict[str, float]`
call, with no assumption from the interface about how many losses or optimizers are
involved. `StubTranslator` builds a private `Adam` at construction as the reference case.

**Deferred, not decided — how the M0.7 detection-consistency term plugs into `fit()`.**
The likely shape is a `Translator`-conforming wrapper around any base translator that adds
the coupling loss without changing the protocol or touching `stub.py`, but this is not
committed to; revisit when M0.7 actually needs it.

## M0.7 — Coupling ✅

- [x] Port `coupling/detection_loss.py` from `../Clean-SeAFusion/src/seafusion/losses/task.py`.
      Drop the YCbCr recombination (our translator emits RGB directly). **Keep the
      batch-size division** — ultralytics returns the loss pre-multiplied by batch size, and
      not undoing it makes λ silently scale with batch size
- [x] Feed float32 `[0,1]` directly; **never** route generated images through
      `preprocess_batch` (it does `.float()/255` on a uint8 dataloader tensor, creating a
      fresh graph root)
- [x] Saturating reward form (ReFL-style hinge / AlignProp-style `|r - target|`) rather
      than unbounded minimisation
- [x] `coupling/schedule.py` — staged λ_det ramp, `λ_det = 0` a clean no-op that never
      constructs the detector
- [x] Recalibrate the λ_det scale for a *detection* loss — the `[0,1,2,3]` ramp is
      calibrated to SeAFusion's segmentation loss. Reference downscales: ReFL `1e-3`,
      AlignProp `0.01`
- [x] Tests: λ=0 constructs no detector (assert it); loss is differentiable w.r.t. the
      generated image; λ invariant to batch size

**Decision — reward-target hinge is ReFL's `relu(total - target)`, not AlignProp's
symmetric `|total - target|`.** Matches `CouplingConfig.reward_target`'s own docstring
("once the detection loss falls below this the sample stops being rewarded"): the hinge has
zero gradient once a sample is already at or below target, whereas the AlignProp form would
also penalise a detection loss that is *better* than target — not what "stops being
rewarded" describes. `CouplingConfig.grad_scale` is applied as a final multiplicative
downscale on the (possibly-hinged) total, since scaling a scalar loss before `backward()`
scales its gradient by the same constant.

**Decision — `grad_scale`/`reward_target` are `DetectionTaskLoss` constructor args, not
read from `Config` inside it.** Keeps the module testable with plain floats and matches
`FrozenDetector`'s own weights/nc-as-args style; `coupling/schedule.py` is what bridges
`CouplingConfig` to the constructor.

**Decision — `schedule.py` is stateless functions (`weight_for_stage`,
`build_detection_loss`), not a class.** M0.8's not-yet-built loop owns all persistent stage
state (the warm-started translator, the detector-weights pointer that gets reassigned each
stage); schedule.py only needs to answer "what's the weight for this stage" and "should a
detector be constructed," both pure functions of a config + stage index + optional weights
path.

**Verify:** `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"`
(149 passed).

## M0.8 — Engine and tracking

- [x] Port `tracking.py` — `RunTracker`, never raises, context manager
- [x] Point W&B at the self-hosted base URL from env; key from env, never committed —
      no code needed: `wandb.init()` already resolves `WANDB_BASE_URL`/`WANDB_API_KEY`
      from the environment on its own, and `RunTracker` never reads either. Attempted
      live verification against a local `wandb/local` Docker instance; blocked by a
      `crypto.randomUUID is not a function` error in that image's bundled frontend
      (a browser secure-context/Web Crypto issue in the test image itself, unrelated to
      our code). Real end-to-end confirmation deferred to M0.10 ("Confirm W&B
      self-hosted logging works end to end"), which already covers it on the actual
      server and was always server-only territory.
- [x] Port `engine/trainer.py` — epoch loop over `translator.fit()`, generic pixel-L2
      validation via `translate()`, `translator.state_dict()`-based checkpoints (epoch +
      `config_hash` + config), warn-on-config-drift resume. **Not** a mechanical port —
      see Decision below
- [x] Port `engine/export.py` + `imaging.py` `Normalize` enum — per-image (never
      batch-wise) clamp/stretch
- [x] Port `engine/loop.py` — staged alternating loop. **Warm-start the translator across
      stages**; original SeAFusion re-instantiates its generator every stage
      (`train.py:203`), making the loop one-directional
- [x] `cli.py` — argparse subcommands with an explicit flag→config-path override table
- [x] `experiments/smoke.yaml` mirroring every real config at tiny scale
- [x] Tests (`slow` marker): full end-to-end on the fixture with the stand-in translator →
      `metrics.json`; resume produces identical state; same config+seed twice → identical
      hash and metrics

**Decision — `RunTracker` drops Clean-SeAFusion's `DistributedContext` parameter and
rank-0 gate.** PLAN.md §3 rules out DDP entirely on native Windows (no NCCL; one
experiment per GPU via `CUDA_VISIBLE_DEVICES`), and `RuntimeConfig` (M0.2) has no
rank/world-size concept to gate on. `RunTracker(config: Config)` takes one argument;
`enabled = config.runtime.wandb`. Everything else — lazy `wandb` import with an
`ImportError` fallback, run-dir creation before `wandb.init`, never-raising
`log`/`finish` that self-disables on failure, and the context-manager wrapper — ports
unchanged. `wandb>=0.28.1` was already a dependency since M0.1.

**Verify (tracking.py only):** `ruff format`, `ruff check`, `pyright` (0 errors),
`pytest -m "not slow"` (155 passed, 6 new).

**Decision — `engine/trainer.py` does not own an optimizer, scheduler, GradScaler, or
autocast, unlike Clean-SeAFusion's `FusionTrainer`.** M0.6 already decided
`Translator.fit(batch) -> dict[str, float]` is a complete, self-contained optimisation
step (`StubTranslator.fit()` does its own `zero_grad()`/`backward()`/`optimizer.step()`
internally) — there is nothing left for a trainer to wrap. AMP becomes each backbone's
own concern if one ever needs it. Consequence: `TrainConfig.lr`/`lr_gamma`/`amp`/
`amp_dtype` stay in the schema unread by `engine/trainer.py` — they're what a future
translator's own constructor/`fit()` will read when it builds its own optimizer/autocast
(e.g. a `pix2pix.py` wrapper reading `config.train.lr`), not dead fields. Checkpoints
hold `translator.state_dict()` rather than separate optimizer/scheduler/scaler entries,
so a translator's optimizer momentum does not survive resume (`nn.Module.state_dict()`
doesn't see `StubTranslator`'s private `Adam` attribute) — accepted since
`StubTranslator` is dev/test-only; a real backbone needing this can expose it itself via
`get_extra_state()`/`set_extra_state()`. Validation is a single generic pixel-L2 pass
through `translate()` alone under `no_grad` (not `FidelityEvaluator`, which always builds
pretrained AlexNet/Inception backbones and would make every epoch's validation
heavyweight) — `translate()` is the one method every backbone keeps grad-connected and
comparable; `fit()`'s internal loss composition is backbone-specific and, like
Clean-SeAFusion excluding its own task term from validation, the wrong thing to select
checkpoints on. No `DistributedContext`/DDP, matching `tracking.py`. Warm-start across
stages needs no `model=None` constructor branch: the caller always passes an
already-constructed translator, and `engine/loop.py` reusing the same instance across
stages *is* the warm start.

**Verify (trainer.py only):** `ruff format`, `ruff check`, `pyright` (0 errors),
`pytest -m "not slow"` (162 passed, 7 new). The rest of M0.8 is still open.

**Decision — `export.py` built before `loop.py`, reversing this checklist's original
order.** `loop.py`'s stage function calls `export_translated(...)` to feed each stage's
detector fine-tune, so it cannot be written or tested without `export.py` existing first.
`export.py` itself has no such dependency and is fully testable in isolation on the
synthetic fixture, so it moved first.

**`engine/export.py`:** ported from
`../Clean-SeAFusion/src/seafusion/engine/export.py`, dropping the YCbCr recombination —
`Translator.translate()` already returns RGB directly, so `export_split` just calls it.
`export_fused` renamed `export_translated`, reading `config.detector.evaluation.batch`
(not the flat `config.detector.batch` Clean-SeAFusion has) per the M0.2 `in_loop`/
`evaluation` split. `write_data_yaml` and the label-mirroring helper port unchanged.

**Found while writing `tests/test_export.py`:** a `tests` package shipped inside a
dependency's wheel (`.venv/lib/python3.12/site-packages/tests/`) shadows the local
`tests/` directory on pyright's module search path. `from tests.conftest import X` type
-checks (resolving to the wrong module and reporting `X` as an unknown symbol) even though
pytest itself imports the right file at runtime. Fixed by not doing cross-test-module
imports at all — `test_export.py` derives its expected counts/sizes from the fixtures
themselves (`len(dataset)`, `Image.open(dataset.visible_paths[0]).size`) rather than
importing `conftest.py`'s constants. Worth remembering if a future test file is tempted to
import from `tests.conftest` directly.

**Verify (export.py only):** `ruff format`, `ruff check`, `pyright` (0 errors),
`pytest -m "not slow"` (171 passed, 9 new). `engine/loop.py` is next.

**`engine/loop.py` needed two prerequisites `loop.py` itself doesn't touch again once
built:** `Translator.fit()` had no way to receive a per-stage detection loss at all
(`StubTranslator.fit()` was a fixed MSE step — M0.6 flagged this as "revisit when M0.7
needs it" and left it unresolved), and there was no answer yet for which `Path` seeds each
stage's frozen in-loop detector. Resolved:

- `Translator.fit()` gained two optional parameters —
  `task_loss: Callable[[Tensor, TranslationBatch], Tensor] | None = None` and
  `task_weight: float = 0.0` — typed as a plain callable rather than importing
  `coupling.DetectionTaskLoss` directly, since `translators/` sits below `coupling/` in
  PLAN.md §5's layer order and `DetectionTaskLoss.__call__` already satisfies the shape.
  `StubTranslator.fit()` now adds `task_weight * task_loss(pred, batch)` to its total when
  both are given; `Trainer` gained matching constructor params it threads through to every
  `fit()` call unchanged, still composing nothing itself.
- **Decision — the in-loop coupling detector stays fixed at `config.detector.in_loop.weights`
  for an entire run; only the *evaluation* detector warm-starts across stages** (resolved
  with the user). Clean-SeAFusion threads one `detector_weights` variable through both
  roles, so its "detector warm-started across stages" (PLAN.md §8) reads as one thing.
  t2o already split the two roles structurally at M0.2 (`DetectorConfig.in_loop`/
  `.evaluation`, "never conflated" — invariant 7) specifically so they cannot share a
  weights file by accident; letting `loop.py` reassign the in-loop detector to each stage's
  freshly fine-tuned evaluation detector would violate that literally, and would mean the
  coupling loss grades the translator against a detector that was itself trained on the
  translator's own prior, weaker outputs — a strictly worse experiment than a fixed,
  independently-trained reference. `run_loop` therefore holds exactly one variable
  (`eval_weights`) across the stage loop, seeded from `config.detector.evaluation.init_weights`
  and reassigned to each stage's `DetectorResult.weights`; `config.detector.in_loop.weights`
  is read fresh every stage and never written. "Detector warm-started across stages" in
  PLAN.md §8 is now read as describing this evaluation-detector accumulation, the same way
  the translator accumulates, not the in-loop detector's identity.
- Detector fine-tuning (`export_translated` → `train_detector`) is unconditional every
  stage, including stage 0 — matching Clean-SeAFusion's table (stage 0 still bootstraps a
  detector from the translation-only output). Only `build_detection_loss`'s own
  weight-vs-zero check (M0.7, unchanged) decides whether a stage's coupling term exists.
- `run_loop(config, translator, run_dir=None, tracker=None, train_detector_stages=True)`
  returns `list[StageResult]` and writes `run_dir/metrics.json` after every stage.
  `train_detector_stages=False` skips export/detector work entirely — the fast test path,
  and an independently useful translator-only mode (Clean-SeAFusion carries the same flag).
- **Out of scope for this step**, left for the later "Tests" bullet below: stage-level
  resume, and the same-config-same-seed-twice hash/metrics equality check. `Trainer.resume()`
  already covers within-stage resume.

**Verify (loop.py only):** `ruff format`, `ruff check`, `pyright` (0 errors),
`pytest -m "not slow"` (176 passed, 5 new — protocol/`StubTranslator`/`Trainer` coupling
coverage plus one fast `run_loop` test). One new `slow` test
(`tests/test_loop.py::test_full_loop_fine_tunes_a_detector_every_stage`) is the first
end-to-end exercise of `train_detector` against ultralytics' real training loop — deferred
from M0.4, run locally and passing (`pytest -m slow`: 8 passed).

**`cli.py`:** four subcommands — `train` (one stage of `Trainer`), `loop` (`run_loop`),
`export` (`export_translated`), `evaluate` (`metrics.task.evaluate_detector`, standalone).
Two things didn't exist yet and landed as part of this step rather than being deferred:

- **`build_translator(config: Config) -> nn.Module`**, added to
  `translators/__init__.py`. No factory existed anywhere — every prior call site (all in
  `tests/`) constructed `StubTranslator` directly. One `isinstance` branch per
  `TranslatorConfig` union member; the `raise ConfigError` fallback is unreachable today (one
  member) but stops being unreachable the moment M1 adds `pix2pix`.
- **Arbitrary-depth override merging.** Clean-SeAFusion's `overrides_from_args` only handles
  1-tuple/2-tuple config paths (`(section,)` or `(section, key)`). t2o's `DetectorConfig` is
  a level deeper (`detector.evaluation.init_weights`), so `_OVERRIDES` maps each flag to an
  N-tuple walked by `node.setdefault(key, {})`, and multiple flags targeting sibling subpaths
  under the same top-level section (`detector.in_loop.*` and `detector.evaluation.*`) now
  coexist in the merged dict instead of one clobbering the other — covered directly in
  `tests/test_cli.py`.

**Decision — `evaluate` subcommand added now, not deferred to M0.10.** Not in TASKS.md's
literal wording, but `metrics.task.evaluate_detector` already existed with no CLI entry
point, and M0.10's E1 reference bracket needs one. Confirmed with the user before building
it. Standalone — takes `--weights`/`--data`/`--imgsz`/`--batch`/`--device` directly and never
touches `Config`, same spirit as Clean-SeAFusion's own standalone `predict` subcommand. No
`predict`/inference subcommand exists or is planned — PLAN.md §1 rules out an inference
service; `evaluate` only ever scores an existing detector checkpoint.

**Decision — heavy submodules (torch, ultralytics) are imported inside each subcommand
function, not at module level.** Keeps `t2o --version`/`t2o --help` fast and importable
without a GPU-capable torch build, matching Clean-SeAFusion's own lazy-import style in its
`cli.py`.

**Verify (cli.py only):** `ruff format`, `ruff check`, `pyright` (0 errors),
`pytest -m "not slow"` (185 passed, 9 new), `pytest -m slow` (9 passed, 1 new — the
`evaluate` subcommand's real `model.val()` call). Manually smoke-tested `t2o --version`,
`t2o --help`, `t2o train --help` against the installed entry point.

**`experiments/smoke.yaml`:** every section present explicitly (no field left to an implicit
default), sized so the full loop finishes on CPU in seconds — `epochs_per_stage: 1`,
`batch_size: 2`, `workers: 0`, a 2-stage `task_weights: [0.0, 1.0]` (not the real 4-stage
ramp) instead of exercising the weight=0/weight>0 split in one pass each, `translator.backbone:
stub` (the only backbone importable on the Mac). First tracked file under `experiments/`.

**Decision — `data.manifest`/`detector.*.weights` stay at schema defaults
(`dataset/yolo_rgbt/data.yaml`, `yolo11n.pt`), not a fixture path.** These are placeholders by
design, the same way a real experiment config's paths are machine-specific and resolved per
invocation (PLAN.md §9's "a path that is missing locally is the normal case"). Every test that
actually runs the file overrides them to the synthetic fixture, exactly like every other engine
test already does — `Config.load()` never checks path existence at load time (`schema.py`'s own
docstring), so loading the file as-committed and merely inspecting its resolved values needs no
override at all.

**`tests/test_experiments.py`:** three fast tests — loads and is tiny; `config_hash()` is
stable across repeated loads of the same file; one real end-to-end pass through
`engine.loop.run_loop` on the synthetic fixture (translator-only, `train_detector_stages=False`,
matching `test_loop.py`'s own fast path). The fuller end-to-end contract this file will also
serve — `metrics.json` across every stage, resume, same-config-same-seed hash/metrics
reproducibility — is the *next* TASKS.md item, not duplicated here.

**Verify (experiments/smoke.yaml only):** `ruff format`, `ruff check`, `pyright` (0 errors),
`pytest -m "not slow"` (188 passed, 3 new). The M0.8 "Tests" bullet (resume + hash/metrics
reproducibility) is next.

**M0.8's closing "Tests" bullet required real production code, not just tests.**
`engine/loop.py`'s own docstring had explicitly deferred "stage-level resume" to this exact
item; nothing before this step could make "resume produces identical state" true, since
`run_loop` had no resume path at all — confirmed with the user before touching its public
signature, given the operational stakes of getting crash-recovery semantics wrong on a
long-running server job.

- **`run_loop(..., resume: bool = False)`** (`engine/loop.py`). On `resume=True`,
  `run_dir/metrics.json` (if present) is parsed back into `StageResult`s via two new
  functions, `_load_existing_results`/`_stage_result_from_json` — the exact inverse of the
  existing `_write_metrics`/`_stage_result_to_json`; already-recorded stages are never
  re-run. A fresh `run_dir` (no file yet) makes `resume=True` a silent no-op, identical to
  `resume=False`. **Detector fine-tuning is never itself resumed** — it is one bounded
  `model.train()` call per stage, not something tracked epoch-by-epoch the way translator
  training is, so a resumed stage's detector fine-tune always restarts from `eval_weights`
  fresh. `eval_weights` itself is recovered from the last completed stage's recorded
  `DetectorResult.weights` when resuming, exactly mirroring what an unbroken run would have
  carried forward.
- **Warm-start continuity across a resume needed one non-obvious fix.** A resumed process's
  `translator` argument is whatever a fresh call to `build_translator` just constructed —
  correct only for a from-scratch run. If the stage about to run has no checkpoint of its
  own yet (never started), `run_loop` now restores it from the *previous* completed stage's
  `translator_last.pt` before training begins; if the stage does have its own checkpoint
  (crashed mid-epoch), `Trainer.resume()` already restores it more precisely (including
  `start_epoch`). Without the first case, a resumed run would silently train the next stage
  from an unwarm-started translator while every other signal claimed continuity.
- **`translators.build_translator` now seeds the global torch RNG from `config.train.seed`
  before constructing the backbone.** Needed for "same config+seed twice → identical
  metrics" to be true structurally rather than by test-only trickery: translator weight
  init previously depended on whatever torch's ambient RNG state happened to be at
  construction time, and `build_translator` (M0.8 step 5) is the one choke point every real
  caller (`cli.py`, tests) already goes through. `Trainer` never constructs a translator, so
  there was no other natural place for this.
- **`cli.py`'s `loop` subcommand gained `--resume`**, wired straight through to
  `run_loop(resume=args.resume)` — otherwise the new capability would be unreachable outside
  a test.

**`tests/test_loop.py`:** four new tests. `test_resume_on_a_fresh_run_dir_behaves_like_a_normal_run`
and `test_running_the_same_config_and_seed_twice_yields_identical_metrics` are fast
(`train_detector_stages=False`). `test_resume_skips_completed_stages_and_restores_warm_started_weights`
is the direct test of "resume produces identical state": an unbroken 2-stage run vs. one
deliberately interrupted right after stage 0 and then resumed must land on bit-identical
final translator weights and identical `metrics.json` contents — also fast, since the
mechanism under test is `run_loop`'s own bookkeeping, not detector training.
`test_resume_continues_through_a_completed_detector_stage` is the one scenario those fast
tests can't reach — `eval_weights` carrying forward from a stage whose detector fine-tune
already completed — and is `slow`, since it touches real `train_detector`. One more fast
test lives in `tests/test_cli.py` (`test_main_loop_resume_skips_completed_stages`),
confirming `--resume` is actually wired rather than merely present on the parser.

**Deviation from the checklist's literal "(`slow` marker)":** only the detector-fine-tune
resume test is `slow`. The already-existing `test_full_loop_fine_tunes_a_detector_every_stage`
(M0.8 step 4) already satisfies "full end-to-end on the fixture with the stand-in translator
→ metrics.json" and is not duplicated here; the two new *fast* tests are meaningful and
correct without touching ultralytics at all, matching the house convention of reserving
`slow` for real detector training rather than blanket-marking everything in a checklist
bullet.

**Verify (M0.8 Tests bullet):** `ruff format`, `ruff check`, `pyright` (0 errors),
`pytest -m "not slow"` (192 passed, 4 new), `pytest -m slow` (10 passed, 1 new).
**M0.8 is now fully closed.** M0.9 (dataset acquisition) is next.

## M0.9 — Dataset acquisition

- [x] `scripts/fetch_datasets.py` for the trivially-scriptable set: MSRS, CPLID, HIT-UAV,
      FLIR-aligned (HuggingFace mirror `UserNae3/FLIR_aligned` — avoids the Teledyne
      registration form)
- [ ] Fetch LLVIP, M3FD, TTPLA once on the Mac via `gdown`, re-host, then make the server
      path a plain `curl`
  - [x] Script written: `scripts/fetch_datasets.py`'s `SOURCES` registry extended with
        `gdown_file_id`/`gdown_folder_id` (real ids read off each dataset's own README —
        `bupt-ai-cz/LLVIP`, `JinyuanLiu-CV/TarDAL`, `R3ab/ttpla_dataset` — not guessed) and
        `fetch_gdown_file`/`fetch_gdown_folder`, dispatched the same idempotent way as the
        existing git/HuggingFace sources. `gdown` added as a new `scripts` uv dependency
        group (not `dev` — it's a fetch-time tool the server never needs, not a code-quality
        one), so a bare `uv sync` doesn't pull it in.
  - [x] **Fetched — on the server, not on the Mac (2026-08-23).** The script ran directly
        there against all three datasets. The Mac disk constraint that blocked this (~31GB
        free, 93% full) has also since been cleared, but is no longer on the critical path.
  - [x] **Re-host destination: not needed.** PLAN.md's plan was "fetch once on the Mac,
        re-host, then the server script is a plain `curl`", to work around `gdown` being
        interactive on first use. Running `fetch_datasets.py` on the server directly made the
        whole hop unnecessary — there is no intermediate artifact to host. **PLAN.md §9's
        "fetch once on the Mac, re-host" line is superseded**; the registry is the delivery
        mechanism on both machines.
  - Note: the M3FD Google Drive folder (`1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6`) also contains
        TNO and RoadScene as sibling subfolders — bundled that way in TarDAL's own share, not
        a mistake on our end. Expect `dataset/raw/m3fd/` to hold more than just M3FD once
        fetched; a future adapter should read only the M3FD subfolder.
- [x] Adapters normalising each into the internal representation
  - [x] MSRS — `data/adapters/msrs.py::adapt_msrs`. `train`/`ir`+`vi` (1083 pairs) merged with
        the small `detection/` pool (80 labelled pairs, disjoint filenames); `test` → `val`
        (361 pairs, unlabelled). Verified against the real local clone: 1163 train / 361 val.
  - [x] FLIR-aligned — `data/adapters/flir.py::adapt_flir`. Reads directly out of
        `aligned.zip` (never extracted to disk); renames `FLIR_XXXXX_RGB.jpg`/
        `_PreviewData.jpeg` to a shared `FLIR_XXXXX.jpg` stem so `Pairing` can find them;
        converts VOC-XML `bndbox` → normalised YOLO boxes; splits by each XML's own
        `<folder>` field. Scoped to the ~5142 annotated pairs (of 10284 total) — the other
        half has neither a split nor a label. Verified against the real local archive: 4129
        train / 1013 val, matching the literature's known FLIR-aligned split exactly.
  - **CPLID and HIT-UAV are out of scope, confirmed with the user.** Verified against the
        real local clones: CPLID is RGB-only (UAV insulator photos, VOC-XML, class
        `insulator`/`defect`), HIT-UAV is IR-only (thermal aerial shots, YOLO, 4 classes).
        Neither has a counterpart modality, so neither fits the paired
        `{visible,infrared}` contract `data/pairing.py`/`data/dataset.py` assume for every
        sample. Revisit only if a single-modality detector-pretraining need arises (e.g. E9).
- [x] Verify: MSRS `detection/` folder — does it have box annotations usable for mAP?
      **Correction: yes.** The prior answer here was written from browsing GitHub without
      cloning. The real clone has `detection/{vi,ir,labels}` — 80 pairs, YOLO boxes, classes
      `person`/`bicycle`/`car` (generic, not power-line-relevant). Small relative to the
      1083 train / 361 test pairs, so MSRS is still primarily translation/fidelity data, as
      `RESEARCH_FINDINGS.md` frames it — just not *exclusively* unlabelled.
- [x] Verify: InsPLAD annotation format (not stated in its README; Mendeley Data gates the
      files behind a form, so this needs the actual download, not just the repo README).
      **Verified: standard MS-COCO detection JSON**
      (`InsPLAD-det.zip/annotations/instances_{train,val}.json`), `bbox: [x, y, w, h]` in
      absolute pixel coordinates (top-left origin), `segmentation: []` throughout (no masks
      — matches the README's note that pixel-level annotation is a separate, later
      project), `iscrowd: 0` throughout. **18 categories, not the 17 the paper's text
      lists** — the released JSON has an extra `sphere` (id 18) absent from the arXiv
      class list; worth flagging if InsPLAD's numbers are ever cited directly. Train: 7981
      images / 22635 boxes; val: 2626 images / 6324 boxes — image count sums to exactly
      the README's 10,607; box count sums to 28,959 against the README's cited 28,933 (off
      by 26, likely a version/rounding drift between the paper and the released archive).
      **Correction: the "Mendeley Data gates the files behind a form" premise was wrong.**
      Mendeley's public API (`data.mendeley.com/public-api/datasets/5n3fjgvfyz`) hands back
      an unauthenticated, directly-fetchable S3 `download_url` for the one 6.4GB
      `InsPLAD_Dataset.zip` — no login, no request form, in practice. That outer zip nests
      three inner zips (`InsPLAD-det.zip` 4.36GB, `unsupervised_anomaly_detection.zip`
      1.17GB, `supervised_fault_classification.zip` 875MB); verified without downloading any
      of them in full — HTTP range requests plus a streaming zip/deflate reader pulled just
      the two annotation JSONs (~1.5MB fetched total over the network, on a machine with
      ~31GB free disk). No adapter written yet; InsPLAD isn't in `fetch_datasets.py`'s
      registry either — both are natural follow-ups once the E9 cross-dataset generalisation
      work actually wants this dataset (RESEARCH_FINDINGS.md; InsPLAD is single-modality
      RGB, like CPLID/HIT-UAV, so it can only ever be a detector-training/eval source, never
      a translation pair).
- [x] Freeze and hash the splits; commit the manifest. `data/splits.py` (`freeze_split`,
      `write_split_manifest`/`load_split_manifest`, `verify_split` + `SplitDriftError`) plus
      `scripts/freeze_splits.py` — run for real against `dataset/processed/{msrs,flir}`,
      committed as `splits/msrs.json` (1163 train / 361 val) and `splits/flir.json` (4129
      train / 1013 val), matching each adapter's own real-run counts exactly. LLVIP/M3FD/TTPLA
      get frozen the same way once actually fetched.
- [x] **The custom paired dataset is frozen on the server** (`yolo_rgbt_29_jul`), run during
      E3's campaign. `combined_hash 7ede3433adc9c0b8`; train **600** (`4e01a89877c6a943`),
      val **153** (`6b06220c26a9adbc`). The val count independently matches M1.2 step 1's
      reference judge (153 images / 423 instances), so the campaign and its judge are provably
      on the same val images. **The record itself cannot reach git** — the server has no
      outbound git access, so `splits/yolo_rgbt_29_jul.json` stays untracked there and the
      hashes above are this repo's only copy of the split identity. Consequences, both real:
      the stem list exists in exactly one place, and an untracked file inside a *tracked*
      directory is precisely what `git clean -fd` removes — keep a copy outside the checkout.
      `--check` against the on-disk record still works on the server and is the drift guard
      before any future run on this dataset.
**The ~850 pairs are 753 train+val plus ~100 held out as an unseen test set** (confirmed with
the user) — so "850" is the dataset, and 753 is what any experiment in this repo has ever
touched. Worth stating because the frozen record shows only the 753 and the difference reads
like loss otherwise.

**The held-out set is isolated structurally, not by discipline.** `DatasetManifest` reads only
`path`/`train`/`val`/`nc`/`names`/`rgbt` and drops every other key (`data/manifest.py`), so a
`test:` entry in `data.yaml` is invisible to the trainer, the exporter, both detector roles and
`freeze_splits.py` alike. Nothing in the project *can* read those pairs, which is the strongest
form this guarantee takes — stronger than the frozen-split contract, which only detects misuse
after the fact.

Consequence, deliberate for now: the test split's membership is unpinned — `freeze_split` can
only ever record train/val. A leak *into* train/val is still caught (the 753 hashes change);
a reshuffle *within* the held-out set is not. That only starts to matter when the set is first
used, so freeze it then rather than teaching the manifest a third split it must otherwise
ignore. Report every count as "753 train+val of 853" in the paper, not "850 pairs".

**`data/splits.py` decisions:**

- **The frozen record, not the images, is what reaches git.** `dataset/` is wholly
  gitignored with no exceptions (M0.1), so a split can never be proven unchanged by diffing
  images. `splits/<name>.json` records sorted train/val filename stems plus a sha256 hash of
  each (16 hex chars, matching `Config.config_hash()`'s own truncation convention) — small,
  diffable, and enough to detect any reshuffling without needing the data itself.
- **Stems are listed in full, not just hashed.** A bare hash mismatch says "something
  changed" with no way to say what; listing every stem makes `git diff` on
  `splits/<name>.json` show exactly which files were added or removed if MSRS/FLIR ever
  publish a revision. At ~1500–5200 lines per file this is well within normal bounds for a
  text-diffable manifest (the same pattern many ML repos use for train/val list files).
  `PLAN.md`'s "one internal representation" invariant is about the directory contract, not
  about keeping every artifact minimal — diagnosability won out here, same reasoning
  `data/pairing.py`'s `validate_pairs` already uses (report the full mismatch, not just that
  one exists).
- **`verify_split`/`SplitDriftError` exist but nothing calls them yet.** Wiring drift
  detection into `engine/loop.py` or `cli.py train` isn't needed to freeze today's two public
  splits, and the dataset this guard matters most for (the custom ~850 pairs) isn't on this
  machine to test against. Built as reusable library code precisely so a future run-start
  check is a small addition, not a redesign — matches the project's own pattern of landing a
  capability before its first caller when the caller depends on data that doesn't exist yet
  (`coupling/schedule.py`'s stage functions predated `engine/loop.py` the same way).
- **`scripts/freeze_splits.py` discovers datasets by globbing `<data-root>/*/data.yaml`**
  rather than a hardcoded list, so committing LLVIP/M3FD/TTPLA/the custom dataset's frozen
  records later needs no changes here — only running the adapter (or pointing `--data-root`
  at wherever a `data.yaml` already exists) and re-running this script.
- **Re-freezing an existing record overwrites rather than refuses, but logs a warning first
  if the previous record no longer matches.** A silent overwrite would defeat the point of
  freezing; an outright refusal would make "yes, I know it changed, update it" require
  manually deleting the file first. `--check` is the strict, non-writing mode for anything
  that wants to fail loudly on drift instead.

**Verify:** `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"`
(235 passed, 18 new). No `slow` test — nothing here touches ultralytics/torch. Ran for real:
`uv run python scripts/freeze_splits.py` (froze both), then `--check` (both matched,
confirming the round trip is exact). **M0.9 is now fully closed** except the LLVIP/M3FD/TTPLA
real fetch (deferred, disk space — see the gdown decisions below) and, by extension, freezing
their splits once fetched.

**`scripts/fetch_datasets.py` decisions:**

- **Destination `dataset/raw/<name>/`** — stays inside the already-wholesale-ignored
  `dataset/` prefix (`.gitignore`), so no new ignore rule was needed, and sits next to
  where the still-open adapter step will read raw layouts from.
- **Two fetch strategies, chosen per source**: `git clone --depth 1` for MSRS/CPLID/
  HIT-UAV (plain git repos); `huggingface_hub.snapshot_download` for FLIR-aligned.
  `huggingface_hub` needed no new dependency — already installed transitively via
  `transformers`.
- **Idempotent** — a destination that already exists and is non-empty is skipped with a
  log line rather than re-cloned, consistent with `engine/loop.py`'s resume discipline.
- **Not executed for real in the session that wrote it** — confirmed with the user at the
  time. The script itself is tested with `subprocess.run`/`huggingface_hub.snapshot_download`
  both monkeypatched (`tests/test_fetch_datasets.py`, 8 fast tests, no network I/O). The user
  has since run it for real: `dataset/raw/{msrs,cplid,hituav,flir}` exist locally as of the
  adapters step below, which is what let that step be designed against actual raw layouts
  instead of READMEs.
- **`scripts/` is a new top-level package**, sibling to `src/t2o`, not part of it — added
  to `[tool.pytest.ini_options] pythonpath`, `[tool.ruff] src`, and `[tool.pyright]
  include` so it lints/type-checks/imports-in-tests the same as everything else, without
  becoming part of the installed `t2o` distribution.

**Verify:** `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"`
(200 passed, 8 new). No `slow` test — nothing here touches ultralytics/torch.

**`data/adapters/` — MSRS (M0.9 adapters step 1 of 2) decisions:**

- **`data/adapters/common.py` is the one place the paired `rgbt:`-block `data.yaml` shape is
  written** (`write_manifest_yaml`), shared by every per-dataset adapter, so MSRS and the
  next one (FLIR-aligned) can't drift against each other or against `tests/conftest.py`'s
  synthetic-fixture format — the same "one evaluation path" discipline `metrics/` already
  follows, applied to the write side of the data contract instead of the read side.
- **Images are copied verbatim (`shutil.copyfile`), never re-encoded.** MSRS ships PNG;
  nothing in `data/dataset.py`/`data/pairing.py` requires a specific extension
  (`IMAGE_SUFFIXES` already covers it), so there's no reason to touch the bytes.
- **`detection/`'s 80 labelled pairs merge into `train`, not a third split.** Verified
  disjoint by filename stem from `train`/`test` in the real clone; the internal
  representation only has two splits, and `data/labels.py::load_yolo_labels` already treats
  a missing label file as a zero-instance negative, so the other 1083 train images need no
  placeholder `.txt` files — `write_label` writes nothing for them rather than an empty file.
- **Collision check is defensive, not decorative.** Today's MSRS release has zero overlap
  between `train/` and `detection/` stems, but a future release changing that would silently
  overwrite images under a plain merge; `adapt_msrs` raises `AdapterError` naming the
  colliding stems instead, and `tests/test_adapters.py` constructs the collision directly
  rather than trusting it never happens.
- **CPLID and HIT-UAV get no adapter.** Both are single-modality (verified against the real
  local clones, not README claims) — confirmed with the user before scoping this step down
  to MSRS + FLIR-aligned only; see the M0.9 checklist entry above for the corrected dataset
  survey.

**Verify (MSRS adapter only):** `ruff format`, `ruff check`, `pyright` (0 errors),
`pytest -m "not slow"` (207 passed, 7 new). Also run for real against the local
`dataset/raw/msrs`: 1163 train / 361 val pairs, matching the plan's predicted counts exactly;
`DatasetManifest.load` on the result loads cleanly. FLIR-aligned adapter is next.

**`data/adapters/flir.py` — FLIR-aligned (M0.9 adapters step 2 of 2) decisions:**

- **Reads straight out of `aligned.zip` via `zipfile`, never extracts to disk.** The raw
  layout for this one source is genuinely an archive, not a directory tree —
  `huggingface_hub.snapshot_download` leaves it zipped. `common.py` gained
  `write_image_pair_bytes`/`write_label_lines` alongside MSRS's path-based
  `copy_image_pair`/`write_label`, so both source shapes share the same destination-writing
  code without forcing a full extraction of an archive most of which (the unannotated half,
  the duplicate `AnnotatedImages/` copies) this step doesn't use.
- **Scoped to the ~5142 annotated pairs, not all 10284.** Each annotation XML's own
  `<folder>` field (`training`/`validation`) is the *only* place a train/val split is
  recorded — verified against an 800-file sample of the real archive, no external split
  list exists. The unannotated half has neither a split nor a label, so including it would
  only add unlabelled bulk; MSRS's 1083-pair unlabelled pool already serves that role more
  compactly.
- **Filenames are normalised on copy.** `FLIR_00002_RGB.jpg` and `FLIR_00002_PreviewData.jpeg`
  share no stem in the raw archive; `data.pairing.Pairing` requires the visible and infrared
  paths to differ only in their `visible`/`infrared` directory segment, so both get renamed
  to `FLIR_00002.jpg` on copy (bytes untouched — both are already JPEG data regardless of the
  source extension).
- **Class vocabulary is derived from the data, sorted alphabetically** (`bicycle`, `car`,
  `dog`, `person`), not hardcoded — nothing in the archive declares a canonical order the way
  MSRS's `classes.txt` does, and deriving it keeps the adapter correct even if a future FLIR
  release adds or removes a class.
- **Boxes are clamped to the frame, and dropped (not written) if zero-area after clamping.**
  A handful of FLIR's `bndbox` values run past the image edge; failing loudly on that would
  make an otherwise-usable annotation block the whole adapter, and writing a degenerate
  zero-width box would hand `v8DetectionLoss` a nonsensical target. `tests/test_flir_adapter.py`
  constructs this case directly (a box entirely outside a 50×50 frame) rather than trusting
  it's rare enough not to matter.
- **The real-data sanity test is `slow`, unlike MSRS's.** MSRS's real-clone test copies
  already-decompressed PNGs from a directory tree and stayed fast; FLIR's equivalent
  decompresses real image bytes out of a ~1.4GB archive (~6s locally) — cheap enough to run,
  but not free enough to belong in the default `pytest -m "not slow"` pass.

**Verify (FLIR adapter):** `ruff format`, `ruff check`, `pyright` (0 errors),
`pytest -m "not slow"` (214 passed, 7 new), `pytest -m slow` (real-archive test passes,
~6s). Also run for real against the local `dataset/raw/flir/aligned.zip`: 4129 train / 1013
val pairs — matching the FLIR-aligned split reported in the literature exactly; classes
`bicycle`/`car`/`dog`/`person`; `DatasetManifest.load` on the result loads cleanly.
**M0.9's adapters item is now closed.** Remaining M0.9 work: LLVIP/M3FD/TTPLA (`gdown`),
freezing and hashing the splits.

**`fetch_datasets.py` — gdown sources decisions:**

- **One script, one registry, not a second file.** LLVIP/M3FD/TTPLA are still "fetch a
  public dataset to `dataset/raw/<name>/`" — the same job `fetch_datasets.py` already does
  for the git/HuggingFace sources, just via a different `DatasetSource` field selecting a
  different strategy function. Splitting fetch logic across two scripts would just be two
  places to keep the idempotent-skip/CLI/logging conventions in sync.
- **Drive ids are copied from each dataset's own README, not guessed.** A wrong file id
  silently serves an HTML interstitial or 404s rather than data; verified against
  `bupt-ai-cz/LLVIP/download_dataset.md`, `JinyuanLiu-CV/TarDAL/README.md`, and
  `R3ab/ttpla_dataset/README.md` directly.
- **`gdown` lives in a new `scripts` uv dependency group, not `dev`.** It's a fetch-time-only
  tool nothing in `src/t2o` or the server ever imports (the server side of this plan is a
  plain `curl` against wherever these get re-hosted) — grouping it with linters/typecheckers
  would blur why it's there. `uv run --group scripts ...` is required to actually invoke the
  gdown paths; a bare `uv sync`/`uv run pytest` does not install it, matching the "server
  never needs gdown" design.
- **The real download was not run.** Confirmed with the user given ~31GB free disk on this
  machine (93% full) — LLVIP alone is large enough that a blind multi-dataset fetch risked
  filling it. The script itself is verified with `gdown.download`/`gdown.download_folder`
  both monkeypatched (`tests/test_fetch_datasets.py`, 4 new fast tests, no network I/O).
- **Re-hosting is intentionally left undecided**, per the user — a reminder to revisit this
  belongs whenever server training on these datasets actually starts, not now.

**Verify (gdown extension):** `ruff format`, `ruff check`, `pyright` (0 errors),
`pytest -m "not slow"` (217 passed, 3 new). No `slow` test and no real network call —
matching M0.9 step 1's own "not executed for real" precedent, this time by explicit
disk-space decision rather than default caution.

## M0.10 — Server bring-up (cannot be verified locally) ✅

- [x] `uv sync --extra gpu` on the server. **First real attempt hit a hash-verification
      failure** ("unexpected sha mismatch") installing `torchvision`, traced to a real
      upstream bug: `download.pytorch.org`'s index is missing the `sha256` fragment for
      `torchvision-0.28.0+cu130`'s `win_amd64`/`cp312` wheel specifically (every version
      from 0.24.0 through 0.27.1 has one — checked directly against the live index).
      **Fixed** by capping `pyproject.toml`'s `torch`/`torchvision` to
      `>=2.12.1,<2.13.0`/`>=0.27.1,<0.28.0` — the latest pair confirmed to have proper
      hashes on both wheels for `cu130`/`win_amd64`/`cp312`. Re-locked (`uv lock`); the full
      local suite (`pytest -m "not slow"` and `-m slow`, 246 total) still passes against
      2.12.1/0.27.1 CPU wheels, so this is a pure downgrade with no code changes needed.
      **Confirmed on the real Windows/CUDA install** — `uv sync --extra gpu` succeeds
      cleanly with the new pins.
- [x] Confirm dataloader does not hang under Windows spawn — start `num_workers=0`, raise
      slowly. Confirmed clean on the server.
- [x] **Measure actual VRAM** at candidate batch sizes and resolutions before committing to
      any long run. Done on the server (stub translator + detector fine-tuning only — real
      diffusion VRAM profiling is still M2a's job, this only established the dataloader/
      detector floor).
- [x] E1 reference bracket: detector on {raw thermal, real visible} × {detector trained on
      thermal, on visible}, using the existing `.pt` weights. Run for real on the server:

      | detector trained on \ evaluated on | thermal | visible |
      | --- | --- | --- |
      | thermal | **> 0.9 mAP50** | — |
      | visible | **0.1887 mAP50** | **0.9213 mAP50** |

      **Both cross/in-domain numbers on the visible-trained row were corrected at M1.** This
      row originally read "< 0.05" and "> 0.9" from an ad-hoc measurement; M1's gate
      evaluation re-ran them through `metrics.task.evaluate_detector` on the frozen val split
      (153 images / 423 instances) and got 0.1887 on raw thermal, 0.9213 on real visible. The
      directional conclusions below are unchanged; the numbers to cite are these.

      **In-domain detection is excellent on both modalities** (> 0.9 mAP50 whichever side
      the detector is trained and run on) — thermal frames are not fundamentally
      information-poor for this task; the open question behind C4/E1 ("does translation
      even help over direct thermal detection") already has a clear directional answer: the
      signal is there, the *domain gap* is the problem, not the sensor.

      **Cross-domain transfer collapses** (0.1887 mAP50, visible-trained detector run
      directly on raw thermal, untranslated) — and this is the actual number M1's gate
      ("translated mAP must beat raw-thermal mAP on at least one class, or stop") measures
      against. The realistic deployment baseline is "run the detector you can actually label
      data for (visible) on whatever raw sensor frame you have," not the thermal-trained
      detector — training a thermal-domain detector requires exactly the thermal-domain
      annotations the low-annotation framing (E8) exists to avoid depending on. The collapse
      is very uneven across classes (M1's table): Pole survives at 0.5551 — a pole's silhouette
      is thermally obvious — while Switch is 0.0047 and Fuse 0.0377, i.e. gone. **The gate's
      bar is low — a good sign for M1's feasibility**, and M1 cleared it on all four classes.
- [x] Confirm W&B self-hosted logging works end to end. Confirmed on the server.

**Found while running these checks — `--device 0` failed, `--device cuda:0` was needed as a
workaround.** Real bug, not a server environment quirk: `cli.py` and `engine/trainer.py` each
carried a private, byte-identical `_resolve_device` that called `torch.device(device)`
directly on the raw string. Real PyTorch's `torch.device("0")` raises
`RuntimeError: Invalid device string: '0'` — it needs the `cuda:` prefix. Both `--device`
help strings advertised `'0'` as valid (copying ultralytics' own convention, where
`select_device` *does* accept a bare digit), but `train`/`loop`/`export` never go through
ultralytics' resolver, so the documented spelling silently didn't work; `evaluate` (which
forwards the device string straight into `model.val(device=...)`, i.e. ultralytics' own
resolver) tolerated it fine, which is why it only bit during the dataloader/VRAM checks
above, not the E1 evaluate calls. **Fixed**: `engine/trainer.py`'s `_resolve_device` became
the single public `resolve_device`, normalising a bare digit to `cuda:{device}` before
constructing the `torch.device`; `cli.py`'s duplicate copy was deleted and `_run_export` now
imports the one in `engine/trainer.py`. The misleading `'0,1'` example in the shared
`--device` help text was also removed — a single `torch.device` can never represent a
comma-joined multi-GPU list, and `PLAN.md` §3 rules out DDP on Windows entirely anyway (one
experiment per GPU via `CUDA_VISIBLE_DEVICES`). `evaluate`'s own `--device` help text is
untouched since it genuinely forwards to ultralytics' multi-device-capable resolver.
Tests: `tests/test_trainer.py` (`resolve_device("0") == torch.device("cuda:0")`, explicit
spellings pass through, `None` matches `torch.cuda.is_available()`).

**Verify:** `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"`
(238 passed, 3 new). **M0.10 is now fully closed.** M1 (Phase 1 GAN loop) is next.

---

## M1 — Phase 1: GAN loop (the go/no-go gate)

- [x] Vendor `pytorch-CycleGAN-and-pix2pix` at `2a7afba` — **`models/networks.py` only**
      (`define_G`, `define_D`, `GANLoss`). Not `BaseModel`, `options/`, `data/`, `train.py`
- [x] `translators/pix2pix.py` wrapper implementing the Translator protocol.
      `translate` is `netG(x)`
- [x] Wire the detection loss where `fake_B` is already un-detached
- [x] Smoke-test the full loop on the fixture
- [x] Train on the custom dataset, λ_det = 0 → baseline translation quality. Run on the
      server via `experiments/pix2pix_baseline.yaml`: **mAP50 0.9199**
- [x] Train with λ_det > 0 → the loop arm. Run on the server via
      `experiments/pix2pix_loop.yaml`, 100 epochs per stage:

      | stage | task_weight | mAP50 |
      | --- | --- | --- |
      | 0 | 0.0 | 0.9053 |
      | 1 | 1.0 | 0.9191 |
      | 2 | 2.0 | 0.8984 |
      | 3 | 3.0 | 0.9132 |

- [x] Evaluate both through the single evaluation path — **the metric above is the wrong
      one, and finding that out is what this step produced.** See "The reported mAP cannot
      answer the gate" below. `engine/loop.py` now records a second, zero-shot arm per stage.
- [x] Run the zero-shot evaluation on the two completed runs (server; validation passes
      only, no retraining — the exports already existed). Results in the gate section below.

**GATE:** if translated mAP does not beat raw-thermal mAP on at least one class, **stop**.
Re-frame around the low-annotation regime before escalating to diffusion.

- [x] Record the gate decision and evidence in this file before proceeding

### GATE DECISION: **PASS** — proceed to M2 (Phase 2, diffusion)

Every number below is one detector — `optical_best_n_1.pt`, visible-trained, **never
fine-tuned on anything this project produced** — run through `t2o evaluate` on the identical
val split (153 images, 423 instances). This is §12's zero-shot arm.

| arm | mAP50 | mAP50-95 | Fuse | Pole | Switch | Transformer |
| --- | --- | --- | --- | --- | --- | --- |
| raw thermal (floor) | 0.1887 | 0.0829 | 0.0377 | 0.5551 | 0.0047 | 0.1573 |
| pix2pix baseline, λ_det=0 | **0.7751** | 0.5244 | 0.8582 | 0.9592 | 0.4320 | 0.8512 |
| loop stage 0, λ_det=0 | 0.7622 | 0.5031 | 0.8199 | 0.9526 | 0.3966 | 0.8799 |
| loop stage 1, λ_det=1 | 0.8218 | 0.5784 | 0.8874 | 0.9558 | 0.5478 | 0.8961 |
| loop stage 2, λ_det=2 | 0.8332 | 0.5536 | 0.8729 | 0.9670 | 0.5781 | 0.9149 |
| loop stage 3, λ_det=3 | **0.8696** | 0.5841 | 0.8942 | 0.9695 | 0.7127 | 0.9019 |
| real visible (ceiling) | 0.9213 | 0.6530 | 0.9448 | 0.9624 | 0.8226 | 0.9554 |

**The gate passes on the clean arm, by a wide margin and on all four classes.** The bar was
"beat raw thermal on at least one class". The λ_det=0 baseline — which never constructs the
in-loop detector at all (`coupling/schedule.py`), so it cannot be contaminated by it — scores
0.7751 against the thermal floor's 0.1887, **+0.586 mAP50**, with every class improving and
Fuse (0.0377 → 0.8582) and Switch (0.0047 → 0.4320) going from unusable to usable. C4's
premise ("does translation beat direct thermal detection") is answered affirmatively without
needing the loop at all.

**Correction to M0.10's E1 bracket: the raw-thermal floor is 0.1887, not < 0.05.** That entry
was recorded from an ad-hoc measurement; this one goes through `metrics.task.evaluate_detector`
on the same val split as every other row, so it is the number to cite. It does not change any
conclusion — the floor is still far below every translated arm — but the paper must not
report < 0.05.

**λ_det improves the zero-shot metric monotonically, and this is the finding that needs
defending, not celebrating yet.** Against the epoch-matched control (loop stage 0, λ=0,
0.7622): λ=1 → +0.060, λ=2 → +0.071, λ=3 → **+0.107**. Against the baseline arm, stage 3 is
+0.095. For scale, the noise floor between two runs of the *same* configuration (baseline vs.
loop stage 0, identical computation) is 0.0129 on this arm — so stage 3's gain is ~8x noise
and comfortably past PLAN.md §16's "+2–4 mAP50 over the strongest baseline" criterion. The
gain is concentrated in **Switch**, the hardest class and the one raw thermal fails on
completely (0.0047): 0.3966 → 0.7127. Stage 3 reaches 94% of the real-visible ceiling, up
from 83%.

**Why this is not yet reportable, and what has to happen before M2 results are trusted:** the
λ_det > 0 stages are graded by *the same checkpoint that supplied their training gradient*
(one optical `.pt` serves as `in_loop.weights` and, by fallback, `detector.reference.weights`
— `engine/loop.py` warns about exactly this). A monotone gain in λ_det is precisely the shape
reward hacking produces. Two pieces of evidence would separate them, neither of which exists
yet:

1. **An independently-trained visible detector as judge** (different seed, ideally different
   architecture). Until then the honest claim is the gate itself, which rests entirely on the
   uncontaminated λ_det = 0 arm.
2. **Fidelity metrics.** ~~`FidelityEvaluator` (M0.5) still has no caller in `engine/`~~ —
   **built, see M1.1 below.** If LPIPS/FID degrade as λ_det rises while zero-shot mAP
   climbs, that is reward hacking; if they hold, the gain is real. Runnable against the two
   completed runs with no retraining.

**The metric change is what produced this result.** The adapted arm reported
0.9199 / 0.9053 / 0.9191 / 0.8984 / 0.9132 — flat, and it ranked stage 3 (0.9132) *below* the
baseline (0.9199). The zero-shot arm ranks stage 3 **+0.095 above** it. The two arms do not
merely differ in precision; they invert the ordering. Anything measured on the adapted arm
alone should be treated as uninformative.

**Vendor + wrapper decisions:**

- **Pin resolved to `2a7afba2895d52556dd5dfe07e8555ef657ced6f` (2025-08-06)** — the local
  `../pytorch-CycleGAN-and-pix2pix` sibling checkout was ~500 commits stale (HEAD `c3268ed`,
  2024-03-22); `git fetch origin` in that checkout (read-only, no working-tree change) made
  the pin reachable. `models/networks.py` at that commit is self-contained (no `BaseModel`/
  `opt` imports), confirming PLAN.md §7's vendoring thesis directly rather than assuming it.
- **Vendored to `third_party/pix2pix/networks.py` + `LICENSE`, byte-identical**, plus empty
  `third_party/__init__.py` / `third_party/pix2pix/__init__.py`. Verified byte-identical
  against the pinned commit as the very last step before committing — `ruff format .`
  reformats any file it can reach regardless of `[tool.ruff] src`'s allowlist (that setting
  only affects import-classification, not file discovery), so it silently rewrote the
  vendored file in place the first time this was run. **Fixed**: `[tool.ruff]
  extend-exclude` gained `"third_party"` alongside the existing `"*.md"` entry — the same
  fix `*.md` already needed, for the same reason (never rewrite text we don't own).
  `[tool.pyright] include` is a genuine allowlist (`["src", "scripts", "tests"]`) and already
  excluded `third_party/` by omission with no change needed — confirmed by running `pyright`
  after the fact and checking no diagnostics were reported *for* `third_party/pix2pix/
  networks.py` itself (only for our own call sites into it).
- **Packaging fix required for the vendor path to import at runtime at all**:
  `[tool.hatch.build.targets.wheel] packages` gained `"third_party"` alongside `"src/t2o"`.
  `third_party/` sits beside `src/t2o`, not inside it (PLAN.md §5), and `t2o` runs as an
  installed console script rather than `python -m` from repo root, so cwd-based import
  tricks would not reliably resolve `third_party.pix2pix.networks`. This is the same
  mechanism that already makes `import t2o` cwd-independent, generalised to a second
  top-level package. Verified directly, not assumed: `import third_party.pix2pix.networks`
  from `/tmp` (not the repo root) resolves cleanly after `uv sync`.
- **`init_weights`, not `init_net`.** At this pinned commit, `define_G`/`define_D` accept
  but never *apply* `init_type`/`init_gain` — that used to be `init_net`'s job, called from
  `BaseModel`, out of vendor scope. `init_net` also hardcodes CUDA device placement (PLAN.md
  §7: "our wrapper owns device placement"). `Pix2PixTranslator.__init__` calls
  `networks.init_weights(net, init_type, init_gain)` directly after each `define_G`/
  `define_D` call instead, then does its own `.to(device)` implicitly via the caller moving
  the whole module.
- **Generator defaults to `resnet_9blocks`, not the paper's `unet_256`.** `UnetGenerator`
  needs input divisible by 2**8=256; the custom dataset is 640x480 and `train.crop` only
  guarantees stride-32 divisibility (PLAN.md §8). `ResnetGenerator` only needs
  divisible-by-4 — works on the synthetic fixture, the real dataset, and any future one.
- **Reconstruction/adversarial losses reuse the shared `LossConfig.{l2,lpips,gan}`**, not
  pix2pix's own `lambda_L1=100` — pix2pix becomes the second consumer of the same three
  fields `StubTranslator` already established, so every backbone stays comparable through
  one set of knobs (PLAN.md §8). `loss.gan` defaults to `0.0` (its own existing docstring:
  "the cheaper and more stable starting point"), so **`net_d` is never constructed by
  default** — the same "clean no-op at weight 0" discipline `CouplingConfig` already uses
  for the frozen in-loop detector. `loss.lpips` defaults to `5.0`, so an LPIPS network
  (`torchmetrics` `LearnedPerceptualImagePatchSimilarity`, `normalize=True` — the same module
  `FidelityEvaluator` already uses) is built by default; also only when the weight is `> 0`.
- **New config surface** (`Pix2PixTranslatorConfig`): `net_g`, `net_d`, `ngf`, `ndf`,
  `gan_mode` — the knobs actually worth sweeping. `gan_mode` is `Literal["vanilla",
  "lsgan"]`, deliberately excluding the paper's third option `wgangp`: that needs
  `cal_gradient_penalty` wiring this milestone doesn't build, so it's rejected at
  config-load time rather than silently behaving like a no-op. `norm="batch"`, `init_type=
  "normal"`, `init_gain=0.02`, `beta1=0.5`, `n_layers_d=3`, and the training-time LPIPS
  backbone (`"alex"`) are fixed constants inside the wrapper, not config fields — no signal
  to sweep them in Phase 1, and every config field carries an ongoing documentation cost
  (house style, PLAN.md §13). `input_nc`/`output_nc` are hardcoded too (1/3 channels,
  exactly what the data contract always produces), matching how `StubTranslator` already
  hardcodes its own conv channel counts.
- **One consistent `[0,1]` convention throughout `fit()`, not a `[-1,1]` detour.**
  `ResnetGenerator`'s last layer is `Tanh` (`[-1,1]`), so `translate()` remaps once at its
  own boundary (`(x + 1) / 2`) to satisfy the `Translator` protocol. Inside `fit()`, an
  earlier draft kept a separate raw-Tanh tensor around for the discriminator to mirror
  upstream's "`fake_B` is already un-detached" framing literally — caught in self-review:
  the affine remap is linear and differentiable, so it does not affect detachment either
  way, and keeping two representations around actually introduced a real bug (concatenating
  a `[0,1]` infrared channel with a `[-1,1]` visible channel for the discriminator's real
  pair). Fixed before committing: `fake_b` (already `[0,1]`) is the only tensor used for
  L2/LPIPS/GAN/detection-loss alike, so `real_a`/`real_b`/`fake_b` are always on the same
  scale for the discriminator regardless of which pair it's shown.
- **Tests split fast/slow exactly like `FidelityEvaluator`'s own precedent (M0.5)**: fast
  tests construct with `loss_gan=0.0, loss_lpips=0.0` so neither the discriminator nor the
  LPIPS network is ever built; slow tests cover real defaults (all loss terms finite) and
  one direct `engine.loop.run_loop` pass with `backbone: pix2pix` on the synthetic fixture,
  the pix2pix analogue of `test_experiments.py`'s stub-only loop test. One convergence test
  needed a second self-review fix: driving `loss_l2` down over 20 steps with `loss_lpips`
  also active (its 5x-heavier default weight) is not reliable on random-noise batches, since
  the optimizer follows the *combined* gradient — isolated to `loss_lpips=0.0` too, matching
  `StubTranslator`'s own convergence test, which tests the same thing without the confound.
- **Also found and fixed while running the full suite**: two *existing* tests assumed
  `"pix2pix"` was still an invalid `backbone` tag (`test_config.py`'s "unknown backbone"
  case, predating this milestone) and that `config.translator` was still a one-member union
  (`test_stub_translator.py`'s `hidden_channels` test, no longer narrowable without an
  explicit `isinstance` check now that a second union member exists) — both updated; the
  first now uses a genuinely-still-invalid tag (`pix2pix_turbo`) and gained a sibling
  positive test for the real `pix2pix` tag.

**Verify:** `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"`
(248 passed, 10 new), `pytest -m slow` (14 passed, 3 new). Vendored file re-diffed against
the pinned commit as the final check — byte-identical.

**`experiments/pix2pix_baseline.yaml` (λ_det=0) / `pix2pix_loop.yaml` (the real 4-stage
ramp) — the two tracked configs the remaining M1 bullets run.** Decisions:

- **`loss.gan: 1.0` in both**, not the schema's own `0.0` default — these two experiments
  specifically exist to test pix2pix's actual GAN recipe (TASKS.md M1's own wording: "the
  GAN loop"), so leaving the adversarial term off by default here would make "baseline
  translation quality" a GAN-less regression stand-in instead of the method being gated.
  `train.lr: 2.0e-4` also departs from the schema's generic `1.0e-4` default, matching the
  pix2pix paper's own tuned value.
- **The two files are identical except `coupling.task_weights` and `runtime.name`** — kept
  as two explicit files rather than reaching for the `base:`-include inheritance mechanism
  `config/schema.py`'s M0.2 notes deferred pending "≥3 experiment files to compare." That
  threshold is technically met now (`smoke.yaml` + these two), but building an inheritance
  mechanism is a bigger feature than two YAMLs differing by two fields warrants right now —
  noted here as a real candidate to revisit, not silently dropped.
  `test_pix2pix_experiments.py::test_baseline_and_loop_configs_differ_only_by_design` pins
  this invariant directly so the two files can't silently drift apart on any other field.
- **`train.epochs_per_stage`/`batch_size` are a starting point, not a measured optimum.**
  M0.10's VRAM check only profiled the stub translator + detector fine-tuning, not
  `ResnetGenerator`/`NLayerDiscriminator` specifically — said explicitly in both files'
  header comments rather than presented as tuned numbers.
- **Not total-epoch-budget-matched between the two arms** (the loop arm's 4 stages run
  4x `epochs_per_stage` in total, warm-started per PLAN.md §8's schedule design) —
  acceptable for M1's go/no-go gate check; a budget-matched ablation is E3's job later
  (PLAN.md §11), not this one. Stated in both files' headers so it isn't mistaken for an
  oversight when E3 actually runs.
- **`detector.in_loop.weights` matters only in `pix2pix_loop.yaml`** (stages 1–3 have
  λ_det > 0; `pix2pix_baseline.yaml`'s single `[0.0]` stage never constructs a detector at
  all, per `coupling/schedule.py`) — both left at the `yolo11n.pt` schema placeholder,
  documented as needing `--in-loop-weights`/`--eval-init-weights`/`--data` overrides (or a
  direct edit once the real server paths are known here too), same convention
  `experiments/smoke.yaml` already established.
- **`runtime.wandb: false` in both**, not `true` — there is no `--no-wandb` flag (`cli.py`'s
  `--wandb` can only turn it on, never off), so leaving it off in the tracked file and
  opting in per-invocation with `--wandb` is the reversible choice.
- Tests (`tests/test_pix2pix_experiments.py`): both files load with the right shape; hashes
  are stable across loads; the identical-except-two-fields invariant above; one `slow`
  end-to-end `run_loop` pass on the synthetic fixture through `pix2pix_loop.yaml` itself
  (not just `Pix2PixTranslator` in isolation), proving the tracked file's own schema is
  actually loadable and drives the loop correctly.

**Verify:** `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"`
(253 passed, 5 new), `pytest -m slow` (15 passed, 1 new).

**Out of scope for this step, needs the server** — running these two configs against the
real ~850-pair dataset and detector weights, then evaluating both through
`metrics.task.evaluate_detector`:

```
uv run t2o loop --config experiments/pix2pix_baseline.yaml --data <real data.yaml> \
    --in-loop-weights <visible-trained.pt> --eval-init-weights <visible-trained.pt> \
    --device cuda:0
uv run t2o loop --config experiments/pix2pix_loop.yaml --data <real data.yaml> \
    --in-loop-weights <visible-trained.pt> --eval-init-weights <visible-trained.pt> \
    --device cuda:0
```

Report the mAP numbers back once run; the gate decision above gets recorded here at that
point, not before.

**The reported mAP cannot answer the gate — the zero-shot arm (`detector.reference`).**

Both runs completed and every stage landed between 0.8984 and 0.9199. That number is
`DetectorResult.map50`, which comes from `train_detector`: the evaluation detector
**fine-tuned for 50 epochs on that stage's translated train images, then validated on its
translated val**. It is an in-domain, post-adaptation number, and it cannot discriminate
between the arms for two independent reasons:

- **It is saturated.** M0.10's E1 bracket already put a same-domain-trained detector above
  0.9 mAP50 on *raw thermal* as well. So ~0.92 on translated images says a YOLO fine-tune
  converges on this dataset whichever domain it is shown — not that translation did
  anything.
- **The spread is inside the noise floor.** `pix2pix_baseline.yaml`'s single stage and
  `pix2pix_loop.yaml`'s stage 0 are *the same computation*: same seed, same 100 epochs,
  `task_weight=0`, and the two files differ only in `coupling.task_weights` and
  `runtime.name` (pinned by
  `test_pix2pix_experiments.py::test_baseline_and_loop_configs_differ_only_by_design`).
  They returned 0.9199 and 0.9053 — a 1.5-point gap with no experimental difference behind
  it. Every loop stage falls within that band.

It also quietly defeats the research framing: fine-tuning a detector on translated images
needs exactly the thermal-domain annotations the low-annotation story (E8) exists to avoid
depending on.

The gate's bar is the **unadapted** visible-trained detector on raw thermal — 0.1887 mAP50,
re-measured through this same path as part of the gate evaluation (M0.10's "< 0.05" was
ad-hoc; see the correction in the gate section above). The measurement that compares against
it is that same unadapted detector run on *translated* images — which `run_loop` never
computed. It does now:

- **`ReferenceDetectorConfig`** (`config/schema.py`) is a third detector role beside
  `in_loop` and `evaluation`, extending the M0.2 structural encoding of invariant 7 rather
  than overloading either existing one. `weights: null` falls back to
  `evaluation.init_weights` — the un-fine-tuned bootstrap, i.e. "the detector you already
  have because you could label visible data for it", which is precisely the detector the
  gate is about. Participates in `config_hash()`: it changes the reported number.
- **`StageResult.zero_shot: TaskMetrics | None`** (`engine/loop.py`), filled by
  `metrics.task.evaluate_detector` — the same single evaluation path `evaluate` and E1 use
  (invariant 1), no new metric code. Called **before** `train_detector`, so the gate number
  survives a detector fine-tune that OOMs or diverges, and costs one extra val pass.
  `_stage_result_from_json` reads it with `.get`, so the two completed runs' existing
  `metrics.json` files still `--resume` instead of dying on a `KeyError` mid-run.
- **The adapted arm is kept, not replaced.** It is still the right number for E7 (detector
  identity) and for "how good can a detector get on these images". `cli.py`'s summary now
  labels both explicitly (`zero-shot mAP50 ... | fine-tuned mAP50 ...`) — reading the
  adapted number as the gate metric is the specific mistake that wording exists to prevent.
- **Contamination warning.** Only one optical `.pt` exists on the server, passed as both
  `in_loop.weights` and `eval_init_weights`, so the fallback makes the zero-shot judge the
  same checkpoint that supplied the training gradient. `_resolve_reference_weights` warns at
  run start when that is true *and* some `task_weight > 0`. It warns rather than raising
  because the λ_det = 0 arm never constructs the in-loop detector at all
  (`coupling/schedule.py`), so the baseline's zero-shot number is clean regardless — and
  that arm alone decides the gate. **An independently-trained visible detector is needed
  before the λ_det > 0 numbers are reported anywhere.**

**Server commands to close the gate** (validation only — no training, the exports at
`runs/<name>/stage*/translated/data.yaml` already exist):

```
uv run t2o evaluate --weights <optical.pt> --data <thermal data.yaml> --device 0
uv run t2o evaluate --weights <optical.pt> --data <real paired data.yaml> --device 0
uv run t2o evaluate --weights <optical.pt> \
    --data runs/pix2pix-baseline/stage0/translated/data.yaml --device 0
uv run t2o evaluate --weights <optical.pt> \
    --data runs/pix2pix-loop/stage<N>/translated/data.yaml --device 0   # N = 0..3
```

The first two re-establish the bracket (raw-thermal floor, real-visible ceiling) on the
identical val split and evaluation path. **These were run; results and the gate decision are
in the "GATE DECISION" section above.**

**Verify:** `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"`
(260 passed, 7 new), `pytest -m slow`.

---

## M1.1 — Fidelity metrics wired (the reward-hacking check)

The gate passed, but M1's λ_det trend is monotone *and* self-graded, which is the shape
reward hacking produces. A detection metric alone cannot tell the two apart. This wires the
other half of the check — PLAN.md §8's fidelity floor, which `FidelityEvaluator` (M0.5) has
had the machinery for since it was written but no caller for.

- [x] `metrics/fidelity.py::evaluate_fidelity` — pairs an exported translated split against
      the real visible frames and returns all five metrics through the existing
      `FidelityEvaluator` (invariant 1: no second implementation)
- [x] `engine/loop.py` — `StageResult.fidelity`, computed on the same export the zero-shot
      arm scores, logged to the tracker under `stage{N}/fidelity/*`
- [x] `t2o fidelity` subcommand — post-hoc scoring of any finished run, which is what M1's
      two completed runs need
- [x] Run it on both completed runs and record the numbers beside the gate table

### Result: **no reward hacking** — but λ_det's benefit is not yet causally established

All five metrics on the exported val split (153 images), against the real visible frames,
with M1's zero-shot mAP50 alongside:

| arm | zero-shot mAP50 | LPIPS ↓ | FID ↓ | KID ↓ | SSIM ↑ | PSNR ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| baseline, λ=0, 100 ep | 0.7751 | 0.3059 | **87.24** | **0.0210** | 0.4528 | 15.50 |
| loop stage 0, λ=0, 100 ep | 0.7622 | 0.2988 | 104.82 | 0.0301 | 0.4979 | 15.34 |
| loop stage 1, λ=1, 200 ep cum. | 0.8218 | 0.2828 | 96.13 | 0.0283 | 0.5002 | 15.08 |
| loop stage 2, λ=2, 300 ep cum. | 0.8332 | **0.2811** | 92.69 | 0.0265 | **0.5027** | 15.66 |
| loop stage 3, λ=3, 400 ep cum. | **0.8696** | 0.2825 | 90.85 | 0.0273 | 0.4914 | **15.69** |

**Reward hacking is ruled out, which is what this milestone existed to test.** Its signature
is a detection metric climbing while perceptual fidelity degrades. Across the loop's four
stages mAP50 rises 0.7622 → 0.8696 while **every** perceptual metric improves or holds:
LPIPS falls 0.2988 → 0.2825, FID falls monotonically 104.82 → 90.85, KID falls
0.0301 → 0.0273, SSIM is flat. The translator is not buying detections with adversarial
texture. `reward_target` stays `null` and `grad_scale` stays `1e-2` — nothing here argues
for retuning them before M2.

**But the stage-to-stage trend cannot attribute that gain to λ_det**, and two things in this
table say so:

1. **The stages are not budget-matched and not independent.** Stage 3 is the same translator
   after 400 cumulative warm-started epochs; stage 0 after 100. Every stage-to-stage
   improvement is confounded with simply training longer. TASKS.md flagged this when the
   configs were written ("a budget-matched ablation is E3's job"); it is now the *binding*
   limitation rather than a footnote, because the trend is the headline.
2. **Run-to-run variance is much larger than the mAP noise floor suggested.** The baseline
   and loop stage 0 are the same computation — same config bar `task_weights`, same seed,
   same 100 epochs — yet they differ by **17.6 FID** (87.24 vs 104.82), 0.009 KID, and 0.045
   SSIM. On mAP50 they differed by only 0.0129, which made the noise floor look small. It is
   not: 100 epochs of GAN training on a GPU is not reproducible run-to-run even at a fixed
   seed (cuDNN non-determinism in conv backward), and fidelity metrics expose that far more
   than mAP does. **The loop's whole 14-point FID improvement is inside one sample of that
   variance.**

Note the direction of that discrepancy: the baseline has the *best* FID and KID of any arm
while having the *worst* LPIPS and SSIM. That is not a translator being better or worse; it
is what two draws from a noisy process look like.

**What is and is not established:**

| claim | status |
| --- | --- |
| Translation beats raw thermal | **Established** (M1 gate, uncontaminated λ=0 arm) |
| The loop does not reward-hack | **Established** — the point of this milestone |
| λ_det causally improves detection | **Not established** — confounded with epoch budget and run variance |

Resolving the third needs E3 as designed (PLAN.md §11): budget-matched arms, ≥3 seeds,
and an independently-trained reference detector. Given the variance measured here, **≥3
seeds is not optional rigour, it is the minimum to say anything at all.**

**Also worth reporting honestly: the absolute fidelity is poor.** PSNR ~15.5, SSIM ~0.50,
LPIPS ~0.28, FID ~90 are weak numbers for an image-translation paper — expected for pix2pix
on ~850 pairs, and part of why M2's diffusion backbone exists. It also *supports* PLAN.md
§12's argument rather than undermining it: detection transfer works well (0.87 mAP50, 94% of
the real-visible ceiling) on images these metrics score as mediocre reconstructions. Pixel
fidelity is measuring something other than what the downstream task needs, which is the
paper's own claim.

**Found incidentally — the custom dataset declares 5 classes but only 4 are annotated.**
`DatasetManifest` logged `5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer']`
off the server's real `data.yaml`; PLAN.md §9 recorded 4 and omitted Connector (corrected
there). Connector appears in no per-class AP table from any M1 evaluation. **Resolved with
the user: it is a Label Studio artifact — the class was created in the labelling project and
never used, so it has zero instances in train *and* val.** Not a split problem; the initial
concern that val might under-sample a class train contains does not apply.

Left as-is deliberately rather than dropping it to `nc: 4`. Connector is index **0**
(alphabetically first), so removing it would renumber every other class in every label file
on the server — a migration with real corruption risk for no measurable gain. The
consequences of leaving it are all benign and worth stating so nobody re-derives them later:

- `v8DetectionLoss` carries one class logit that never receives positive supervision. Costs
  nothing but a negligible slice of the head.
- `metrics/task.py::_extract_per_class_ap` already omits zero-instance classes rather than
  reporting a misleading `0.0` (M0.5's own decision), which is exactly why this surfaced as
  an absence rather than as four-fifths-of-nothing dragging the table down.
- ultralytics averages mAP over classes actually present, so **no reported mAP is diluted**
  by the dead class. Every number in M1's gate table is a 4-class average, as intended.
- **The paper must report 4 classes**, not the `nc: 5` the manifest declares.

**Decision — fidelity is scored on the exported PNGs, not the translator's float output.**
Those files are the exact bytes both detectors were scored against, so fidelity and mAP
describe the same artifact; scoring the float tensors would measure an image no reported
detection number ever saw, and would leave `to_uint8`'s quantisation unmeasured. It also
makes the metric recomputable for any finished run without loading a checkpoint — which is
the only reason M1's existing runs can be scored at all. Consequence worth stating: these
numbers include export quantisation, and are therefore very slightly pessimistic relative to
what the translator emitted.

**Decision — `data.dataset._to_tensor` became public `load_image_tensor`.** `evaluate_fidelity`
reads images off disk and must decode them exactly the way training did; a second decoder
with its own scaling convention would silently make fidelity numbers incomparable to
everything else. Reused rather than reimplemented.

**Decision — pools are paired by filename *stem*, and a partial overlap warns.** `export.py`
writes `.png` regardless of the source suffix (`.jpg` here), so full-filename matching would
find nothing. A fully disjoint pair raises `FidelityError` rather than scoring an empty
intersection — that means the two directories are not the same split, which is a bug, not a
degenerate case to average over.

**Decision — `t2o fidelity` is standalone, taking paths rather than a `Config`.** Same
spirit as `evaluate` (M0.8): it only ever scores artifacts that already exist on disk and
needs no experiment definition to say what it is measuring. It accepts either the export root
(`.../translated`) or the images directory itself, because `run_loop` names the former and
the `data.yaml` inside it names the latter, and both are things that get pasted into a shell.

**Not yet answered — what the numbers say.** The code is verified; the experiment is not run.
Commands for the two completed runs:

```
uv run t2o fidelity --translated runs/pix2pix-baseline/stage0/translated \
    --data <real paired data.yaml> --device cuda:0
uv run t2o fidelity --translated runs/pix2pix-loop/stage<N>/translated \
    --data <real paired data.yaml> --device cuda:0      # N = 0..3
```

Read against the zero-shot mAP column of M1's gate table. LPIPS and FID are the ones that
matter (both lower-is-better); PSNR/SSIM reward blur and will look best on the least
interesting translator (PLAN.md §12). **If LPIPS/FID hold flat or improve from stage 0 to
stage 3 while zero-shot mAP climbs 0.7622 → 0.8696, the λ_det gain is real. If they degrade
monotonically as mAP rises, it is reward hacking and the ramp needs `reward_target`/
`grad_scale` retuning before M2.**

---

## M1.2 — E3: is the λ_det gain causal? **Yes, at a calibrated dose** (step 8)

M1.1 left one claim open, and it is the project's central one. Three defects block it, all
measured rather than suspected:

1. **Not budget-matched.** The loop's stage 3 is the translator after 400 warm-started
   epochs; its λ=0 comparator ran 100. Every stage-to-stage gain is confounded with simply
   training longer.
2. **Self-graded.** One `yolo11n` checkpoint is both `detector.in_loop.weights` and — via
   `reference.weights: null` — the zero-shot judge. `engine/loop.py::_resolve_reference_weights`
   already warns about exactly this.
3. **n = 1.** Baseline and loop stage 0 are the same computation at the same seed, yet differ
   by 17.6 FID and 0.045 SSIM.

PLAN.md §11 calls E3 "the most important experiment in the project"; §16 makes it the
**causality** acceptance criterion. Scope here is the **pix2pix arm only** — the
`pix2pix-turbo` cell waits on M2a and reuses every piece of tooling below.

**Outcome — read the steps in order, because the answer reverses at step 8.**

| step | dose | stage-3 `zero_shot.map50` | verdict |
| --- | --- | --- | --- |
| 6 | `grad_scale: 1.0e-2` (0.9–2.3% of the objective) | +0.0070, p = 0.66 | negative |
| 7 | — | — | the null is **dose-limited**, not a mechanism result |
| 8 | `grad_scale: 0.15` (10–19.8%) | **+0.0512, p = 0.031** | **positive** |

§16's causality criterion is **satisfied for pix2pix**, with two caveats step 8 states in full:
a wide stage-0 null (resolved by the trajectory contrast, +0.0909) and a fidelity cost
(+0.0097 LPIPS) that step 9 exists to characterise. The effective λ_det is
`task_weights × grad_scale` — 0.15/0.30/0.45 in the reported campaign, never 1/2/3.

### Design

| arm | `coupling.task_weights` | translator epochs | in-loop detector |
| --- | --- | --- | --- |
| control | `[0, 0, 0, 0]` | 400, warm-started | **never constructed** |
| loop | `[0, 1, 2, 3]` | 400, warm-started | constructed from stage 1 |

Everything else — warm-start, four exports, four evaluation-detector fine-tunes, the seed —
is identical per pair. **The only difference in the entire computation is λ_det.** The
control needs no new machinery: `build_detection_loss` returns `None` at weight 0
(`coupling/schedule.py`), so an all-zero ramp is the same code path with the coupling term
absent, four times over.

**Stage 0 is a free null control.** Both arms are λ=0 there, so the paired stage-0 difference
at each seed measures run-to-run noise *from inside the experiment itself*, at no extra GPU
cost. If the stage-3 effect is not clearly larger than the stage-0 difference, E3 is negative
and gets reported that way.

**Six seeds, and the number is not arbitrary.** The comparison is paired (seed *i*'s control
vs seed *i*'s loop), and the assumption-free test on paired data is an exact sign-flip
permutation over 2ⁿ assignments. Smallest attainable two-sided p: 0.25 at n=3, 0.0625 at
n=5, **0.031 at n=6** — n=6 is the first size where a distribution-free two-sided test can
clear 0.05 *at all*, whatever the effect size. At ~6h per 4-stage run that is ~72 GPU-hours.
Falling back to 3 seeds is allowed but then the writeup says "consistent across 3 seeds", not
"significant".

### Step 1 — an honest judge, and a gate before spending three days

- [x] `engine/detector_stage.py::train_detector` takes explicit parameters instead of a
      `Config`. It now has two callers with different provenance: `engine/loop.py` (evaluation
      detector, settings from `detector.evaluation`/`train`) and `cli train-detector`
      (reference detector, paths only). One implementation, no role flag inside it — the roles
      stay separated by *who calls it with which weights* (invariant 1 and invariant 7 both
      hold)
- [x] `t2o train-detector` subcommand, standalone like `evaluate`/`fidelity` (M0.8). Defaults
      chosen to make independence the easy path: `--init-weights yolo11s.pt` (a different
      architecture from the in-loop `yolo11n`) and `--seed 1` (not `train.seed`'s 0)
- [x] **Bug found by the first server run: a relative `project` escapes to another repo.**
      ultralytics does not resolve a relative `project` against the cwd —
      `cfg/__init__.py::get_save_dir` appends it under `SETTINGS["runs_dir"]/<task>`, and
      that setting is a machine-global default frozen to whichever git root was current the
      first time ultralytics wrote its `settings.json`. `--out runs/reference-yolo11s` from
      this repo therefore landed in
      `D:\Atick\GitHub\Thermal-Image-Research\runs\detect\runs\reference-yolo11s` — the
      doubled `runs` being the `runs_dir / task / project` composition. Fixed by resolving
      inside `train_detector`, which covers the loop's per-stage detector directories too.
      **M1's numbers are unaffected**: `_resolve_weights` reads the trainer's real save_dir,
      so stage-to-stage warm-starting always chained the correct checkpoint — only the files
      sat in the wrong repository. `evaluate_detector` still writes ultralytics' throwaway
      `val/` scratch dirs there; nothing reads them, so this is left alone rather than
      inventing a location for output no one consumes
- [x] **Server:** train the judge on the **visible train split only**, validating on visible
      val — `yolo11s`, seed 1, 100 epochs: **P 0.8931 R 0.9030 mAP50 0.9364 mAP50-95 0.6822**
      on the 153-image / 423-instance val split
- [x] **Server:** re-score everything M1 already produced under the new judge — validation
      passes only, minutes, no retraining. Results and gate decision below

```
uv run t2o train-detector --data <real paired data.yaml> --init-weights yolo11s.pt \
    --epochs 100 --seed 1 --out runs/reference-yolo11s --device cuda:0

# then, with the printed weights path:
uv run t2o evaluate --weights runs/reference-yolo11s/weights/best.pt --data <thermal data.yaml> --device 0
uv run t2o evaluate --weights runs/reference-yolo11s/weights/best.pt --data <real paired data.yaml> --device 0
uv run t2o evaluate --weights runs/reference-yolo11s/weights/best.pt \
    --data runs/pix2pix-baseline/stage0/translated/data.yaml --device 0
uv run t2o evaluate --weights runs/reference-yolo11s/weights/best.pt \
    --data runs/pix2pix-loop/stage<N>/translated/data.yaml --device 0     # N = 0..3
```

**GATE — three things to read off that table before any campaign starts:**

- **Does the λ_det ordering survive an honest judge?** If stage 3 no longer beats stage 0,
  M1's trend was self-grading and E3's hypothesis changes before 72 GPU-hours are spent.
- **Was the old `yolo11n` trained on val?** Its recorded provenance is "the dataset with
  ultralytics defaults", which does not confirm a held-out split. If the old judge scores
  *real visible* far above what the new train-split-only judge does, the old one memorised
  val — which would inflate M1's 0.9213 "ceiling" **and** every translated arm sharing val's
  scene layout. **Answered, favourably: no evidence of leakage.** The new judge, trained on
  train and validated on held-out val, scores **0.9364** where the old one scored 0.9213.
  Memorisation would have pushed the old number *above* the clean one, not below; the gap is
  the other way and is the size expected from `yolo11s` being the larger model. M1's ceiling
  row stands.
- **Judge-to-judge agreement** is itself a reportable number.

### GATE DECISION: **proceed with E3** — the trend survives, the noise floor is 4.6× larger

Same 153-image / 423-instance val split, same `metrics.task.evaluate_detector` path, only the
judge differs. The old judge is `yolo11n` — the checkpoint that supplied the in-loop gradient.
The new one is `yolo11s`, seed 1, trained on visible train and never on anything this project
produced.

| arm | old mAP50 | **new mAP50** | Δ | new mAP50-95 | Fuse | Pole | Switch | Transformer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw thermal (floor) | 0.1887 | **0.1552** | −0.034 | 0.0675 | 0.0321 | 0.3949 | 0.0130 | 0.1810 |
| baseline, λ=0, 100 ep | 0.7751 | **0.7851** | +0.010 | 0.5082 | 0.8515 | 0.9514 | 0.5279 | 0.8095 |
| loop stage 0, λ=0, 100 ep | 0.7622 | **0.7260** | −0.036 | 0.4806 | 0.8172 | 0.9440 | 0.4128 | 0.7300 |
| loop stage 1, λ=1, 200 ep | 0.8218 | **0.8244** | +0.003 | 0.5503 | 0.8791 | 0.9679 | 0.5336 | 0.9169 |
| loop stage 2, λ=2, 300 ep | 0.8332 | **0.8106** | −0.023 | 0.5421 | 0.8596 | 0.9724 | 0.5127 | 0.8978 |
| loop stage 3, λ=3, 400 ep | 0.8696 | **0.8470** | −0.023 | 0.5593 | 0.8852 | 0.9696 | 0.6175 | 0.9156 |
| real visible (ceiling) | 0.9213 | **0.9366** | +0.015 | 0.6831 | 0.9746 | 0.9719 | 0.8507 | 0.9492 |

**1. M1's gate is confirmed by a detector that supplied no gradient to anything.** Baseline
0.7851 against the thermal floor's 0.1552 is **+0.630 mAP50**, all four classes improving.
Nothing about the headline claim depended on the contaminated judge.

**2. Self-grading was real but small — about 2 points.** The λ>0 arms lose ~0.023 on average
when the judge changes; the λ=0 arms move −0.036 and +0.010, i.e. in both directions. So the
in-loop checkpoint inflated the arms it had trained, but by less than run-to-run noise. This
is a genuinely favourable result: the translator was not exploiting that specific checkpoint
in any large way, which is consistent with M1.1's fidelity finding.

**3. Monotonicity was partly an artifact.** Old: 0.7622 → 0.8218 → 0.8332 → 0.8696, strictly
increasing. New: 0.7260 → 0.8244 → **0.8106** → 0.8470 — stage 2 dips below stage 1. The clean
ramp does not survive an honest judge; the overall direction does. Any writeup must show the
new column, and must not describe λ_det's effect as monotone.

**4. The binding finding: the noise floor is 0.059, not 0.013.** Baseline and loop stage 0 are
the same computation at the same seed. Under the old judge they differed by 0.0129; under the
new one, by **0.0591**. That reframes everything measured against it:

| comparison | Δ mAP50 | vs. the 0.059 noise floor |
| --- | --- | --- |
| stage 3 vs. the **baseline** arm (λ=0) | +0.062 | ~1× — indistinguishable from one noise draw |
| stage 3 vs. **loop stage 0** (λ=0, epoch-matched) | +0.121 | ~2×, but confounded by 400 vs 100 epochs |

Both λ=0 comparators are the same condition, and they disagree by more than the effect being
claimed. **λ_det's benefit is of the same order as run-to-run variance and cannot be separated
from it at n = 1.** That is not a setback — it is the exact situation E3's design anticipated,
and it converts the six-seed budget-matched campaign from rigour into necessity.

**Where the gain lives, still: Switch.** 0.4128 → 0.6175 stage 0 → stage 3, the hardest class
and the one raw thermal fails on completely (0.0130). Pole is saturated at ~0.95 in every
translated arm and will not separate anything.

**Consequences for the rest of M1.2:** none of the design changes. `detector.reference.weights`
in both E3 configs points at this judge. The measured 0.059 noise floor is the number the
aggregator's stage-0 null control has to beat, and it is what the paired test is up against.

### Step 2 — make "differ only by seed" true rather than merely likely

E3's whole claim is that paired runs differ only by seed; today that rests on an implicit
audit of which RNGs the training path touches. `torch.cuda.manual_seed_all`, `numpy.random`
and Python's global `random` are never seeded, and the train DataLoader has no
`worker_init_fn` (masked for now by `workers: 0`).

- [x] `t2o/seeding.py::seed_everything(seed)` — torch, CUDA, numpy, `random` — called from
      `translators/__init__.py` (replacing the bare `torch.manual_seed`) and from `run_loop`
- [x] `worker_init_fn` on the train DataLoader

**Deliberately not** setting `torch.use_deterministic_algorithms(True)`: cuDNN
non-determinism is precisely the variance E3 exists to quantify across seeds, so forcing it
away would hide the measurement and cost throughput. Both arms use `workers: 16` — see step
2b, which made the worker count result-neutral and therefore safe to raise for throughput.

**Decision — `run_loop` reseeds once *per stage* from `train.seed + stage`, not once per
run.** This is the only non-obvious choice in the step, and a single top-of-run seed is
actively wrong rather than merely coarser:

- It would **break** `test_resume_skips_completed_stages_and_restores_warm_started_weights`.
  An unbroken run enters stage 1 carrying stage 0's leftover RNG state; a resumed process
  re-seeds and enters stage 1 freshly seeded. Final weights diverge.
- Per-stage seeding makes that test pass *structurally* rather than by the accident it
  relied on before — both paths happening to draw the same number of samples in one
  process.
- It severs a real dependency the E3 campaign would otherwise carry: stage *N*'s
  augmentation stream currently depends on how many RNG draws stage *N−1*'s export and
  ultralytics' detector fine-tune happened to make. E3 runs four stages with a detector
  fine-tune between each, and its whole claim is that paired runs differ only by seed.
- It is the convention already in the file next door: `Trainer.train()` reseeds
  `_train_generator` from `seed + epoch` once per epoch for exactly this reason
  (`engine/trainer.py`, "required for resume to reproduce training exactly").

`tests/test_loop.py::test_a_run_is_unaffected_by_the_ambient_rng_state_it_inherits` is the
direct test — it consumes differing amounts of ambient RNG before each of two otherwise
identical runs and demands identical `metrics.json` and identical final weights. Confirmed
to **fail** with the per-stage reseed removed, so it has teeth rather than passing
vacuously.

**Decision — `worker_init_fn` on the train loader only.** Val (`engine/trainer.py`) and the
export loader (`engine/export.py`) neither shuffle nor augment, so no worker of theirs draws
a random number; giving them one would imply a stream that does not exist. Also note the
division of labour that makes this correct: torch already seeds each *worker's torch RNG*
from the loader's `generator`, so today's hflip/crop draws (which are `torch.rand`/
`torch.randint`) were already covered at `workers > 0`. `seed_worker` covers `random` and
numpy, which were not — and is what keeps the augmentation stream auditable if a future
augmentation reaches for either.

**`seed_worker` must stay a module-level function** — asserted directly by
`tests/test_seeding.py::test_seed_worker_is_picklable`. The training machine is native
Windows and spawns rather than forks (PLAN.md §3), so `worker_init_fn` is pickled to reach
the worker; a lambda or a closure over `Trainer` would fail there and nowhere else, which is
precisely the silent-hang class of bug §3 warns about.

**Verified, not assumed: `torch.manual_seed` already forwards to `torch.cuda.manual_seed_all`**
(`torch/random.py::_manual_seed_impl`, installed torch 2.12.1). `seed_everything` calls it
explicitly anyway so the set of covered RNGs reads off the function itself rather than off
torch's source — the cost is one line and the failure it guards against (a future torch
dropping the forwarding, silently unseeding CUDA on the only machine that has one) is
invisible from the dev Mac.

**Verify:** `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"`
(274 passed, 6 new), `pytest -m slow` (19 passed, unchanged — `test_resume_continues_through_a_completed_detector_stage`
is the one test that exercises per-stage seeding across a real ultralytics fine-tune).

**No server run needed for this step.** But it *changes the RNG streams*, so it must land
before the E3 campaign starts — a run made now is not comparable to one made after. Steps 3
and 4 still stand between here and the campaign.

### Step 2b — `workers` was silently experiment identity; now it is not

Raised while closing step 2, and it turned out to be a real bug rather than the stale
comment it looked like. `TrainConfig.workers` defaulted to `4` under a comment reading "0 is
the safe start" — but the deeper problem was that **changing the worker count changed the
training run**. `TranslationPairDataset.__getitem__`'s hflip/crop drew from the ambient
global torch RNG, which at `workers = 0` is the main process's and at `workers > 0` is
torch's per-worker derivation from the loader's generator. Two different augmentation
streams. Raising `workers` for throughput therefore changed the reported numbers, and
raising it via `--workers` changed `config_hash()` at the same time — so a paired E3 run
launched with a different worker count would not have been paired at all.

**User confirmed up to 16 workers runs clean on the server**, which is what made this worth
fixing rather than documenting.

- [x] `data/dataset.py` — augmentation draws from a per-sample generator seeded by
      `(augment_seed, epoch, index)`, never the ambient RNG. Keying on the *sample index*
      rather than call order is the load-bearing part: a worker only ever sees a strided
      subset of the epoch, so any order-dependent stream necessarily varies with the worker
      count. New `set_epoch(epoch)`, called by `Trainer.train()`, advances it so epoch *N*
      does not replay epoch *N−1*'s flips
- [x] `workers` moved `TrainConfig` → `RuntimeConfig`, i.e. out of `config_hash()` — now
      true rather than merely asserted, and structurally so, matching how M0.2 handled
      `device`/`name` and the reason `seed` went the other way
- [x] `tests/test_trainer.py::test_training_is_bit_identical_at_any_worker_count` — the same
      config trained at `workers=0` and `workers=2` must land on bit-identical weights.
      Confirmed to **fail** with the per-sample generator reverted, so it is the actual
      regression guard for this, not a restatement
- [x] `tests/test_dataset.py` — augmentation ignores ambient RNG *and* call order (built
      forwards, then rebuilt in reverse); `set_epoch` advances the stream; a different
      `augment_seed` gives a different stream

**Decision — `persistent_workers` must stay off, and `set_epoch` is why.** The epoch reaches
the workers only because a `DataLoader` without `persistent_workers` re-pickles the dataset
when each epoch's iterator is created. Turning it on for throughput without also propagating
the epoch would silently replay epoch 0's augmentation forever — a bug that costs nothing
visible and quietly removes most of the augmentation. Stated in
`TranslationPairDataset.set_epoch`'s own docstring, where anyone about to enable it will
read it.

**Decision — `experiments/pix2pix_*.yaml` keep `workers: 0`,** now under `runtime:`. The
move changes their `config_hash()` regardless (a field left the hashed section), but the
*value* is documentary: it records how M1 actually ran. E3's configs (step 3) take
**`workers: 16`**.

**Consequence for step 4's aggregator — do not load an old snapshot through `Config`.** M1's
two completed server runs have `train.workers` in their `runs/*/config.yaml`, which
`extra="forbid"` now rejects (verified). Since the aggregator's whole reason for joining runs
to their sibling snapshot is to keep those two runs readable, it must parse the snapshot as
plain YAML and read the keys it needs, not round-trip it through `Config.load`.

**M1's and M1.2's recorded numbers stand** — those runs are finished and their `metrics.json`
files are untouched. But be explicit about what does *not* follow: the augmentation stream
changed at **every** worker count, `0` included, since it no longer comes from the ambient
RNG at all. Re-running `experiments/pix2pix_baseline.yaml` today would not reproduce M1's
0.7851 exactly. That is the same "must land before the campaign starts" caveat as step 2, and
it is why E3's six seeds are run fresh rather than reusing M1's two runs as a control.

### Step 3 — the two E3 configs ✅

- [x] `experiments/e3_pix2pix_control.yaml` / `e3_pix2pix_loop.yaml`, differing in
      `coupling.task_weights` and `runtime.name` only, both pointing
      `detector.reference.weights` at step 1's judge and both at `runtime.workers: 16`
- [x] A test mirroring `test_baseline_and_loop_configs_differ_only_by_design` for the E3
      pair. That test **is** the ablation's integrity check — it is what stops a stray
      hyperparameter edit turning the comparison into a confound
- [x] `RuntimeConfig.group` + `group=`/`tags=` on `wandb.init`; 12 runs are unnavigable as
      untagged siblings

**Decision — `detector.reference.weights` is the one concrete path in these files, while
`data.manifest`/`detector.in_loop.weights`/`detector.evaluation.init_weights` stay
placeholders** (confirmed with the user). Those three are machine-specific and follow
`pix2pix_baseline.yaml`'s convention: supplied per-invocation by `--data`/`--in-loop-weights`/
`--eval-init-weights`. The reference judge is different in kind — `runs/reference-yolo11s/weights/best.pt`
is repo-relative and is exactly where step 1's own documented `t2o train-detector --out
runs/reference-yolo11s` command writes, so it is reproducible rather than machine-specific.
Leaving it `null` would make a forgotten `--reference-weights` fall back silently to
`evaluation.init_weights` — the same `yolo11n` checkpoint that supplies the loop arm's training
gradient — producing a self-graded number that looks entirely normal and that this whole
milestone exists to eliminate. A missing file at a concrete path fails loudly at load; `null`
fails quietly at the level of the scientific claim. Still overridable with
`--reference-weights`.
`test_the_judge_is_not_the_detector_that_supplies_the_gradient` pins all three inequalities.

**Decision — W&B tags are *derived* from the resolved config (`tracking.py::run_tags`), not
authored in YAML.** The campaign launches 12 runs off two files with `--seed` and `--name`
overridden on every one, so a YAML-authored `seed:0` tag would be wrong on ten of them and
right on two — the worst possible failure mode for a label whose only job is filtering.
Derived tags cannot disagree with the run they describe. Three of them, the facts a campaign
is actually filtered on: `backbone:pix2pix`, `seed:<train.seed>`, and `lambda_det:on|off`
(computed from whether *any* `task_weights` entry exceeds 0, so the control's four-stage
all-zero ramp tags `off` rather than being mistaken for a coupled run with a zero first stage).
`RuntimeConfig.group` is a plain field, set to `e3-pix2pix` in both arms and overridable with
the new `--group` flag; it sits in `RuntimeConfig` (outside `config_hash()`) because regrouping
runs must never change what is being measured.

**Decision — `runtime` is compared field-by-field in the integrity test, not wholesale.**
`name` is one of the two intended differences, so a wholesale `control.runtime == loop.runtime`
is impossible and skipping the section entirely would leave `workers` and `group` unguarded —
a campaign where one arm ran at a different worker count is exactly the drift this test exists
to catch. The fields that must not differ are therefore named explicitly.

**Both arms carry `runtime.workers: 16`,** unlike `experiments/pix2pix_*.yaml`'s documentary
`0`. Safe since step 2b made the worker count result-neutral, and M1's runs are not comparable
to these regardless — step 2b changed the augmentation stream at *every* worker count.

**Test teeth confirmed, not assumed:** `test_control_and_loop_configs_differ_only_by_design`
was verified to **fail** with a one-field edit (`ngf: 64` → `32`) to one arm, then reverted.
The two configs also hash differently (`a08e1b86` vs `bcf00e33`) — they are different
experiments, as they must be, while every non-`coupling` section is identical.

**Verify:** `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"`
(291 passed, 13 new — 10 in `tests/test_e3_experiments.py`, 2 in `test_tracking.py`, 1 in
`test_cli.py`), `pytest -m slow` (20 passed, 1 new — the control arm's four all-zero stages
completing end to end on the synthetic fixture).

**No server run for this step.** Step 4's aggregator stands between here and the campaign.

### Step 4 — aggregation and statistics ✅

Nothing in the repo reads more than one run today. `metrics.json` records **no seed and no
config hash**, so the aggregator joins each run to its sibling `config.yaml` snapshot rather
than changing `metrics.json`'s format — which keeps M1's two completed server runs readable.

- [x] `t2o/analysis/aggregate.py`: tidy rows per (run, seed, arm, stage); mean ± std per
      arm × stage; paired per-seed differences with **stage 0 reported beside stage 3**;
      exact sign-flip permutation test (2ⁿ enumeration, n ≤ 6 → 64 cases, no scipy, no
      distributional assumption); bootstrap CI on the mean difference
- [x] Refuse to test unpaired arms — a missing seed on one side raises rather than silently
      dropping the pair
- [x] `t2o aggregate`, standalone (paths, no `Config`). Runs on the server; `runs/` is
      gitignored and results must not travel back through git

**Decision — the config snapshot is parsed as plain YAML, never through `Config.load`.**
Step 2b already predicted this: M1's completed runs carry `train.workers` in their
`runs/*/config.yaml`, and `extra="forbid"` now rejects that key. Only two values are read
(`train.seed`, `coupling.task_weights`), both in the schema since M0.2, so raw YAML costs
nothing and is what keeps *every* run ever produced readable rather than only runs written
by today's schema. Directly tested
(`test_a_snapshot_with_a_key_the_current_schema_rejects_still_loads`).

**Decision — the arm is derived, not declared, and derived from `config.yaml` rather than
`metrics.json`.** `Arm.LOOP if any(w > 0 for w in task_weights)` is deliberately the *same*
predicate `tracking.py::run_tags` uses for its `lambda_det:on|off` W&B tag, so a run's arm in
the CSV and its tag in W&B cannot disagree about the same run. Reading it from `metrics.json`
instead would look equivalent and is not: the loop arm's stage 0 is itself λ = 0, so a run
interrupted after stage 0 records an all-zero `task_weight` column and would be silently
misfiled as a control (`test_a_loop_run_that_only_finished_stage_zero_is_still_the_loop_arm`).

**Decision — every stage common to all runs is computed, so stage 0's null control needs no
flag.** `--stage` only selects which line the verdict marker points at; the paired test runs
at every shared stage regardless, and stage 0 is labelled `(null)` in the output. The two are
only useful side by side — if the stage-3 effect is not clearly larger than the stage-0
difference, E3 is negative — so making the null opt-in would be the wrong default.

**Decision — `metric_value` raises on a null arm rather than returning a sentinel.** A stage
run with `--no-detector` records `zero_shot: null`/`fidelity: null`. A hole in one cell of a
paired comparison must not become a number; the error names the metric path and the stage.

**Decision — `sign_flip_p_value` refuses above n = 20.** 2²⁰ ≈ 1M assignments is about a
second; past that a mis-globbed `--runs` would hang instead of failing. The comparison uses
`>= observed - 1e-12`, not `>`, because floating-point summation makes the identity
assignment's own mean differ from `observed` in the last bits — excluding it would let p dip
below its true floor of 2/2ⁿ. `test_sign_flip_p_value_is_exact_and_bottoms_out_at_two_over_two_to_the_n`
pins 2/64 = 0.03125 at n = 6 against a hand-computed answer, so it pins E3's six-seed
justification itself, not only the code.

**Decision — `pair_runs` refuses two runs in the same `(arm, seed)` cell**, not just a
missing one. That is what a forgotten `--name` produces: `runs/<name>` has no seed component
(step 5's own warning), so the second launch overwrites the first and the survivor looks
perfectly normal. Both refusals name the offending seeds/paths.

**Decision — `--metric` takes a list; `--csv` is opt-in** (confirmed with the user). One
invocation reporting `zero_shot.map50` beside `fidelity.lpips` is M1.1's reward-hacking read
in a single command — mAP rising while LPIPS degrades is the signature, and it is only a
convenient check if both come out of the same pass. Per-class paths work too
(`zero_shot.per_class_ap50.Switch` — the class the gain lives in). `--csv` writes the tidy
long-format rows for re-analysis; a bare invocation writes nothing, since `runs/` is
gitignored and the printed table is what actually travels back to this repo.

**Not built, deliberately:** no paired t-test (PLAN.md §12 allows "paired t-test *or*
bootstrap CIs"; scipy is not a dependency and this step names the distribution-free test); no
formal test on the difference-of-differences between stage 3 and stage 0 (the step says
stage 0 reported *beside* stage 3, and both are — a test on their difference is scope this
does not ask for); `metrics.json`'s format is untouched.

**The difference-of-differences was built later, and the reversal is worth recording.** Step 8's
campaign drew a stage-0 null of −0.0397 with the loop arm's sd 2.4× the control's, which left
"clearly larger than the stage-0 difference" undecidable by eye at 0.0512 vs 0.0397. That is the
case this omission did not anticipate: reporting the two side by side is sufficient only while
the null lands near zero. `TrajectoryResult` now computes it at every stage from the *same*
per-seed paired differences (so it cannot drift from the paired block), reusing
`sign_flip_p_value` and `bootstrap_ci` unchanged. It is a sensitivity analysis, added after
seeing the data, and step 8's finding 7 states the caveat that goes with it.

**Tests (`tests/test_aggregate.py`, 21 fast + 1 slow; 3 more in `test_cli.py`).** Run
directories are hand-written, the same reasoning M0.3 applied to the synthetic dataset: an
unpaired seed, a duplicate cell, a run one stage short, a null metric, and a stale snapshot
can each be constructed exactly instead of hoping a real run contains one. The one `slow`
test covers the seam no hand-written fixture can — it runs `run_loop` for real (a control arm
and a loop arm, one stage each, on the synthetic fixture) and aggregates the result, so the
fast tests cannot all keep passing if `engine/loop.py`'s output format drifts.

**Verify:** `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"`
(315 passed, 24 new), `pytest -m slow` (21 passed, 1 new). Also exercised for real against a
hand-built 12-run campaign in a scratch directory — the glob, the arm × stage table, the
paired blocks at all four stages and `--csv` all render correctly end to end.

**No server run for this step.** Step 5's campaign is the next thing that needs the server,
and this was the last piece of tooling it waited on.

### Step 5 — the campaign (server)

`--name` is mandatory per run. `runs/<runtime.name>` has no seed component, so a bare
`--seed 1` writes straight over seed 0's `metrics.json`, `config.yaml` and every
`stage*/translator_*.pt`.

The three machine-specific paths are flags, not config (step 3's decision), so they appear on
every launch. `detector.reference.weights` is already correct in both files and needs no flag.
Both arms **must** be launched with identical flags — a difference in any of them is a
confound, which is the same thing
`test_control_and_loop_configs_differ_only_by_design` guards inside the files.

```bash
DATA=<real paired data.yaml>
OPTICAL=<visible-trained yolo11n.pt>

for s in 0 1 2 3 4 5; do
  for arm in control loop; do
    uv run t2o loop --config experiments/e3_pix2pix_$arm.yaml \
      --data "$DATA" --in-loop-weights "$OPTICAL" --eval-init-weights "$OPTICAL" \
      --seed $s --name e3-$arm-s$s --wandb --device cuda:0
  done
done
uv run t2o aggregate --runs 'runs/e3-*' --stage 3 --metric zero_shot.map50
```

`--wandb` is opt-in per invocation (there is no `--no-wandb`); the group is already
`e3-pix2pix` in both files, overridable with `--group` if a second campaign needs its own.
With two GPUs, split the seed range across two shells via `CUDA_VISIBLE_DEVICES` rather than
reaching for DDP (PLAN.md §3) — but keep each *pair* on one card, since the paired comparison
is per seed.

**Fixed mid-campaign — W&B dropped stages 1–n's loss curves, and the campaign was *not*
stopped.** Fourteen hours in, the server logged thousands of `Tried to log to step 46 that is
less than the current step 258`. Cause: `Trainer.train()` logged with an explicit
`step=epoch`, and `epoch` restarts at 0 in every stage while W&B's counter only increases.
Everything else in a loop run (zero-shot, fidelity, the detector callback) logs implicitly and
rides the auto-increment, so the counter was already at 151 by the time stage 1's epoch 0
arrived; all 100 were rejected, and so on per stage.

**Diagnosed as cosmetic before deciding not to stop, on three specific grounds:** the
per-epoch history is in `metrics.json` regardless (`asdict(StageResult)` carries every
`EpochStats`, written after each stage); `t2o aggregate` reads `metrics.json` plus the
`config.yaml` snapshot and never touches W&B, so **no number E3 is read off was affected**; and
the stage-level metrics in W&B (`stage*/zero_shot/*`, `stage*/fidelity/*`,
`stage*/detector/*`) all log implicitly and were all accepted. The only loss was the live view
of the generator curves for stages 1–3.

The fix drops the explicit `step=`, carries the epoch as a *value*, and adds
`Trainer(metric_prefix=...)` — set to `stage{N}` by `run_loop`. The prefix repairs a second
latent defect the step problem masked: the trainer's keys were the only ones in a loop run not
namespaced by stage, so stage 1's `train/l2` would have plotted on top of stage 0's even with
monotonic steps. Both now match `detector_stage.py::_forward_epoch_metrics`, which uses this
shape already and is why the detector curves never hit the bug.

**Not pulled onto the server until the campaign finished.** Logging touches no RNG, so the
change is result-neutral — but pulling mid-campaign would have left seeds 0–2 and 3–5 on
different trees, which is the same comparability rule step 2b's caveat states.
`test_epoch_metrics_are_namespaced_and_never_carry_an_explicit_step` pins both halves.

### Step 6 — record the outcome ✅

- [x] Results below. **Superseded by step 8** — this campaign's dose was 2.3% of the
      objective (step 7), and the recalibrated re-run is positive. Kept in full as the
      record of the uncalibrated dose: the two campaigns side by side *are* the dose
      argument, and neither is publishable without the other.

**E3's pix2pix arm is negative on its pre-registered endpoint** *(at `grad_scale: 1.0e-2` —
read step 7 and step 8 before quoting anything below)*. Twelve runs, six paired seeds,
400 warm-started epochs in both arms, judged by step 1's independent `yolo11s`. Per-arm
mean ± sd:

| metric | arm | stage 0 | stage 1 | stage 2 | stage 3 |
| --- | --- | --- | --- | --- | --- |
| `zero_shot.map50` | control | 0.7632 ± .0408 | 0.7679 ± .0560 | 0.7913 ± .0352 | 0.7987 ± .0309 |
| `zero_shot.map50` | loop | 0.7568 ± .0438 | 0.7920 ± .0483 | 0.8151 ± .0390 | 0.8057 ± .0324 |
| `fidelity.lpips` | control | 0.3104 ± .0098 | 0.2964 ± .0115 | 0.2893 ± .0124 | 0.3009 ± .0139 |
| `fidelity.lpips` | loop | 0.3200 ± .0293 | 0.3021 ± .0211 | 0.2968 ± .0196 | 0.2989 ± .0099 |
| Switch AP50 | control | 0.4131 ± .1091 | 0.4434 ± .1488 | 0.5064 ± .0710 | 0.5310 ± .0693 |
| Switch AP50 | loop | 0.4561 ± .1076 | 0.5519 ± .0398 | 0.5733 ± .0539 | 0.5626 ± .0939 |

Paired loop − control, exact two-sided sign-flip over 2⁶ assignments, with bootstrap CI:

| metric | stage 0 (null) | stage 1 | stage 2 | **stage 3 (headline)** |
| --- | --- | --- | --- | --- |
| `zero_shot.map50` | −0.0063, p=.875, [−.045, +.040] | +0.0241, p=.281, [−.011, +.058] | +0.0238, p=.312, [−.014, +.055] | **+0.0070, p=.656, [−.020, +.032]** |
| `fidelity.lpips` | +0.0096, p=.875, [−.011, +.040] | +0.0056, p=.344, [−.003, +.015] | +0.0075, p=.500, [−.003, +.023] | **−0.0019, p=.812, [−.016, +.011]** |
| Switch AP50 | +0.0430, p=.406, [−.050, +.136] | +0.1085, p=.031, [+.016, +.214] | +0.0669, p=.094, [+.019, +.114] | **+0.0316, p=.562, [−.046, +.115]** |

The Design section above fixed the decision rule before the campaign ran: *"If the stage-3
effect is not clearly larger than the stage-0 difference, E3 is negative and gets reported that
way."* Stage 3 is **+0.0070** against a null of **−0.0063** — the same magnitude — at p = 0.66.
By its own rule, E3's pix2pix arm is negative.

**1. The null control behaved as a null, so this is a measurement and not a broken campaign.**
−0.0063 at p=0.875 with a CI straddling zero almost symmetrically. That is the single most
important line in the table: step 2's RNG audit and step 2b's `workers` fix were exactly the
work that makes the stage-0 contrast interpretable, and it came out where it had to.

**2. Step 1's noise-floor estimate, taken at n=1, was right.** It measured 0.0591 from one
paired λ=0 draw. Stage 0's CI half-width here (0.042 ≈ 1.96·SE) implies a per-seed sd of
**≈0.053**. The number the whole six-seed budget was justified against holds.

**3. What six paired seeds actually bought: ±0.026 resolution** — 2.3× tighter than a single
paired draw. So the honest claim is *"no effect larger than about +3 mAP50 points"*, not "no
effect": a true +0.02 sits inside the stage-3 CI. §16's causality criterion is still not
satisfied — an unmeasurable effect cannot establish causality — but the writeup must state the
bound rather than assert a zero.

**4. There is no dose-response in λ.** *(Superseded by step 8's campaign: at `grad_scale: 0.15`
the same six seeds give 0 → +0.028 → +0.036 → +0.051, monotone. The absence recorded here was a
property of the dose, not of λ_det.)* The paired difference goes 0 → +0.024 → +0.024 → +0.007,
peaking at the *smallest* nonzero λ and decaying as λ triples. Both arms converge over stages
(control 0.7632 → 0.7987, loop 0.7568 → 0.8057), which is what a shared 400-epoch budget plus a
per-stage detector fine-tune produces on its own. Extends step 1's finding 3: λ_det's effect is
not monotone in the *stage*, and now not monotone in the *weight* either.

**5. Fidelity-neutral, with a bound — M1.1's reward-hacking question is closed at n=6.** λ_det
moves stage-3 LPIPS by −0.0019, CI [−0.016, +0.011]. The detection gradient at the `grad_scale`
PLAN.md §8 prescribes neither degrades nor improves perceptual fidelity. That is worth reporting
whatever happens to C1.

**6. Switch — where step 1 said the gain lived — does not survive.** Its own stage-0 null is
**+0.043, larger than its stage-3 effect of +0.032**; the loop arm is simply noisier on the
rarest class. The one cell under 0.05 (stage 1, +0.1085) is 1 of 12 reported tests sitting
*exactly* on the n=6 p-floor of 0.031, where Bonferroni would need 0.0042 — so **no per-class
claim is attainable in this design at any effect size.** Logged as a hypothesis to pre-register
for the turbo arm, not as a finding. (The one detail worth carrying: at stage 1 the loop arm's
Switch sd is 0.0398 against the control's 0.1488. If λ_det does anything here it may be variance
reduction on the hardest class rather than mean improvement — but it does not reappear at stages
2 or 3, so it is an observation, not a result.)

**7. E8 is not an escape route from this, and the earlier note here saying otherwise was
wrong.** `data.annotation_fraction` gates only the batch's `cls`/`bboxes`
(`data/dataset.py:188-191`) — that is, only the *loop* arm's own supervision, since the control
never reads annotations at all. Lowering it makes the two arms **more** alike, not less, so a
low-annotation E3 cannot recover a causality claim that failed at full annotation. E8 remains a
valid and probably headline-worthy question about the *translator's* data efficiency (PLAN.md
§11), but it answers a different question than C1's.

**The one thing that blocks reading this as a claim about coupling: λ_det was never 1/2/3.**
`translators/pix2pix.py:161-163` adds `task_weight * detection` to the total, while
`coupling/detection_loss.py:103` has *already* multiplied by `grad_scale`. Both E3 configs set
`grad_scale: 1.0e-2`, so the ramp the optimiser saw was **0.01 / 0.02 / 0.03**, against
`l2: 1.0` + `lpips: 5.0` + `gan: 1.0`. That downscale is PLAN.md §8's anti-reward-hacking
guardrail — and M1.1, then finding 5 above, established there was no hack to guard against. The
guardrail set aggressively against a problem that did not materialise is now the prime suspect
for the null.

So the campaign **cannot yet distinguish "coupling does not help pix2pix at this data scale"
from "λ_det was too small to move the optimiser"** — a mechanism result and a dose result, with
very different consequences. **Do not launch M2a's turbo arm until step 7 settles it**: same
`grad_scale`, same possible null, another ~72 GPU-hours.

### Step 7 — was λ_det ever large enough to matter?

- [x] **Answered: no. The detection term was 0.9 / 1.7 / 2.3% of the objective.** E3's null is
      **dose-limited, not a mechanism result.**

Pooled over epochs and over the six loop runs (`scripts/loss_share.py --runs 'runs/e3-loop-*'`):

| stage | w | λ_eff | `loss_l2` | `loss_lpips` | `loss_gan` | `loss_det` | `loss_total` | w·det | share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0.0415 | 1.8062 | 1.5747 | — | 3.4223 | 0 | 0.0% |
| 1 | 1 | 0.01 | 0.0361 | 1.5940 | 1.6886 | 0.0298 | 3.3485 | 0.0298 | **0.9%** |
| 2 | 2 | 0.02 | 0.0336 | 1.5146 | 1.6508 | 0.0275 | 3.2540 | 0.0550 | **1.7%** |
| 3 | 3 | 0.03 | 0.0323 | 1.4758 | 1.7490 | 0.0261 | 3.3354 | 0.0782 | **2.3%** |

The decision rule fired on the ≲5% branch, and not marginally: at the *top* of the ramp the
detection term was **1/43rd of the objective**, roughly **19× smaller than the LPIPS term** it
had to compete against for the optimiser's attention. (It was not the objective's *smallest*
term — `loss_l2` at 1.0% is smaller still; see the composition paragraph below.)
E3 did not test the coupling hypothesis at a dose capable of refuting it.

**PLAN.md §8 predicted exactly this and the instruction was skipped.** Its second guardrail
bullet reads: *"The `[0,1,2,3]` ramp is calibrated to SeAFusion's **segmentation** loss scale,
not a detection loss — recalibrate empirically in Phase 0."* That recalibration was never done,
and `grad_scale: 1.0e-2` (AlignProp's `loss_coeff`, for a different objective on a different
task) went into E3 unexamined. The 72 GPU-hours were not wasted — they produced a validated
null-control, a confirmed noise floor and a fidelity bound — but the headline they were spent on
cannot be read as evidence about coupling.

**The objective is GAN + LPIPS, and `l2` is 1% of it.** At stage 3: LPIPS 1.4758 (44%), GAN
1.7490 (52%), l2 0.0323 (**1.0%**), detection 0.0782 (2.3%). `LossConfig`'s comment calling l2
"Dominant, and the reason a translator with no other term produces a blurred conditional mean"
is true of an l2-only translator and false of this configuration — corrected in the schema.
Cross-check that the recorded losses mean what we think: `loss_lpips / 5.0` = 0.295 against the
measured val `fidelity.lpips` of 0.2989. Also note the GAN term is the only one that *rises*
across stages (1.575 → 1.749) while LPIPS falls (1.806 → 1.476): the discriminator is winning
more as training goes on.

**Finding 4 of step 6 is superseded.** "No dose-response in λ" spanned shares of 0.9% → 2.3% —
a range too narrow to be informative about λ at all. It is not evidence against a dose-response;
it is evidence that no dose was applied.

**The candidate λ, from the numbers rather than from taste.** `loss_det` is recorded
post-`grad_scale`, so the raw per-image detection loss is `loss_det / grad_scale` = **2.98 / 2.75
/ 2.61** across stages — stable and slowly declining, which is what makes a linear extrapolation
usable. With fidelity terms summing to F = 3.257 at stage 3 and a target share S, the required
weight is `w·g = F·S/((1−S)·L_raw)`:

| target share at stage 3 | required `w·g` | `grad_scale` at w=3 |
| --- | --- | --- |
| 20% | 0.312 | 0.104 |
| 25% | 0.416 | 0.139 |
| 30% | 0.535 | 0.178 |

**`grad_scale: 0.15`** — 15× the E3 value — is the candidate, predicting a ramp of ≈12% / 21% /
27% (λ_eff 0.15 / 0.30 / 0.45). Chosen at the upper half of the band deliberately: the
extrapolation holds `L_raw` fixed, but a term 15× stronger will *drive the detection loss down*,
which lowers its own share at equilibrium. The achieved share will land below the prediction.

**Two honest limits on this number.** (1) Loss share is a proxy for gradient influence, not a
measurement of it — `grad_scale` multiplies the detection gradient exactly linearly, so 15× more
gradient is certain, but whether that is 15× more *influence on the update* depends on the other
terms' gradient norms, which nothing here measured. (2) Raising `grad_scale` 15× removes most of
§8's anti-reward-hacking downscale. The replacement guard is already in place and now
quantified: LPIPS is reported per stage, and step 6 finding 5 bounds λ_det's current fidelity
effect at ±0.016. `coupling.reward_target` stays `null` for the probe — with `L_raw` ≈ 2.6–3.0 a
target near 1.5–2.0 would start biting, and that is the knob to reach for *if* fidelity degrades
at the new dose. Fidelity degrading would itself be a finding, not a failure: it would mean the
loop can trade fidelity for detection, which is precisely what §8 anticipated and what the
guardrails exist to bound.

Zero GPU cost: every loop run's `metrics.json` already holds `task_weight` per stage and
`loss_det`/`loss_total` per epoch (`StageResult` → `asdict`). `loss_det` as recorded is already
post-`grad_scale`, so `task_weight * loss_det / loss_total` *is* the detection term's share of
the objective, with no reconstruction of the weight chain needed.

`scripts/loss_share.py` reports it, pooled over epochs and over the campaign's loop runs:

```bash
uv run python scripts/loss_share.py --runs 'runs/e3-loop-*'
```

It exists as a script rather than a `t2o aggregate --metric` path because
`analysis/aggregate.py::metric_value` walks dicts and `epochs` is a list. W&B is no help either:
the step bug fixed in `accfe56` dropped precisely the stages that have a detection term.
Control runs are skipped rather than pooled in (they record no `loss_det` at all), and runs that
disagree on `task_weight` or `grad_scale` are refused, so a mis-globbed `--runs` cannot quietly
average two experiments.

**The decision rule, written before the number is known:**

- **share ≲ 5%** → the null is *dose-limited*. Re-calibrate λ before any further six-seed
  campaign, in either backbone. The share also gives the candidate λ for free: `L_det`'s
  magnitude is roughly stable across training, so scale `grad_scale` by the ratio needed to
  bring the term to ~20–30% of the objective.
- **share ≳ 20%** → the null is a *mechanism* result. E3's pix2pix arm is written up negative as
  it stands, and the turbo arm becomes the test of whether a stronger prior changes that.

**The trap to avoid: a λ sweep must not be read off mAP at one seed.** With a per-seed sd of
≈0.053 (finding 2), a single run cannot resolve the +0.02 being chased — that is precisely the
n=1 mistake M1.2 exists to correct. A one-seed run at the candidate λ is a *stability* check
(does LPIPS collapse, does the GAN diverge, does the detection term stay bounded), never a
measurement. The measurement is another paired six-seed campaign, and it only earns its GPU time
once the share says the dose is real.

### Step 8 — recalibrate the dose, then re-run E3

- [x] Probe: one seed at the candidate `grad_scale`, share re-read, stability confirmed
- [x] `scripts/loss_share.py --first-epochs N` and an `epochs` column on the report — the probe
      is a quarter the length of a campaign run and nothing on the row said so
- [x] Both E3 configs bumped to the calibrated value — `grad_scale: 0.15`
- [x] The twelve-run campaign relaunched; its tables sit below, beside step 6's

**The probe is a calibration, not a measurement.** With a per-seed sd of ≈0.053 (step 6 finding
2) one run cannot resolve the +0.02 being chased, so nothing about mAP is to be concluded from
it. It answers three narrower questions: does the achieved share land in the 20–30% band, does
the run stay stable, and does fidelity survive. No new config is needed — both knobs are already
CLI overrides (`cli.py::_OVERRIDES`), and `config_hash` covers `coupling`, so a probe run cannot
be confused with a campaign run on resume.

```bash
uv run t2o loop --config experiments/e3_pix2pix_loop.yaml \
  --data "$DATA" --in-loop-weights "$OPTICAL" --eval-init-weights "$OPTICAL" \
  --grad-scale 0.15 --epochs 25 --seed 0 --name e3-probe-g015 --device cuda:0
uv run python scripts/loss_share.py --runs 'runs/e3-probe-g015'
```

`--epochs 25` keeps all four stages (the ramp is what needs measuring) at ~1/4 the cost — the
share is a loss-composition ratio and is legible within a few epochs. Read three things off it:

1. **Share** per stage, from `loss_share.py`. In the 20–30% band → proceed. Still under ~10% →
   raise `grad_scale` by the shortfall ratio and re-probe. Over ~40% → lower it; the detection
   term should not dominate the objective it is meant to inform.
2. **Fidelity**, from the run's own `stage*/fidelity/lpips`, read **against its own stage 0 and
   never against E3's ~0.30** — see the probe result below for why that instruction, as first
   written, was wrong. Rising sharply within the run is reward hacking becoming affordable — set
   `coupling.reward_target` (config-file-only) and re-probe rather than pressing on.
3. **Stability**: `loss_det` should fall rather than oscillate, and `loss_gan` should not
   diverge. A one-step check that the dose is trainable at all.

#### Probe result — `grad_scale: 0.15` lands the dose

```
stage runs epochs   w lambda_eff  loss_l2 loss_lpips loss_gan loss_det loss_total   w*det  share
    0    1     25 0.0     0.0000   0.0492     2.1006   1.7234       --     3.8732  0.0000   0.0%
    1    1     25 1.0     0.1500   0.0430     1.9252   1.6868   0.4462     4.1012  0.4462  10.9%
    2    1     25 2.0     0.3000   0.0461     2.0174   2.4022   0.4456     5.3570  0.8913  16.6%
    3    1     25 3.0     0.4500   0.0431     1.9002   2.3294   0.4136     5.5136  1.2409  22.5%
```

**In band.** 10.9 / 16.6 / 22.5% against a predicted 12 / 21 / 27, and the ramp now spans a real
dose range rather than step 7's 0.9–2.3%. Stability held: four stages completed, `loss_gan` did
not diverge, nothing collapsed. `grad_scale: 0.15` is the calibrated value.

**The epoch-length trap, recorded because it cost a round trip.** The probe ran `--epochs 25`;
`epochs_per_stage` in both E3 configs is **100**. Every figure here is a mean over epochs, and
the opening epochs are the loud ones, so the probe's numbers sit above a campaign run's in
*every* term at once — which reads as a changed condition rather than a shorter one. Two false
alarms came out of that before the cause was found:

* Stage 0 is provably inert to `grad_scale` (`translators/pix2pix.py::fit` gates the detector on
  `task_weight > 0.0`), so the probe's stage 0 should reproduce `e3-loop-s0`'s. It did not —
  `loss_lpips` 2.1006 vs 1.8608, val `map50` 0.7451 vs 0.7951, val `lpips` 0.3451 vs 0.3020.
  That looked like a determinism failure large enough to invalidate step 2/2b. It was the epoch
  count. **Determinism is not in question; nothing about steps 2, 2b or 6 changes.**
* Raw detection loss (`loss_det / grad_scale`) reads 2.97 / 2.97 / 2.76 here against step 7's
  2.98 / 2.75 / 2.61 — i.e. *worse* under 15× the pressure. **Withdrawn**: a 25-epoch mean
  against a 100-epoch mean is not a comparison. The `loss_gan` observation (this run rises 35%
  across stages 0→3 where `e3-loop-s0` rises 12%) is within-run on both sides so it survives the
  level offset, but 25 epochs leaves less room to re-equilibrate; logged as a watch item for the
  campaign's per-stage LPIPS readout, not a finding.

`--first-epochs` exists so this cannot recur silently: it truncates the pool, and the `epochs`
column puts the length on the row. The apples-to-apples check is
`scripts/loss_share.py --runs 'runs/e3-loop-s0' --first-epochs 25`, which reads E3's share over
the same window the probe covers. The share verdict does not depend on it — 2.3% → 22.5% is a
10× shift against a 15× change in `grad_scale`, which no epoch count can manufacture — but the
raw-detection-loss and `loss_gan` questions above are only answerable there.

**Nothing about the probe's mAP is usable**, and not only for step 6 finding 2's reason: at 25
epochs the translator is a quarter trained, so its 0.7451 → 0.8158 stage ramp measures training
length as much as anything else. That confound is exactly what the control arm exists to
subtract, and the probe has no control arm.

#### Matched-epoch comparison — and the loss-space noise floor

`--first-epochs 25` on `e3-loop-s0` puts both runs on equal terms (same seed, same arm, same
length; `grad_scale` 0.01 vs 0.15). Stage 0 is λ-inert, so its residual *is* the run-to-run
noise floor, now measured in loss space for the first time:

| stage 0, λ inert | `loss_l2` | `loss_lpips` | `loss_gan` | fidelity total |
| --- | --- | --- | --- | --- |
| gap between two nominally identical runs | +3.4% | +3.6% | +11.8% | +7.1% |

**This is expected and by design, not a determinism defect.** `seeding.py`'s docstring is
explicit that `torch.use_deterministic_algorithms` and the cuDNN flags are deliberately absent,
because "cuDNN non-determinism is precisely the variance E3 exists to quantify across seeds";
`test_training_is_bit_identical_at_any_worker_count` is a CPU test of the data pipeline, not of
kernels. Two GPU runs at one seed are not expected to reproduce. Step 6 finding 1 already priced
this in mAP space (sd ≈0.053); the table above is its counterpart in loss space, and it is the
resolution limit for every `loss_share.py` comparison from here on.

Read against that floor:

* **Raw detection loss does not detectably move.** `loss_det / grad_scale` goes 3.13 → 2.97,
  2.86 → 2.97, 2.72 → 2.76 across stages 1–3 under **15× the weight** — i.e. −5.0 / +3.9 / +1.4%,
  every one of them inside the floor. Not "λ_det fails to reduce its own loss"; the comparison
  cannot resolve anything under ~10%. Worth restating that this is not the endpoint either: the
  in-loop detector is the frozen yolo11n, and E3's claim rests on the independent yolo11s judge.
* **Fidelity improvement is what the dose costs.** Within-run across stages 0→3, `loss_lpips`
  falls 25.2% at `grad_scale` 0.01 but only 9.5% at 0.15 — a 15.7-point gap against a 3.6% floor,
  the one comparison here that clears it by a wide margin. The fidelity total follows: −6.0%
  versus +10.3%. `loss_gan` (+20.1% vs +35.2%) points the same way but sits close enough to its
  own 11.8% floor to stay a watch item.

So the dose is doing something measurable, and what it measurably does is trade fidelity
improvement. Whether it buys detection is exactly what the campaign's control arm is for.

**`reward_target` stays `null` for the relaunch.** Setting it now would change the dose and its
guard together, and any fidelity result would then be unattributable. The campaign has a control
arm at every stage and a per-stage LPIPS readout; if the trade turns out worse than E3's
measured −0.002 ± 0.016 bound, that is a finding, and `reward_target` is the response to it
rather than a precaution against it.

Then bump `coupling.grad_scale` in **both** `experiments/e3_pix2pix_{control,loop}.yaml` — the
control's value is inert (no detector is built at weight 0) but must match, which
`test_control_and_loop_configs_differ_only_by_design` enforces — and relaunch step 5's loop
verbatim **except for the run names** — see below. ~72 GPU-hours. Step 6's tables stay in place
as the record of the uncalibrated dose, with the new campaign's beside them.

**Relaunch under new names, or the uncalibrated campaign is destroyed.** There is no
`config_hash` guard on a run directory: without `--resume`, `run_loop` starts from
`results = []` and rewrites `metrics.json` stage by stage, so reusing `e3-<arm>-s<n>` overwrites
seed-for-seed. Those twelve run dirs are the only copy of the 0.01 campaign — step 6's tables
record its *results*, but the per-epoch loss curves behind step 7's shares and step 8's noise
floor live nowhere else, and `runs/` is gitignored.

```bash
for s in 0 1 2 3 4 5; do
  for arm in control loop; do
    uv run t2o loop --config experiments/e3_pix2pix_$arm.yaml \
      --data "$DATA" --in-loop-weights "$OPTICAL" --eval-init-weights "$OPTICAL" \
      --seed $s --name e3b-$arm-s$s --group e3-pix2pix-g015 --wandb --device cuda:0
  done
done
uv run t2o aggregate --runs 'runs/e3b-*' --stage 3 \
  --metric zero_shot.map50 fidelity.lpips zero_shot.per_class_ap50.Switch
uv run python scripts/loss_share.py --runs 'runs/e3b-loop-*'
```

**Split across two cards by seed, never by arm** — a pair must stay on one card or the paired
difference absorbs whatever differs between the GPUs. Seeds 0–2 on `cuda:0`, 3–5 on `cuda:1`,
two shells, ~36h wall clock instead of ~72h. PowerShell, since the server is native Windows:

```powershell
# shell A
foreach ($s in 0,1,2) { foreach ($arm in 'control','loop') {
  uv run t2o loop --config "experiments/e3_pix2pix_$arm.yaml" `
    --data $DATA --in-loop-weights $OPTICAL --eval-init-weights $OPTICAL `
    --seed $s --name "e3b-$arm-s$s" --group e3-pix2pix-g015 --wandb --device cuda:0
} }

# shell B -- identical but for the seed range and the card
foreach ($s in 3,4,5) { foreach ($arm in 'control','loop') {
  uv run t2o loop --config "experiments/e3_pix2pix_$arm.yaml" `
    --data $DATA --in-loop-weights $OPTICAL --eval-init-weights $OPTICAL `
    --seed $s --name "e3b-$arm-s$s" --group e3-pix2pix-g015 --wandb --device cuda:1
} }
```

Concurrency is safe on disk: `run_loop` passes `project=stage_dir / "detector"`, so every
ultralytics write is scoped to one run's own stage directory, and each stage's detector trains
on that stage's own exported images — there is no shared label `.cache` for the two processes to
race on. The one machine-global file in play is ultralytics' `settings.json` (see
`detector_stage.py`'s note on it), so start shell B a minute after shell A and let any
first-touch write land once.

Both shells must carry **identical** `$DATA`/`$OPTICAL`, for step 5's reason: those flags are
machine-specific paths rather than config, so a difference between the shells is a confound that
`test_control_and_loop_configs_differ_only_by_design` cannot see.

The `e3b-` prefix also keeps `'runs/e3-*'` unambiguous: that glob still selects the old campaign
alone, and `pair_runs` would refuse a mixed glob anyway (two runs per `(arm, seed)` cell — the
same guard that correctly refused the probe against E3 in step 8). The trailing `loss_share`
call confirms the achieved share at full length, since the 11/17/23% ramp was measured over 25
epochs and the campaign runs 100.

#### Campaign result — **E3's pix2pix arm is positive at the calibrated dose**

Twelve runs, six paired seeds, 400 warm-started epochs in both arms, same independent `yolo11s`
judge, `grad_scale: 0.15`. Per-arm mean ± sd:

| metric | arm | stage 0 | stage 1 | stage 2 | stage 3 |
| --- | --- | --- | --- | --- | --- |
| `zero_shot.map50` | control | 0.7579 ± .0372 | 0.7519 ± .0238 | 0.8071 ± .0263 | 0.7975 ± .0330 |
| `zero_shot.map50` | loop | 0.7182 ± .0907 | 0.7799 ± .0887 | 0.8427 ± .0215 | 0.8487 ± .0140 |
| `fidelity.lpips` | control | 0.3304 ± .0364 | 0.3165 ± .0150 | 0.2992 ± .0101 | 0.2905 ± .0054 |
| `fidelity.lpips` | loop | 0.3272 ± .0299 | 0.3230 ± .0414 | 0.2997 ± .0205 | 0.3002 ± .0106 |
| Switch AP50 | control | 0.4000 ± .0943 | 0.4340 ± .0726 | 0.5601 ± .0442 | 0.5271 ± .0787 |
| Switch AP50 | loop | 0.4511 ± .1534 | 0.5007 ± .1620 | 0.6211 ± .0579 | 0.6306 ± .0335 |

Paired loop − control, exact two-sided sign-flip over 2⁶ assignments, with bootstrap CI:

| metric | stage 0 (null) | stage 1 | stage 2 | **stage 3 (headline)** |
| --- | --- | --- | --- | --- |
| `zero_shot.map50` | −0.0397, p=.469, [−.127, +.037] | +0.0280, p=.500, [−.051, +.078] | +0.0357, p=.062, [+.013, +.055] | **+0.0512, p=.031, [+.025, +.081]** |
| `fidelity.lpips` | −0.0033, p=.844, [−.049, +.037] | +0.0065, p=.844, [−.030, +.045] | +0.0005, p=.938, [−.017, +.018] | **+0.0097, p=.125, [+.002, +.017]** |
| Switch AP50 | +0.0511, p=.594, [−.119, +.201] | +0.0667, p=.406, [−.098, +.178] | +0.0610, p=.062, [+.024, +.093] | **+0.1035, p=.094, [+.025, +.183]** |

**1. The headline clears the pre-registered endpoint, at the only p this design can reach.**
p = 0.0312 is exactly 2/2⁶ — the sign-flip floor — so all six seeds moved the same way *and* the
observed mean was the most extreme of all 64 assignments. Nothing about n=6 can produce a
smaller number; step 5's six-seed budget was sized for exactly this outcome.

**2. Dose-response appeared, and its absence was step 6's finding 4.** The paired difference goes
0 → +0.0280 → +0.0357 → +0.0512, monotone in λ. At `grad_scale: 1.0e-2` it went
0 → +0.024 → +0.024 → +0.007, peaking at the *smallest* nonzero λ and decaying. This is the
single most persuasive line in the table, because it is not something a lucky draw produces: the
same six seeds, the same machinery, the same judge, ordered by dose.

**3. The mechanism is now visible in loss space, where before it was not.** Raw detection loss
(`loss_det / grad_scale`, from `scripts/loss_share.py --runs 'runs/e3b-loop-*'`) runs
2.56 → 2.12 → 1.84 across stages 1–3, against the uncalibrated campaign's 2.98 → 2.75 → 2.61 —
stage 3 about **30% lower**, well outside step 8's ~10% loss-space noise floor. Step 8 could only
report that raw detection loss "does not detectably move" under 15× the weight, measured over 25
epochs; at full length it moves, and downward.

```
stage runs epochs   w lambda_eff  loss_l2 loss_lpips loss_gan loss_det loss_total   w*det  share
    0    6    100 0.0     0.0000   0.0419     1.8288   1.6460       --     3.5167  0.0000   0.0%
    1    6    100 1.0     0.1500   0.0362     1.6131   1.8033   0.3834     3.8359  0.3834  10.0%
    2    6    100 2.0     0.3000   0.0334     1.5358   1.7464   0.3181     3.9518  0.6362  16.1%
    3    6    100 3.0     0.4500   0.0317     1.4905   1.8211   0.2758     4.1707  0.8274  19.8%
```

**4. The dose landed where the probe predicted, and the epoch-length trap barely bit.**
10.0 / 16.1 / 19.8% at 100 epochs against the probe's 10.9 / 16.6 / 22.5% at 25. The top of the
ramp sits just under the 20–30% target band. **Not re-tuned**: the band was a calibration target,
not an endpoint, and moving `grad_scale` again after seeing the mAP result would make the
campaign unreportable. Record the achieved share; do not chase the band.

**5. `loss_gan` did not diverge, closing step 8's watch item.** +10.6% across stages 0→3 against
the uncalibrated campaign's +11.1% — indistinguishable. The probe's alarming +35% was the
25-epoch artifact step 8 suspected it of being, and is now confirmed as one.

**6. The stage-0 null drew wide, and this is the result's main soft spot.** −0.0397 (p = 0.469,
CI [−.127, +.037]) against the uncalibrated campaign's −0.0063, with the loop arm's stage-0 sd at
**0.0907 versus the control's 0.0372**. Stage 0 is provably λ-inert in *both* arms —
`build_detection_loss` returns `None` at weight ≤ 0 so no `FrozenDetector` is ever constructed
(`coupling/schedule.py`), `translators/pix2pix.py:161` gates the term on `task_weight > 0.0`, and
`config_hash` is read only by `Trainer`'s resume drift check and seeds nothing — so the arms run
the identical computation there and this is an unlucky draw, not a code path. But the decision
rule says the stage-3 effect must be *clearly larger* than the stage-0 difference, and 0.0512
against 0.0397 is 1.3× by magnitude. **Not clearly larger on that reading alone.**

**7. The within-arm trajectory corroborates 6 but does not confirm it.** Measured, not
estimated (`TrajectoryResult`, `analysis/aggregate.py`):

| stage | control gain | loop gain | difference | p | 95% CI |
| --- | --- | --- | --- | --- | --- |
| 1 | −0.0060 | +0.0617 | +0.0677 | .312 | [−.061, +.184] |
| 2 | +0.0491 | +0.1245 | +0.0754 | .156 | [+.003, +.160] |
| **3** | **+0.0396** | **+0.1305** | **+0.0909** | **.094** | **[+.019, +.181]** |

Over the same 400-epoch budget the loop arm travels 3.3× further. Two readings, and the second
is the one that must not be skipped:

* **It says the effect is not an artifact of the stage-0 offset.** If the finish-line +0.0512
  were the offset showing through, removing the offset would collapse the contrast toward zero.
  It grows instead, and grows *monotonically in dose* (+0.068 → +0.075 → +0.091) — a second
  dose-response, in a contrast the offset cannot touch, independent of finding 2's.
* **It cannot rescue significance, and p = 0.094 is the honest number.** Differencing two paired
  quantities adds their variances (`Var(A−B) = Var(A) + Var(B) − 2Cov`), and the stage-0
  difference is precisely the noisy one, so subtracting it injects exactly the noise finding 6
  is about. This is the ordinary bias–variance trade: the trajectory removes a possible bias at
  the cost of a wider interval. **A larger point estimate at a worse p is not a stronger
  result.**

So the defensible sentence is *"the pre-registered endpoint is significant at p = 0.031, and a
contrast immune to the stage-0 draw points the same way, larger, at p = 0.094"* — never
"+0.0909, CI excludes zero", which invites the reader to take the sensitivity analysis as the
finding. Step 4 declined to build this test ("scope this does not ask for"); the wide draw is
what changed that. **It was added after seeing the data and is a sensitivity analysis, never the
endpoint.** The one mitigating argument the paper may make: the decision rule already directs
stage 3 to be read against stage 0, so this formalises the rule's own arithmetic rather than
introducing a second hypothesis.

**Finding 6 is therefore narrowed, not closed.** The stage-0 draw is still the result's softest
edge. What can be said is that all three available readings agree in direction — the stage-0
null's CI [−.127, +.037] contains zero comfortably, stage 3's [+.025, +.081] excludes it, and
the trajectory grows monotonically in dose — and that findings 2 and 3 do not depend on stage 0
at all.

**8. Variance collapses in the loop arm as λ rises.** Its `zero_shot.map50` sd runs
0.0907 → 0.0887 → 0.0215 → 0.0140 while the control stays in 0.024–0.037; by stage 3 the coupled
arm is **2.4× tighter** than the control. Step 6 finding 6 saw this on Switch alone at stage 1
and logged it as a hypothesis to pre-register; it now appears on the headline metric and across
the ramp. Still not a claim — regression from an unlucky stage-0 draw predicts some of it — but
it is the pre-registerable hypothesis for the turbo arm, and it is the second-most interesting
thing in this table.

**9. Fidelity is no longer neutral: +0.0097 LPIPS at stage 3, CI [+.002, +.017].** The
uncalibrated campaign's −0.0019 ± 0.016 bound does not survive the dose. Both arms still improve
over the ramp (control 0.3304 → 0.2905, loop 0.3272 → 0.3002); the loop arm improves **less** —
0.0270 against 0.0399, so **about a third** of the control's fidelity gain is traded away. Step 8 predicted exactly this ("what the
dose measurably does is trade fidelity improvement"). Two things must be said about it:
the sign-flip test gives p = 0.125 while the bootstrap CI excludes zero, so it is *suggestive,
not significant at n=6* — report both, and do not lean on the CI alone; and detection rising
while fidelity falls **is** the reward-hacking signature (PLAN.md §8), which the independent
judge argues against but does not measure. `t2o faithfulness` exists to settle it — see below.

**The trajectory reading weakens this further, and is the one to quote.** Within-arm, the
control's LPIPS improves −0.0399 and the loop's −0.0270, a difference of **+0.0129, p = 0.562,
CI [−.025, +.054]** — straddling zero. So the fidelity cost is *directionally consistent across
both contrasts and statistically established by neither*. The strongest defensible claim is a
bound: **λ_det at this dose does not cost more than about 0.02 LPIPS**, and may cost nothing.
Writing it as a demonstrated trade would be overclaiming in the one place a reviewer looking for
reward hacking will read most carefully.

**10. Switch is still not claimable, and the trajectory removes any remaining doubt.**
+0.1035 at stage 3 against its own stage-0 null of +0.0511 and p = 0.094. Under the trajectory
contrast it dissolves outright: control +0.1270, loop +0.1794, difference **+0.0524, p = 0.781,
CI [−.102, +.248]**. Nearly all of Switch's apparent gain is movement *both* arms make over the
ramp. Step 6 finding 6's arithmetic is unchanged besides: 3 metrics × 4 stages is 12 reported
tests against an n=6 floor of 0.031, where Bonferroni needs 0.0042. **Descriptive only**, and the
per-class table in the paper carries no inferential claim at all.

**11. Where the bootstrap CI and the sign-flip p disagree, believe the p.** It happens three
times above (map50 trajectory at stage 3, LPIPS at stage 3, Switch at stage 2), always the same
way: CI excludes zero, p does not clear 0.05. Both facts about n = 6. The sign-flip test is
exact but **coarse** — 2⁶ = 64 assignments, so p can only take 0.031, 0.062, 0.094, … and there
is no value between the floor and 0.062. The percentile bootstrap resamples 6 numbers with
replacement and is known to be anti-conservative at that size, giving intervals narrower than
their nominal coverage. The CIs stay in the tables as descriptive spread; **no claim in this
project rests on a bootstrap interval excluding zero.** PLAN.md §12 permits either, which is
what made this ambiguity available — that permission needs narrowing in the writeup.

**Consequence for §16.** PLAN.md's causality criterion is **satisfied for the pix2pix backbone**,
with findings 6/7 and 9 as its two stated caveats. The dose caveat that gated step 6's null is
resolved: coupling was tested at 19.8% of the objective and moved the endpoint.

### Step 9 — the reward-hacking question the fidelity cost opens

- [x] `TrajectoryResult` + `t2o faithfulness` built and tested locally (355 fast, 22 slow)
- [ ] Both run over the finished `runs/e3b-*` on the server, and finding 9 resolved

`metrics/faithfulness.py` has been complete since M0.5 and was wired to **nothing** — no engine,
no analysis, no CLI. Finding 9 is what makes it load-bearing: an LPIPS cost that coincides with a
detection gain is either a fidelity trade or hallucination, and only a count of invented and
erased objects separates them. Every stage's export is still on disk, so this needs no retraining.

```powershell
uv run t2o aggregate --runs 'runs/e3b-*' --stage 3 `
  --metric zero_shot.map50 fidelity.lpips zero_shot.per_class_ap50.Switch `
  --csv runs/e3b-tidy.csv
uv run python scripts/loss_share.py --runs 'runs/e3b-control-*' --terms-only
foreach ($arm in 'control','loop') { foreach ($s in 0,1,2,3,4,5) {
  uv run t2o faithfulness --translated "runs/e3b-$arm-s$s/stage3/translated" `
    --data $DATA --weights runs/reference-yolo11s/weights/best.pt --device cuda:0
} }
```

The `--csv` also answers finding 6 directly: whether the wide stage-0 draw is one outlier seed or
spread across all six. `loss_share.py --terms-only` on the **control** arm is the missing half of finding 9 — the loop
arm's loss trajectory is known and the control's is not, so the +0.0097 cannot yet be located in
loss space. (`--terms-only` was added for this: the default path refuses a control-only glob,
correctly, since a control run has no `loss_det` and so no share. Asking it for one was an error
in this step's first draft.)

**Read on the faithfulness pass:** false-object rate flat or falling while mAP50 is +0.0512 means
the LPIPS cost is a fidelity trade, and C2 gets its first real number. False objects rising with
λ means reward hacking, `coupling.reward_target` becomes live, and **the turbo campaign must not
launch before it is understood** — the same gate step 6 applied to the dose question, for the
same reason.

**`--weights` is deliberately required and un-defaulted.** It must be the reference `yolo11s`.
Scoring hallucination with the in-loop `yolo11n` measures how well the translator learned to
please the checkpoint that trained it, which is the confound step 1's judge exists to remove.

---

## M2 — Phase 2: Diffusion loop

The M1 gate passed, so M2a is detailed below. M2b stays a bullet list until M2a's backbone
has produced numbers worth comparing against.

Note the ordering against M1.2: E3's **pix2pix** campaigns are server work that does not
touch any of this — their two configs are `backbone: pix2pix` and stay that way. M2a builds
the arm PLAN.md §11 calls the strong one; the pix2pix campaign is the cheap control it is
read against.

**The gate M1.2 step 6 put on this milestone is released.** "Do not launch M2a's turbo arm
until step 7 settles it" was about the dose, and step 8 settled it: E3's pix2pix cell is
positive at `grad_scale: 0.15`. Two conditions carry forward in its place, both from step 8:

1. **Calibrate `grad_scale` for turbo before its campaign** (step 4 below, and PLAN.md §8).
   0.15 is a property of pix2pix's objective composition, not a constant. sd-turbo is
   pretrained, so its `loss_det` starts at a different magnitude and the same value can land
   back near 2%. A 25-epoch `loss_share.py` probe costs hours; skipping it cost 72 GPU-hours
   last time.
2. **Resolve step 9's faithfulness pass first if it comes back badly.** The calibrated dose
   costs +0.0097 LPIPS on pix2pix; turbo has far more capacity to find a genuine hack. If
   false objects rise with λ on the finished pix2pix exports, `reward_target` and the step 3
   fidelity floor land *before* the turbo campaign, not after.

### M2a — pix2pix-turbo (primary)

#### Step 1 — the backbone behind the `Translator` interface ✅

- [x] Vendor `src/model.py` at `463b2d3` (`my_vae_encoder_fwd`/`my_vae_decoder_fwd`) and
      reimplement `Pix2Pix_Turbo` against *current* diffusers rather than inheriting the
      2023-era pinned stack
- [x] Wrapper owns device placement (upstream hardcodes `.cuda()` in six places)
- [x] `Backbone.PIX2PIX_TURBO` + `Pix2PixTurboTranslatorConfig` in the discriminated union;
      one new branch in `build_translator`

**Decision — `Pix2Pix_Turbo` is reimplemented, not vendored** (PLAN.md §7's row corrected).
It cannot be vendored verbatim: the file opens with `sys.path.append("src/")` +
`from model import ...` and does not import outside upstream's own cwd, and 100 of its 229
lines are `requests`/`tqdm` downloads for two pretrained tasks we never use. What is left is
the random-init branch and a 12-line forward. This is the same split M1 made — `networks.py`
vendored, `Pix2PixModel`/`BaseModel` reimplemented. `make_1step_sched` comes with the
vendored file but is never called; it does `set_timesteps(1, device="cuda")` and a network
fetch.

**Decision — components are injected, not loaded inside `__init__`.** `load_sd_turbo()` is
the only function that touches HuggingFace. That is what lets the fast tests build a small
`AutoencoderKL`/`UNet2DConditionModel` locally and still exercise the real LoRA adapters, the
real vendored skip forwards and real gradient flow with no network — the fixture philosophy
M0.3 applied to the dataset, applied to a 1.3B-parameter checkpoint.

**Decision — the skip-conv widths are derived from `vae.config["block_out_channels"].`**
Upstream writes `Conv2d(512,512)`, `(256,512)`, `(128,512)`, `(128,256)` as literals, which
is sd-turbo's geometry spelled out. Deriving them is what makes a tiny test VAE possible at
all, and `test_skip_widths_reproduce_upstreams_hardcoded_convs` pins the derivation against
those four literals so the generalisation cannot drift from the model it was read off.

**Decision — the timestep is named, not inferred.** Upstream calls `set_timesteps(1)` and
gets `[999]` only because sd-turbo's scheduler config happens to say
`timestep_spacing: "trailing"`. Under diffusers' `"leading"` default the identical call
yields `[0]` and `step()` denoises from the wrong end of the chain **without erroring** —
found by a tiny-model test, which is exactly the class of bug a tiny model exists to find.
`set_timesteps(timesteps=[999])` states the thing the distillation is about.

**Decision — input is reflect-padded to a multiple of 64 inside the wrapper.** The VAE is f8
and the UNet downsamples 8× more. The dataset is 640×480 and 480 % 64 == 32, so a full frame
shape-mismatches inside the UNet's skip concat. `train.crop` cannot fix this: validation and
export always run whole images. Reflect rather than zero — a black band at the frame edge is
a structure the VAE has never seen and would bleed inside the crop-back region.

**Decision — `state_dict()` carries only what trains** (LoRA + `unet.conv_in` + the four
skip convs, the same selection upstream's `save_model` makes). Measured on the real
checkpoint: **9.5M trainable parameters, a 38MB checkpoint**, against ~2.5GB for the full
module. `Trainer` writes one twice per stage and `engine/loop.py` warm-starts from it, so
unreduced this is ~20GB per run and ~240GB for a twelve-run campaign, all of it frozen base
weights `load_sd_turbo` reproduces bit for bit. `load_state_dict` accepts the partial dict
but **raises on unexpected keys** — `strict=False` alone would silently accept a checkpoint
from a different model.

**Decision — evaluation uses the VAE posterior's mode, training keeps upstream's sample.**
Upstream calls `.latent_dist.sample()` in both. Sampling at eval makes an exported image a
function of RNG state, so two evaluations of the same checkpoint disagree — not acceptable in
something whose entire job is measurement. Training keeps the stochasticity, which is where
it was doing work.

**Decision — `loss.gan` reuses M1's already-vendored PatchGAN** (`define_D`/`GANLoss`), not
upstream's `vision_aided_loss` CLIP discriminator (confirmed with the user). PLAN.md §8's
loss is the same four knobs for every backbone; a different adversarial objective per
backbone would confound E2's and E3's backbone comparison with a change nobody asked for. It
also avoids the dependency PLAN.md §7 already declined.

**Decision — `train.amp`/`amp_dtype` are finally consumed, CUDA only.** They have sat unused
in the schema since M0.8 with `engine/trainer.py`'s docstring saying they exist "for a future
translator's own constructor to read". This is that translator. No `GradScaler`: the schema's
own comment already records that `float16` without one produces silent NaNs, which is why the
default is `bfloat16`.

**Tests (`tests/test_pix2pix_turbo_translator.py`, 20 fast + 1 slow).** The fast ones run in
~5s on tiny modules. The `slow` one loads the genuine `stabilityai/sd-turbo` and runs a
forward and a `fit` step at 96×128 on CPU in ~15s — the only check that `add_adapter`,
`latent_dist`, `scaling_factor` and `sched.step().prev_sample` still mean what upstream's
2023-era pinned stack meant by them under diffusers 0.39 / transformers 5.15 / peft 0.20. It
**skips itself** unless the checkpoint is already in the HuggingFace cache, so the suite
stays offline for anyone who has not fetched it. Weights live in `~/.cache/huggingface`
(~4.8GB), never in the repo.

Also fixed: `test_unknown_backbone_lists_the_valid_discriminators` used `"pix2pix_turbo"` as
its example of an invalid tag, which this step made valid. A test whose meaning depends on
what has not shipped yet stops testing anything the moment it does; it now uses a name no
milestone will claim.

**Verify:** `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"`
(336 passed, 21 new), `pytest -m slow` (22 passed, 1 new).

**No server run for this step.** The sd-turbo fetch happened on the Mac, as PLAN.md §17's
risk row anticipated.

#### Step 2 — data prep and pretrain

- [ ] Constant-caption pipeline end to end: the prompt is a config field already; confirm the
      normalisation asymmetry upstream carries (input [0,1], target [-1,1]) has no analogue
      left in our path, which stays [0,1] throughout
- [ ] Pretrain → custom fine-tune. **Unblocked** (M0.9 closed 2026-08-23: every dataset is
      on the server). **FLIR-aligned is the pretrain corpus, not LLVIP** — confirmed with the
      user. `data/adapters/flir.py::adapt_flir` is written and verified against the real
      archive (4129 train / 1013 val paired, read straight out of `aligned.zip`), and it is
      the same FLIR camera family as the custom 640×480 pairs, where LLVIP is 1024×1280
      street scenes. LLVIP stays available on the server as a later corpus ablation (E9), not
      on this path.

#### Step 3 — the fidelity floor

- [ ] Early-stop / clamp when LPIPS rises past a threshold while λ_det > 0. The LPIPS network
      is already in the loss assembly at `loss.lpips > 0`, so this is a schedule decision, not
      new machinery. M1.1 measured no reward hacking on pix2pix; the turbo arm has far more
      capacity to find it

#### Step 4 — the turbo experiment configs

- [ ] `experiments/e3_turbo_{control,loop}.yaml`, mirroring the pix2pix pair and pinned by the
      same differ-only-by-design test. Batch size and `train.crop` need a VRAM measurement on
      the server first — a 1.3B model at 640×512 is not the same budget as `resnet_9blocks`

#### Step 5 — the turbo campaign (server)

- [ ] Six seeds × two arms, aggregated with `t2o aggregate` exactly as M1.2 step 5 does. This
      is E3's strong arm; the pix2pix campaign is the control it is read against

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
- [ ] Report the effective λ_det as `coupling.task_weights × coupling.grad_scale`. The paper
      must never say "λ_det = 3": `DetectionTaskLoss.forward` already applies `grad_scale`
      before `fit` applies the stage weight. The reported campaign ran at
      **0.15 / 0.30 / 0.45** (`grad_scale: 0.15`, M1.2 step 8); the superseded one ran at
      0.01/0.02/0.03 (step 6) and is reported beside it as the dose-limited null.
- [ ] Pre-register **one** endpoint. At n=6 the exact sign-flip test resolves ±0.026 mAP50 and
      bottoms out at p = 0.031, so with 3 metrics × 4 stages reported no per-class or
      per-stage claim can survive a multiplicity correction (Bonferroni would need 0.0042).
- [ ] λ_det's effect **is** monotone in λ once the dose can show it: +0.028 → +0.036 → +0.051
      at `grad_scale: 0.15` (M1.2 step 8). The opposite claim, drawn from the 0.9%–2.3%
      campaign's 0 → +0.024 → +0.024 → +0.007, was withdrawn by step 7 as too narrow a span
      to be a dose-response test at all — and step 8 then measured the real one. Both
      campaigns belong in the paper: the pair *is* the dose argument.
- [ ] Report the objective's actual composition, not its nominal weights. At `l2: 1.0`,
      `lpips: 5.0`, `gan: 1.0` the reported campaign's stage-3 split is **detection 19.8% /
      LPIPS 35.7% / GAN 43.7% / l2 0.8%** (M1.2 step 8); the uncalibrated campaign's was
      LPIPS 44% / GAN 52% / l2 1.0% / detection 2.3% (step 7). Any sentence calling the pixel
      term dominant is wrong in both.
- [ ] Report the fidelity cost as a **bound, not a measured trade**. λ_det at the calibrated
      dose moves stage-3 LPIPS by +0.0097 (p = 0.125) and the within-arm gain by +0.0129
      (p = 0.562): consistent in direction, significant in neither. The defensible sentence is
      "does not cost more than about 0.02 LPIPS", and may cost nothing.
- [ ] Report significance from the **exact sign-flip p, never a bootstrap CI excluding zero**.
      At n=6 they disagree in three of E3's cells (M1.2 step 8 finding 11): the sign-flip test
      is exact but coarse (p ∈ {0.031, 0.062, 0.094, …}), the percentile bootstrap over six
      values is anti-conservative. PLAN.md §12 has been narrowed to say so; the tables keep the
      CIs as descriptive spread only.
- [ ] State that the difference-of-differences (M1.2 step 8 finding 7) was added **after**
      seeing the campaign, and that the pre-registered endpoint is the paired stage-3
      difference. Reporting it as if it had been planned would be the one genuinely
      indefensible move available here.
- [ ] Pre-register variance reduction for the turbo arm. The loop arm's `zero_shot.map50` sd
      runs 0.0907 → 0.0887 → 0.0215 → 0.0140 across the ramp, ending 2.4× tighter than the
      control (M1.2 step 8 finding 8). Not claimable from this campaign — it is confounded
      with regression from a wide stage-0 draw — but it is a real, testable prediction.
