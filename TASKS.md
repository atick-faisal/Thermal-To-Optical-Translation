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
  - [ ] **Real fetch not executed — confirmed with the user.** This machine has ~31GB free
        disk (93% full), shared across both checkouts on this volume; LLVIP alone (15,488
        pairs at 1024×1280) plausibly doesn't fit safely alongside M3FD/TTPLA. Run
        `uv run --group scripts python scripts/fetch_datasets.py --dataset llvip m3fd ttpla`
        yourself once disk/bandwidth allow, same spirit as M0.1's "not verifiable locally."
  - [ ] **Re-host destination not decided.** PLAN.md's plan is "fetch once on the Mac,
        re-host, then the server script is a plain `curl`" — where (HF dataset repo, S3,
        etc.) is an open decision. User asked to be reminded when server training
        approaches, rather than deciding it now.
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
- [ ] Verify: InsPLAD annotation format (not stated in its README; Mendeley Data gates the
      files behind a form, so this needs the actual download, not just the repo README)
- [ ] Freeze and hash the splits; commit the manifest

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
InsPLAD annotation-format verification, freezing and hashing the splits.

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
