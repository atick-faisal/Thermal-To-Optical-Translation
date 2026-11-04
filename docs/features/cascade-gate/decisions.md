# Enhancer cascade faithfulness gate — Decisions

## ADR-01 — Stage 2 is PiSA-SR's default setting, reimplemented on stock diffusers, with no new dependency

**Date:** 2026-11-03
**Status:** accepted
**Tasks:** CASCADE-GATE-01, CASCADE-GATE-05, CASCADE-GATE-06

**Context**
Plan Q5 left the zero-shot stage-2 model to this task: OSEDiff or PiSA-SR, both one-step
restorers on SD-2.1-base. The spike read both upstream repos on 2026-11-03:

- OSEDiff (`cswry/OSEDiff`) at `cdd3b7b04850fa9bb7faf8f2187edd279cc96507` (2026-04-09);
- PiSA-SR (`csslc/PiSA-SR`) at `9c76e12e261d1980148738a0cff41e5e27d18c9f` (2026-03-15).

What it found, point by point:

- **Neither repo has a network of its own.** Both ship `unet_2d_condition.py` and
  `autoencoder_kl.py`. Diffed against diffusers `v0.25.0`, these are the stock classes, with two
  changes: absolute instead of relative imports, and an appended `merge_and_unload` helper
  copied from peft. The VAE files also add a dtype cast before `quant_conv`. Neither file
  imports under the pinned diffusers 0.39: `diffusers.models.unet_2d_blocks` and
  `diffusers.loaders.FromOriginalVAEMixin` no longer exist there (checked against this repo's
  `uv.lock`). This is the same finding as modtr-baseline ADR-01. Each model's contribution is
  its LoRA weights plus about 10 lines of one-step inference.
- **Licence.** OSEDiff ships an Apache-2.0 `LICENCE` file. PiSA-SR's README (`:182-183`)
  says "released under the Apache 2.0 license" and links to a `LICENSE` file. No such file
  exists at the pinned commit, and none has existed in any of its 30 commits. GitHub therefore
  reports no licence.
- **Native Windows.** At inference, neither repo needs triton, xformers or `torch.compile`.
  `xformers` appears only in training paths and behind a `try/except ImportError` in the tiled
  VAE hook (`my_utils/vaehook.py:74-78` in OSEDiff, `src/my_utils/vaehook.py:74-78` in PiSA-SR).
  Both hardcode `"cuda"` throughout, for example OSEDiff `osediff.py:274-278` and PiSA-SR
  `pisasr.py:311`. A reimplementation drops that.
- **The one-step recipe.**
  - OSEDiff runs the UNet at `t = 999` and takes a full `DDPMScheduler.step` from the encoded
    input (`osediff.py:277,406`). It treats the input latent as if it were pure noise.
  - PiSA-SR runs the UNet at `t = 1` and subtracts the prediction as a residual,
    `x_denoised = z − ε̂` (`pisasr.py:341,487`). It uses no scheduler step at all.
- **Prompts and extra weights.**
  - OSEDiff captions every image with DAPE, a LoRA fine-tune of the RAM Swin-L tagger, and
    feeds `"<tags>, "` as the prompt (`test_osediff.py:28-35,72-78`). That adds two more
    checkpoints: RAM's `ram_swin_large_14m.pth` from a HuggingFace Space, and the 7.2 MB
    `DAPE.pth`, which is committed in the repo. The vendored `ram/models/bert.py:39-44` imports
    `apply_chunking_to_forward`, `find_pruneable_heads_and_indices` and `prune_linear_layer`
    from `transformers.modeling_utils`. The locked transformers 5.15 no longer has them, so the
    tagger does not import without editing vendored code or downgrading transformers.
  - PiSA-SR's inference uses an empty prompt (`test_pisasr.py:50`). RAM appears only in
    training. Its one checkpoint, `pisa_sr.pkl`, holds UNet LoRA weights only, with no VAE
    LoRA (`pisasr.py:296-305`). It is hosted on Google Drive and Baidu, not in the repo.
- **×1 on 640×480.** Both test scripts accept `--upscale 1`. Both resize any image whose short
  side is under `process_size` (512) up to that size, then crop it to a multiple of 8. A
  640×480 frame therefore becomes 682×512 and then 680×512.
  - PiSA-SR resizes the output back to 640×480 (`test_pisasr.py:69-70`), so each frame goes
    through two resamplings.
  - OSEDiff computes the resized-back image and discards it (`test_osediff.py:132-133`), so
    its output is 680×512, a different size from the input.

  The latent of a 640×512 frame is 80×64, under the 96×96 tiling threshold, so neither model
  tiles at this size.
- **Pinned dependencies upstream.** Both pin the same 2023-era stack: diffusers 0.25,
  torch 2.0.1, transformers 4.28.1, xformers 0.0.20, peft 0.9 and basicsr. At inference,
  PiSA-SR needs only `diffusers`, `transformers` (`CLIPTextModel` and `AutoTokenizer`) and
  `peft`, all of which `t2o` already pins. OSEDiff also needs the RAM tagger's dependencies.
- **SD-2.1-base.** `stabilityai/stable-diffusion-2-1-base` returns 401 to an anonymous request.
  OSEDiff's README now points to the mirror `Manojb/stable-diffusion-2-1-base`, which is
  reachable at revision `0094d483a120f3f33dafbd187ea4aa60d10de75c` and licensed
  CreativeML Open RAIL++-M.

**Decision**
- **The model is PiSA-SR, in its default setting**: the pixel and semantic LoRAs both merged at
  weight 1.0 into the SD-2.1-base UNet, with one UNet pass (`pisasr.py:333-335`,
  `_predict_no_tiling`'s `default` branch at `:507-510`). This is the published setting.
  λ_pix and λ_sem are not tuned. The setting is fixed here, before any stage-2 output exists.
- **Nothing is vendored under `third_party/`.** The UNet and VAE are diffusers' own
  `UNet2DConditionModel` and `AutoencoderKL`, imported from their defining modules as
  `pix2pix_turbo.py` does. LoRA loading and merging use peft's own API, not upstream's copied
  `merge_and_unload`. The module docstring cites the pinned commit and `file:line` for every
  piece of the recipe it reimplements.
- **The recipe, reimplemented in `src/t2o/`:**
  1. The input is the stage-1 RGB image in [−1, 1].
  2. It is reflect-padded to a multiple of 64, as `pix2pix_turbo.py:33-35` already does.
  3. It is VAE-encoded and scaled by `scaling_factor`.
  4. The prediction is `ε̂ = UNet(z, t = 1, emb(""))`, and `z' = z − ε̂`.
  5. `z'` is decoded, clamped to [−1, 1] and sliced back to the input size.
  6. AdaIN colour matching to the stage-1 input is applied: per-channel mean and std, which is
     upstream's default `align_method` (`test_pisasr.py:64-65,90`).
- **Five deliberate deviations from upstream**, each named in the wrapper's docstring:
  1. **No resizing.** The frame is padded rather than resized to `process_size`, so the output
     is the input's 640×480 with no resampling round trip. Two Lanczos/bicubic passes would blur
     exactly the thin components the gate scores.
  2. **The VAE posterior's mode replaces `latent_dist.sample()`** (`pisasr.py:481`), so a
     refined image is a deterministic function of its input. Without this, the same stage-1
     image would refine differently on each run.
  3. **The empty prompt is encoded once and the text encoder dropped.** This is
     `pix2pix_turbo.py`'s deviation 3.
  4. **Components are injected, not loaded inside the wrapper.** This is `pix2pix_turbo.py`'s
     deviation 1, so CPU tests can use tiny local models.
  5. **No device is hardcoded.** The precision is a config field (CASCADE-GATE-06).
- **No new dependency.** CASCADE-GATE-05 adds nothing to `pyproject.toml` or `uv.lock`. Weights
  are fetched on the server, never committed:
  - SD-2.1-base from the mirror, pinned by `revision="0094d483a120f3f33dafbd187ea4aa60d10de75c"`;
  - `pisa_sr.pkl` from upstream's Google Drive folder with `gdown`, which is already in the
    `scripts` dependency group. Its sha256 is recorded in the server log the first time it is
    fetched.

**Alternatives considered**
- **OSEDiff.** It has the cleaner provenance: an Apache-2.0 `LICENCE` file, and weights
  committed in the repo, so the commit pins them. It is also the more cited of the two.
  Against it:
  - The DAPE tagger cannot import under the locked transformers without editing vendored code
    or downgrading transformers for the whole lock, and it adds a third checkpoint.
  - Its per-image tags inject semantics into stage 2. A tag such as "pole" can prompt the model
    to draw a pole, which confounds exactly the insertion rate the gate measures.
  - The `t = 999` step treats the input as noise. That is the strongest generative prior of
    the two recipes, so it is the more likely to invent objects for reasons that have nothing
    to do with the cascade idea.
  - Running it with an empty prompt removes the tagger, but it is then no longer OSEDiff as
    published, because it was trained on DAPE prompts.
- **PiSA-SR at λ_sem = 0 (pixel LoRA only).** It is the least generative setting and the most
  likely to pass. But it barely tests the professor's idea of semantic enhancement, and picking
  the setting most likely to pass the gate is the post-hoc choice the gate exists to prevent.
- **Both, side by side.** Plan Q5 already rejected this: it doubles the install work before the
  2026-11-12 meeting.
- **Vendor upstream's model files anyway.** Making them import under diffusers 0.39 would mean
  editing vendored code, which invariant 4 forbids. Pinning diffusers 0.25 would break
  `pix2pix_turbo.py` and the shared lock.

**Consequences**
- **CASCADE-GATE-05's row says "vendor only the network file under `third_party/`".** That no
  longer describes what gets built: there is no network file to vendor, so `third_party/` gains
  nothing. CASCADE-GATE-05 takes the reimplementation path, as MODTR-BASELINE-03 did after
  modtr-baseline ADR-01.
- **Cost: PiSA-SR's licence rests on a README sentence.** The code is not copied, so for the
  code the question is moot. For the weights, the paper relies on the README's Apache-2.0
  statement and must disclose that no licence file exists. The SD-2.1 base weights also carry
  Open RAIL++-M use restrictions, which allow research use. `goal.md` Constraints' licence list
  does not name either one yet. The paper's licence statement must add both.
- **Cost: the SD-2.1 base comes from a third-party mirror.** It is pinned by revision, but
  whether it matches StabilityAI's original weights is believed, not verified.
- **Cost: the checkpoint loads under the current stack only in principle.** `pisa_sr.pkl` was
  saved with torch 2.0 and peft 0.9. It holds only lists of module names, integer ranks and a
  tensor `state_dict` (`pisasr.py:296-305`), so it should load under torch 2.12's
  `weights_only=True` default. Its LoRA keys use peft's `lora_A.<adapter>.weight` scheme, which
  peft 0.20 still uses. Both are believed, not verified: the weights were not downloaded in the
  spike.
- **CASCADE-GATE-05 must load LoRA weights strictly.** It fails if any checkpoint key is left
  unused or any expected LoRA parameter is left unfilled. If the weights silently fail to load,
  peft's zero-initialised `lora_B` leaves stock SD-2.1. At `t = 1`, stock SD-2.1 predicts a
  near-zero residual, so stage 2 becomes close to an identity and passes the gate trivially.
- **Cost: deviation 2 changes upstream's output distribution slightly.** The SD VAE's posterior
  variance is small, so the mode and a sample should differ very little. Believed, not
  measured.
- **Cost: how PiSA-SR behaves at ×1 on a clean-ish synthetic image is unknown.** It was trained
  to restore degraded, upsampled inputs. It may sharpen, add texture, or do almost nothing.
  The gate measures exactly this, so it is a finding, not a defect.
- **Changing the model or its λ setting after any stage-2 output is scored needs a superseding
  ADR**, and the paper must then disclose both versions.

## ADR-02 — Do not run `scripts/loss_share.py` for stage 2; a strict-load and non-identity check replaces it

**Date:** 2026-11-03
**Status:** accepted
**Tasks:** CASCADE-GATE-05, CASCADE-GATE-07

**Context**
The repo rule (CLAUDE.md, "Things that will trip you up") is to re-run
`scripts/loss_share.py` (25 epochs) for every new backbone before its campaign, and to keep the
detection term at 20–30% of the objective. Skipping it once cost about 72 GPU-hours and an
uninterpretable null (record 006). The script measures `task_weight * loss_det / loss_total`
over training epochs (`scripts/loss_share.py:1-14`).

Stage 2 is zero-shot and frozen (ADR-01; plan, Tech Stack). It has no training, no epochs, no
objective and no gradient path, so there is nothing to measure. modtr-baseline ADR-03 is the
precedent: it skipped the script for a backbone whose objective had a single term, and put a
direct check in its place.

**Decision**
- `scripts/loss_share.py` is not run for stage 2.
- The rule exists so that a campaign never runs on a component that silently does nothing. A
  detection term too weak to train anything is how that happened before. For a frozen stage 2,
  the same failure is a LoRA checkpoint that silently does not load (ADR-01, Consequences),
  which turns stage 2 into a near-identity that passes the gate trivially. Two checks guard it:
  - **at load:** CASCADE-GATE-05 consumes every checkpoint key and fills every expected LoRA
    parameter, or raises;
  - **on the server:** the first step of CASCADE-GATE-07's runbook refines a handful of val
    images. It logs each output's size and its mean absolute change from the stage-1 input.
    A run whose outputs are not the input's size, or whose mean change is zero, stops there.

**Alternatives considered**
- **Run `loss_share.py` as written.** That is not possible: there is no training loop for it
  to read, so it would only error.
- **Skip the rule with no replacement.** It is the cheapest option. But a stage 2 that loaded
  no weights would look like a clean gate pass, which is the worst outcome a faithfulness gate
  can have.
- **A fidelity threshold (e.g. LPIPS to the input) as the smoke pass rule.** It is more
  informative. But it needs a threshold picked before any output exists, with nothing to
  calibrate it against. "Not identical, right size" needs no tuning, and it catches the failure
  that matters.

**Consequences**
- The CLAUDE.md rule stays as written. This entry is its documented exception for a frozen,
  loss-free component. If stage 2 is ever trained (paper-2 work, out of scope here), it runs the
  script as usual.
- CASCADE-GATE-07's runbook gains a smoke step before its main runs. Its log is part of the
  server log saved under `logs/`.
- Cost: a non-zero mean change shows that stage 2 changes the image. It does not show that the
  intended weights loaded. The strict load in CASCADE-GATE-05 is what checks that, so the two
  checks are needed together.

## ADR-03 — The gate targets fuses and switches: erase only them, and judge their per-class rates

**Date:** 2026-11-03
**Status:** accepted
**Tasks:** CASCADE-GATE-02, CASCADE-GATE-03, CASCADE-GATE-04, CASCADE-GATE-07

**Context**
The cascade's hypothesis is narrower than the plan states it. Stage-1 outputs often draw switches
and fuses deformed and noisy, so the detector misses them. Perhaps the component was not hot enough
to stand out from the background, or the thermal frame lacked texture. The hope is that stage 2
reconstructs a clean component from that partial evidence. That is a win. A component drawn where
the thermal frame holds no trace is a red flag (user, 2026-11-03).

Two things in the plan as resolved do not fit that:

- **Plan Q4 erases every ground-truth box.** After step 1 of the stage-1 runbook, the user looked at
  the erased frames. Pole boxes are large, and inpainting them left the frames badly deformed. That
  large Telea fill is itself an artefact the translator then reacts to. Poles are also easy to
  detect, and they are not what the hypothesis is about.
- **The insertion/deletion audit pools all four classes** (fuse, pole, switch, transformer). Poles
  are many and easy, so they dilute exactly the fuse and switch gain, and the fuse and switch
  inventions, that the gate is meant to show.

**Decision**
- **The erasure probe erases only fuse and switch boxes.** Poles and transformers stay in the frame
  untouched. Telea inpainting and the dilation stay as plan Q4 resolved them.
- **The erased tree's labels hold only the erased boxes.** `t2o erase` gains a class filter and
  writes each frame's label file with only the boxes it inpainted. It records the classes under
  `erasure:` in its `data.yaml`. The completion scorer is unchanged: "its labels are the erased
  boxes" stays true.
- **The faithfulness audit reports per-class missed-object and false-object rates beside the pooled
  ones.** The gate reads the fuse and switch rates of both as well as the pooled rates:
  - per-class missed-object rate: that class's ground-truth boxes left unmatched, over that class's
    ground-truth boxes. This is the win.
  - per-class false-object rate: that class's predictions left unmatched, over that class's
    predictions. This is the red flag on real frames.

  Matching is already same-class, so the per-class counts add up exactly to the pooled counts. No
  match changes.
- **The completion rate stays lower-is-better.** On an erased box no trace is left, so a component
  that comes back was invented by definition. A pass needs three things against stage 1: fewer
  missed fuses/switches on real val, no rise in invented fuses/switches on real val, and no rise in
  the completion rate on erased val.
- Margins for the new per-class rates are not set here. Gate ADR-04 (CASCADE-GATE-04) sets them from
  the stage-1 loop arm's seed spread, the same rule plan Q3 gives the completion rate.

**Alternatives considered**
- **Erase every ground-truth box (plan Q4 as resolved).** It is the purest counterfactual. But the
  pole fills deform the frame, and poles are not the hypothesis.
- **Erase fuses and switches but copy the labels unchanged.** Poles and transformers would still be
  in the frame and in the labels. The judge finds them easily, so they would count as "put back",
  and the completion rate would mostly measure pole recall.
- **Filter by class in the completion scorer instead of in the labels.** That keeps the erased
  labels identical to the source. But then the scorer needs to know which classes were erased, and
  a mismatch between the erase flags and the score flags would be silent. Writing only the erased
  boxes keeps one source of truth on disk.
- **Judge the pooled rates only (the status quo).** It needs no code. But a fuse/switch gain or a
  fuse/switch invention can vanish into the pole count, so a pass or fail would not answer the
  cascade's question.
- **Per-class missed-object rates only.** It covers the win. But an invented fuse or switch on real
  val would then be caught only by the pooled false-object rate, where poles dilute it.
- **Also erase transformers.** The user ruled it out: the hypothesis concerns fuses and switches
  only.

**Consequences**
- **Cost: two code changes come before the stage-1 probe run,** a class filter on `t2o erase` and
  per-class missed/false rates in the audit. That pushes the first reading closer to the 2026-11-12
  meeting.
- **Cost: record 026's `judges.rtdetr.faithfulness` has no per-class rates.** So the e3b and e3t
  stage-1 exports must be re-scored with the judge (a server pass of minutes). The write-back
  replaces those keys. The pooled values should come out identical because the scorer is the same,
  and the record of that run must check this.
- **Cost: per-class rates rest on fewer boxes than the pooled ones,** so their seed spread, and the
  margins built from it, will be wider. A per-class verdict may come out unresolved rather than pass
  or fail. That is reported as it falls.
- The probe no longer tests invented poles or transformers on erased input. On real val, the pooled
  false-object rate still catches those.
- The all-class erased split from the runbook's step 1 stays on the server unused. The fuse/switch
  split gets a new folder.
- **The gate's own ADR moves to ADR-04.** CASCADE-GATE-04's row still says "Gate ADR-03", so it
  needs a `/task-plan` update. That update can add the two code tasks too.

## ADR-04 — The cascade gate: zero tolerance on completion, stage-1 spread margins on fuse and switch, and every test must pass

**Date:** 2026-11-04
**Status:** accepted
**Tasks:** CASCADE-GATE-04, CASCADE-GATE-07

**Context**
Plan Q3 takes the completion-rate margin, and ADR-03 the fuse and switch per-class margins, from the
stage-1 loop arm's seed spread on the probe. Record 034 has now measured it. Stage 1 put back none
of the 234 erased fuses and switches on any of the nine e3b and e3t loop runs, so the completion
rate is 0.0000 on every seed and its spread is 0 (F217). With a margin of 0, the pass rule from
out-of-family-judge ADR-01 fails every cascade, whatever stage 2 does: it needs the interval to lie
"below +margin", and stage 1 sits at zero, so cascade − stage 1 can never be negative (F218). The
per-class spreads are all non-zero, but they are 1.2–4.7× the pooled ones (F219). Two more things
were open. ADR-03 says a pass needs "fewer missed fuses/switches", while the task row says "on the
safe side of each margin". And nothing said how the per-cell, per-rate verdicts combine into the one
answer the gate exists to give: does the cascade enter the IEEE TIM paper?

This entry supersedes plan Q3 for the completion rate only. ADR-03 stays accepted: only its "fewer
missed" wording is read here, and its erasure decisions stand. As of this entry, no stage-2 output
exists (CASCADE-GATE-05 is pending).

**Decision**
- **The comparison.** For each cell, e3b (pix2pix, seeds 0–5) and e3t (turbo, seeds 0, 1, 3), the
  comparison is cascade − stage 1, paired by seed. Stage 1 is the loop arm's stage-3 export, and the
  cascade is stage 2 applied to that same export. Everything is scored on val (out-of-family-judge
  ADR-02) under the `rtdetr` checkpoint `runs/reference-rtdetr-l/weights/best.pt`, with the default
  scorer settings (IoU 0.5, conf 0.25, imgsz 640, batch 16). The erased frames come from
  `runs/cascade-gate/erased-custom-val-fuse-switch` (Fuse and Switch, Telea, 4 px, radius 3.0), as
  record 034 scored them.
- **Completion rate (erased val): zero tolerance.** A cell passes only if the cascade's completion
  rate is 0.0000 on every seed. Because stage 1 is 0 on every seed, this is the same as the paired
  95% CI's upper end being ≤ 0. This replaces plan Q3's spread margin, for the completion rate only.
- **Fuse and Switch missed- and false-object rates (real val): not worse, within margin.** A rate
  passes when the 95% seed-paired bootstrap CI of cascade − stage 1 lies below +margin, in the same
  form as ADR-01. The margin is the stage-1 loop arm's seed sd (ddof = 1) from record 034:

  | cell (seeds) | missed Fuse | false Fuse | missed Switch | false Switch |
  | --- | --- | --- | --- | --- |
  | e3b (6) | 0.0229 | 0.0284 | 0.0602 | 0.0487 |
  | e3t (3) | 0.0209 | 0.0342 | 0.0546 | 0.0558 |

  ADR-03's "fewer missed fuses/switches" is read as this not-worse test. Stage 2 does not have to
  show a gain.
- **Pooled false- and missed-object rates (real val):** ADR-01's margins, unchanged. They are e3b
  0.0179 / 0.0094 and e3t 0.0315 / 0.0112, under the same "below +margin" rule.
- **Pooled detection-consistency is reported, not gated.**
- **The gate's answer.** That makes seven gated tests per cell, fourteen in all. The cascade enters
  the paper only if all fourteen pass. Any fail keeps it in paper 2. Each test's verdict is still
  reported per cell and per rate, and a fail is reported as a fail (out-of-family-judge ADR-02).
- **How passes and fails read** follows out-of-family-judge ADR-02. A fail is robust. A pass rests
  on an anti-conservative interval at n = 3–6 and is reported beside its exact sign-flip p.
- **Scope.** These margins belong to record 026's `rtdetr` checkpoint and to record 034's erased
  tree, and to nothing else.

**Alternatives considered**
- **Plan Q3's spread margin for completion, as written.** The margin comes out at 0, so the test
  cannot pass under ADR-01's strict rule even when stage 2 puts back nothing (F218).
- **A one-box resolution margin, 1/234 ≈ 0.0043, under the strict rule.** It tolerates a stray box
  on one seed and fails a repeatable invention. It lost because ADR-03 already counts any put-back
  as an invention, and the judge found nothing at the erased spots in 2,106 box checks after stage
  1. A stray box is not noise to forgive.
- **Report the completion rate without gating on it.** This is the weakest guard against the red
  flag the gate exists for: a component drawn where the thermal frame holds no trace.
- **Missed fuses and switches must improve (CI upper below 0).** This reads ADR-03 literally. But it
  turns a faithfulness gate into a detection-gain claim, which the plan puts out of scope ("Margin
  or Transfer numbers for the cascade"). At n = 3 (e3t), the CI may rarely clear zero.
- **Per-cell entry**, where each backbone's cascade enters on its own verdict. It was rejected for
  one answer that needs every test on both cells, which is how ADR-03's "a pass needs three things"
  reads.
- **Gate consistency too, at ADR-01's margins.** The task row gated insertion and deletion only.
  e3t's consistency margin, 0.0038, is the tightest in ADR-01 and may fail on noise alone.

**Consequences**
- **Cost: this is the strictest possible completion test.** One fuse or switch put back on one seed
  of nine fails the gate, and with it the cascade's entry into the January paper. A spurious judge
  detection at an erased spot also counts. Stage 1 showed none, but stage 2 changes the image, so
  that is not guaranteed.
- **Cost: the gate no longer uses one method.** Plan Q3 chose spread margins so a reviewer sees one
  rule in one unit. The completion test is now a count rule beside them, and the paper has to say
  why (F218).
- **Cost: fourteen tests must all pass.** Each one passing on an anti-conservative interval makes
  the overall pass weaker than it looks. Any single fail decides the answer.
- **The per-class margins are coarse** (F219). e3b's Switch missed-object margin, 0.0602, is about
  four of val's 66 switches. A cascade can pass that test while missing a few more switches per
  seed.
- **e3t's margins rest on three seeds** and are unstable estimates, as ADR-01 noted for its own.
- **Consistency is ungated.** A cascade can pass while it changes what the detector finds compared
  with real visible frames. Only the report shows that.
- **The CI needs a cascade-vs-stage-1 pairing.** Record 034 found that `t2o aggregate` refuses runs
  from a single arm. CASCADE-GATE-06 and -07 must produce a seed-paired comparison it accepts, or
  compute the same interval another way.
- **Re-erasing val or retraining the judge voids these margins.** `runs/reference-rtdetr-l` and
  `runs/cascade-gate/erased-custom-val-fuse-switch` must never be overwritten.
