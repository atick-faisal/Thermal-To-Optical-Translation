---
slug: cascade-gate
title: Enhancer cascade faithfulness gate
status: in-progress
created: 2026-11-03
---

# Enhancer cascade faithfulness gate

## Summary

This is the professor's idea: a paired thermal→visible translator (stage 1, our existing
translators), then an unpaired image enhancer (stage 2), then the frozen detector. `docs/goal.md`
(Non-Goals) calls this cascade a second paper. It enters the IEEE TIM paper only if it passes a
faithfulness gate that is fixed before its first score, by 2027-01-10.

This feature builds **only that gate**, with one zero-shot (untrained) stage-2 model:

- an ADR that fixes the gate's margins and pass rule before any stage-2 output is scored;
- a stage-2 runner that refines existing stage-1 exports;
- the existing insertion/deletion audit, reused as is;
- a new counterfactual-erasure probe. It erases a real object from the thermal input and
  measures whether the pipeline puts it back.

The question the gate answers: **does stage 2 invent or restore objects that stage 1 did not?**
Training stage-2 models on unpaired data (CycleGAN-Turbo, StegoGAN, CUT/UNSB) is paper-2 work and
out of scope. Target for the 2026-11-12 update meeting: this plan, the gate ADR, and a first
zero-shot reading, if the server install allows.

## Tech Stack / Approach

- **Stage 2 is zero-shot and frozen.** Candidates: OSEDiff or PiSA-SR, both one-step diffusion
  restorers built on SD-2.1-base, run at ×1 scale on 640×480 frames. With no training there is
  no gradient path and no loss, so the detector loop does not apply. Stage 2 sees only the
  stage-1 RGB output, because neither model takes a second input.
- **Wrapping follows `src/t2o/translators/pix2pix_turbo.py:1-45,154`.** Vendor only the network
  file under `third_party/` at a pinned commit, never edited. Reimplement the inference
  pipeline in `src/t2o/`. Inject the VAE, UNet and LoRA components, so CPU tests use tiny local
  models, as the turbo tests do.
- **How stage 2 is composed and which stage-1 outputs it reads:** Q1 and Q2.
- **Insertion/deletion audit, reused as is.** `t2o faithfulness` (`src/t2o/cli.py:525-563`)
  scores any image directory through `evaluate_faithfulness`
  (`src/t2o/metrics/faithfulness.py:196-317`). It uses the `rtdetr` judge checkpoints from
  record 026, which must never be overwritten (out-of-family-judge ADR-01).
- **Counterfactual-erasure probe, new.** Nothing in `src/`, `scripts/` or `docs/features/`
  covers it today.
  - **Probe:** inpaint each ground-truth box out of the thermal input (Q4 settles how). Run
    stage 1 alone and stage 1 + stage 2 on the erased input, and score both with the `rtdetr`
    judge.
  - **Completion rate:** the fraction of erased boxes where the judge finds a same-class object
    at IoU ≥ 0.5.
  - **Reuse:** box matching comes from `_greedy_match` (`faithfulness.py:105`) and detections
    from `detections_from_result` (`faithfulness.py:85`).
- **The gate ADR follows out-of-family-judge ADR-01/02**
  (`docs/features/out-of-family-judge/decisions.md:3-140`). It takes a margin per cell and per
  rate, a seed-paired difference with a 95% bootstrap CI, and a verdict per cell and per rate
  that is reported as a fail when it fails. Margin source: Q3.
- **Task order enforces "fixed before the first score":**
  1. build and test the probe;
  2. run the probe on stage-1 outputs only;
  3. write the gate ADR;
  4. build and run stage 2.

  No stage-2 image is scored before the ADR lands.
- **Fuse/switch targeting (ADR-03).** `write_erased_split` takes the class names to erase,
  resolved against `names:`; an unknown name fails, listing the valid ones. A kept label line is
  copied as text, never reformatted. `FaithfulnessMetrics` gains `missed_object_rate_by_class`
  and `false_object_rate_by_class`, `dict[str, float]` keyed by class name and empty by default,
  so record 026's `metrics.json` still loads through `FaithfulnessMetrics(**…)`
  (`src/t2o/engine/loop.py:398,409`). `aggregate.metric_value` already walks nested keys, so
  `faithfulness.missed_object_rate_by_class.fuse` needs no aggregate change.
- **`scripts/loss_share.py` is skipped by ADR.** A frozen zero-shot stage 2 has no objective.
  The precedent is modtr-baseline ADR-03 (`docs/features/modtr-baseline/decisions.md:132-176`).
- **Constraints:**
  - native Windows on one A100-40GB (GPU1), with no triton, xformers or `torch.compile`;
  - a licence check against `goal.md` Constraints;
  - every component tested on synthetic pairs on CPU in seconds.

## Open Questions

## Resolved

### Q1: How is stage 2 composed with stage 1?

- [x] A config-driven `t2o refine` command over an existing stage-1 export (recommended) — the
  experiment stays one config file (invariant 6). It reuses the stage-1 val exports already on
  the server, and `t2o faithfulness` already scores any directory. Nothing in the translator
  protocol changes.
- [ ] A composite `Translator` that wraps a frozen stage-1 checkpoint plus stage 2 — it plugs into
  `t2o export` and `build_translator` unchanged in spirit. But `translate()` must stay on the
  gradient graph (`protocol.py:46-53`), it has to load SD-2.1 next to stage 1 on one card, and it
  re-exports stage 1 for nothing.
- [ ] A standalone `scripts/refine.py` — the fastest to write. But the experiment is no longer a
  config file, and it bypasses `snapshot()` and `config_hash()`.

### Q2: Which stage-1 outputs feed stage 2?

- [x] The custom set's loop arm, e3b (pix2pix) and e3t (turbo), every seed (recommended) — the
  power-line set is the paper's core, both cells already have `rtdetr` margins, and val is only
  153 images per seed, so a first reading fits before 2026-11-12.
- [ ] Loop and control arms of e3b and e3t — this also shows whether stage 2 treats the loop
  output differently from the no-loop output, at twice the runs. The gate doesn't need it.
- [ ] The loop arm of all four cells (e3b, e3t, e3f, e3md) — the full picture across datasets,
  but FLIR's val is much larger and its margins barely constrain (ADR-01). Better as a follow-up
  once the custom reading is in.

### Q3: Where do the gate's margins come from?

- [x] Insertion/deletion: reuse ADR-01's per-cell margins as they stand. Erasure: the stage-1
  loop arm's seed-to-seed spread on the probe, measured before any stage-2 output exists
  (recommended) — both are fixed from data that stage 2 cannot have influenced. Same rule and
  same unit as ADR-01, so a reviewer sees one method, not two.
- [ ] Fixed absolute margins on every rate, e.g. 0.02 — the purest pre-registration, but
  arbitrary and hard to defend. ADR-01 already rejected it for the same reason.
- [ ] Margins from the stage-1 control arm's spread on every rate, the probe included — it
  matches ADR-01's "control spread" wording exactly. But the probe would need control-arm runs
  too, and the cascade is compared against its own stage 1, not against control.

> Superseded by ADR-04 (2026-11-04)

### Q4: How does the erasure probe remove an object from the thermal input?

- [x] OpenCV Telea inpainting of each GT box, slightly dilated (recommended) — a standard,
  deterministic, parameter-light fill that blends with the local background texture. opencv
  already comes in with ultralytics, so no new dependency.
- [ ] Fill each box with the median of a ring of surrounding pixels — the simplest to explain and
  test. But the flat patch is itself an artefact a diffusion model may "repair" into something,
  which confounds the completion rate.
- [ ] Gaussian-blur each box heavily — it keeps the low-frequency heat signature, so a trace of the
  object survives. That tests robustness rather than erasure.

> Superseded by ADR-03 (2026-11-03)

### Q5: Which zero-shot stage-2 model goes first?

- [x] Decide in the first build task, after reading both upstream repos for licence, Windows
  compatibility, ×1 behaviour and extra weights (recommended) — the explorer could not verify
  any of these. A wrong pick costs days of install work before the meeting.
- [ ] OSEDiff — the most cited one-step diffusion restorer, but it also needs a RAM/DAPE tagger
  model: a second set of weights and another licence to check.
- [ ] PiSA-SR — dual-LoRA on SD-2.1, so pixel fidelity versus semantic strength can be turned
  down, which suits a faithfulness gate. Whether it needs the tagger is unverified.
- [ ] Both, side by side — this doubles the vendoring and server install for the first reading.

## Out of Scope

- **Training any stage-2 model** (CycleGAN-Turbo, StegoGAN, CUT/UNSB). That needs an unpaired
  visible target set and an extension of the frozen data contract (invariant 2), and it is
  paper-2 work under `goal.md`'s "Nothing outside the publication" non-goal.
- **Joint or alternating training** of stage 1 through stage 2 with the detector loss.
- **Stage 2 conditioned on thermal** (a second input or cross-network skips): no zero-shot
  candidate supports it.
- **Margin or Transfer numbers for the cascade.** The gate is faithfulness only. If the cascade
  passes, its detection numbers are a later feature.
- **Editing `docs/goal.md`, `docs/design.md`, or any record**: a pass or fail is applied by a
  `/log-experiment` record, as out-of-family-judge ADR-02 did.
