# Paired→unpaired cascade (pix2pix → CycleGAN) — Decisions

## ADR-01 — RGB infrared input for the stage-1 tree

**Date:** 2026-11-13
**Status:** accepted
**Tasks:** CYCLEGAN-CASCADE-03, and a loader task `/task-plan` has yet to add

**Context**
Stage 2's input is pix2pix's RGB output, stored as an RGB PNG under the infrared token of the
stage-1 tree (CYCLEGAN-CASCADE-01). The loader always reads the infrared side as one grey channel:
`src/t2o/data/dataset.py:321` decodes it with `"L"`, and `TranslationSample.infrared` is
`(1, H, W)`. Left as is, CycleGAN would only see the luminance of stage 1's output. Stage 2's job
is to move that output closer to real optical images, so throwing away its colour defeats the
cascade. Q8's precompute route still stands, but its note that "the data contract stays intact"
no longer fully holds: the contract gains an opt-in channel count.

**Decision**
Add a `DataConfig` field `infrared_channels`, which takes `1` or `3` and defaults to `1`. At `3`,
`TranslationPairDataset` decodes the infrared side as `"RGB"`, giving `(3, H, W)` per sample and
`(B, 3, H, W)` per batch. The field is inside `config_hash()`, but its default leaves every
existing experiment config's hash unchanged, the same pattern `misalignment_px` and `calibration`
follow. The `cyclegan` backbone takes its input channel count from this field. Both cascade
configs set it to `3`.

**Alternatives considered**
- Accept greyscale, with no loader change. It needs no new code, but stage 2 trains on a grey
  copy of stage 1's colour output, so it cannot build on what stage 1 learned. It becomes
  another grey-to-visible translator.
- A composite `cascade` translator that holds frozen pix2pix and CycleGAN and passes stage 1's
  RGB in memory. That reopens Q8: it needs new pretrained-weights config fields and a second
  checkpoint path into `build_translator`, which the precompute route was chosen to avoid.

**Consequences**
- The loader's infrared side is no longer always single-channel. The `(1, H, W)` /
  `(B, 1, H, W)` shapes in `TranslationSample` and `TranslationBatch` become 1 or 3 channels.
- Only `cyclegan` accepts 3 channels. pix2pix fixes its input at one channel
  (`translators/pix2pix.py:85`), ModTr repeats a single channel to three (`modtr.py:119`), and
  pix2pix-turbo expands one channel (`pix2pix_turbo.py:433`). Each would break or misbehave on 3,
  so `build_translator` must refuse any other backbone at `infrared_channels: 3`, at startup.
- The data-contract change costs one more hash-neutrality test, pinned against an existing
  config's hash.
