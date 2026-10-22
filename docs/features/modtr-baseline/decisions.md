# ModTr baseline — the non-inferiority anchor for Margin — Decisions

## ADR-01 — Reimplement ModTr's U-Net on torchvision instead of vendoring or adding smp

**Date:** 2026-10-22
**Status:** accepted
**Tasks:** MODTR-BASELINE-01, MODTR-BASELINE-03, MODTR-BASELINE-04

**Context**
The plan (Q2) said to vendor ModTr's network under `third_party/` at a pinned commit, and to
reimplement it from the paper if vendoring was not clean. The spike read upstream at
`cace11eeb51f1421149ff4740f14e3b38f03d58c` (2024-11-28, MIT). It found that ModTr's repo contains
no network of its own. `src/translator.py:25-30` builds `segmentation_models_pytorch.Unet` with
encoder `resnet34`, depth 5 and `encoder_weights="imagenet"`, and swaps the head's last layer for
a sigmoid. The paper confirms this: Sec. 4.1(b) names "PyTorch Segmentation Models [20]", Table 3
gives 24.4M parameters for the ResNet34 default, and Sec. 3(b) says the framework "is not limited
by the translation architecture". ModTr's own contribution in code is about 10 lines of fusion
(`src/engine/utils/utils.py:507-516`, the paper's Eq. 3). `segmentation-models-pytorch` is not a
`t2o` dependency, and adding it would also bring in `timm`, which is absent from `uv.lock`.

**Decision**
Reimplement the network in `src/t2o/translators/` on torchvision's `resnet34`, matching
`smp.Unet`'s defaults:
- an ImageNet-pretrained ResNet34 encoder, depth 5;
- decoder channels (256, 128, 64, 32, 16);
- in each decoder block, a nearest ×2 upsample, concatenation with the encoder skip, then two
  3×3 conv–BN–ReLU layers;
- a 3×3 conv head to 3 channels with a sigmoid.

The input is 1-channel thermal in [0, 1], repeated to 3 channels with no ImageNet normalization
(upstream `utils.py:96-97`). The output is fused as `clip(x ⊙ h(x), 0, 1)`, the paper's ModTr⊙,
which is the README's `--fuse-data cross`. The module docstring cites the paper section and the
upstream `file:line` for each piece and names every deviation. A test pins the parameter count at
24.44M (the paper's 24.4M; a shape-equivalent stand-in measured in the spike gives 24.44M).

**Alternatives considered**
- Add `segmentation-models-pytorch` as a dependency: the exact upstream network with no code to
  own. But it is a new dependency that pulls in `timm`, for one model's worth of code that
  torchvision already covers.
- Vendor smp's `Unet` and ResNet encoder files under `third_party/`: those files import smp's
  package internals (base classes, encoder registry). Making them import standalone means editing
  vendored code, which invariant 4 forbids.
- Vendor ModTr's own files: there is no network to vendor. `translator.py` is a wrapper around the
  smp call, and `modtr.py` imports the torchvision detectors at module level. Only the fusion
  function is ModTr's, and that is shorter than a citation.

**Consequences**
- No new dependency. The network rests on torchvision, which is already pinned.
- Cost: `t2o` owns about 80 lines that have to match smp's decoder. A silent mismatch would make
  the arm "not ModTr" with no error. The parameter-count test is the guard, and it catches
  structural drift but not every detail (for example, the upsample mode).
- Cost: the encoder weights are torchvision's ResNet34 ImageNet weights. That they are the same
  weights smp 0.3.3 loaded as `"imagenet"` is believed, not verified in the spike.
- The paper must disclose that ModTr is reimplemented: its network on torchvision and its
  detection term as YOLO's loss (margin-gate ADR-01). It cites upstream at the pinned commit for
  the training values the main paper does not state (they are in upstream `main.py:53, :304-321,
  :404-418` and `config.py`; the paper defers them to supplementary material that is not in the
  repo).
- MODTR-BASELINE-03 takes its reimplementation path. Nothing goes under `third_party/modtr/`, so
  plan.md's "upstream network vendored" paragraph no longer describes what gets built. The
  resolved Q2 fallback covers it.
