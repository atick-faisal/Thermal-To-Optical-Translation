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

## ADR-02 — Margin's non-inferiority test: δ = 5 mAP@50 points, paired image bootstrap on seed-averaged mAP

**Date:** 2026-10-22
**Status:** accepted
**Tasks:** MODTR-BASELINE-02, MODTR-BASELINE-05

**Context**
`goal.md:48-53` scores Margin as a +2 mAP@50 gain whose 95% CI excludes zero, and falls back to
"non-inferiority to ModTr" when the CI cannot resolve +2. Record 027 showed it cannot: the widest
image-level half-width is 9.41 points at n = 100 (F189), and even the loop against its own no-loop
control is 3.03–6.43 points (F190). Seed-to-seed spread is as large as one seed's image noise
(F192: sd 3.84 against a mean half-width of 4.90). So Margin is now a non-inferiority test, and
`goal.md` leaves both its margin δ and its test unstated. Both must be fixed before ModTr's first
number exists, or any choice reads as post hoc.

**Decision**
- **δ = 5 mAP@50 points, absolute.** It is drawn from record 027's e3b campaign
  (`docs/experiments/027-margin-resolvability-gate.md:323`):
  - the loop's mean gain over its no-loop control is 5.12 points (sd 3.84, six seeds);
  - the loop sits about 68 points above raw thermal (per-seed CIs about 57–79, `:106-201`);
  - the loop-vs-control half-width is 4.90 on average at n = 100, which scales to about 4 at
    n = 153.

  δ is therefore about the loop's own gain over no-loop. It is under 8% of the gain over raw
  thermal, so a passing loop keeps at least about 92% of ModTr's improvement over the thermal
  floor. And it sits just above image noise, so a loop that truly ties ModTr can pass on 153
  images.
- **The interval** is a paired 95% bootstrap over the validation images:
  - each resample draws one image set;
  - each seed's mAP@50 is recomputed for both arms on that draw, then averaged across seeds;
  - the CI is taken over the seed-averaged delta (loop − ModTr);
  - each seed's delta and their sd are reported beside it.

  The detector is the zero-shot `yolo11s` reference (margin-gate ADR-01), and the split is the
  validation split, which is the reporting split.
- **The pass rule** is that the lower bound of that CI is greater than −δ, that is, > −5 points.
- **The record:** this entry, committed before any ModTr server run, the smoke run included. That
  dated commit is the pre-registration.

**Alternatives considered**
- δ = 2.5 points, which keeps half the loop's gain over no-loop (the usual "preserve 50%" rule):
  stricter and conventional. But with a half-width of about 4 points, a loop that truly ties ModTr
  fails it most of the time. On 153 images it is close to unwinnable.
- δ as a fraction of ModTr's own mAP@50 (e.g. 10%): reads naturally in a table. But δ then depends
  on the arm under test and is only known after ModTr runs, which a reviewer reads as post hoc.
- Fixing δ after ModTr's first seed: the most information, and exactly the post-hoc choice a
  pre-set margin exists to prevent.
- A hierarchical bootstrap over both seeds and images: the most honest about F192's seed spread.
  But it is wider, so with 3 seeds a tied loop likely fails even δ = 5. It would also reopen
  margin-gate ADR-01's choice of unit.
- The seed-level sign-flip test (design.md:660-666): matches the repo's rule for seed intervals.
  But it tests superiority, not non-inferiority, cannot express a margin well with 3–6 seeds, and
  margin-gate ADR-01 already rejected it for Margin.
- Recording δ as a `goal.md` amendment: the most visible place. But `goal.md` is immutable by
  design, and δ fills a parameter it left open rather than changing a criterion. Faithfulness's
  margin is fixed outside `goal.md` the same way.

**Consequences**
- MODTR-BASELINE-05 builds the multi-seed paired bootstrap and a `non_inferior(lower_bound, delta)`
  gate that defaults to δ = 5.
- Cost: δ = 5 is loose. A loop that is truly up to 5 points worse than ModTr passes. A reviewer
  can call that generous, and the paper has to defend it with the numbers above, not just state it.
- Cost: the interval hides seed variance inside the average. The per-seed deltas and their sd are
  reported to show it, but they do not widen the CI that decides the verdict.
- Margin passing on non-inferiority alone is not a C1 win. Transfer and Faithfulness must carry
  the claim (`goal.md:52-53`).
- Changing δ or the test after a ModTr number exists needs a superseding ADR, and the paper must
  then disclose both versions.

## ADR-03 — Do not run `scripts/loss_share.py` for ModTr; a smoke run replaces it

**Date:** 2026-10-22
**Status:** accepted
**Tasks:** MODTR-BASELINE-02, MODTR-BASELINE-06

**Context**
The repo rule (CLAUDE.md, "Things that will trip you up") is to re-run `scripts/loss_share.py`
(25 epochs) for every new backbone before its campaign and keep the detection term at 20–30% of
the objective. Skipping it once cost ~72 GPU-hours and an uninterpretable null (record 006: the
detection term was 0.9 / 1.7 / 2.3% of E3's objective). The script measures
`task_weight * loss_det / loss_total`, the balance between reconstruction and detection
(`scripts/loss_share.py:1-13`). ModTr's objective is detection only (ADR-01; plan Q1), so there
is no balance to calibrate: the share is 100% by construction.

**Decision**
- `scripts/loss_share.py` is not run for ModTr.
- What the rule protects against is a detection term too weak to train anything. For ModTr that
  is checked directly instead: a short server smoke run must show that the training detection
  loss falls. Pass rule: the mean `loss_det` over the last 5 epochs is below the mean over the
  first 5. A run that fails this does not proceed to the campaign.
- The smoke run's length and form are fixed by the training-budget decision for this feature.

**Alternatives considered**
- Run `loss_share.py` as written: follows the rule literally, but costs 25 epochs and returns
  100% by construction. Its only value would be as a smoke run, which a shorter run gives more
  cheaply.
- Skip both the script and a smoke run: the cheapest. But it repeats the exact failure the rule
  exists for, a campaign launched without evidence that the detection gradient moves the
  translator.

**Consequences**
- The CLAUDE.md rule stays as written. This entry is its documented exception, for a backbone
  whose objective has a single term. Any future backbone that adds a reconstruction term runs the
  script as usual.
- Cost: whether `grad_scale` matters for ModTr is inferred, not verified. With Adam and one loss
  term, a constant scale on the loss should cancel in the update
  (`src/t2o/coupling/detection_loss.py:103`). It cancels only while per-parameter gradients stay
  well above Adam's ε (1e-8). At the default `grad_scale: 1.0e-2` that is not guaranteed for a
  24M-parameter network, so a small `grad_scale` could quietly damp training. The smoke run is
  the only check on this.
- Cost: `reward_target` is a hinge that zeroes a sample's gradient once its loss is below the
  target (`detection_loss.py:96-102`). Upstream ModTr has no such hinge. The ModTr configs must
  leave it at its default `null`, or the arm is no longer upstream's objective.

## ADR-04 — ModTr's training budget: a fixed 400 epochs, last checkpoint, with a pre-set calibration run

**Date:** 2026-10-22
**Status:** accepted
**Tasks:** MODTR-BASELINE-02, MODTR-BASELINE-04, MODTR-BASELINE-06

**Context**
Plan Q1 resolved that ModTr uses upstream's optimiser and schedule "where the spike finds them".
Upstream (at the commit ADR-01 pins) has no fixed length. It trains for at most `--epochs`
(200 in every README command, `README.md:23-29`) and stops early with patience 5 on
`val_map_50` (`main.py:407-411`). That metric comes from an inner 80/20 split of the training
set (`src/engine/config/config.py:27`), and the reported model is the best checkpoint by it
(`main.py:392-401`). Two things here make upstream's stopping rule a poor fit:
- the custom set has 600 training images, so an inner split costs 120 of them, and patience 5 on
  a 120-image mAP stops at an epoch that varies by seed;
- the loop selects nothing. It reports its final checkpoint, and `translator_best.pt` is never
  loaded (`src/t2o/engine/trainer.py:158-165`). Selecting on the validation split is ruled out
  because validation is the reporting split (`goal.md:134`).

So the arms would differ in both training data and checkpoint selection.

**Decision**
- **Budget:** a fixed 400 epochs, the loop's total of 4 stages × 100
  (`experiments/e3_pix2pix_loop.yaml:21`). It trains on all 600 training images and reports the
  last checkpoint. No early stopping, and no selection on any split.
- **Upstream's optimiser and schedule are kept:**
  - AdamW at lr 1e-4 (`main.py:53`) with torch's default weight decay
    (`src/engine/config/config.py:233-239`);
  - batch 8 with gradient accumulation 2 (`main.py:413`);
  - per-step cosine annealing with `T_max = epochs × train batches per epoch`
    (`main.py:310-311, :370`).

  The schedule is kept as written, quirk included. `T_max` counts batches, but with
  accumulation 2 Lightning steps a per-step scheduler once per optimiser step, so a full-length
  run ends at about half the starting lr, not near zero. That reading of Lightning's behaviour is
  inferred, not run.
- **Calibration run, fixed before it runs:**
  - one seed, on an 80/20 split of the 600 training images, logging inner mAP@50 every epoch
    against the in-loop `yolo11n`. Validation images are never touched;
  - it doubles as ADR-03's smoke run, and ADR-03's loss pass rule applies to it;
  - **trigger:** take the 10-epoch trailing mean of inner mAP@50. If its final value is more than
    2.0 points below its maximum, the budget for the campaign becomes the last epoch of the
    best 10-epoch window. Otherwise it stays at 400;
  - whatever the outcome, it is recorded through `/log-experiment` before the campaign starts.
- **Disclosure:** the paper states that ModTr was trained for a fixed budget instead of upstream's
  early stopping, and why.

**Alternatives considered**
- Upstream's early stopping as written (an inner 80/20 split, patience 5, best checkpoint): the
  most literal recipe. But it gives ModTr 20% less training data than the loop, a stopping epoch
  that varies by seed, and a selection step the loop does not get.
- Early stopping or best-checkpoint selection on the validation split: uses all 600 images for
  training. But validation is the reporting split, and selecting on it inflates ModTr's reported
  number.
- A fixed budget with no calibration: the simplest. But 400 epochs on 600 images could overfit,
  and nothing would show it.
- Upstream's 200 epochs as a fixed budget: closer to the README's number. But it is upstream's
  ceiling, not a length it ever trained to, and it gives ModTr half the loop's passes over the data.
- A raw-peak trigger (final value more than 2 points below the single best epoch): the rule as
  first sketched. But the maximum of 400 noisy readings on 120 images sits above the true curve,
  so it would fire on noise alone and pick a lucky epoch.
- A clean full cosine to zero instead of upstream's half-length one: arguably what upstream
  intended. But it is not what upstream's code does, and a reviewer comparing against upstream
  would find a different schedule.

**Consequences**
- MODTR-BASELINE-04 builds a schedule that runs a fixed number of epochs and reports the last
  checkpoint, through the same `fit`/export path as the loop. MODTR-BASELINE-06 sets the
  campaign's epoch count from the calibration result.
- Cost: this is a deviation from the published recipe, and a reviewer can call it "not ModTr".
  The defence is parity with the loop, and it must be disclosed.
- Cost: if the trigger fires, the campaign's cosine schedule (`T_max`) is set for a different
  length than the calibration run's. The calibration then speaks to a schedule the campaign does
  not exactly repeat.
- Cost: the calibration trains on 480 images, so it has fewer steps per epoch than the campaign's
  600. An epoch count transfers between them only approximately.
- Changing the budget or the trigger after the calibration result exists needs a superseding ADR.
