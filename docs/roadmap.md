# Roadmap

What is still open, and where the drafting criteria stand. *What* the project must show is
[goal.md](goal.md); *how* the code is built is [design.md](design.md); what has been measured is
the run ledger, [experiments/](experiments/index.md). This file only says what is next.

Every open row of `TASKS.md` as tagged `pre-spec-migration` is carried below, in its order there.
Three rules hold:

- **Rows keep their legacy section label** — `M2a step 2`, `M2b`, `M3`, `E8`, `E9`, `M4` — and
  are never renumbered. Commit subjects, code comments and the frozen records cite those labels.
- **Row text is verbatim**, wraps included, and each section names its `TASKS.md` lines at the
  tag (`git show pre-spec-migration:TASKS.md`). A row that was already out of date when tagged
  keeps its text and gains a *Migration note* beneath it.
- **Check a box when its work closes.** What the work measured goes in a record under
  `docs/experiments/`, never here.

## Drafting criteria

Drafting begins when all six of [goal.md](goal.md)'s Success Criteria hold. Each line grades
`goal.md`'s wording and cites the [findings](experiments/index.md#findings) that bear on it.
Scoring against `RESEARCH_FINDINGS.md` §10's legacy five, wherever a record holds it, is evidence
here, never a status.

- **Margin — not measured.** The held-out test split has not been read, and of the
  frozen-detector baselines only raw thermal and the no-loop translator exist; ModTr, HalluciDet
  and zero-shot Grounding DINO wait on `M3`'s baseline suite. On validation, the loop beats its
  no-loop translator (F35, F72) and translation beats raw thermal (F03, F10). F82 is measured
  against a thermal-*trained* detector, the comparison `goal.md`'s Non-Goals concede.
- **Transfer — not measured.** Every score so far comes from the reference `yolo11s`, the same
  family as the in-loop `yolo11n`. No detector from another family has scored a translation.
- **Consistency — 2 of 5 datasets.** The loop beats its no-loop control on the custom set (F35;
  turbo F72) and on FLIR-aligned (F136). LLVIP, M3FD and MSRS have no cell yet. Translation
  against raw thermal is reported, not graded: above the floor on the custom set (F10), on it on
  FLIR (F137).
- **Causality — met on the custom set.** E3 at the calibrated dose (F35), replicated on turbo
  (F72); the earlier null was dose-limited, not a mechanism result (F25).
- **Stability — met for pix2pix.** F35 and F136 are sign-flip tested over six paired seeds;
  turbo's three seeds can corroborate but not reach significance (F72). On FLIR, stage 0 —
  identical in both arms — collapsed on two control seeds and no loop seed, by draw rather than
  by arm (F140).
- **Faithfulness — partial.** C2's insertion and deletion rates favour the loop or match it on
  every campaign (F49, F76, F143), but they are scored by `yolo11s`, the in-loop family, and no
  margin has been fixed for the test split.

## M2a step 2

`TASKS.md:2884-2893`. Why the pretrain runs after E3's turbo campaign rather than before it, and
the checkpoint seam it needs first, are [design.md](design.md) §10, Phase 2a.

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

## M2a step 3

`TASKS.md:2909-2912`. Context is [design.md](design.md) §10, Phase 2a; turbo's C2 result is F76
(record [010](experiments/010-m2a-e3-turbo-replicates.md)).

- [ ] Early-stop / clamp when LPIPS rises past a threshold while λ_det > 0. The LPIPS network
      is already in the loss assembly at `loss.lpips > 0`, so this is a schedule decision, not
      new machinery. M1.1 measured no reward hacking on pix2pix; the turbo arm has far more
      capacity to find it

## M2b

`TASKS.md:3608-3614`. The lower-priority comparison arm; [design.md](design.md) §10, Phase 2b
locates its seam.

- [ ] Vendor at `02c3b13`; grad-enabled copy of `LatentBrownianBridgeModel.decode()`
- [ ] Fix the known template config bugs (issue #47), esp. `UNetParams.image_size` needing
      the *latent* size
- [ ] VQGAN f4 weights from LDM; verify the checkpoint actually loaded
      (`init_from_ckpt` uses `strict=False` and will silently keep random weights)
- [ ] ReFL coupling via `predict_x0_from_objective` / `log_dict["x0_recon"]`
- [ ] Truncated-gradient arm using AlignProp's per-step detach inside a checkpointed loop

## M3

`TASKS.md:3620-3631`, open rows only: the done E8 row between them is record
[011](experiments/011-e8-annotation-sweep.md).

- [ ] Full baseline suite (`RESEARCH_FINDINGS.md` §8)
- [ ] E9 cross-dataset generalisation — **steps 0–4 and 3b all done; the twelve-run cell RAN
      2026-10-01 and its readout is pending** (step 5 below). FLIR-aligned. The kill-test it was
      gated on passed weakly at **+0.2066** 3-class headroom against the custom set's +0.733, which
      is the thing this cell exists to report. The cell's cost was re-measured by step 4 (b) at
      **77–103 GPU-h**, a floor rather than a forecast — not the ~126 GPU-h first recorded here,
      and not the ~72 before that.

  > *Migration note:* already out of date at the tag. E9's FLIR cell has been read — record
  > [016](experiments/016-e9-flir-twelve-run-cell.md), F136 and F137 — and "step 5 below" is that
  > record.

- [ ] E10 faithfulness stress tests
- [ ] E4 coupling comparison: cascaded vs bilevel-**reimplemented** (TarDAL's released code
      severs the generator gradient — see PLAN.md §11)

## E8

`TASKS.md:3899-3913`. The sweep is record [011](experiments/011-e8-annotation-sweep.md); this is
the one decision it left open. The public-dataset cell it waited on is E9, records 012–016.

- [ ] **Deferred: `RESEARCH_FINDINGS.md` §10 criterion 1 is not edited yet.** E8 makes the
      margin *measured and conditional* (+0.650 at `N`=0, −0.125 at `N`=600) rather than
      unknown, which would normally be the moment to rewrite the scorecard. Held deliberately
      until the public-dataset cell runs, on the user's reasoning: **the custom dataset may
      simply be a bad case for this method.** Power lines are close to invisible in thermal,
      which is what makes the zero-annotation gap enormous (+0.650) — and the very same
      legibility gap is why a fine-tuned thermal detector then catches up so fast once it has
      a few hundred boxes. A dataset where thermal is *moderately* legible could behave
      differently in both directions. If it does, the right response may be to **loosen the
      criterion to admit a public dataset** rather than to record a conditional pass here.
      Decide once there is a second dataset to decide with — not before. **The second dataset
      now exists (E9 step 5, 2026-10-01); the decision is still open.** Its crux is that
      step's finding 2: on FLIR the loop beats the control by +0.351 but only reaches the
      raw-thermal floor, so "loop beats control" holds on two datasets and "translation beats
      thermal" holds on one.

  > *Migration note:* possibly settled by `docs/goal.md` Margin/Consistency — human to close.
  > `goal.md` keeps Margin on the power-line test split, counts public datasets through
  > Consistency, and makes "translation beats raw thermal" a reported outcome, not a criterion.
  > The row's +0.650 and −0.125 were written at n = 3; F82 holds the n = 6 values.

## E9

`TASKS.md:5354-5357`, the open items from the twelve-run cell's readout, record
[016](experiments/016-e9-flir-twelve-run-cell.md). "The fixed report" is `campaign_report.py`
after F147's and F148's fixes; the ×2.03 growth is F145.

- [ ] **Still wanted from W&B:** the Runtime column for the twelve runs (wall time including
      stalls, beside the fixed report's span) and the campaign `--group` — the resolved config
      prints `runtime.group: e3-pix2pix-flir600`, which should match.
- [ ] **The ×2.03 per-stage growth**, now measured on two datasets and in an arm with no detector.

## M4

`TASKS.md:5361-5364`.

- [ ] ≥3 seeds on every headline result
- [ ] Significance testing (paired t-test or bootstrap CIs on mAP)
- [ ] Complete ablation grid
- [ ] Check the five acceptance criteria (`RESEARCH_FINDINGS.md` §10)

  > *Migration note:* `goal.md` has six criteria, not `RESEARCH_FINDINGS.md` §10's five; where
  > they stand is [Drafting criteria](#drafting-criteria) above.

## Paper obligations

`TASKS.md:5370-5419`, the corrections owed to the manuscript. Legacy citations inside them resolve
as everywhere else: a step or finding label by grep under `docs/experiments/`, `PLAN.md §N` as
[design.md](design.md) §N.

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
      term dominant is wrong in both. The composition is also **not the same across backbones**:
      turbo's 25-epoch probe splits **detection 24.5% / GAN 50.9% / LPIPS 24.1% / l2 0.5%**
      (M2a step 4), GAN-heavy because the pretrained backbone makes the LPIPS term small.
- [ ] State that `grad_scale: 0.15` was calibrated **separately** for each backbone and happened
      to coincide, rather than being carried over. It is the reason the two E3 cells differ by
      the backbone alone, so the paper should say it explicitly — and say that turbo's raw
      `loss_det` does sit lower (1.98 vs 2.76 at stage 3), the share holding only because the
      fidelity terms fell with it (M2a step 4). Written as inheritance it would read as the
      shortcut PLAN.md §8 forbids.
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
