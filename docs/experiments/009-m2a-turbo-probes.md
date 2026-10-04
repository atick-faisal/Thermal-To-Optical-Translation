# 009 — M2a step 4: turbo's VRAM probe and two 25-epoch `grad_scale` probes, at 0.15 and 0.75

**Date:** 2026-08-25 · **Task:** M2a · **Machine:** Windows server, 2× A100 40 GB · **Wall clock:** not recorded
**git_sha:** not recorded — results committed in c54afbb · **W&B:** none — no command in the source carries `--wandb` · **Log:** none saved — transcribed from TASKS.md 2914-3071 @ pre-spec-migration

## Command

As written in `TASKS.md`, in PowerShell since the server is native Windows. The `$DATA` and
`$OPTICAL` paths actually passed are not recorded. The VRAM probe:

```powershell
uv run t2o train --config experiments/e3_turbo_loop.yaml --data $DATA `
  --in-loop-weights $OPTICAL --stage 3 --epochs 1 --name vram-probe --device cuda:0
```

The two calibration probes have **no command written in the source**. It states only their shape:
both seed 0, both `--epochs 25 --no-detector` off `experiments/e3_turbo_loop.yaml`, one per card,
differing only in `--grad-scale` — 0.15 into `runs/turbo-probe-g015`, 0.75 into
`runs/turbo-probe-g075` — each covering all four stages. The `scripts/loss_share.py` invocation
that printed the tables below is not recorded either.

## Configuration

**What step 4 was for.** `experiments/e3_turbo_{control,loop}.yaml` landed in 7a35413 mirroring
the pix2pix pair, with two fields deliberately unvalidated and the headers saying so loudly:
`train.batch_size` and `coupling.grad_scale`. This record is the measurement of both; after it,
both headers carry the numbers instead of the warnings.

- **The pair is pinned.** The differ-only-by-design test is parametrised over both pairs
  (`tests/test_e3_experiments.py:35`, `_PAIRS`), so adding a backbone to E3 is one row. Covering
  only pix2pix would have left the arm PLAN.md §11 calls the strong one unguarded.
- **`train.batch_size: 2`, `train.crop: null`** — the VRAM-determined pair. Full frames keep the
  backbone as the only difference from the pix2pix campaign. The wrapper reflect-pads 640×480 to
  640×512 for the f8 VAE and slices back after decode, so geometry never forces a crop — only
  memory would. The documented fallback, a stride-32 `[512, 512]` (also upstream's recipe), stays
  unused.
- **Why `t2o train` for the VRAM probe, not `t2o loop`.** One stage, no export, no evaluation
  detector, and `--stage 3` selects the top of the ramp so the `FrozenDetector` is resident
  alongside the generator — the actual peak. `--epochs` maps to `train.epochs_per_stage` only;
  the detector fine-tune reads `detector.evaluation.epochs` and is untouched by it, which is why
  `t2o loop --epochs 1` still runs a full 50-epoch ultralytics job per stage. That fine-tune's
  memory is ultralytics' own and unchanged by the translator backbone, so it was not re-measured.
- **`train.lr: 1.0e-4`**, not the pix2pix pair's 2.0e-4: this trains LoRA adapters on a
  pretrained model rather than a generator from scratch, and it is what
  `Pix2PixTurboTranslator.__init__` defaults to. The probe is where a bad LR would have shown, and
  it did not: over stage 0 → 1 `loss_lpips` falls 1.2561 → 0.8148 and `loss_l2` 0.0278 → 0.0169.
- **`coupling.grad_scale` candidates.** 0.15 is **pix2pix's** value (F30, F38), the one informed
  starting point — same loss weights (`l2: 1.0`, `lpips: 5.0`, `gan: 1.0`), so the same order of
  magnitude is where the search begins. The trap was that it looked validated: sd-turbo starts
  pretrained, so its `loss_det` sits at a different magnitude from epoch 0, and 0.15 could have
  landed back near the 2.3% that made E3's first campaign uninterpretable (F25). 0.75, 5× the
  dose, is the bracket. The target is a detection share of 20–30% of the objective.
- **The epoch-length trap (F31)** applies to every cross-horizon comparison below: 25-epoch means
  sit above 100-epoch means in every term.

**Decision — 0.15 kept, not re-tuned.** The alternative was ~0.20, to centre the band under the
epoch decay. Rejected on four counts: the identical-knobs property across backbones is an
explicit design commitment (PLAN.md §8, both config headers) and is worth more than 2pp of share;
F58 measures a higher dose costing fidelity; step 8 already ruled that the band is a calibration
target and not something to chase after the fact (F38); and 24.5% meets the criterion as it
stands. The failure being guarded against — F25's 2.3% — is an order of magnitude away.

## Results

As recorded in c54afbb.

**VRAM probe:** batch 2, full frames, stage 3 with the `FrozenDetector` resident, one epoch —
**34.14 GB peak, no OOM.** The source carries no console output for it.

**Calibration probes**, `scripts/loss_share.py` output verbatim.

`runs/turbo-probe-g015` — `grad_scale: 0.15`, the candidate:

```
stage runs  epochs     w lambda_eff     loss_l2  loss_lpips    loss_gan    loss_det  loss_total     w*det   share
    0    1      25   0.0     0.0000      0.0278      1.2561      0.8444          --      2.1283    0.0000    0.0%
    1    1      25   1.0     0.1500      0.0173      0.8148      0.8002      0.3249      1.9572    0.3249   16.6%
    2    1      25   2.0     0.3000      0.0159      0.7874      1.1686      0.2931      2.5583    0.5863   22.9%
    3    1      25   3.0     0.4500      0.0169      0.8770      1.8513      0.2963      3.6343    0.8890   24.5%
```

`runs/turbo-probe-g075` — `grad_scale: 0.75`, 5× the dose, as the bracket:

```
stage runs  epochs     w lambda_eff     loss_l2  loss_lpips    loss_gan    loss_det  loss_total     w*det   share
    0    1      25   0.0     0.0000      0.0279      1.2527      0.8449          --      2.1255    0.0000    0.0%
    1    1      25   1.0     0.7500      0.0186      0.8712      0.8739      1.5408      3.3046    1.5408   46.6%
    2    1      25   2.0     1.5000      0.0179      0.8737      1.3142      1.3343      4.8745    2.6686   54.7%
    3    1      25   3.0     2.2500      0.0183      0.9097      1.9392      1.2876      6.7299    3.8627   57.4%
```

The source's derived tables, verbatim. Detection share against pix2pix at the same `grad_scale`:

| stage | turbo @0.15, 25 ep | pix2pix @0.15, 25 ep | pix2pix @0.15, 100 ep |
| --- | --- | --- | --- |
| 1 | 16.6% | 10.9% | 10.0% |
| 2 | 22.9% | 16.6% | 16.1% |
| 3 | **24.5%** | 22.5% | 19.8% |

Raw `loss_det` (`loss_det / grad_scale`, since `DetectionTaskLoss.forward` applies `grad_scale`
before `fit` applies the stage weight):

| stage | @0.15 | @0.75 | Δ |
| --- | --- | --- | --- |
| 1 | 2.166 | 2.054 | −5.2% |
| 2 | 1.954 | 1.779 | −9.0% |
| 3 | 1.975 | 1.717 | −13.1% |

Stage 3, both backbones at 25 epochs and `grad_scale: 0.15`:

| term | pix2pix | turbo | Δ |
| --- | --- | --- | --- |
| `loss_l2` | 0.0431 | 0.0169 | −61% |
| `loss_lpips` (weighted) | 1.9002 | 0.8770 | −54% |
| `loss_gan` | 2.3294 | 1.8513 | −21% |
| raw `loss_det` | 2.757 | 1.975 | −28% |
| `loss_total` | 5.5136 | 3.6343 | −34% |

Re-render from the run directories without re-running, on the server:
`uv run python scripts/loss_share.py --runs runs/turbo-probe-g015` and the same for
`runs/turbo-probe-g075`.

## Findings

### F55 — `train.batch_size: 2` fits turbo at full frame: 34.14 GB peak on a 40 GB A100 at stage 3 with the `FrozenDetector` resident, ~5.8 GB of headroom (legacy: M2a step 4, `batch_size` confirmed)

Measured against the card's 40 GB with one stage, no evaluation detector, and the in-loop
detector resident — a peak-with-activations figure. So 2 is a real ceiling now, not just
upstream's conservative recipe for the *paired* model. What it does not cover is the stage
boundary, where ultralytics' detector fine-tune allocates beside everything the translator still
holds; see Next.

### F56 — `grad_scale: 0.15` lands turbo in the 20–30% band at the first candidate: 16.6 / 22.9 / 24.5% at 25 epochs, better placed than pix2pix's 10.9 / 16.6 / 22.5% at the same value (legacy: M2a step 4, calibration result finding 1)

**The probe pair is its own control.** Stage 0 is λ-inert in both runs and its totals agree to
**0.13%** (2.1283 vs 2.1255). The two runs differ by dose and nothing else, which is what licenses
reading every stage-1-to-3 difference in F57 and F58 as an effect of the dose.

Applying pix2pix's own measured 25 → 100 epoch decay (22.5 → 19.8, ×0.88; F30 → F38) projects
turbo to **~21.6% at 100 epochs** — inside the band, where pix2pix's campaign ran just below it
and was recorded rather than re-tuned. Interpolating between the two probes, the band at stage 3
spans roughly `grad_scale ∈ [0.11, 0.21]`, so 0.15 sits near its centre. The share is steep in
this parameter — `w·det` scales close to linearly while the fidelity total barely moves — which
is why 0.75 overshot to 57.4% rather than to something intermediate. That the same value lands
in the same place is no coincidence of the backbones: numerator and denominator fell by similar
factors (F59).

### F57 — The dose bites in loss space within a 25-epoch probe: raw `loss_det` is −5.2 / −9.0 / −13.1% at 0.75 against 0.15, monotone in stage (legacy: M2a step 4, calibration result finding 2)

The further up the ramp, the more the extra dose buys. This is the signature that was **absent**
on pix2pix at `1.0e-2` (step 7, record 006), did not move in pix2pix's own 25-epoch probe (F33),
and only appeared there at 0.15 over a full 100-epoch stage (F37). Here it is visible in 25
epochs.

### F58 — The dose costs fidelity: weighted `loss_lpips` is +6.9 / +11.0 / +3.7% worse at 5× the dose across stages 1 / 2 / 3, against a stage-0 null that agrees to 0.3% (legacy: M2a step 4, calibration result finding 3)

The pix2pix campaign's +0.0097 stage-3 LPIPS (F43) reproduces here as a dose-response rather than
a single point. There is no headroom argument for pushing `grad_scale` above the band's centre,
which is the second of the four counts behind keeping 0.15.

### F59 — Turbo is lower than pix2pix on every term of the objective at stage 3, `loss_total` −34% — descriptive only (legacy: M2a step 4, calibration result finding 4)

Both backbones at 25 epochs and `grad_scale: 0.15` (table under Results): `loss_l2` −61%,
weighted `loss_lpips` −54%, `loss_gan` −21%, raw `loss_det` −28%. Turbo's raw LPIPS at 25 epochs
(0.175) already beats pix2pix's at 100 (0.298). This is also *why* the same `grad_scale` lands in
the same place (F56). **No claim attaches to it** — training loss on the train split, one seed,
25 epochs, not an evaluation metric. PLAN.md §11 calls turbo the strong arm; the campaign is what
tests that.

### F60 — Turbo's objective is GAN-heavy: detection 24.5 / GAN 50.9 / LPIPS 24.1 / l2 0.5% at stage 3, against the pix2pix campaign's 19.8 / 43.7 / 35.7 / 0.8% (legacy: M2a step 4, calibration result finding 5)

Turbo's LPIPS term is small because the pretrained backbone is good at it, so the GAN term takes
the space. The composition of the objective is therefore not a property of the loss weights
alone: it differs by backbone at identical knobs. 25 epochs against 100, so the comparison is
indicative rather than matched.

### F61 — `loss_gan` climbs +119% across stages 0 → 3, and it reads as catch-up rather than divergence: it ends below pix2pix's (1.8513 vs 2.3294) from a stage-0 base half as high (0.8444 vs 1.7234) (legacy: M2a step 4, calibration result finding 6)

A pretrained generator fools a fresh PatchGAN easily, and the discriminator closes the gap over
four warm-started stages — convergence toward a common equilibrium from a much lower base, not a
runaway. The 0.75 arm climbs +130%, near-identically, so this is not the coupling. Precedent is
on the benign side: pix2pix's probe showed +35% and its campaign +10.6%, the probe's figure being
the short-horizon artifact it was suspected to be (F39). But this is a bigger climb than pix2pix
ever showed, and 100 epochs is where a 25-epoch artifact separates from a real one — an
interpretation to check, not a result; see Next.

## Provenance caveats

- **The source mislabels one citation.** It credits the pix2pix campaign's +0.0097 stage-3 LPIPS
  to "step 8 finding 6". That number is step 8 finding 9 (F43), both at c54afbb, which wrote the
  sentence, and at the tag; finding 6 is the wide stage-0 null (F40). F58 cites F43.
- **Cross-references are re-keyed to F-IDs**: "step 8 finding 3" is F37, "step 8 finding 5" and
  the probe's "+35%" are F39, step 7's "2.3%" is F25, the pix2pix 25-epoch share is F30 and its
  100-epoch share F38.
- **Every pix2pix column comes from record 007's runs, not this record's.** The 25-epoch column is
  007's single-seed probe; the 100-epoch column is the campaign's six-run loop-arm mean. The
  comparisons cross both horizon and n.
- **One seed per probe, and turbo's noise floor was not measured.** The source's control is
  stage-0 agreement (0.13% on the total, 0.3% on LPIPS). Pix2pix's single-run loss-space floor,
  which F44 says still governs probes, is +3.4 / +3.6 / +11.8 / +7.1% for l2 / lpips / gan /
  total (F32) — as large as several of F57's and F58's deltas (stage 3's +3.7% LPIPS sits on it).
  The source does not compare against it; whether it transfers to turbo is unknown.
- **Two numbers are derived, not measured.** The ~21.6% projection assumes pix2pix's ×0.88 decay
  transfers to turbo. The `[0.11, 0.21]` band's interpolation method is not stated; assuming
  `w·det` linear in `grad_scale` and a fixed fidelity total gives about [0.12, 0.20] from the 0.15
  run and [0.14, 0.24] from the 0.75 run, so "roughly" holds.
- **"No detector" in the VRAM probe means no evaluation detector.** `t2o train` builds none; the
  in-loop `FrozenDetector` is resident at `--stage 3`, as the source's checkbox states.
- **The `--epochs` mapping was re-checked at the tag**: `src/t2o/cli.py:50` maps `epochs` to
  `train.epochs_per_stage` and `:59` maps `detector_epochs` to `detector.evaluation.epochs`
  (50 in both turbo configs). A reading of the code, not re-executed.
- **Date is c54afbb's.** The source names no run date. The configs landed in 7a35413
  (2026-08-23 17:13) with both fields unmeasured, and the results in c54afbb (2026-08-25 10:20),
  so all three runs fall in that window. The SHA they ran at, the wall clock, `$DATA`,
  `$OPTICAL`, the probes' exact commands and the `loss_share.py` invocation are not recorded.
- **Lines 2763–2913 are not in this record**: the M2 preamble, step 1's build decisions and
  steps 2–3. Their open rows go to the roadmap (SPEC-MIGRATION-22); the rest is left for the
  coverage map (SPEC-MIGRATION-25). c54afbb also rewrote both config headers; those stay in the
  repo and are not migrated.
- **Nothing later is spliced in.** `git blame` at the tag shows lines 2914–3071 written only by
  3e20cf6, 7a35413 and c54afbb.

## Next

The E3 turbo campaign (M2a step 5) at `grad_scale: 0.15`, batch 2, full frames. Posted before it
runs:

- **The one unmeasured risk: stage 0 → 1 of the first run may OOM.** Both probes ran
  `--no-detector`; the campaign does not. `run_loop` builds the translator once and holds it for
  the whole run, so at each stage boundary ultralytics allocates at `detector.evaluation.batch:
  16` while the translator, the reference `yolo11s`, the LPIPS/KID nets and the just-finished
  `Trainer`'s optimizer state are all still resident. 34.14 GB was a peak-with-activations
  measurement, so torch's caching allocator should absorb the detector's demand out of blocks
  the training step has already freed — but that is an argument, not a measurement, and
  fragmentation is real. If it OOMs, drop `detector.evaluation.batch` to 8: it changes only the
  adapted arm's fine-tune and never the zero-shot gate metric E3 is decided on.
- **Watch `loss_gan` at stage 0 → 1** (F61): catch-up predicts it ends below pix2pix's at 100
  epochs; divergence would not.
- **The achieved stage-3 share at 100 epochs should land near ~21.6%** (F56), inside the band.
