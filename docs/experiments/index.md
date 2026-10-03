# Experiments

## Runs

| Run | Date | What ran | Task | Machine | Headline | Findings |
| --- | --- | --- | --- | --- | --- | --- |
| [001](001-e1-reference-bracket.md) | 2026-08-12 | E1 reference bracket: {thermal, visible}-trained detector × {raw thermal, real visible} | M0.10 | Windows server, 2× A100 | visible-trained detector: 0.1887 mAP50 on raw thermal vs 0.9213 on visible | F01, F02 |
| [002](002-m1-phase1-gate.md) | 2026-08-13 | M1 Phase 1 go/no-go: pix2pix λ_det=0 baseline + 4-stage λ_det loop, adapted and zero-shot arms | M1 | Windows server, 2× A100 | λ_det=0 pix2pix: 0.7751 zero-shot mAP50 vs the 0.1887 raw-thermal floor — gate PASS | F03, F04, F05 |

## Findings

| ID | Claim | Run | Status | Acts on / acted on by |
| --- | --- | --- | --- | --- |
| F01 | In-domain detection > 0.9 mAP50 on both modalities; the domain gap, not the sensor, is the problem | 001 | open | |
| F02 | A visible-trained detector collapses to 0.1887 mAP50 on raw thermal (0.9213 on visible); Switch and Fuse near zero, Pole 0.5551 | 001 | open | |
| F03 | The Phase 1 gate passes on the uncontaminated λ_det=0 arm: 0.7751 vs 0.1887 floor, +0.586 mAP50, all four classes | 002 | open | |
| F04 | λ_det raises zero-shot mAP50 monotonically (+0.107 at λ=3 over the epoch-matched control, ~8x the 0.0129 noise floor), but the judge supplied the training gradient — not reportable | 002 | open | |
| F05 | The adapted (fine-tuned) arm is saturated (0.8984–0.9199, inside a 1.5-point same-config gap) and inverts the zero-shot ranking — uninformative | 002 | open | |
