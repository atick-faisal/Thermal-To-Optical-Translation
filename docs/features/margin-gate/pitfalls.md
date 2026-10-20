# Margin gate — resolvability check and test-split pin — Pitfalls

- [ ] [MARGIN-GATE-01] The untrained `detector_weights` fixture outputs only its biases (activations shrink ~4x per layer to ~1e-9 at the head): one confidence for every anchor, every box past the image edge, so it scores mAP@50 0.0 on any labels, its own predictions included — any test that compares, bootstraps or gates on mAP from it passes without testing anything (MARGIN-GATE-04's end-to-end run would see zero-width intervals and `resolves()` true)
