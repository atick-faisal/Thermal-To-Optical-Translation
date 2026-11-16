# Clipped-box audit — Pitfalls

- [ ] [CLIPPED-BOX-AUDIT-05] A pair tree's `data.yaml` (the frozen `yolo_rgbt_29_jul` and every `clipped_box_drop_tree.py` copy) points `val:` at `visible/images`, so passing it straight to `clipped_box_rescore.py` scores a thermal detector on optical pixels with no error — thermal arms need a `write_budget_manifest(..., 1.0, infrared=True)` manifest per tree, or their re-score deltas are silently meaningless
