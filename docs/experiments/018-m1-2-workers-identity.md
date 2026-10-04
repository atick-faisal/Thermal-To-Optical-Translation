# 018 — M1.2 step 2b: `workers` was silently experiment identity

**Date:** 2026-08-13 · **Task:** M1.2
**git_sha:** ba6da8e — the fix, which also wrote the source text below, except one sentence added in e0c4a27 (see Provenance)

## Evidence

Lifted verbatim from `TASKS.md:1753-1811` @ pre-spec-migration (M1.2 step 2b):

> Raised while closing step 2, and it turned out to be a real bug rather than the stale
> comment it looked like. `TrainConfig.workers` defaulted to `4` under a comment reading "0 is
> the safe start" — but the deeper problem was that **changing the worker count changed the
> training run**. `TranslationPairDataset.__getitem__`'s hflip/crop drew from the ambient
> global torch RNG, which at `workers = 0` is the main process's and at `workers > 0` is
> torch's per-worker derivation from the loader's generator. Two different augmentation
> streams. Raising `workers` for throughput therefore changed the reported numbers, and
> raising it via `--workers` changed `config_hash()` at the same time — so a paired E3 run
> launched with a different worker count would not have been paired at all.
>
> **User confirmed up to 16 workers runs clean on the server**, which is what made this worth
> fixing rather than documenting. **Scope of that confirmation, narrowed 2026-09-27:** it holds for
> *one* process on t2o's own single-pool `DataLoader` at `batch_size: 2`. It does **not** transfer to
> an ultralytics detector run — which builds a second, permanently resident validation pool at
> `workers * 2` — and certainly not to two of those at once. M3 E9 step 2 exhausted host RAM that
> way; the mechanism is recorded there and pinned as a comment in `engine/detector_stage.py`.
>
> - [x] `data/dataset.py` — augmentation draws from a per-sample generator seeded by
>       `(augment_seed, epoch, index)`, never the ambient RNG. Keying on the *sample index*
>       rather than call order is the load-bearing part: a worker only ever sees a strided
>       subset of the epoch, so any order-dependent stream necessarily varies with the worker
>       count. New `set_epoch(epoch)`, called by `Trainer.train()`, advances it so epoch *N*
>       does not replay epoch *N−1*'s flips
> - [x] `workers` moved `TrainConfig` → `RuntimeConfig`, i.e. out of `config_hash()` — now
>       true rather than merely asserted, and structurally so, matching how M0.2 handled
>       `device`/`name` and the reason `seed` went the other way
> - [x] `tests/test_trainer.py::test_training_is_bit_identical_at_any_worker_count` — the same
>       config trained at `workers=0` and `workers=2` must land on bit-identical weights.
>       Confirmed to **fail** with the per-sample generator reverted, so it is the actual
>       regression guard for this, not a restatement
> - [x] `tests/test_dataset.py` — augmentation ignores ambient RNG *and* call order (built
>       forwards, then rebuilt in reverse); `set_epoch` advances the stream; a different
>       `augment_seed` gives a different stream
>
> **Decision — `persistent_workers` must stay off, and `set_epoch` is why.** The epoch reaches
> the workers only because a `DataLoader` without `persistent_workers` re-pickles the dataset
> when each epoch's iterator is created. Turning it on for throughput without also propagating
> the epoch would silently replay epoch 0's augmentation forever — a bug that costs nothing
> visible and quietly removes most of the augmentation. Stated in
> `TranslationPairDataset.set_epoch`'s own docstring, where anyone about to enable it will
> read it.
>
> **Decision — `experiments/pix2pix_*.yaml` keep `workers: 0`,** now under `runtime:`. The
> move changes their `config_hash()` regardless (a field left the hashed section), but the
> *value* is documentary: it records how M1 actually ran. E3's configs (step 3) take
> **`workers: 16`**.
>
> **Consequence for step 4's aggregator — do not load an old snapshot through `Config`.** M1's
> two completed server runs have `train.workers` in their `runs/*/config.yaml`, which
> `extra="forbid"` now rejects (verified). Since the aggregator's whole reason for joining runs
> to their sibling snapshot is to keep those two runs readable, it must parse the snapshot as
> plain YAML and read the keys it needs, not round-trip it through `Config.load`.
>
> **M1's and M1.2's recorded numbers stand** — those runs are finished and their `metrics.json`
> files are untouched. But be explicit about what does *not* follow: the augmentation stream
> changed at **every** worker count, `0` included, since it no longer comes from the ambient
> RNG at all. Re-running `experiments/pix2pix_baseline.yaml` today would not reproduce M1's
> 0.7851 exactly. That is the same "must land before the campaign starts" caveat as step 2, and
> it is why E3's six seeds are run fresh rather than reusing M1's two runs as a control.

The code as it stands (`src/` and `tests/` are unchanged since the tag):

- `src/t2o/data/dataset.py:276-283` — `_augment_generator`, seeded by
  `(augment_seed, epoch, index)` and nothing else.
- `src/t2o/data/dataset.py:265-273` — `set_epoch`, whose docstring carries the
  `persistent_workers` warning; `src/t2o/engine/trainer.py:212` points at it.
- `tests/test_trainer.py:337` — `test_training_is_bit_identical_at_any_worker_count`.

Observed: the bit-identical test was "Confirmed to **fail** with the per-sample generator
reverted", and `extra="forbid"` rejecting M1's old snapshots was "verified". Reasoned, not
executed: the `persistent_workers` replay (no test enables it), and the claim that M1's run would no
longer reproduce.

## Findings

### F151 — Changing `workers` changed the training run and its `config_hash()`: augmentation drew from the ambient RNG, whose stream at `workers = 0` differs from `workers > 0` (legacy: M1.2 step 2b)

Compared against what the config asserted. Before ba6da8e, `TranslationPairDataset.__getitem__`
drew its flips and crops from the global torch RNG. At `workers = 0` that is the main process's
stream; at `workers > 0` it is torch's per-worker derivation. Two runs that differed only in
`--workers` therefore trained on different augmentation and landed in different `config_hash()`
cells, so a paired E3 run launched at another worker count would not have been paired at all. Fixed
by a per-sample generator keyed on the sample *index*, not call order. A worker sees only a strided
subset of the epoch, so any order-dependent stream varies with the worker count. `workers` also
moved to `RuntimeConfig`, out of the hash. The guard is the bit-identical test at `workers=0` and
`workers=2`. It fails with the fix reverted.

### F152 — `persistent_workers` must stay off: `set_epoch` reaches the workers only because a non-persistent `DataLoader` re-pickles the dataset each epoch (legacy: M1.2 step 2b, "Decision — `persistent_workers` must stay off, and `set_epoch` is why")

Compared against turning it on for throughput, which looks free. With persistent workers the epoch
never reaches the worker copies, so every epoch after the first silently replays epoch 0's
augmentation. Nothing errors and no number flags it; most of the augmentation simply disappears. The
warning lives in `set_epoch`'s docstring, where anyone about to enable the flag will read it.

### F153 — M1's recorded numbers stand, but M1's run no longer reproduces: the augmentation stream changed at every worker count, `0` included (legacy: M1.2 step 2b, "M1's and M1.2's recorded numbers stand")

Compared against re-running `experiments/pix2pix_baseline.yaml` after ba6da8e: it would not land on
the M1 baseline's 0.7851 (F10) exactly. The finished runs' `metrics.json` files are untouched, so
the numbers stand as records of what ran. This is why E3's six seeds were run fresh rather than
reusing M1's two runs as a control. M1's run directories also stay readable only as plain YAML. Their
snapshots carry `train.workers`, which `Config.load` now rejects under `extra="forbid"`, so
`t2o aggregate` never round-trips a snapshot through `Config`.

## Provenance caveats

- **One sentence in the lifted text is not this record's.** "Scope of that confirmation, narrowed
  2026-09-27 …" was added in e0c4a27, six weeks after ba6da8e, and is F97's (record 013). It is
  kept because it sits inside the source paragraph. F151–F153 do not rest on it.
- "User confirmed up to 16 workers runs clean" is a report from the user, not a measurement. F97,
  F120 and F130–F133 bound it for ultralytics runs and for two runs at once.
- The lifted text's "M1's 0.7851" is the reference `yolo11s` judge's score of M1's baseline (F10,
  record 004), not M1's own gate figure of 0.7751 (F03).
- The bit-identical test runs on CPU on the synthetic fixture at `workers` 0 and 2. It was never run
  at `workers: 16` on the server.
- F153's "would not reproduce" is reasoned from the change, not from a re-run.

## Next

None of its own. `workers` as a throughput lever, now result-neutral, is carried by F120 and F130.
