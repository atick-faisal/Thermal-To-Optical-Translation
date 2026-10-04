# 017 — M0.10: `--device 0` raised in `train`/`loop`/`export` while `evaluate` accepted it

**Date:** 2026-08-12 · **Task:** M0.10
**git_sha:** 2a6b17d — the fix, which also wrote the source paragraph below

## Evidence

Lifted verbatim from `TASKS.md:1022-1041` @ pre-spec-migration:

> **Found while running these checks — `--device 0` failed, `--device cuda:0` was needed as a
> workaround.** Real bug, not a server environment quirk: `cli.py` and `engine/trainer.py` each
> carried a private, byte-identical `_resolve_device` that called `torch.device(device)`
> directly on the raw string. Real PyTorch's `torch.device("0")` raises
> `RuntimeError: Invalid device string: '0'` — it needs the `cuda:` prefix. Both `--device`
> help strings advertised `'0'` as valid (copying ultralytics' own convention, where
> `select_device` *does* accept a bare digit), but `train`/`loop`/`export` never go through
> ultralytics' resolver, so the documented spelling silently didn't work; `evaluate` (which
> forwards the device string straight into `model.val(device=...)`, i.e. ultralytics' own
> resolver) tolerated it fine, which is why it only bit during the dataloader/VRAM checks
> above, not the E1 evaluate calls. **Fixed**: `engine/trainer.py`'s `_resolve_device` became
> the single public `resolve_device`, normalising a bare digit to `cuda:{device}` before
> constructing the `torch.device`; `cli.py`'s duplicate copy was deleted and `_run_export` now
> imports the one in `engine/trainer.py`. The misleading `'0,1'` example in the shared
> `--device` help text was also removed — a single `torch.device` can never represent a
> comma-joined multi-GPU list, and `PLAN.md` §3 rules out DDP on Windows entirely anyway (one
> experiment per GPU via `CUDA_VISIBLE_DEVICES`). `evaluate`'s own `--device` help text is
> untouched since it genuinely forwards to ultralytics' multi-device-capable resolver.
> Tests: `tests/test_trainer.py` (`resolve_device("0") == torch.device("cuda:0")`, explicit
> spellings pass through, `None` matches `torch.cuda.is_available()`).

The code as it stands (`src/` and `tests/` are unchanged since the tag):

- `src/t2o/engine/trainer.py:91-104` — `resolve_device`, the single resolver. Its docstring
  restates the split: `evaluate` "forwards straight into ultralytics' own resolver and never hits
  this function".
- `tests/test_trainer.py:40-43` — `resolve_device("0") == torch.device("cuda:0")`, with `:46` and
  `:51` covering explicit spellings and `None`.

Observed: `RuntimeError: Invalid device string: '0'`, quoted above from the server checks. Read
from the code, not traced: which subcommands reach `torch.device` directly and which go through
ultralytics' `select_device`.

## Findings

### F150 — `--device 0` raised in `train`, `loop` and `export` but worked in `evaluate`, because only `evaluate` goes through ultralytics' resolver (legacy: M0.10, "`--device 0` failed, `--device cuda:0` was needed")

Compared against what the help strings advertised: both `--device` help texts offered `'0'` as
valid, copying ultralytics' convention, where `select_device` accepts a bare digit.
`torch.device("0")` does not, and `train`/`loop`/`export` built their device with it directly. So
the documented spelling failed in exactly the subcommands that never touch ultralytics' resolver.
It bit during M0.10's dataloader and VRAM checks and never during the E1 `evaluate` calls. Fixed in
2a6b17d: one public `resolve_device` normalises a bare digit to `cuda:{N}`, and the `'0,1'` example
is gone, because a single `torch.device` cannot hold a multi-GPU list.

For reading the ledger: records 002 and 004 quote `t2o evaluate … --device 0`. Those commands are
valid as written, because `evaluate` accepted a bare digit before and after the fix.

## Provenance caveats

- The source has no finding number; the legacy label quotes the paragraph's bold lead.
- The command that raised, and the torch version it raised under, are not recorded. Only the error
  string is.
- Which subcommands bypass ultralytics' resolver is a reading of the code at 2a6b17d, not a trace.
  The test checks the normalisation on any machine. It does not open a CUDA device.

## Next

None. A closed bug; the help text and the resolver now agree.
