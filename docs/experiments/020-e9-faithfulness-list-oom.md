# 020 — E9 step 5: `t2o faithfulness` never streamed a list, so every split ran as one batch

**Date:** 2026-10-01 · **Task:** M3 E9
**git_sha:** 7a3ca94 — the fix, which also wrote the source text below; record 016's report ran at it

## Evidence

Lifted verbatim from `TASKS.md:5123-5149` @ pre-spec-migration (M3 E9 step 5):

> ##### Command 2 OOM'd on the first attempt — `t2o faithfulness` never streamed a list
>
> `torch.OutOfMemoryError: Tried to allocate 9.89 GiB` in a YOLO backbone concat, on a card
> already holding 37.4 GiB of this one process's tensors. **A bug, not a tuning problem**, and
> `evaluate_faithfulness`'s own docstring claimed the opposite.
>
> `model.predict` was handed a `list[Path]`, which in ultralytics takes a different route from a
> path or a directory: `check_source` sends a list through `autocast_list` — decoding every
> element up front — and then to `LoadPilAndNumpy`, whose `bs = len(im0)`. `batch=` is passed
> only to the *other* branch, `LoadImagesAndVideos`, so it is silently dropped here, and
> `stream=True` yields a generator over a single iteration. **The whole split was always one
> batch.**
>
> Which is why it surfaced on this cell and no earlier one: the custom set's val split is **153
> images** and fit; FLIR's is **1,013** (step 1's count), 6.6× larger, and its input tensor alone
> is 4.9 GiB. `evaluate_detector` was never affected — it goes through `model.val(batch=16)`.
>
> **The two earlier cells' C2 numbers stand.** `rect` is false by ultralytics' own default, so
> `pre_transform` letterboxes every image to a fixed square `imgsz` independently of its
> neighbours and NMS is per-image: a chunk boundary cannot move a rate. The fix pins `rect=False`
> rather than inheriting it, so that stays true across an ultralytics bump (PLAN.md invariant 1),
> and the test asserts the **chunk sizes** — the failure returned perfectly correct rates right
> up until the card ran out, so nothing else would catch a reversion.
>
> Nothing wrote back before the crash: the traceback is in the *second* detector pass and
> `record_faithfulness` runs after both. All twelve are a clean re-run, and command 3 would have
> produced a valid report with blank faithfulness columns either way.

The code as it stands (`src/` and `tests/` are unchanged since the tag):

- `src/t2o/metrics/faithfulness.py:248-292` — `detect`, which chunks the path list by `batch` and
  pins `rect=False`. Its docstring traces the ultralytics route: `check_source` → `autocast_list`
  → `LoadPilAndNumpy`, `bs = len(im0)`.
- `tests/test_faithfulness.py:436-460` — the chunk sizes are asserted (`[1, 1, 1, 1]`) alongside
  equality with the unchunked rates; `:463` covers a short final chunk.

Observed on the server: `torch.OutOfMemoryError: Tried to allocate 9.89 GiB`, on a card holding
37.4 GiB of the process's tensors. Read from ultralytics' source, not traced: the list route and the
dropped `batch=`.

## Findings

### F156 — A `list[Path]` handed to `model.predict` is decoded up front and run as one batch, `batch=` silently dropped: FLIR's 1,013-image val split asked for 9.89 GiB and OOM'd (legacy: M3 E9 step 5, "Command 2 OOM'd on the first attempt")

Compared against the two splits that did not fail. The custom set's 153-image val split fit in one
batch on a 40 GiB card, which is why both earlier C2 campaigns passed. FLIR's split is 6.6× larger,
and its input tensor alone is 4.9 GiB. `evaluate_detector` was never affected, because it goes
through `model.val(batch=16)`. `evaluate_faithfulness`'s own docstring had claimed the opposite.
Fixed in 7a3ca94 by chunking the list.

### F157 — The earlier C2 numbers stand: with `rect=False` every image is letterboxed on its own and NMS is per image, so a chunk boundary cannot move a rate (legacy: M3 E9 step 5, "The two earlier cells' C2 numbers stand")

Compared against the unchunked pass that scored e3b's C2 (F49–F52, record 008) and e3t's (F76,
record 010). With `rect` false, `pre_transform` letterboxes each image to a fixed square `imgsz`,
independent of its neighbours, so splitting the list changes nothing per image. The fix pins
`rect=False` rather than inheriting it, so this stays true across an ultralytics bump. The test
asserts chunk sizes because the failure "returned perfectly correct rates right up until the card
ran out". Nothing wrote back before the crash: the traceback is in the second detector pass, and
`record_faithfulness` runs after both.

## Provenance caveats

- The equality test monkeypatches `ultralytics.YOLO` with a fake (`_fake_yolo`,
  `tests/test_faithfulness.py:339`). It proves `t2o`'s chunking scores every frame once. It does not
  exercise ultralytics' letterbox. That a chunk boundary cannot move a real rate is reasoned from
  `rect=False`, not tested.
- The ultralytics version installed on the server on 2026-10-01 is not recorded; the pin is
  `>=8.4.108,<8.5`.
- "Its input tensor alone is 4.9 GiB" is the source's arithmetic, not a measurement.
- Record 016 cites this finding as SPEC-MIGRATION-19's. All twelve of its C2 passes are the clean
  re-run after 7a3ca94.

## Next

None of its own. Any other `model.predict` call fed a list needs the same chunking.
