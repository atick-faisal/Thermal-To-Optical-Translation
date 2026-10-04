# Working in t2o

## Verification

**After implementing anything, run `just verify`.** It runs every check in
`.pre-commit-config.yaml`: ruff, ruff-format, pyright and the fast pytest suite.

Do not report work as done until `just verify` exits zero. If a check fails for
a reason unrelated to your change, say so explicitly rather than removing the
check.

Run `just` with no arguments to see every recipe. The ones you will need most:

| Command | Purpose |
| --- | --- |
| `just verify` | Full verification. The entrypoint. |
| `just lint` / `just format` / `just types` / `just test` | One check at a time. |
| `just hooks` | Every prek hook; these **rewrite** files. |

## House style and repo conventions

`AGENTS.md` carries them. It is imported below, so every session loads it.

@AGENTS.md

## Things that will trip you up

- **This Mac has no dataset and no detector weights.** Every component must run
  end-to-end on the synthetic pairs the tests build with `tmp_path_factory`, on CPU,
  in seconds. Anything longer gets the `slow` marker.
- **Never commit `dataset/`**, not even the 9-pair smoke fixture. The remote is
  public and the pairs are unpublished research data.
- **`third_party/` is vendored at pinned commits and never edited in place.** ruff
  skips it — and `*.md`, whose quoted upstream code carries `file:line` citations —
  on purpose.
- **ultralytics honours `data.yaml`'s `path:` literally.** `t2o`'s own loader falls
  through a stale `path:`, so a moved tree passes `t2o` and then raises
  `FileNotFoundError` the moment it reaches ultralytics.
- **ultralytics' `labels.cache` is keyed on file sizes and paths, never contents.**
  A label rewrite that keeps every file the same length is silently ignored. Delete
  the split's `labels.cache` after changing labels.
- **Re-run `scripts/loss_share.py` (25 epochs) for every new backbone before its
  campaign**, and keep the detection term at 20–30% of the objective. Skipping it
  once cost ~72 GPU-hours and an uninterpretable null.

## Planning

Work here is planned before it is built.

- `docs/goal.md` is the north star: Problem, Success Criteria, Non-Goals,
  Constraints. Read it first. It is immutable — changing it is a conversation,
  not a routine edit.
- `docs/features/` holds one directory per feature, named for its slug:
  `plan.md` (whose `## Resolved` section is settled law), `tasks.md` (the
  checklist and its task IDs), `decisions.md` (ADRs), and `pitfalls.md`
  (footguns hit along the way). These are agent state, not published docs.
- `/feature-plan` starts a new feature. `/implement` builds one task from an
  existing one. `/task-plan`, `/write-adr`, `/log-pitfall` and
  `/resolve-pitfall` are on-demand. None of them run on their own — a human
  types them.
- Every commit names its task in the subject, e.g.
  `✨ feat(api): add refresh endpoint (AUTH-ROTATION-04)`. A commit that
  genuinely belongs to no task says `[no-task]` instead.
