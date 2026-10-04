# General Guidelines
- Think Before Coding. Don't assume. Don't hide confusion. Surface tradeoffs.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.
- Simplicity First. Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.
- If you write 200 lines and it could be 50, rewrite it.
- No error handling for impossible scenarios.

# Working Order
**The working loop lives in `CLAUDE.md` `## Planning`.** Read `docs/goal.md` first,
then the feature's `docs/features/<slug>/` (`plan.md`, whose `## Resolved` is settled
law, and `tasks.md`). Plan with `/feature-plan`, build one task with `/implement`,
gate on `just verify`, commit with `/commit-task`. This file does not describe a
second loop.

`PLAN.md` and `TASKS.md` are frozen source material for the `spec-migration` feature
until it deletes them. Read them; never add to them.

# Where the work lives

- **What we are proving, and why** — [docs/goal.md](docs/goal.md). Immutable.
- **How the code is built** — [docs/design.md](docs/design.md), `PLAN.md`'s `§1`–`§16` under
  their original numbers.
- **Planned work** — `docs/features/<slug>/`: `plan.md` and `tasks.md`.
- **What is still open** — [docs/roadmap.md](docs/roadmap.md): every open row under its legacy
  label, the paper obligations, and a status line per drafting criterion.
- **What was measured** — [docs/experiments/](docs/experiments/index.md), starting at
  `index.md`: the run ledger and the findings ledger.
- **The proposal of record** — `RESEARCH_FINDINGS.md`, frozen. It defines `E1`–`E10` and
  `C1`–`C4`.

# The experiment loop

```
docs/roadmap.md + open findings       the direction. A human picks it; nothing automates this
        |  /feature-plan              one stage or campaign, never one run
docs/features/<slug>/                 plan.md (the design) + tasks.md (the buildable steps)
        |  /implement                 on the Mac, gated on `just verify`
   -- the server run --               not a task, not gated
logs/<YYYY-MM-DD>-<task>-<what>.txt   pasted back unfiltered
        |  /log-experiment
docs/experiments/NNN-*.md + findings
```

**A run is not a task.** A task is code this Mac can verify; the run happens *between* two
tasks, and its result is a record. A `TASKS.md` step used to hold the code, the server run and
its analysis in one row, which is how that file reached 5,419 lines and stopped being a
checklist.

**Findings live in `docs/experiments/`** — never in a `tasks.md` row, never appended to a
`plan.md`, never in `docs/roadmap.md`.

# Server runs

Save every server log as `~/.claude/guidelines/research.md` §Conventions says: to
`logs/<YYYY-MM-DD>-<task>-<what-ran>.txt` (git-ignored), unfiltered. How code reaches the server
is [docs/design.md](docs/design.md) §14 and `docs/goal.md`'s Constraints.

No server log was saved before the ledger existed. Records 001–016 therefore read
`Log: none saved — transcribed from TASKS.md <lines>`, or name a file under `runs/` — a report
or a CSV. `runs/` is git-ignored, so those files exist only on the machine that wrote them, never
on a fresh clone.

# Citation legend

`PLAN.md` and `TASKS.md` are retired, and frozen byte for byte at the annotated tag
`pre-spec-migration` (`git show pre-spec-migration:<file>`). Code comments, configs, the frozen
records and commit subjects still cite both by section. Those citations stay exactly as written
— a record is never edited to chase a reference that moved — and this table resolves them.

| Citation | Resolves to |
| --- | --- |
| `PLAN.md §N` | [docs/design.md](docs/design.md) `§N`, same number; `§13` → this file's `# House style` |
| `TASKS.md M<x>[ step <n>][ finding <k>]`, commit subjects like `(M3 E9)` or `(M1.2 step 8)` | measured work → the record whose `Task:` names that section (`# Conventions` below says where a step or finding sits); open work → the same label in [docs/roadmap.md](docs/roadmap.md); build narrative → `git show pre-spec-migration:TASKS.md` |
| `RESEARCH_FINDINGS.md §N`, `E1`–`E10`, `C1`–`C4` | unchanged — the file stays at the root, frozen; `E` and `C` are defined in its §7 and §1 |

**`C1`–`C4` are numbered differently in `docs/goal.md`.** Legacy C1 is goal C1; legacy C2
(faithfulness) is goal C3; legacy C3 (protocol) is goal C4; legacy C4 (when translation pays) is
goal C2. Every record, `src/` and the experiment configs use the legacy numbers.

# Conventions

**Legacy labels are never renumbered.** `M0.1`–`M4` (milestones) and `E1`–`E10`
(experiments) are cited by commit subjects, code comments and the frozen records. A legacy label
is tracked in one of **two** places — or in neither:

| Where | What it looks like | Example |
| --- | --- | --- |
| a record's `Task:` field | the section only, `**Task:** M1.2`. A step is in `index.md`'s What ran column; a finding is its F-heading's `(legacy: …)` label | `M1.2` → records 004–008; `M1.2 step 8` → record 007; F31's `(legacy: M1.2 step 8, probe result)` |
| a `docs/roadmap.md` section | a `## <label>` heading over rows carried verbatim from `TASKS.md` | `## M2b` at `roadmap.md:72` |
| neither | build narrative the code has superseded | `M0.8` → `git show pre-spec-migration:TASKS.md` |

**Grep both before saying anything about a label.** Some live in both places: `M2a` is records
009–010 *and* the roadmap's `## M2a step 2` and `## M2a step 3` (`roadmap.md:46`, `:62`); `E9`
is records 012–016 and 020 *and* `## E9` (`roadmap.md:132`). The spelling differs too: E8 and E9
ran under M3, so records write `Task: M3 E9` where the roadmap heading says `E9`. F01–F14
(records 001–004) carry no `(legacy: …)` label; they resolve through their record's `Task:`
field and `Log:` line anchors.

**New work uses new IDs.** A task is `<SLUG>-NN` in its feature's `tasks.md`, and a commit names
it in parentheses — `(SPEC-MIGRATION-23)` — or says `[no-task]`. The commit guard, a Claude
Code hook, refuses a subject in the legacy `(M3 E9)` form.

**F-IDs live only in `docs/experiments/index.md`**, minted by `/log-experiment` from its one
counter. A claim's fate goes in its `Status` cell. Never restate the ledger head ("runs to
record 020") in prose anywhere else: every new record would make it stale.

# House style

Moved here from `PLAN.md §13` at `pre-spec-migration`; [docs/design.md](docs/design.md) `§13`
keeps only a pointer. One edit since: the `basicConfig` bullet names the standalone
`scripts/*.py`, which each own theirs by design.

Match the mature repos (`../Clean-SeAFusion`, `../RGBT-Fusion-Detection`), not the older
`PYTHON_CODING_GUIDELINES.md`:

- `src/` layout; Python ≥3.12; `from __future__ import annotations` in every module.
- Fully annotated. `@dataclass(frozen=True, slots=True)` for value objects; `TypedDict` for
  batch dicts; `Protocol` for pluggable hooks; `StrEnum` for choice-typed config.
- pyright `standard`; ruff line-length 100.
- Module docstrings explaining *why*, citing upstream `file:line` for every deviation.
- **An inline comment on every config field naming its failure mode.**
- `logging.getLogger(__name__)` with **%-style lazy formatting — never f-strings, never
  `print`**. `basicConfig` only in `cli.py` and in each standalone `scripts/*.py`.
- Custom exception subclasses; fail fast at startup with messages saying what was tried.
- argparse CLI with an explicit flag→config-path override table.
- pytest, `tmp_path_factory` synthetic datasets, a `slow` marker for CPU end-to-end runs.
- gitmoji + conventional commits.

**Deliberate deviation:** pydantic replaces the hand-rolled YAML→dataclass `_build`/
`_coerce` loader in `../Clean-SeAFusion/src/seafusion/config.py`. Keep that file's two best
behaviours — **unknown-key rejection** (pydantic `extra="forbid"`) and **`snapshot()`** of
the resolved config into the run dir.

**`config_hash()` covers the experiment, not the invocation.** The `runtime` section
(device, run name, run dir, W&B flags) is excluded wholly, so the same experiment run on
two GPUs under two names carries one hash — which is what lets invariant 6 mean anything.
`seed` therefore lives under `train`, not `runtime`: it is scientific, and a
silently-changed seed on resume is precisely the drift M0.8's warn-on-drift check exists to
catch. Clean-SeAFusion hashed everything (`engine/fusion_trainer.py:60`) and consequently
warned on a renamed run.

# Non-negotiables

- **The seven invariants in [docs/design.md](docs/design.md) §5** — one evaluation path, the
  frozen data contract, interchangeable backbones, vendored code never edited, the loop
  switchable off, an experiment is a config file, three detector roles never conflated. They
  are stated there once and not repeated here.
- **Negative results are recorded, not dropped.** A null is a finding — record 005 is one.
- **A record is never edited after it lands.** What later work did to a claim goes in the
  findings ledger's `Status` and `Acts on` cells.
