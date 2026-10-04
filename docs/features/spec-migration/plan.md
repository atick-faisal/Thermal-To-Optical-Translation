---
slug: spec-migration
title: Spec-Workflow Migration
status: shipped
created: 2026-10-02
---

# Spec-Workflow Migration

## Summary

The project runs on three root documents. `RESEARCH_FINDINGS.md` (304 lines) is the original
proposal. `PLAN.md` (806 lines) is live design with results narrative woven through it.
`TASKS.md` (5,419 lines, 377 KB) stopped being a checklist long ago: next to its 147 done and 31
open rows it holds every campaign's tables, 57 bold-numbered finding paragraphs plus two
`##### Finding` headings and many unnumbered findings, and a 13-item list of corrections owed to
the paper. The global `~/.claude/CLAUDE.md` tells agents to skip the spec workflow while a root
`PLAN.md`/`TASKS.md` exists, so `/feature-plan`, `/implement` and `/log-experiment` cannot be used
here until this migration lands.

This feature moves the project onto the spec workflow — `docs/goal.md`, `docs/features/`,
`docs/experiments/`, `docs/design.md`, `docs/roadmap.md`, `just verify` — and then deletes the
two root files, **without losing a finding, a number, an open row or a citation**. Every byte of
the old record stays recoverable from the annotated tag `pre-spec-migration` (on `e70760c`,
created 2026-10-02 before any migration commit).

The sister repo, `../Thermal-Image-Registration`, did the same migration in PR #1 (merged
2026-09-10, 47 commits) as two features: `docs/features/experiment-ledger/` and
`docs/features/plan-tasks-retirement/`. Their `## Resolved` sections are the precedent this plan
cites, and the gaps found after that merge are the guard rails below.

## Tech Stack / Approach

Documentation only. No `src/`, `scripts/` or `tests/` logic changes. `just verify` passes
vacuously on every task (`*.md` is ruff-excluded at `pyproject.toml:116`), so **human review is
the real gate**, exactly as the sister's `experiment-ledger/plan.md` established.

### End state

| Path | Holds |
| --- | --- |
| `docs/goal.md` | Problem, Success Criteria (the six drafting criteria), Non-Goals, Constraints. Written by `/project-init`, immutable afterwards. |
| `CLAUDE.md` | `just verify`, footguns, the fixed `## Planning` section, and `@AGENTS.md`. |
| `justfile` | non-fail-fast `verify` (lint, format, types, test) + `hooks`, as in `../Thermal-Image-Registration/justfile`, derived from `.pre-commit-config.yaml`. |
| `AGENTS.md` | General Guidelines (kept verbatim), where the work lives, the experiment loop, server runs, the citation legend, conventions, house style, non-negotiables. |
| `docs/experiments/` | `index.md` (run + findings ledgers) and one record per campaign. |
| `docs/design.md` | `PLAN.md` with its `§1`–`§16` numbering preserved, results narrative replaced by pointers to records. |
| `docs/roadmap.md` | Every open row, verbatim, under its legacy label; the paper obligations; the drafting-criteria status. |
| `RESEARCH_FINDINGS.md` | Unchanged, frozen in place (Q11). |

### Source material — line anchors at `pre-spec-migration`

Read sources with `git show pre-spec-migration:TASKS.md | sed -n 'A,Bp'`, never from the working
tree, so the anchors below stay valid whatever happens to the branch.

| `TASKS.md` lines | Section | Destination |
| --- | --- | --- |
| 1–19 | preamble, legacy working order | superseded by `AGENTS.md` / `CLAUDE.md` |
| 20–969 | M0.1–M0.9, the instrument build | superseded by the code and its module docstrings; named findings become no-run records |
| 970–1047 | M0.10 server bring-up, E1 bracket | record 001; the `--device 0` bug as a no-run record |
| 1048–1365 | M1, the Phase 1 gate | record 002 |
| 1366–1510 | M1.1, fidelity on M1's runs | record 003 |
| 1511–2762 | M1.2, E3 pix2pix (steps 1–9) | records 004–008; build-time findings (step 2b's `workers`, step 9's absent-vs-null) as no-run records |
| 2763–3071 | M2 / M2a steps 1–4 | step 4 → record 009; open rows of steps 2–3 → roadmap |
| 3072–3605 | M2a step 5 + turbo result | record 010 |
| 3606–3617 | M2b | roadmap |
| 3618–3632 | M3 status rows | roadmap |
| 3633–3914 | E8 | record 011; the deferred criterion-1 row (3899) → roadmap |
| 3915–5358 | E9 | records 012–016 + no-run records; open readout rows (5354, 5357) → roadmap |
| 5359–5367 | M4 | roadmap |
| 5368–5419 | Corrections to fold into the paper | roadmap `## Paper obligations` (Q10) |

Also read: `runs/e3f-report-2026-10-01.txt`, `runs/flir-probe-report-2026-09-28.txt`,
`runs/inloop-flir-yolo11n-results-2026-09-28.csv` (git-ignored, on disk) and the tracked
`docs/results/{e3f,e8}-tidy.csv`. There is no `logs/` directory: no server log was ever saved.

### How this differs from the sister migration

| Here | There | Consequence |
| --- | --- | --- |
| `PLAN.md`/`TASKS.md` are **tracked** | git-ignored | The tag is the archive. No Desktop backups — the sister's `~/Desktop/PLAN.md` has already vanished. `TASKS.md` is never collapsed or edited (the sister's `EXPERIMENT-LEDGER-14` is unnecessary). |
| Findings are **locally numbered** ("M1.2 step 8 finding 11") | global `F1`–`F98` | F-numbers are minted fresh (Q5); each heading keeps its legacy label as the crosswalk. |
| **No saved logs** | ten logs in `logs/` | Records are lifted from `TASKS.md` text; `Log:` needs a convention (Q6). |
| `PLAN.md` is mostly **live design**; no `GRID.md` | `PLAN.md` §§1–14 superseded by `GRID.md` | `PLAN.md` moves to `docs/design.md` with its numbering, rather than being declared superseded (Q2). |
| 188 citations in 70 tracked files (123 `PLAN.md §N`, 61 `TASKS.md M…`, 4 `RESEARCH_FINDINGS.md §N`), and 90 commit subjects like `(M3 E9)` | 440 citations, `P<phase>-<n>` IDs | One citation legend in `AGENTS.md` (Q8). |
| Public remote | private | Nothing new is exposed; all three documents are already public. |

### Lessons the sister migration paid for

1. **It lost 17 open rows.** The retirement scoped "Phases 4–9" and its audit checked findings
   only; the rows were restored after merge (`8d22470`). → Net 2 tiles every line; Net 3a counts
   every open checkbox.
2. **`AGENTS.md`'s legacy Working Order contradicted `CLAUDE.md`** until late in the feature. →
   It is fixed by the first task.
3. **`CLAUDE.md` must import `AGENTS.md` with `@AGENTS.md`** (`6b39437`); a prose "read it first"
   made half the house rules optional.
4. **Roadmap prose restated the ledger head** ("runs to record 027 and finding F215"), costing
   seven `[no-task]` commits that only bumped it. → No counter anywhere in roadmap prose; point at
   `docs/experiments/index.md`.
5. **Name every tracker an ID can live in** (`917cb47`). → `AGENTS.md` says where an M/E label
   lives: a record's `Task:` field, a roadmap row, or neither.
6. **`just verify` is Markdown-blind** (sister pitfall `EXPERIMENT-LEDGER-03`). A malformed ledger
   table passes green.
7. **Lift-and-shift reproduces contradictions** (`EXPERIMENT-LEDGER-07`). A derived number that
   disagrees with its own table is caveated in Provenance, never corrected in a frozen record.
8. **`core.ignorecase=true` holds here too.** Never add `PLAN.md`/`TASKS.md` to `.gitignore`
   unanchored: it shadows every `docs/features/*/plan.md` and `tasks.md`
   (`EXPERIMENT-LEDGER-01`).
9. **The commit guard switches on with the first `docs/features/*/tasks.md`.** The legacy
   `(M3 E9)` subject style fails its regex from that commit on.
10. **"A run is not a task"** (`d1a3e26`). One `TASKS.md` step used to mix code, a server run and
    its analysis; in the new loop the code is a task, the run falls between tasks, and the result
    is a record.

### The experiment ledger

Built with `~/.claude/templates/experiment.md` and `experiments-index.md`, instantiated as
`~/.claude/skills/log-experiment/SKILL.md` §3 says (strip block removed, every placeholder
filled, never `cp`). Each record goes through that skill's draft → STOP → write gate. The
no-run variant already exists in the template (the sister's `EXPERIMENT-LEDGER-02`).

Draft inventory — boundaries settle per Q7:

| Run | Source | What |
| --- | --- | --- |
| 001 | M0.10 (+ M1's re-measure) | E1 reference bracket — 0.1887 raw-thermal floor, 0.9213 visible |
| 002 | M1 | Phase 1 go/no-go — PASS; the adapted arm saturates and cannot decide it |
| 003 | M1.1 | fidelity on M1's runs — no reward hacking, causality still open |
| 004 | M1.2 step 1 | the independent `yolo11s` judge; noise floor 0.059 |
| 005 | M1.2 steps 5–6 | E3 pix2pix at `grad_scale: 1.0e-2` — null, +0.0070 |
| 006 | M1.2 step 7 | the dose measured at 2.3% of the objective — the null is dose-limited |
| 007 | M1.2 step 8 | probes + E3 pix2pix at 0.15 — +0.0512, p = 0.031 |
| 008 | M1.2 step 9 | C2 on step 8's exports — reward hacking ruled out |
| 009 | M2a step 4 | turbo probes at 0.15 and 0.75; 34.14 GB peak |
| 010 | M2a step 5 | E3 turbo, n = 3 — replicates |
| 011 | E8 | annotation sweep + C/D top-up — crossover N ≈ 150 |
| 012 | E9 step 0 | pix2pix wall clock, 10.5 h a run |
| 013 | E9 steps 2–3 | FLIR judge + kill-test — WEAK PASS, +0.2123 |
| 014 | E9 step 3b | de-rolled labels + re-gate — +0.2066 |
| 015 | E9 step 4(b) | throughput probe — 77–103 GPU-h; host RAM binds concurrency |
| 016 | E9 step 5 | twelve-run FLIR cell — loop +0.351 over control, at the raw-thermal floor |
| 017+ | M0.9, M0.10, M1.2 steps 2b/9, E9 steps 3b/4/5 | no-run: `--device 0` string bug; `workers` as hidden experiment identity; absent vs null metrics; label-roll direction; stale ultralytics label cache; list-source OOM in `t2o faithfulness` |

Inherited from the sister's `## Resolved`: one record per stage (their Q1), the no-run shape for
findings from writing code (Q2), the finding status vocabulary, and the two-cell rule for acting
on an earlier finding. Records are lifted verbatim; the Results block is the distilled tables
already in `TASKS.md` (the sister's record `001` precedent), never a console block reconstructed
from memory. Header fields `TASKS.md` does not state — the command, wall clock, W&B group, and
the SHA the run executed at — read `not recorded`, never reconstructed from the code; `git_sha`
may name the commit that recorded the result (`not recorded — results committed in 7297601`),
and `Date` is that commit's date unless the source names the run's own. Record `013`'s step-2b
CSV matches `TASKS.md`'s table except in two places, both Provenance caveats: it gives 4,864 s
stall-free (not 4,881), and its mAP50 peaks at epoch 76 (0.5153), because ultralytics 8.4 picks
`best.pt` on mAP50-95 alone (`ultralytics/utils/metrics.py:1009`). A `best` row is that peak, not
the mAP50 one. Record `014` mints E9 step 3b's stale-label-cache and label-roll-direction findings
itself, their evidence in its Results: both came off server runs (the bit-identical first re-gate,
the 60-pair α sweep), as records 010 and 013 kept their run-found footguns. The 017+ row above
loses those two, so SPEC-MIGRATION-19 carries four. Record `016` mints E9 step 5's two
`campaign_report.py` wall-clock bugs and the `control-s0` 95-epoch explanation by the same rule,
while the list-source OOM stays with SPEC-MIGRATION-19. Its saved `runs/e3f-report-2026-10-01.txt`
predates both fixes (generated at `7a3ca94`), so block 5's spans and stage 0–2 `bound_h` are the
buggy ones: `train_h` stands, and stage 0–2 boundaries re-derive as `bound_h` minus the next
stage's `train_h`, but the stage-3 boundaries, the 9.51–10.44 h spans and ~116.8 GPU-h exist only
in `TASKS.md`, so `Log:` reads `none saved — transcribed from TASKS.md <lines>` for them. The
launch command reads `not recorded`; block 3's flag set is quoted verbatim, never pieced together
into a `t2o loop` line.

SPEC-MIGRATION-19 writes one no-run record per finding — 017 `--device 0`, 018 `workers`, 019
absent vs null (which also judges record 008's consequences 1 and 3), 020 list-source OOM — because
the no-run header holds a single Date, Task and `git_sha`; their dates precede 016's, and numbers
are never reordered. It also replaces F97's `acts on M1.2 step 2b (legacy; …)` cell with 018's
F-ID, and caveats step 2b's 2026-09-27 narrowing as F97's.

SPEC-MIGRATION-20's census counts claims, not paragraphs. One source paragraph can yield two F-IDs
under distinct labels (F43 and F44, both M1.2 step 8 finding 9), one F-ID can hold two source
findings (F12, M1.2 step 1 findings 2 and 3), and a claim the source withdrew before it got an F-ID
lives in the F-ID that explains the withdrawal (`TASKS.md:2285`, inside F31). None of these is an
orphan or a double count. A "narrowed" relation keeps the old row `open` with a link in both
directions (F40 ← F41, F151 ← F97), since the status vocabulary has no `narrowed`.

### `PLAN.md` → `docs/design.md`

Every `§N` heading survives under the same number so `PLAN.md §N` resolves by one legend rule.
Design rules and calibrated values stay verbatim — §8's "calibrate `grad_scale` per backbone",
§9's data contract, §12's two task arms and the exact sign-flip rule. Measurement narrative
becomes a one-line pointer to the record that holds it: the status paragraphs in §8, §10's
"Result" paragraphs, §11's E3 row, §15's status cells and §16's status section. §13 moves into
`AGENTS.md` (loaded every session) and `docs/design.md §13` keeps only its heading and a pointer,
so there is one copy. §1 and §16 point at `docs/goal.md` for the objective and the criteria.

Inside those spots, a sentence a tracked file cites by `PLAN.md §N` stays verbatim: §1's "not a
product" line (`src/t2o/cli.py:7`), §11's E3 design sentences (`tests/test_e3_experiments.py:4`,
`experiments/e3_*.yaml:3`) and §16's `annotation_fraction` paragraph (`src/t2o/data/budget.py:6`).
§1's C1–C4 table also stays, with one line under it: `goal.md` renumbers C2–C4, so legacy C2
(faithfulness) is its C3, C3 (protocol) its C4, and C4 (when translation pays) its C2. Every
record, `src/` and the experiment configs use the legacy numbers.

### The roadmap

All 31 open rows, verbatim, grouped under their legacy section labels (Q9) — `M2a step 2`,
`M2a step 3`, `M2b`, `M3`, `E8`, `E9`, `M4` — and the 13 paper corrections as
`## Paper obligations` (Q10). The six drafting criteria in `docs/goal.md` get a status line each
that cites F-IDs rather than restating numbers (Q12). Row 3899's open decision travels with it whole: whether to
loosen criterion 1 to admit a public dataset, now that E9 says "loop beats control" holds on two
datasets and "translation beats thermal" on one. It carries a note that `docs/goal.md`'s Margin
and Consistency wording may have settled it, for the human to close (Q13).

The corrections list is 12 rows (`TASKS.md:5370-5416`), not 13: the 31 are M2a step 2 (2), step 3
(1), M2b (5), M3 (4), E8 (1), E9 (2), M4 (4) and the 12 obligations. Rows already stale at the tag
keep their text and gain a one-line pointer like 3899's — 3623's "readout is pending" → record 016,
5364's "five acceptance criteria" → the six status lines. A status line grades `goal.md`'s wording,
never a legacy verdict: the reference judge (`yolo11s`) and the in-loop detector (`yolo11n`) are
one family, so no finding yet meets Transfer's or Faithfulness's "never in the loop" clause, and
record 011's "all satisfied" was scored against the legacy five.

### The citation legend (in `AGENTS.md`)

| Citation | Resolves to |
| --- | --- |
| `PLAN.md §N` | `docs/design.md §N`; `§13` → `AGENTS.md` House style |
| `TASKS.md M<x>[ step <n>][ finding <k>]`, commit subjects `(M3 E9)` | the record whose `Task:` names that section — grep the legacy label under `docs/experiments/`; open work → the same label in `docs/roadmap.md`; build narrative → `git show pre-spec-migration:TASKS.md` |
| `RESEARCH_FINDINGS.md §N`, `E1`–`E10`, `C1`–`C4` | unchanged — the file stays (Q11); `E`/`C` are defined in its §7 and §1 |

The `C1`–`C4` row carries the same crosswalk, since `goal.md` and the legacy files now number
C2–C4 differently.

`AGENTS.md`'s other End-state sections restate nothing that already has a home: server runs point
at `~/.claude/guidelines/research.md` §Conventions for the `logs/` rule, non-negotiables at
`docs/design.md §5` Invariants. "Where an M/E label lives" works at three levels: a record's
`Task:` holds the section only (`M1.2`, `M3 E9`), a step is in the index's What ran column, a
finding in its F-heading's `(legacy: …)` label. A label can live in a record and a roadmap section
at once (`M2a`, `E9`), so the table says to grep both.

### Loss prevention

| Net | What | Catches |
| --- | --- | --- |
| 1. Snapshot | The annotated tag `pre-spec-migration` on `e70760c`, pushed to `origin`. | Anything: `git show pre-spec-migration:<file>`. |
| 2. Coverage map | A table appended to this plan giving every line range of `TASKS.md` (1–5419) and `PLAN.md` (1–806) at the tag exactly one disposition — record `NNN`, roadmap, `design.md §N`, `AGENTS.md`, or superseded with a reason. A throwaway awk check (scratchpad, never committed) confirms the ranges tile with no gap and no overlap. | Whole sections nobody migrated — the sister's 17 rows. |
| 3a. Open rows | Each of the 31 `- [ ]` rows found verbatim in `docs/roadmap.md`, or listed as dropped with a reason the human accepts. | Lost forward work. |
| 3b. Numbers | Every decimal with three or more significant figures in `TASKS.md` must appear somewhere under `docs/`; misses go to a triage list the human reviews. | Silent transcription gaps. |
| 3c. Findings | Every finding maps to exactly one F-ID whose heading carries its legacy label; every "superseded", "withdrawn" or "narrowed" relation in the source is a two-ended chain in the ledger. | Orphaned or double-counted findings. |
| 3d. Citations | Every citation shape in tracked files and in `git log` subjects resolves through the legend. | Dangling references. |
| 4. Fresh clone | Clone the branch head into a scratch directory (no `PLAN.md`, no `TASKS.md`) and answer ten fixed questions from the new docs alone, each with a path: E3 pix2pix's headline and p; why `grad_scale` is per backbone; E9 FLIR against its floor; what runs next; sign-flip versus bootstrap; 753 of 853 and the `Connector` class; why ultralytics is pinned `>=8.4.108,<8.5`; the three detector roles; E8's crossover; the open paper obligations. | Present but unfindable. |
| 5. Human review | `/log-experiment`'s draft → STOP → write gate on every record. | Wrong claims and invented numbers. |

Nets 2 and 3 run before the deletion task, and the deletion task is blocked on them.
`git diff pre-spec-migration -- src scripts tests` must show no logic change.

**How SPEC-MIGRATION-25 runs the nets.**

- Net 2 splits a range wherever a carve-out sits inside it. Where two records cite the same
  lines, the record whose `Task:` names the section owns them. The other is listed in an "also
  lifted into" column, and the tile check stays strict.
- In `TASKS.md:20-969` and `1681-1965`, each discovery paragraph gets its own row with its
  reason (Q15).
- Net 3b counts a decimal as `\d+\.\d+` with at least 3 digits once the leading zeros and the
  point are dropped. It matches on number boundaries, never as a substring, against
  `docs/**/*.{md,csv}`. A number found only outside `docs/experiments/`, `design.md` and
  `roadmap.md` goes on the triage list too, marked as such.
- Net 3d lists every shape found, not only `§N` and `M<x>`: `PLAN.md §N` (N must exist in
  `design.md`), `TASKS.md` labels (each classed record, roadmap or neither), bare `(TASKS.md)`
  cites, `TASKS.md:<line>` anchors, and commit subjects up to the tag, parenthesised and bare.
  Counts are taken fresh, not reconciled to the 188 and 90 in this plan's table.
- The results (the map, the 3a result, the 3b triage list, the 3d shape table) go under the map
  in this file. The scripts stay in the scratchpad.

**How SPEC-MIGRATION-26 runs Net 4.** The clone is HEAD with the uncommitted deletion applied
(`git diff HEAD --binary | git -C <clone> apply`), and no answer may come from
`git show pre-spec-migration:`. "What runs next" is answered by `docs/roadmap.md`'s open rows and
`AGENTS.md`'s "a human picks it", not by a ranking. The ten answers, question → path, go under the
Net 3 results as `### Net 4 — fresh clone`. An unanswerable question is fixed in `design.md`,
`roadmap.md` or `AGENTS.md`, never in a record, before the deletion lands.

### Sequencing

1. *Done in 5b37655.* `/project-init` lands `docs/goal.md`, `CLAUDE.md` and `justfile` **before** `tasks.md`:
   `/implement` gates on `just verify`, and this plan's Out of Scope must be re-checked against
   `goal.md`'s Non-Goals once they exist. Its `goal.md` sources are `RESEARCH_FINDINGS.md` §1,
   §10, §12, §13 and `PLAN.md` §2, §3, §9. The human writes the Non-Goals.
2. `/feature-plan spec-migration` continues this feature: Q5–Q14 move to `## Resolved`, then
   `tasks.md`, `decisions.md` and `pitfalls.md` are written. That commit switches the guard on.
3. Task order: fix `AGENTS.md`'s Working Order → ledger scaffold → records → finding census →
   `docs/design.md` → `docs/roadmap.md` → `AGENTS.md` rewrite → `README.md` links →
   coverage map and censuses → delete `PLAN.md` and `TASKS.md`. (The `CLAUDE.md` import
   landed in 5b37655; memory is a post-merge chore, Q14.)
4. Research is paused until the PR merges (Q3), so main stays at `e70760c` and nothing has to be
   re-migrated.

### Memory housekeeping (post-merge, no task row — Q14)

Five memories sit orphaned at the project's former path,
`~/.claude/projects/-Users-ai--GoogleDrive-Python-Thermal-To-Optical-Translation/memory/`.
Port them to the current project memory. Rewrite `biweekly-progress-updates` to read
`docs/experiments/` and `docs/roadmap.md` instead of the three root documents. Drop
`dataset-rehost-deferred` — M0.9 closed 2026-09-19 with LLVIP and M3FD adapted on the server.

## Open Questions

None.

## Resolved

### Q1: What happens to `PLAN.md` and `TASKS.md` once everything is migrated?

- [x] Tag `pre-spec-migration` first, then delete both in the last task, after Nets 2–3 pass (recommended) — every byte stays recoverable with `git show`, and the root stops signalling the legacy workflow to agents
- [ ] Move both to `docs/archive/` with an archived header — greppable on a fresh clone without git commands, but ~430 KB of stale text agents can mistake for current state
- [ ] Keep both at the root, stamped archived — the sister's approach, forced on it by untracked files; here it contradicts the global rule that a root `PLAN.md`/`TASKS.md` means the legacy workflow

### Q2: Where does `PLAN.md`'s live design go?

- [x] `docs/design.md`, keeping the `§1`–`§16` numbering, results narrative replaced by pointers, §13 moved into `AGENTS.md` (recommended) — one move, and every `PLAN.md §N` citation resolves by a single legend rule
- [ ] Split like the sister — §6/§7 to `docs/reference-implementations.md`, §13 to `AGENTS.md`, the rest to `docs/design.md`; more files, and the legend needs a row per section
- [ ] Keep `PLAN.md` in place and strip only the results — least churn, but leaves a root `PLAN.md`

### Q3: How does research proceed during the migration?

- [x] Full pause until the PR merges (recommended by the human) — main stays at `e70760c`, so the branch never re-migrates new material
- [ ] Pause, but allow server runs whose logs are saved to `logs/` and recorded after the ledger exists — keeps the GPUs busy, at the cost of records written against a half-built ledger
- [ ] Continue on main in parallel — every new `TASKS.md` edit would have to be re-migrated and re-audited

### Q4: One feature or two?

- [x] One feature, `spec-migration` (recommended) — the whole scope is known up front, so one plan holds every decision, one coverage map and one audit
- [ ] Two features like the sister, `experiment-ledger` then `plan-tasks-retirement` — shorter task tables and a proven precedent, but the sister split only because it found the second half later, and the checks would span two plans

### Q5: What form do the new finding numbers take?

- [x] Template form `F01`, `F02`, … allocated chronologically, each heading carrying its legacy label, e.g. `### F23 — … (legacy: M1.2 step 8 finding 7)` (recommended) — no F-number is cited anywhere yet, so there is nothing to preserve and no reason to deviate from the template; the label is the crosswalk, found by grep
- [ ] Unpadded `F1`, `F2`, … as the sister kept — matches the sister's ledger, but the sister only kept it because 98 numbers were already cited, which is not the case here
- [ ] Numbers derived from the legacy IDs (e.g. `F-M1.2-8-7`) — self-describing, but breaks the ledger's single counter and its sort order

### Q6: What goes in a run record's `Log:` field when no log was saved?

- [x] `none saved — transcribed from TASKS.md <lines> @ pre-spec-migration`, or the `runs/*.txt` path where one exists (recommended) — honest about provenance, and the anchor is resolvable from any clone
- [ ] Delete the field, as a no-run record does — tidier, but it hides that a run did happen and leaves no pointer to the source text
- [ ] Use the no-run record shape for every migrated record — no template deviation at all, but it strips wall clock, machine and W&B group from records that have them

### Q7: What is the unit of a record?

- [x] One per campaign or stage, splicing its steps, as in the inventory above (recommended) — the sister's Q1; a campaign is the unit that has a question, a design and findings, and it keeps chains like step 6 → step 7 → step 8 inside few records
- [ ] One per `TASKS.md` step — strict "one run, one file", but mints records for build steps that ran nothing and splits findings that were reasoned about together

### Q8: How are the 188 code and config citations handled?

- [x] The legend only; no sweep (recommended) — the sister kept 209 such citations by design; § numbers are preserved, so every `PLAN.md §N` still resolves, and comment churn across 70 files buys nothing
- [ ] Repoint every comment to its new path — no legend needed, but touches 70 files and still cannot reach the 90 commit subjects

### Q9: How are roadmap rows identified?

- [x] By their legacy section label, verbatim and never renumbered (`M2b`, `E4`, `M4`, `M2a step 2`, …) (recommended) — commit subjects and code comments already cite those labels
- [ ] New stable IDs (e.g. `R-01`) — uniform and short, but a third naming scheme that cites nothing that already exists

### Q10: Where do the 13 corrections owed to the paper go?

- [x] `docs/roadmap.md`, under `## Paper obligations` (recommended) — one place for everything still owed, beside the drafting criteria they serve
- [ ] A new `docs/paper-notes.md` — room to grow into a methods-section draft, at the cost of a fifth document to keep current

### Q11: What happens to `RESEARCH_FINDINGS.md`?

- [x] It stays at the root, frozen, and `docs/goal.md` takes over its north-star role (recommended) — it is the proposal of record, defines `E1`–`E10` and `C1`–`C4`, and its four code citations stay valid
- [ ] Move it to `docs/research-proposal.md` — a clearer name next to `docs/experiments/`, but it breaks four citations and the README link for a rename

### Q12: Which drafting criteria do `docs/design.md §16` and the roadmap's status lines cite?

`docs/goal.md` (5b37655) now has **six** criteria with new thresholds (Margin ≥ +2 mAP@50 on the
held-out power-line test split; Consistency on ≥3 of 5 datasets against the no-loop control).
`PLAN.md §16` and `RESEARCH_FINDINGS.md §10` still say five. This plan's End state table and
"The roadmap" section say "the five drafting criteria".

- [x] `goal.md`'s six; `§16` becomes a pointer to `goal.md`, and each roadmap status line cites F-IDs against the six (recommended) — `goal.md` is the north star, this plan already says `§16` points at it, and grading progress against retired criteria would mislead whoever reads the roadmap next
- [ ] The legacy five from `RESEARCH_FINDINGS.md §10` — matches the migrated record word for word, but scores progress against criteria the project no longer uses
- [ ] Both, side by side — a full crosswalk, at the cost of two scorecards to keep current

### Q13: What happens to the deferred criterion-1 decision at `TASKS.md:3899`?

The row holds back an edit to criterion 1 "until the public-dataset cell runs", on the reasoning
that the custom set may be a bad case for the method. `goal.md` may already have answered it:
Margin stays on the power-line split, public datasets count through Consistency, and "translation
beats raw thermal" is no longer a criterion.

- [x] Carry it verbatim into the roadmap, annotated "possibly settled by `docs/goal.md` Margin/Consistency — human to close" (recommended) — Net 3a wants every open row verbatim, and closing a research decision is not the migration's call
- [ ] Drop it as superseded by `goal.md`, listed in Net 3a's dropped list with that reason — a cleaner roadmap, but the migration silently makes a research decision
- [ ] Carry it verbatim with no annotation — faithful, but leaves a settled decision looking live

### Q14: The memory-housekeeping step writes under `~/.claude/`, which Out of Scope excludes. Which wins?

- [x] Move it out of the feature: a post-merge chore with no task row (recommended) — it changes nothing in the repo, so a task row could never end in a commit, and the rewritten `biweekly-progress-updates` memory needs `docs/experiments/` and `docs/roadmap.md` to exist first anyway
- [ ] Keep it as the last task and narrow Out of Scope to "templates, skills, guidelines" — keeps it tracked, but a row whose only output is outside git breaks the one-commit-per-task rhythm
- [ ] Drop it entirely — the five memories stay orphaned at the old path, where no session will ever load them

### Q15: Which paragraphs of `TASKS.md:20-969` and `1681-1965` does SPEC-MIGRATION-19 mint as findings?

- [x] Only step 2b's `workers` (1751–1811); every other discovery paragraph in both ranges goes into Net 2's map as "superseded by the code and its docstrings", each with its reason (recommended) — M0.1–M0.9 labels nothing a finding, and its bold paragraphs are Decisions, which the no-run bar excludes as preferences
- [ ] Also mint the short list that arguably clears the bar: pyright's wheel-shipped `tests` package shadowing `tests/` (446), InsPLAD's 18 categories against the paper's 17 (740), the ultralytics 8.4.117 API checks including `unwrap_model`'s rename at ~8.4.112 (180), and the §7 dependency maze not materialising (57) — keeps these facts findable once `TASKS.md` is deleted, at the cost of four more draft gates for records no experiment cites
- [ ] Every bold-led discovery or decision paragraph — nothing has to be judged, but it pads the ledger with design choices

### Q16: How does the finding census crosswalk F01–F14, whose headings carry no `(legacy: …)` label?

Records 001–004 landed without the label Q5 asks for (pitfall SPEC-MIGRATION-07), and Out of
Scope forbids editing a record after it lands. Only three of the 14 map to a numbered source
finding (F10, F12, F13 ← M1.2 step 1 findings 1, 2–3, 4); the other 11 would read
"`<section>`, unnumbered", repeating their record's `Task:` field. No tracked file and no commit
subject cites any of these labels.

- [x] Leave records 001–004 untouched; the census crosswalks F01–F14 by hand through each record's `Task:` field and `Log:` line anchors (recommended) — the never-edited rule holds, and the labels it gives up are cited nowhere; new work cites F-IDs
- [ ] Append `(legacy: …)` to the 14 headings, label only, with a one-line carve-out in Out of Scope — meets Q5 and this task's wording exactly, but edits four landed records for labels nothing cites
- [ ] Leave the records untouched and append a 14-row F-ID → legacy-label table to this plan, beside Net 2's coverage map — durable, but a crosswalk that serves no existing citation

### Q17: §13 says `basicConfig` only in `cli.py`; four `scripts/*.py` own theirs by design. What lands in `AGENTS.md`?

`scripts/fetch_datasets.py:176`, `mirror_thermal_labels.py:72`, `adapt_datasets.py:73` and
`gate_table.py:267` each call `logging.basicConfig`, and each docstring says why. Answered by the
human on 2026-10-04.

- [ ] §13 verbatim, plus one *Migration note* under it naming the four scripts and that "invariant N" means `docs/design.md §5` (recommended) — the house rule keeps its wording, and the note stops an agent from "fixing" a script
- [x] Amend the bullet to "`basicConfig` only in `cli.py` and in each standalone `scripts/*.py`", as the sister did — matches the code, but rewrites a house rule inside a migration that promises no content change
- [ ] §13 verbatim, no note — fully faithful, but every session loads a rule four files break on purpose

## Net 2 and Net 3 results

What SPEC-MIGRATION-25 produced, against the tag `pre-spec-migration` (`e70760c`). Line numbers are
at the tag. The scripts that built and checked these tables were throwaway and stayed in the
scratchpad; the rules they ran are in Tech Stack / Approach, *How SPEC-MIGRATION-25 runs the
nets*. Net 3c is SPEC-MIGRATION-20's. The censuses below skip `docs/features/`: those files are
agent state, and this section would otherwise cite itself.

### Net 2 — the coverage map

Every line of both files has one disposition, and the two tables tile `TASKS.md` 1–5419 and
`PLAN.md` 1–806 with no gap and no overlap (checked by parsing this section back out of the file).

- `record NNN` — lifted into that record. Only one range is shared: 1080-1105 is record 002's
  (section M1) and is also lifted into 001.
- `roadmap`, `design.md §N`, `AGENTS.md`, `CLAUDE.md`, `goal.md` — carried there.
- `superseded` — nothing lifted, with a code for why:
  - **(C)** done build checklist; the code and its tests are the record.
  - **(D)** build-time decision or note, realised in the module it names — a preference, not a
    finding (Q15). Where the module is not named in the paragraph, the last column says.
  - **(V)** check-run transcript; `just verify` supersedes it.
  - **(X)** the last column gives the specific reason: a closed item, or a fact whose home is a code
    comment or another document.
- `T1`–`T5` under *Also lifted into* mark a paragraph whose forward-looking content lives only in
  the tag. See *Triage for the human*.
- A record's `Log:` can skip lines inside its own section. M1's vendor and wrapper decisions
  (1139-1229) and two `Verify` lines (1226-1229, 1360-1365) are `superseded`, not record 002's.

#### `TASKS.md` — 5,419 lines at `pre-spec-migration`

| Lines | Disposition | Also lifted into | What it is / why |
| --- | --- | --- | --- |
| 1-8 | AGENTS.md; CLAUDE.md |  | Title and the legacy working order — superseded by AGENTS.md `# Working Order` and CLAUDE.md `## Planning` |
| 9-12 | CLAUDE.md |  | Every item verifiable on the synthetic fixture, on CPU, in seconds — CLAUDE.md `## Things that will trip you up`, goal.md Constraints |
| 13-19 | CLAUDE.md | design.md §9 | `dataset/` is never tracked; tests build pairs with `tmp_path_factory` — CLAUDE.md `## Things that will trip you up` |
| 20-56 | superseded (X) | goal.md | M0.1 — Repo hygiene and packaging ✅ — Done checklist; the code and `pyproject.toml` are the record, and the ultralytics pin (line 43) is goal.md Constraints |
| 57-63 | design.md §7 |  | Resolved: the PLAN.md §7 dependency maze did not materialise. One `uv lock` reso… — `PLAN.md` §7's thesis, confirmed: the dependency maze is the frameworks', not the models' (`design.md:233`); versions are in `uv.lock` |
| 64-68 | superseded (X) |  | Note: `transformers` resolved to 5.x, well past the `>=4.45` floor and two major… — Closed: `transformers 5.x` against the vendored turbo wrapper was verified in M2a step 1 (`TASKS.md:2866`) and the turbo runs, records 009–010, exercised it |
| 69-72 | superseded (X) |  | Deviation: `[tool.ruff] extend-exclude = ["*.md"]`. ruff formats fenced Python i… — Kept as `extend-exclude = ["*.md", "third_party"]` at `pyproject.toml:116`; CLAUDE.md names it |
| 73-75 | superseded (X) |  | Not verifiable locally: `uv sync --extra gpu`. `uv lock` resolves the CUDA path,… — Closed: `uv sync --extra gpu` was the first M0.10 check (`TASKS.md:972-983`) |
| 76-94 | superseded (C) |  | M0.2 — Config layer ✅ |
| 95-98 | superseded (D) |  | Sections: `data` (manifest + the E8 `annotation_fraction`/`annotation_seed` hook… |
| 99-106 | AGENTS.md |  | Decision — `config_hash()` excludes `runtime` wholly. Device, run name, run dir … — House style: `config_hash()` covers the experiment, not the invocation |
| 107-110 | superseded (D) |  | Decision — `detector` splits into `in_loop` and `evaluation` sub-sections. PLAN.… |
| 111-117 | superseded (D) |  | Decision — `translator` is a pydantic discriminated union on `backbone`, current… |
| 118-121 | superseded (X) | T2 | Deferred: experiment inheritance (a `base:` include key). `experiments/smoke.yam… — Deferred idea (a `base:` include key) — forward-looking, no home outside the tag; triage T2 |
| 122-150 | superseded (C) |  | M0.3 — Data layer ✅ |
| 151-157 | superseded (D) |  | Decision — `annotation_fraction` zeroes labels, not images. Every image still tr… |
| 158-162 | superseded (D) |  | Deviation from Clean-SeAFusion's own test fixture: its `tests/conftest.py` reuse… |
| 163-165 | superseded (V) |  | Verify: `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"… |
| 166-179 | superseded (C) |  | M0.4 — Detection layer ✅ |
| 180-186 | superseded (X) |  | Verified against the installed `ultralytics 8.4.117`, not assumed: `unwrap_model… — ultralytics 8.4.117 API checks; the `unwrap_model` rename is a code comment at `src/t2o/detection/frozen.py:29`, the pin is goal.md Constraints |
| 187-193 | superseded (D) |  | Decision — `train_detector` reads `config.detector.evaluation.*`, not flat |
| 194-200 | superseded (D) |  | Decision — `detection/evaluation.py` is new, not a port. Clean-SeAFusion has no |
| 201-205 | superseded (X) |  | Not yet exercised: an actual end-to-end `train_detector(...)` call against ultra… — Closed: the real `train_detector` end-to-end ran in M0.8, passing (`TASKS.md:505-507`) |
| 206-208 | superseded (V) |  | Verify: `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"… |
| 209-260 | superseded (C) |  | M0.5 — Metrics (all from scratch — none exist in any of our repos) |
| 261-269 | superseded (X) |  | Decision — Hallucination Index deferred. The MICCAI 2024 metric it's adapted fro… — Hallucination Index deferred: kept in the module docstring at `src/t2o/metrics/faithfulness.py:13` and `design.md` §12 |
| 270-275 | superseded (D) |  | Decision — `MetricsConfig` added to `config/schema.py`. `lpips_net` and |
| 276-284 | superseded (D) |  | Decision — detection-consistency's formula. RESEARCH_FINDINGS.md §9 names the me… |
| 285-291 | superseded (D) |  | Decision — matching is greedy, confidence-ordered, single-IoU-threshold. Same pr… |
| 292-306 | superseded (C) |  | M0.6 — Translator interface and CPU stand-in ✅ |
| 307-312 | superseded (D) |  | Decision — round trip scoped to data → translate → fidelity-eval, not the full |
| 313-318 | superseded (D) |  | Decision — each translator owns its own optimizer(s) internally; `fit()` is a co… |
| 319-323 | superseded (X) |  | Deferred, not decided — how the M0.7 detection-consistency term plugs into `fit(… — Closed in M0.8: `engine/loop.py`'s prerequisites decided how the coupling term plugs into `fit()` (`TASKS.md:459-472`) |
| 324-342 | superseded (C) |  | M0.7 — Coupling ✅ |
| 343-351 | superseded (D) |  | Decision — reward-target hinge is ReFL's `relu(total - target)`, not AlignProp's |
| 352-356 | superseded (D) |  | Decision — `grad_scale`/`reward_target` are `DetectionTaskLoss` constructor args… |
| 357-363 | superseded (D) |  | Decision — `schedule.py` is stateless functions (`weight_for_stage`, |
| 364-366 | superseded (V) |  | Verify: `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"… |
| 367-393 | superseded (C) |  | M0.8 — Engine and tracking |
| 394-402 | superseded (D) |  | Decision — `RunTracker` drops Clean-SeAFusion's `DistributedContext` parameter a… |
| 403-405 | superseded (V) |  | Verify (tracking.py only): `ruff format`, `ruff check`, `pyright` (0 errors), |
| 406-429 | superseded (D) |  | Decision — `engine/trainer.py` does not own an optimizer, scheduler, GradScaler,… |
| 430-432 | superseded (V) |  | Verify (trainer.py only): `ruff format`, `ruff check`, `pyright` (0 errors), |
| 433-438 | superseded (D) |  | Decision — `export.py` built before `loop.py`, reversing this checklist's origin… |
| 439-445 | superseded (D) |  | `engine/export.py`: ported from |
| 446-455 | superseded (X) |  | Found while writing `tests/test_export.py`: a `tests` package shipped inside a — pyright's wheel-shipped `tests` package: kept as code comments at `tests/test_export.py:6` and `tests/test_splits.py:23` (Q15) |
| 456-458 | superseded (V) |  | Verify (export.py only): `ruff format`, `ruff check`, `pyright` (0 errors), |
| 459-472 | superseded (D) |  | `engine/loop.py` needed two prerequisites `loop.py` itself doesn't touch again o… |
| 473-496 | superseded (D) |  | Decision — the in-loop coupling detector stays fixed at `config.detector.in_loop… |
| 497-500 | superseded (X) |  | Out of scope for this step, left for the later "Tests" bullet below: stage-level — Closed: stage-level resume and the same-seed-twice check landed in M0.8's closing Tests bullet (`TASKS.md:569-628`) |
| 501-507 | superseded (V) |  | Verify (loop.py only): `ruff format`, `ruff check`, `pyright` (0 errors), |
| 508-511 | superseded (D) |  | `cli.py`: four subcommands — `train` (one stage of `Trainer`), `loop` (`run_loop… |
| 512-516 | superseded (D) |  | `build_translator(config: Config) -> nn.Module`, added to |
| 517-524 | superseded (D) |  | Arbitrary-depth override merging. Clean-SeAFusion's `overrides_from_args` only h… |
| 525-532 | superseded (D) |  | Decision — `evaluate` subcommand added now, not deferred to M0.10. Not in TASKS.… |
| 533-537 | superseded (D) |  | Decision — heavy submodules (torch, ultralytics) are imported inside each subcom… |
| 538-542 | superseded (V) |  | Verify (cli.py only): `ruff format`, `ruff check`, `pyright` (0 errors), |
| 543-548 | superseded (D) |  | `experiments/smoke.yaml`: every section present explicitly (no field left to an … |
| 549-557 | superseded (D) |  | Decision — `data.manifest`/`detector.*.weights` stay at schema defaults |
| 558-564 | superseded (D) |  | `tests/test_experiments.py`: three fast tests — loads and is tiny; `config_hash(… |
| 565-568 | superseded (V) |  | Verify (experiments/smoke.yaml only): `ruff format`, `ruff check`, `pyright` (0 … |
| 569-575 | superseded (D) |  | M0.8's closing "Tests" bullet required real production code, not just tests. |
| 576-586 | superseded (D) |  | `run_loop(..., resume: bool = False)` (`engine/loop.py`). On `resume=True`, |
| 587-594 | superseded (D) |  | Warm-start continuity across a resume needed one non-obvious fix. A resumed proc… |
| 595-601 | superseded (D) |  | `translators.build_translator` now seeds the global torch RNG from `config.train… |
| 602-605 | superseded (D) |  | `cli.py`'s `loop` subcommand gained `--resume`, wired straight through to |
| 606-618 | superseded (D) |  | `tests/test_loop.py`: four new tests. `test_resume_on_a_fresh_run_dir_behaves_li… |
| 619-626 | superseded (D) |  | Deviation from the checklist's literal "(`slow` marker)": only the detector-fine… |
| 627-628 | superseded (V) |  | Verify (M0.8 Tests bullet): `ruff format`, `ruff check`, `pyright` (0 errors), |
| 629-630 | superseded (D) |  | M0.8 is now fully closed. M0.9 (dataset acquisition) is next. |
| 631-635 | superseded (C) |  | M0.9 — Dataset acquisition ✅ |
| 636-658 | superseded (C) |  | Fetch LLVIP, M3FD, TTPLA via `gdown`. The planned "fetch once on the Mac, re-hos… |
| 659-726 | superseded (C) |  | Adapters normalising each into the internal representation |
| 727-732 | design.md §9 |  | Verify: MSRS `detection/` folder — does it have box annotations usable for mAP? — MSRS `detection/` has 80 YOLO-labelled pairs, merged into `train` — `design.md` §9 Public datasets |
| 733-757 | superseded (X) | T1 | Verify: InsPLAD annotation format (not stated in its README; Mendeley Data gates… — InsPLAD verified: 18 categories not the paper's 17, counts, Mendeley's open S3 link — `design.md` §9 keeps only the format; the rest only in the tag, triage T1 |
| 758-764 | superseded (C) |  | Freeze and hash the splits; commit the manifest. `data/splits.py` (`freeze_split… |
| 765-775 | design.md §9 |  | The custom paired dataset is frozen on the server (`yolo_rgbt_29_jul`), run duri… — The custom set frozen on the server as `yolo_rgbt_29_jul`, and why that record cannot reach git — `design.md` §9 The custom dataset |
| 776-780 | goal.md | design.md §9 | The ~850 pairs are 753 train+val plus ~100 held out as an unseen test set (confi… — 753 train+val of ~850 (853); the held-out ~100 — goal.md Constraints, `design.md` §9 |
| 781-793 | design.md §9 | T5 | The held-out set is isolated structurally, not by discipline. `DatasetManifest` … — The held-out set is isolated structurally (`DatasetManifest`); the test split's membership being unpinned until first use is only in the tag, triage T5 |
| 794-795 | superseded (D) |  | `data/splits.py` decisions: |
| 796-800 | superseded (D) |  | The frozen record, not the images, is what reaches git. `dataset/` is wholly |
| 801-809 | superseded (D) |  | Stems are listed in full, not just hashed. A bare hash mismatch says "something |
| 810-816 | superseded (X) | T3 | `verify_split`/`SplitDriftError` exist but nothing calls them yet. Wiring drift — `verify_split` is not called at run start — only by `scripts/freeze_splits.py`; the deferred wiring is only in the tag, triage T3 |
| 817-820 | superseded (D) |  | `scripts/freeze_splits.py` discovers datasets by globbing `<data-root>/*/data.ya… |
| 821-826 | superseded (D) |  | Re-freezing an existing record overwrites rather than refuses, but logs a warnin… |
| 827-834 | superseded (V) |  | Verify: `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"… |
| 835-836 | superseded (D) |  | `scripts/fetch_datasets.py` decisions: |
| 837-839 | superseded (D) |  | Destination `dataset/raw/<name>/` — stays inside the already-wholesale-ignored |
| 840-843 | superseded (D) |  | Two fetch strategies, chosen per source: `git clone --depth 1` for MSRS/CPLID/ |
| 844-845 | superseded (D) |  | Idempotent — a destination that already exists and is non-empty is skipped with … |
| 846-851 | superseded (X) |  | Not executed for real in the session that wrote it — confirmed with the user at … — Closed: the user has since run `fetch_datasets.py` for real (same bullet) |
| 852-856 | superseded (D) |  | `scripts/` is a new top-level package, sibling to `src/t2o`, not part of it — ad… |
| 857-859 | superseded (V) |  | Verify: `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"… |
| 860-861 | superseded (D) |  | `data/adapters/` — MSRS (M0.9 adapters step 1 of 2) decisions: |
| 862-866 | superseded (D) |  | `data/adapters/common.py` is the one place the paired `rgbt:`-block `data.yaml` … |
| 867-869 | superseded (D) |  | Images are copied verbatim (`shutil.copyfile`), never re-encoded. MSRS ships PNG… |
| 870-874 | superseded (D) |  | `detection/`'s 80 labelled pairs merge into `train`, not a third split. Verified |
| 875-879 | superseded (D) |  | Collision check is defensive, not decorative. Today's MSRS release has zero over… |
| 880-884 | superseded (D) |  | CPLID and HIT-UAV get no adapter. Both are single-modality (verified against the… |
| 885-889 | superseded (V) |  | Verify (MSRS adapter only): `ruff format`, `ruff check`, `pyright` (0 errors), |
| 890-891 | superseded (D) |  | `data/adapters/flir.py` — FLIR-aligned (M0.9 adapters step 2 of 2) decisions: |
| 892-898 | superseded (D) |  | Reads straight out of `aligned.zip` via `zipfile`, never extracts to disk. The r… |
| 899-904 | superseded (D) |  | Scoped to the ~5142 annotated pairs, not all 10284. Each annotation XML's own |
| 905-909 | superseded (D) |  | Filenames are normalised on copy. `FLIR_00002_RGB.jpg` and `FLIR_00002_PreviewDa… |
| 910-913 | superseded (D) |  | Class vocabulary is derived from the data, sorted alphabetically (`bicycle`, `ca… |
| 914-919 | superseded (D) |  | Boxes are clamped to the frame, and dropped (not written) if zero-area after cla… |
| 920-924 | superseded (D) |  | The real-data sanity test is `slow`, unlike MSRS's. MSRS's real-clone test copie… |
| 925-929 | superseded (V) |  | Verify (FLIR adapter): `ruff format`, `ruff check`, `pyright` (0 errors), |
| 930-933 | superseded (X) |  | M0.9's adapters item is now closed. Remaining M0.9 work: LLVIP/M3FD/TTPLA (`gdow… — Status note, closed by the M0.9 closing paragraph (963-969) |
| 934-935 | superseded (D) |  | `fetch_datasets.py` — gdown sources decisions: |
| 936-940 | superseded (D) |  | One script, one registry, not a second file. LLVIP/M3FD/TTPLA are still "fetch a |
| 941-944 | superseded (D) |  | Drive ids are copied from each dataset's own README, not guessed. A wrong file i… |
| 945-950 | superseded (D) |  | `gdown` lives in a new `scripts` uv dependency group, not `dev`. It's a fetch-ti… |
| 951-954 | superseded (X) |  | The real download was not run. Confirmed with the user given ~31GB free disk on … — Closed: M0.9 closed 2026-09-19 with LLVIP and M3FD fetched and adapted on the server (`TASKS.md:963-969`) |
| 955-957 | superseded (X) |  | Re-hosting is intentionally left undecided, per the user — a reminder to revisit… — Closed: the re-hosting reminder lapsed when M0.9 closed (`TASKS.md:963-969`); plan Q14 drops the `dataset-rehost-deferred` memory |
| 958-962 | superseded (V) |  | Verify (gdown extension): `ruff format`, `ruff check`, `pyright` (0 errors), |
| 963-969 | goal.md | T4 | M0.9 is now fully closed (2026-09-19). Four paired public datasets are adapted a… — M0.9 closed: the four public datasets and their split sizes — goal.md Constraints; DroneVehicle and InsPLAD 'until E9 needs it' only in the tag, triage T4 |
| 970-989 | superseded (C) |  | M0.10 — Server bring-up (cannot be verified locally) ✅ |
| 990-1019 | record 001 |  | M0.10 E1 reference bracket and its corrected table |
| 1020-1021 | superseded (C) |  | Confirm W&B self-hosted logging works end to end. Confirmed on the server. |
| 1022-1041 | record 017 |  | `--device 0` string bug, lifted verbatim |
| 1042-1047 | superseded (V) |  | Verify: `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"… |
| 1048-1079 | record 002 |  | M1 Phase 1 checklist and the gate bar |
| 1080-1105 | record 002 | record 001 | GATE DECISION: PASS and the corrected raw-thermal floor 0.1887 — 001 lifts the re-measure |
| 1106-1138 | record 002 |  | Gate evidence: monotone λ_det, why it is not yet reportable, the metric change |
| 1139-1141 | superseded (D) |  | Vendor + wrapper decisions: → `translators/pix2pix.py`, `third_party/pix2pix/` |
| 1142-1146 | superseded (D) |  | Pin resolved to `2a7afba2895d52556dd5dfe07e8555ef657ced6f` (2025-08-06) — the lo… → `translators/pix2pix.py`, `third_party/pix2pix/` |
| 1147-1158 | superseded (D) |  | Vendored to `third_party/pix2pix/networks.py` + `LICENSE`, byte-identical, plus … → `translators/pix2pix.py`, `third_party/pix2pix/` |
| 1159-1166 | superseded (D) |  | Packaging fix required for the vendor path to import at runtime at all: → `translators/pix2pix.py`, `third_party/pix2pix/` |
| 1167-1173 | superseded (D) |  | `init_weights`, not `init_net`. At this pinned commit, `define_G`/`define_D` acc… → `translators/pix2pix.py`, `third_party/pix2pix/` |
| 1174-1177 | superseded (D) |  | Generator defaults to `resnet_9blocks`, not the paper's `unet_256`. `UnetGenerat… → `translators/pix2pix.py`, `third_party/pix2pix/` |
| 1178-1186 | superseded (D) |  | Reconstruction/adversarial losses reuse the shared `LossConfig.{l2,lpips,gan}`, … → `translators/pix2pix.py`, `third_party/pix2pix/` |
| 1187-1197 | superseded (D) |  | New config surface (`Pix2PixTranslatorConfig`): `net_g`, `net_d`, `ngf`, `ndf`, → `translators/pix2pix.py`, `third_party/pix2pix/` |
| 1198-1208 | superseded (D) |  | One consistent `[0,1]` convention throughout `fit()`, not a `[-1,1]` detour. → `translators/pix2pix.py`, `third_party/pix2pix/` |
| 1209-1217 | superseded (D) |  | Tests split fast/slow exactly like `FidelityEvaluator`'s own precedent (M0.5): f… → `translators/pix2pix.py`, `third_party/pix2pix/` |
| 1218-1225 | superseded (D) |  | Also found and fixed while running the full suite: two *existing* tests assumed → `translators/pix2pix.py`, `third_party/pix2pix/` |
| 1226-1229 | superseded (V) |  | Verify: `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"… |
| 1230-1359 | record 002 |  | pix2pix configs, the saturated adapted arm, the `detector.reference` zero-shot arm and its server commands |
| 1360-1365 | superseded (V) |  | Verify: `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"… |
| 1366-1510 | record 003 |  | M1.1 fidelity on M1's runs |
| 1511-1680 | record 004 |  | M1.2 design and step 1, the `yolo11s` judge |
| 1681-1691 | superseded (C) |  | Step 2 — make "differ only by seed" true rather than merely likely |
| 1692-1696 | superseded (D) |  | Deliberately not setting `torch.use_deterministic_algorithms(True)`: cuDNN → `engine/loop.py`, `seeding.py` |
| 1697-1720 | superseded (D) |  | Decision — `run_loop` reseeds once *per stage* from `train.seed + stage`, not on… → `engine/loop.py`, `seeding.py` |
| 1721-1729 | superseded (D) |  | Decision — `worker_init_fn` on the train loader only. Val (`engine/trainer.py`) … → `engine/loop.py`, `seeding.py` |
| 1730-1735 | superseded (D) |  | `seed_worker` must stay a module-level function — asserted directly by → `engine/loop.py`, `seeding.py` |
| 1736-1742 | superseded (D) |  | Verified, not assumed: `torch.manual_seed` already forwards to `torch.cuda.manua… → `engine/loop.py`, `seeding.py` |
| 1743-1746 | superseded (V) |  | Verify: `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"… |
| 1747-1750 | superseded (X) |  | No server run needed for this step. But it *changes the RNG streams*, so it must… — Ordering note: step 2 changes the RNG streams, so it had to land before the E3 campaign — a constraint on a campaign since run (records 005–007) |
| 1751-1812 | record 018 |  | Step 2b: `workers` was silently experiment identity (lifted verbatim, 1753-1811) |
| 1813-1823 | superseded (C) |  | Step 3 — the two E3 configs ✅ |
| 1824-1838 | superseded (D) |  | Decision — `detector.reference.weights` is the one concrete path in these files,… → `experiments/e3_*.yaml`, `t2o aggregate` |
| 1839-1850 | superseded (D) |  | Decision — W&B tags are *derived* from the resolved config (`tracking.py::run_ta… → `experiments/e3_*.yaml`, `t2o aggregate` |
| 1851-1856 | superseded (D) |  | Decision — `runtime` is compared field-by-field in the integrity test, not whole… → `experiments/e3_*.yaml`, `t2o aggregate` |
| 1857-1860 | superseded (D) |  | Both arms carry `runtime.workers: 16`, unlike `experiments/pix2pix_*.yaml`'s doc… → `experiments/e3_*.yaml`, `t2o aggregate` |
| 1861-1865 | superseded (D) |  | Test teeth confirmed, not assumed: `test_control_and_loop_configs_differ_only_by… → `experiments/e3_*.yaml`, `t2o aggregate` |
| 1866-1870 | superseded (V) |  | Verify: `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"… |
| 1871-1872 | superseded (D) |  | No server run for this step. Step 4's aggregator stands between here and the cam… → `experiments/e3_*.yaml`, `t2o aggregate` |
| 1873-1887 | superseded (C) |  | Step 4 — aggregation and statistics ✅ |
| 1888-1895 | superseded (D) |  | Decision — the config snapshot is parsed as plain YAML, never through `Config.lo… → `experiments/e3_*.yaml`, `t2o aggregate` |
| 1896-1903 | superseded (D) |  | Decision — the arm is derived, not declared, and derived from `config.yaml` rath… → `experiments/e3_*.yaml`, `t2o aggregate` |
| 1904-1909 | superseded (D) |  | Decision — every stage common to all runs is computed, so stage 0's null control… → `experiments/e3_*.yaml`, `t2o aggregate` |
| 1910-1913 | superseded (D) |  | Decision — `metric_value` raises on a null arm rather than returning a sentinel.… → `experiments/e3_*.yaml`, `t2o aggregate` |
| 1914-1921 | superseded (D) |  | Decision — `sign_flip_p_value` refuses above n = 20. 2²⁰ ≈ 1M assignments is abo… → `experiments/e3_*.yaml`, `t2o aggregate` |
| 1922-1926 | superseded (D) |  | Decision — `pair_runs` refuses two runs in the same `(arm, seed)` cell, not just… → `experiments/e3_*.yaml`, `t2o aggregate` |
| 1927-1934 | superseded (D) |  | Decision — `--metric` takes a list; `--csv` is opt-in (confirmed with the user).… → `experiments/e3_*.yaml`, `t2o aggregate` |
| 1935-1940 | superseded (D) |  | Not built, deliberately: no paired t-test (PLAN.md §12 allows "paired t-test *or… → `experiments/e3_*.yaml`, `t2o aggregate` |
| 1941-1949 | record 007 |  | The difference-of-differences was built later, and the reversal is worth recordi… — The difference-of-differences was built after the campaign — record 007 (F41) |
| 1950-1957 | superseded (D) |  | Tests (`tests/test_aggregate.py`, 21 fast + 1 slow; 3 more in `test_cli.py`). Ru… → `experiments/e3_*.yaml`, `t2o aggregate` |
| 1958-1962 | superseded (V) |  | Verify: `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"… |
| 1963-1965 | superseded (D) |  | No server run for this step. Step 5's campaign is the next thing that needs the … → `experiments/e3_*.yaml`, `t2o aggregate` |
| 1966-2119 | record 005 |  | M1.2 steps 5–6, E3 pix2pix at `grad_scale: 1.0e-2` |
| 2120-2223 | record 006 |  | M1.2 step 7, the dose at 2.3% of the objective |
| 2224-2613 | record 007 |  | M1.2 step 8, probes and E3 pix2pix at 0.15 |
| 2614-2710 | record 008 |  | M1.2 step 9, C2 on step 8's exports |
| 2711-2738 | record 019 |  | `--write-back`: an absent metric and a null one are different facts (lifted verbatim, 2711-2738) |
| 2739-2762 | record 008 |  | C2 campaign result: reward hacking is ruled out |
| 2763-2776 | superseded (X) |  | M2 — Phase 2: Diffusion loop — M2 header, the M2b-stays-a-list note and the ordering against M1.2 — status narrative; step 6's gate on this milestone is released in the paragraph below |
| 2777-2789 | design.md §8 | CLAUDE.md; record 010 | 1. Calibrate `grad_scale` for turbo before its campaign (step 4 below, and PLAN.… — Two conditions carried forward: calibrate `grad_scale` per backbone before its campaign (design.md §8, CLAUDE.md) — the second, step 9's faithfulness pass, was released clean and became turbo's C2 pass, record 010 |
| 2790-2791 | superseded (D) |  | M2a — pix2pix-turbo (primary) |
| 2792-2800 | superseded (C) |  | Step 1 — the backbone behind the `Translator` interface ✅ |
| 2801-2809 | superseded (D) |  | Decision — `Pix2Pix_Turbo` is reimplemented, not vendored (PLAN.md §7's row corr… → `translators/pix2pix_turbo.py` |
| 2810-2815 | superseded (D) |  | Decision — components are injected, not loaded inside `__init__`. `load_sd_turbo… → `translators/pix2pix_turbo.py` |
| 2816-2821 | superseded (D) |  | Decision — the skip-conv widths are derived from `vae.config["block_out_channels… → `translators/pix2pix_turbo.py` |
| 2822-2828 | superseded (D) |  | Decision — the timestep is named, not inferred. Upstream calls `set_timesteps(1)… → `translators/pix2pix_turbo.py` |
| 2829-2834 | superseded (D) |  | Decision — input is reflect-padded to a multiple of 64 inside the wrapper. The V… → `translators/pix2pix_turbo.py` |
| 2835-2843 | superseded (D) |  | Decision — `state_dict()` carries only what trains (LoRA + `unet.conv_in` + the … → `translators/pix2pix_turbo.py` |
| 2844-2849 | superseded (D) |  | Decision — evaluation uses the VAE posterior's mode, training keeps upstream's s… → `translators/pix2pix_turbo.py` |
| 2850-2855 | superseded (D) |  | Decision — `loss.gan` reuses M1's already-vendored PatchGAN (`define_D`/`GANLoss… → `translators/pix2pix_turbo.py` |
| 2856-2861 | superseded (D) |  | Decision — `train.amp`/`amp_dtype` are finally consumed, CUDA only. They have sa… → `translators/pix2pix_turbo.py` |
| 2862-2866 | superseded (D) |  | Tests (`tests/test_pix2pix_turbo_translator.py`, 20 fast + 1 slow). The fast one… → `translators/pix2pix_turbo.py` |
| 2867-2875 | superseded (D) |  | skips itself unless the checkpoint is already in the HuggingFace cache, so the s… → `translators/pix2pix_turbo.py` |
| 2876-2878 | superseded (V) |  | Verify: `ruff format`, `ruff check`, `pyright` (0 errors), `pytest -m "not slow"… |
| 2879-2881 | superseded (D) |  | No server run for this step. The sd-turbo fetch happened on the Mac, as PLAN.md … → `translators/pix2pix_turbo.py` |
| 2882-2894 | roadmap |  | `## M2a step 2`: the two open rows (2884, 2887) |
| 2895-2906 | design.md §10 | roadmap | Why the pretrain runs after the turbo campaign, and that nothing can seed a translator from an external checkpoint — design.md §10 Phase 2a; roadmap `## M2a step 2` points at it |
| 2907-2913 | roadmap |  | `## M2a step 3`: the open LPIPS early-stop row (2909) |
| 2914-3071 | record 009 |  | M2a step 4, turbo probes at 0.15 and 0.75 |
| 3072-3605 | record 010 |  | M2a step 5, E3 turbo n = 3 |
| 3606-3617 | roadmap |  | `## M2b`: five open rows (3608-3614) |
| 3618-3620 | roadmap |  | `## M3`: heading and the open baseline-suite row (3620) |
| 3621-3622 | record 011 |  | M3's done E8 row (the crossover N ≈ 150) |
| 3623-3632 | roadmap |  | `## M3`: the E9 row (stale at the tag — readout pending, see record 016), E10, E4 |
| 3633-3898 | record 011 |  | E8 low-annotation sweep and C/D top-up |
| 3899-3913 | roadmap |  | `## E8`: the deferred criterion-1 row, annotated per Q13 |
| 3914-3914 | record 011 |  | Section separator after the deferred row |
| 3915-4146 | record 012 |  | E9 pricing and step 0, pix2pix wall clock |
| 4147-4477 | record 013 |  | E9 steps 1–3: FLIR judge, in-loop `yolo11n`, the kill-test gate |
| 4478-4728 | record 014 |  | E9 step 3b: de-rolled labels and the re-gate |
| 4729-5056 | record 015 |  | E9 step 4: corpus-cap seam, per-epoch clock, throughput probe |
| 5057-5122 | record 016 |  | E9 step 5: the twelve-run FLIR cell, launch and pre-flight |
| 5123-5149 | record 020 |  | `t2o faithfulness` never streamed a list (lifted verbatim, 5123-5149) |
| 5150-5353 | record 016 |  | E9 step 5: results, wall clock, the two `campaign_report.py` bugs |
| 5354-5357 | roadmap |  | `## E9`: the two open readout rows (5354, 5357) |
| 5358-5358 | record 016 |  | Section separator after the open rows |
| 5359-5367 | roadmap |  | `## M4`: four open rows (5361-5364) |
| 5368-5419 | roadmap |  | `## Paper obligations`: the twelve corrections (5370-5416) |

#### `PLAN.md` — 806 lines at `pre-spec-migration`

| Lines | Disposition | Also lifted into | What it is / why |
| --- | --- | --- | --- |
| 1-8 | design.md |  | Title, preamble and the pointer to `TASKS.md` — design.md's header; the `TASKS.md` pointer is superseded by `docs/features/` and `docs/roadmap.md` |
| 9-10 | design.md §1 |  | §1 heading |
| 11-15 | goal.md | design.md §1 | The research question — goal.md Problem; design.md §1 keeps a pointer |
| 16-30 | design.md §1 |  | §1 body: the instrument, the C1–C4 table, scope (verbatim, plus the C-renumbering line) |
| 31-49 | design.md §2 |  | §2 Three facts that shape everything below (verbatim) |
| 50-74 | design.md §3 |  | §3 Environment and constraints (verbatim) |
| 75-122 | design.md §4 |  | §4 Backbone strategy and ladder (verbatim) |
| 123-169 | design.md §5 |  | §5 Repository architecture and the seven invariants (verbatim) |
| 170-205 | design.md §6 |  | §6 What ports from Clean-SeAFusion (verbatim) |
| 206-265 | design.md §7 |  | §7 Vendoring strategy, extra dependencies, licensing (verbatim) |
| 266-331 | design.md §8 |  | §8 Coupling design: schedule, ultralytics pin, loss contract (verbatim) |
| 332-343 | design.md §8 | record 006 | Saturating-reward and downscale guardrails — verbatim, with the dose result replaced by a pointer to 006 |
| 344-348 | design.md §8 | record 007 | Calibrated `grad_scale: 0.15` for pix2pix — result narrative replaced by a pointer to 007 |
| 349-377 | design.md §8 |  | §8 remainder: the reward-tuning guardrails, object-aware content loss (verbatim) |
| 378-477 | design.md §9 |  | §9 Data: contract, custom dataset, smoke fixture, public datasets (verbatim) |
| 478-504 | design.md §10 |  | §10 Phases 0–1 design (verbatim) |
| 505-513 | design.md §10 | records 002, 004–007 | Phase 1 result narrative — replaced by pointers to the gate and E3 records |
| 514-585 | design.md §10 |  | §10 Phases 2a–4 (verbatim) |
| 586-589 | design.md §11 |  | §11 heading and intro |
| 590-601 | design.md §11 | records 005–010 | §11 experiment table — the E3 row's result narrative replaced by pointers; design sentences kept (Q2) |
| 602-623 | design.md §11 |  | §11 remainder: why E4's comparison arms must be reimplemented (verbatim) |
| 624-670 | design.md §12 |  | §12 Metrics, including the exact sign-flip rule (verbatim) |
| 671-671 | design.md §13 |  | §13 heading — design.md keeps the heading and a pointer |
| 672-700 | AGENTS.md |  | §13 House style body — AGENTS.md `# House style` (one edit: the `basicConfig` bullet, plan Q17) |
| 701-703 | design.md §13 |  | Blank lines and separator |
| 704-728 | design.md §14 |  | §14 Dev-on-Mac / train-on-server workflow (verbatim) |
| 729-741 | design.md §15 | records 002, 007, 008 | §15 Risks — the Phase-1 and reward-hacking status cells replaced by pointers; the rest verbatim |
| 742-744 | design.md §15 |  | Blank lines and separator |
| 745-746 | design.md §16 |  | §16 heading |
| 747-750 | goal.md | design.md §16 | The five drafting criteria — goal.md Success Criteria lists six (Q12); design.md §16 points at it |
| 751-788 | superseded | records 005–008 | Status after M1.2 step 8 (causality satisfied for pix2pix; stability caveats; the stage-0 null) — records 005–008 hold the measurements; roadmap `## Drafting criteria` holds the status |
| 789-793 | design.md §16 |  | The superseded 1.0e-2 campaign and the two campaigns as the dose argument (verbatim) |
| 794-796 | superseded |  | 'Margin and consistency remain untouched' — stale status (the turbo arm has since replicated, record 010); roadmap `## Drafting criteria` carries both |
| 797-806 | design.md §16 |  | Fallback framing and why `annotation_fraction` cannot reach it (verbatim) |

### Net 3a — open rows

31 `- [ ]` rows at the tag, and 31 in `docs/roadmap.md`. Each is verbatim in full, wraps included
(whitespace collapsed), not only its first line. All 31 source line numbers fall in a `roadmap` row
of the map above. Per section they match this plan's count: M2a step 2 (2), M2a step 3 (1), M2b (5),
M3 (4), E8 (1), E9 (2), M4 (4), Paper obligations (12). **Dropped: none.** The tag also holds 147
`- [x]` rows and no other checkbox shape (`[~]`, `[-]`, `TODO`).

### Net 3b — numbers

817 distinct decimals with three or more significant figures in `TASKS.md`. 811 appear in
`docs/experiments/`, `design.md` or `roadmap.md`. The census took one spelling normalisation beyond
the Approach rule: a leading `0` is optional, so `.469` matches `0.469`. Without it, 8 were missing;
`0.469` and `0.875` are the two it resolves. The other 6 are below, and none is a transcription
gap.

| ID | Number | Source | Where it lives today | Verdict |
| --- | --- | --- | --- | --- |
| N1 | `4.45` | 64 | `transformers>=4.45` at `pyproject.toml:29` | not a gap — a version floor |
| N2 | `2407.12780` | 262 | `src/t2o/metrics/faithfulness.py:14` | not a gap — an arXiv ID |
| N3 | `0.03125` | 1919 | `tests/test_aggregate.py:11` and `:240`; the docs carry it as p = .031 | not a gap — the n = 6 sign-flip floor, 2/64 |
| N4 | `22.6` | 4987 | record 015 carries `22.59` GiB (F131) | not a gap — the source rounded |
| N5 | `39.7` | 4987 | record 015 carries `39.70` GiB (F131) | not a gap — the same sentence's spelling |
| N6 | `3.0000` | 5190 | record 016 line 65 gives only `std 0.0000`; the 3-class endpoint implies the mean | accept or restore — the one number whose value is not stated |

### Net 3d — citations

Every shape in tracked files and in `git log` subjects resolves through the legend or its opening
sentence (the files are frozen at the tag). **Dangling: none.**

| Shape | Cites | Resolves through |
| --- | --- | --- |
| `PLAN.md §N` | 130, in 62 files | `design.md §N`. Every N cited is 1–16 and has a `## N.` heading there; §13 is `AGENTS.md`'s `# House style`. |
| `TASKS.md <label>` | 59, 22 distinct labels | The section's record `Task:`, its roadmap heading, or neither. 20 labels resolve to a record or a roadmap section; `M0.5` and `M0.9` (4 cites) are build narrative and resolve to `git show pre-spec-migration:TASKS.md`. A step-level label (`M1.2 step 8`) resolves to its section's records; the index's *What ran* column places the step. |
| bare `TASKS.md`, no label | 8 sites in `src/`, `tests/`, `scripts/` (`src/t2o/data/adapters/msrs.py:3`, `src/t2o/data/mirror.py:16`, `src/t2o/metrics/task.py:68`, `src/t2o/seeding.py:17`, `tests/test_experiments.py:6`, `tests/test_loop.py:8` and `:152`, `scripts/gate_table.py:206`) | The legend's opening sentence: the files are frozen at the tag. Q8 keeps code comments as written. The rest of the 52 bare mentions are provenance lines in records, `AGENTS.md` and `docs/roadmap.md`, which already say `@ pre-spec-migration`. |
| `TASKS.md:<line>` | 20, in 8 files (records 007, 012, 013 and 017–020; 8 in `docs/roadmap.md`) | The tag, by the same sentence. |
| `RESEARCH_FINDINGS.md §N` | 3 | The file stays at the root. |
| commit subjects up to the tag | 111; 89 carry a parenthesised legacy label, 6 name a label bare, 16 name none | The 89 split into record 25, roadmap 1, both 43, neither 20 (the `M0.6`–`M0.9` build steps, which resolve to the tag). The 6 bare ones are `E3` ×5, defined in `RESEARCH_FINDINGS.md` §7, and `M0.8` ×1. Counted fresh, not reconciled to the plan's earlier 90. |
| commit subjects after the tag | none in the legacy `(M3 E9)` form | The commit guard refuses it. |

One `PLAN.md §17` cite exists, at `TASKS.md:2879`. `PLAN.md` has no §17, but the cite sits in a file
that is being deleted, so no tracked file carries it.

### Triage for the human

Nothing here blocks SPEC-MIGRATION-26 unless you say so. N1–N6 above are the number census's
output. T1–T5 came from reading `TASKS.md` for Net 2: each is a "revisit when…" note, not a
`- [ ]` row, so Net 3a could not catch it. Each lives only in the tag today.

| ID | Source | What the tag holds | Today |
| --- | --- | --- | --- |
| T1 | 733-757 | InsPLAD has 18 categories, not the paper's 17 (an extra `sphere`, id 18); train 7,981 and val 2,626 images; Mendeley serves the 6.4 GB archive with no form | `design.md` §9 keeps only "MS-COCO detection JSON". Q15 chose not to mint it. It matters only if InsPLAD numbers are ever cited. |
| T2 | 118-121 | Experiment inheritance (a `base:` include key), "revisit at M0.8 with ≥3 files" | `experiments/` now holds seven YAMLs and the idea was never revisited. A speculative idea, not an obligation. |
| T3 | 810-816 | `verify_split` and `SplitDriftError` were built so "a future run-start check is a small addition" | Only `scripts/freeze_splits.py` calls them. `engine/loop.py` and `cli.py` do not. |
| T4 | 704-716, 963-969 | CPLID, HIT-UAV, TTPLA, InsPLAD and DroneVehicle as single-modality detector data "if E9 wants it". DroneVehicle has images here and no labels, "not now". | E9 ran on FLIR alone. `goal.md` Constraints names CPLID, TTPLA and HIT-UAV as unpaired. DroneVehicle appears nowhere else. |
| T5 | 781-793 | The held-out test split's membership is unpinned: `freeze_split` records only train and val, and the note says to freeze it when it is first used. A leak into train or val is caught; a reshuffle within the held-out set is not. | `design.md` §9 says the isolation is structural, not that the membership is unfrozen. `goal.md` rests on frozen, committed splits, and the custom test split is the one that is not. **The one I would act on**, as a roadmap row, before the test split is first read. |

### No logic change

`git diff --exit-code pre-spec-migration -- src scripts tests` prints nothing and exits 0.

### Net 4 — fresh clone

What SPEC-MIGRATION-26 produced. The clone is `5af2ce6` with the task's uncommitted diff applied:
no `PLAN.md`, `TASKS.md`, `runs/` or `dataset/`. Every answer was read inside the clone, none from
the tag. **Unanswered: none.**

| # | Question | Answer | Path |
| --- | --- | --- | --- |
| 1 | E3 pix2pix's headline and p | Stage-3 paired zero-shot mAP50 +0.0512, p = .031, CI [+.025, +.081] | F35, `docs/experiments/007-m1-2-e3-pix2pix-positive.md:255` |
| 2 | Why `grad_scale` is per backbone | 0.15 is a property of one objective's composition with one generator; turbo coinciding was measured, not predictable, so every backbone re-runs the 25-epoch `scripts/loss_share.py` probe (20–30% share) | `docs/design.md:360-370`, `:709`; `CLAUDE.md:42` |
| 3 | E9 FLIR against its floor | Loop − control +0.3512 (p = .031), but the loop lands on the raw-thermal floor, 0.4415 against 0.4499: it prevents a loss, it does not beat no translation | F136, F137, `docs/experiments/016-e9-flir-twelve-run-cell.md:278`, `:288` |
| 4 | What runs next | The open rows of the roadmap, chosen by a human — nothing ranks them, and the tag did not either | `AGENTS.md:33`; `docs/roadmap.md:5` |
| 5 | Sign-flip versus bootstrap | The exact sign-flip p is the test; at n = 6 the percentile bootstrap is anti-conservative, so CIs are descriptive only | `docs/design.md:660-666` |
| 6 | 753 of 853 and the `Connector` class | 600 train + 153 val of 853, ~100 held out; `nc: 5` but `Connector` (index 0) is an unused Label Studio class — report 4 | `docs/goal.md:130-133`; `docs/design.md:403-411`; record 003 |
| 7 | Why ultralytics is pinned `>=8.4.108,<8.5` | The `Detect` head's output format changed between 8.3 and 8.4, and code written against one silently fails on the other. No reason is given for the `.108` patch floor itself, here or at the tag; `frozen.py` notes the `de_parallel` → `unwrap_model` rename just above it | `docs/design.md:302-311`, `:329`; `docs/goal.md:145`; `src/t2o/detection/frozen.py:27-30` |
| 8 | The three detector roles | In-loop (gets generator gradients), evaluation (fine-tuned on exports, never gets them), reference (never trained, zero-shot gate) — three `DetectorConfig` sub-sections | `docs/design.md:168-173` |
| 9 | E8's crossover | Translation is worth ≈150 annotated thermal images, ±1 sd [114, 185]; ≈214 against arm D | F81, `docs/experiments/011-e8-annotation-sweep.md:188`, `:210` |
| 10 | The open paper obligations | 12 unchecked rows | `docs/roadmap.md` `## Paper obligations` |

## Out of Scope

- Any logic change under `src/`, `scripts/` or `tests/`. Comment edits only, and only if a task names them (Q8 recommends none).
- Editing a record after it lands, or editing any finding's claim text. A claim's fate goes in the ledger's `Status` cell.
- Renumbering or renaming legacy labels — `M`/`E`/`C` IDs, step and finding numbers.
- Research and server runs, until the PR merges (Q3).
- `docs/progress/*`, `docs/results/*` and `runs/*`. They stay as they are.
- Changes under `~/.claude/` (templates, skills, guidelines). A gap found there is a separate conversation.
- The sister repo, including its missing `~/Desktop/PLAN.md` backup.
- Pushing the tag or the branch — the human's step.
