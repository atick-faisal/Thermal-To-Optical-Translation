---
slug: spec-migration
title: Spec-Workflow Migration
status: in-progress
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
and `Date` is that commit's date unless the source names the run's own.

### `PLAN.md` → `docs/design.md`

Every `§N` heading survives under the same number so `PLAN.md §N` resolves by one legend rule.
Design rules and calibrated values stay verbatim — §8's "calibrate `grad_scale` per backbone",
§9's data contract, §12's two task arms and the exact sign-flip rule. Measurement narrative
becomes a one-line pointer to the record that holds it: the status paragraphs in §8, §10's
"Result" paragraphs, §11's E3 row, §15's status cells and §16's status section. §13 moves into
`AGENTS.md` (loaded every session) and `docs/design.md §13` keeps only its heading and a pointer,
so there is one copy. §1 and §16 point at `docs/goal.md` for the objective and the criteria.

### The roadmap

All 31 open rows, verbatim, grouped under their legacy section labels (Q9) — `M2a step 2`,
`M2a step 3`, `M2b`, `M3`, `E8`, `E9`, `M4` — and the 13 paper corrections as
`## Paper obligations` (Q10). The six drafting criteria in `docs/goal.md` get a status line each
that cites F-IDs rather than restating numbers (Q12). Row 3899's open decision travels with it whole: whether to
loosen criterion 1 to admit a public dataset, now that E9 says "loop beats control" holds on two
datasets and "translation beats thermal" on one. It carries a note that `docs/goal.md`'s Margin
and Consistency wording may have settled it, for the human to close (Q13).

### The citation legend (in `AGENTS.md`)

| Citation | Resolves to |
| --- | --- |
| `PLAN.md §N` | `docs/design.md §N`; `§13` → `AGENTS.md` House style |
| `TASKS.md M<x>[ step <n>][ finding <k>]`, commit subjects `(M3 E9)` | the record whose `Task:` names that section — grep the legacy label under `docs/experiments/`; open work → the same label in `docs/roadmap.md`; build narrative → `git show pre-spec-migration:TASKS.md` |
| `RESEARCH_FINDINGS.md §N`, `E1`–`E10`, `C1`–`C4` | unchanged — the file stays (Q11); `E`/`C` are defined in its §7 and §1 |

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

## Out of Scope

- Any logic change under `src/`, `scripts/` or `tests/`. Comment edits only, and only if a task names them (Q8 recommends none).
- Editing a record after it lands, or editing any finding's claim text. A claim's fate goes in the ledger's `Status` cell.
- Renumbering or renaming legacy labels — `M`/`E`/`C` IDs, step and finding numbers.
- Research and server runs, until the PR merges (Q3).
- `docs/progress/*`, `docs/results/*` and `runs/*`. They stay as they are.
- Changes under `~/.claude/` (templates, skills, guidelines). A gap found there is a separate conversation.
- The sister repo, including its missing `~/Desktop/PLAN.md` backup.
- Pushing the tag or the branch — the human's step.
