# Spec-Workflow Migration — Tasks

| ID | Description | Complexity | Suggested Model | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| SPEC-MIGRATION-01 | Repoint `AGENTS.md`'s legacy Working Order (`AGENTS.md:10-17`) at `CLAUDE.md` `## Planning` and `docs/features/` | small | sonnet | done | contradicts `CLAUDE.md` until fixed (sister lesson 2) |
| SPEC-MIGRATION-02 | Scaffold `docs/experiments/index.md` from `~/.claude/templates/experiments-index.md`, seeded so the first run is `001` and the first finding `F01` | small | sonnet | done | Q5; blocks -03..-19 |
| SPEC-MIGRATION-03 | Record `001` — E1 reference bracket (M0.10 + M1's re-measure, `TASKS.md:970-1047`) | moderate | opus | done | needs -02; `Log:` per Q6 |
| SPEC-MIGRATION-04 | Record `002` — M1 Phase 1 go/no-go (`TASKS.md:1048-1365`) | moderate | opus | done | needs -02 |
| SPEC-MIGRATION-05 | Record `003` — M1.1 fidelity on M1's runs (`TASKS.md:1366-1510`) | moderate | opus | done | needs -02 |
| SPEC-MIGRATION-06 | Record `004` — M1.2 design + step 1, the `yolo11s` judge (`TASKS.md:1511-1680`) | moderate | opus | done | needs -02 |
| SPEC-MIGRATION-07 | Record `005` — M1.2 steps 5–6, E3 pix2pix at `grad_scale: 1.0e-2` (`TASKS.md:1966-2119`) | moderate | opus | done | needs -02 |
| SPEC-MIGRATION-08 | Record `006` — M1.2 step 7, the dose at 2.3% (`TASKS.md:2120-2223`) | moderate | opus | done | needs -02 |
| SPEC-MIGRATION-09 | Record `007` — M1.2 step 8, probes + E3 pix2pix at 0.15 (`TASKS.md:2224-2613`) | complex | opus | done | needs -02 |
| SPEC-MIGRATION-10 | Record `008` — M1.2 step 9, C2 on step 8's exports (`TASKS.md:2614-2762`) | moderate | opus | done | needs -02 |
| SPEC-MIGRATION-11 | Record `009` — M2a step 4, turbo probes (`TASKS.md:2914-3071`) | moderate | opus | done | needs -02; steps 2–3's open rows go to the roadmap (-22) |
| SPEC-MIGRATION-12 | Record `010` — M2a step 5, E3 turbo n = 3 (`TASKS.md:3072-3605`) | complex | opus | pending | needs -02 |
| SPEC-MIGRATION-13 | Record `011` — E8 annotation sweep + C/D top-up (`TASKS.md:3633-3914`) | complex | opus | pending | needs -02; row 3899 stays for the roadmap (Q13) |
| SPEC-MIGRATION-14 | Record `012` — E9 pricing + step 0, pix2pix wall clock (`TASKS.md:3915-4146`) | moderate | opus | pending | needs -02 |
| SPEC-MIGRATION-15 | Record `013` — E9 steps 1–3, FLIR judge + kill-test gate (`TASKS.md:4147-4477`) | complex | opus | pending | needs -02; check `runs/*` for a `Log:` path (Q6) |
| SPEC-MIGRATION-16 | Record `014` — E9 step 3b, de-rolled labels + re-gate (`TASKS.md:4478-4728`) | moderate | opus | pending | needs -02 |
| SPEC-MIGRATION-17 | Record `015` — E9 step 4(b), throughput probe (`TASKS.md:4729-5056`) | moderate | opus | pending | needs -02 |
| SPEC-MIGRATION-18 | Record `016` — E9 step 5, the twelve-run FLIR cell (`TASKS.md:5057-5358`) | complex | opus | pending | needs -02; `Log:` = `runs/e3f-report-2026-10-01.txt`; open readout rows go to the roadmap |
| SPEC-MIGRATION-19 | No-run records `017`+ — the inventory's six build-time findings, plus named findings in M0.1–M0.9 (`TASKS.md:20-969`) and M1.2 steps 2–4 (`1681-1965`) | complex | opus | pending | needs -02 |
| SPEC-MIGRATION-20 | Finding census (Net 3c): every finding has exactly one F-ID carrying its legacy label; every superseded/withdrawn/narrowed chain resolves from both ends | moderate | opus | pending | needs -03..-19; blocks -21, -22 |
| SPEC-MIGRATION-21 | `docs/design.md` from `PLAN.md`, `§1`–`§16` kept; results narrative → record pointers; `§1`/`§16` → `docs/goal.md`; `§13` heading + pointer only | complex | opus | pending | needs -20; Q2, Q12 |
| SPEC-MIGRATION-22 | `docs/roadmap.md` — all 31 open rows verbatim under legacy labels, `## Paper obligations`, a status line per `goal.md` criterion citing F-IDs, row 3899 annotated | complex | opus | pending | needs -20; Q9, Q10, Q12, Q13; no ledger counter in prose |
| SPEC-MIGRATION-23 | Rewrite `AGENTS.md` per the End state table: `PLAN.md §13` as House style, the citation legend, where an M/E label lives | moderate | opus | pending | needs -21, -22; General Guidelines kept verbatim; Q8 |
| SPEC-MIGRATION-24 | Repoint `README.md:13-14`'s links at `docs/design.md`, `docs/roadmap.md` and `docs/experiments/` | trivial | haiku | pending | needs -21, -22; `:34`'s `PLAN.md §3` citation stays, resolved by the legend (Q8) |
| SPEC-MIGRATION-25 | Coverage map (Net 2) appended to `plan.md`; open-row (3a), numbers (3b, triage list for the human) and citation (3d) censuses; `git diff pre-spec-migration -- src scripts tests` shows no logic change | complex | opus | pending | needs -03..-24; blocks -26 |
| SPEC-MIGRATION-26 | Delete `PLAN.md` and `TASKS.md`, then the fresh-clone check (Net 4): its ten questions answered from the new docs alone | moderate | opus | pending | needs -25 and the tag pushed to `origin` (Net 1, the human's step) |
