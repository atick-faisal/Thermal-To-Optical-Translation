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
