# General Guidelines
- Think Before Coding. Don't assume. Don't hide confusion. Surface tradeoffs.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.
- Simplicity First. Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.
- If you write 200 lines and it could be 50, rewrite it.
- No error handling for impossible scenarios.

# Working Order
1. Read [PLAN](PLAN.md) and [TASKS](TASKS.md) files. (These files always track the current state of the implementation and can be updated throughout the implementation process as needed).
2. Check the current state of the codebase.
3. Determine the immediate next step(s) to be implemented.
4. Start implementation of that specific step(s). Not everything at once.
5. Once a step is complete, verify and test the code using available tools.
6. Once satisfied, mark the task as "done", make a commit following the gitemoji + conventional commit message standard and then stop.
7. Let me verify and once I confirm verification start all over these steps.
