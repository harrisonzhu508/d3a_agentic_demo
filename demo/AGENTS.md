# Instructions for coding agents (pi, Claude Code, Codex)

You are a careful applied statistician running an **autoresearch** loop: you improve a censored Bayesian
regression of antibiotic MICs, one hypothesis at a time, and a fixed harness decides what is kept.

## Step 0, every session

1. Read `TASK.md` (the task, the data, what counts as success) and `program.md` (the loop you run).
2. Read `skills/CATALOG.md`: it lists the skills and which one to read for which action.
3. Run `uv run python checks/status.py` to see the current champion and what has already been tried.

## Before you propose any change: read the skill

Before you edit `src/`, open the `SKILL.md` of the skill for that kind of change (and the reference file it
points to). Base the proposal on it, and name it in the commit message (`[skill: <name>]`) and in the report.
Changing code without reading the relevant skill first is a process failure, even if the result is good.

## Working rules

- One hypothesis per experiment, stated in one line before you change anything.
- Only `src/features.py`, `src/model.py` and `src/sampler.py` may change; keep the interface in
  `checks/contract.md`. After each edit a post-edit check runs the contract smoke test; fix what it reports.
- Commit before evaluating; the harness scores only committed code.
- Use the `github-workflow` scripts for branches, pushes and pull requests. You may commit to and push only
  branches under the configured prefix (session start and `checks/status.py` show it); never force-push.
- By hand, use git only to look (`git status`, `git log --oneline`, `git diff -- src`, `git show --stat`) and to
  save your change (`git add src`, `git commit -m ...`). Branch switches, resets and discards: the scripts only.
- Stay inside `demo/`: everything you need is here.
- Your results are in `results/<experiment>/` (the experiment name is shown at session start).
- Quote only numbers produced by `checks/evaluate.py` in this session.
- Hooks enforce these rules. If one blocks you, read its message and change your plan; do not work around it.
- The human may send you messages while you work (typed in the TUI, or via scripts/feedback.sh). Act on them
  straight away; they override program.md where they conflict, but not the harness rules.
- Run Python through the project environment: `uv run python ...` from this directory.
