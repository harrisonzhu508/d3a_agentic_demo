# Autoresearch demo: an agent improves a censored Bayesian MIC model

An agent (pi, with Qwen3.8-27B or an Anthropic / OpenAI / DeepSeek model; Codex; or Claude Code) improves a NumPyro regression of
ciprofloxacin MIC on AMR genotype in *E. coli*, one hypothesis at a time. A fixed harness scores every change
(5-fold CV ELPD + sampler gates) and keeps it only if it beats the current best by more than 2 SE. Every
hypothesis becomes a git branch; kept ones become pull requests with a written report, and each experiment ends in
one pull request into `main`. Runs and every agent step are logged to Weights & Biases.

## Quick start

Needs Linux, `uv`, Node.js >= 22.19, `git` with an SSH key for GitHub, and (for a local model) a 48 GB NVIDIA GPU.
For the Codex agent also the Codex CLI (`npm install -g @openai/codex`, then `codex login`).

```bash
# first fork https://github.com/harrisonzhu508/d3a_agentic_demo on GitHub (see "Your own fork" below)
git clone git@github.com:<you>/d3a_agentic_demo.git && cd d3a_agentic_demo/demo
bash scripts/setup.sh --vllm      # Python env, data split, pi, configs; vLLM + the model (--vllm is optional)
# edit config/*.toml and config/secrets.env (below), then:
bash scripts/vllm.sh start        # only for the local model
uv run python scripts/check.py    # model endpoint, keys, where PRs and W&B runs go
bash scripts/run_pi.sh --experiment demo1 --experiments 5
```

## Configuration (all files gitignored, created from the `.example` files; the agent cannot edit them)

| File | What |
|---|---|
| `config/autoresearch.toml` | experiment name and budget; `repository` (the GitHub repository for branches and PRs, `owner/name`; empty = the one you cloned), `branch_prefix` (the only branches agents may push), `base_branch`, `push`, `auto_merge` (agent merges its own kept PRs into the experiment branch); `[evaluation] budget_seconds` (time limit per evaluation, 0 = none); `[tracking] wandb_project` (`d3a_demo` = your own W&B account, or `<entity>/<project>`) |
| `config/endpoint.toml` | the agent's model: `local/qwen3.8-27b`, or uncomment `anthropic/claude-sonnet-5-5`, `openai/gpt-6-luna` or `deepseek/deepseek-v4-pro`; your own OpenAI-compatible servers as `[endpoints.<name>]` |
| `config/vllm.toml` | the local vLLM server: model, GPU, port, context length |
| `config/secrets.env` | `GITHUB_TOKEN` (fine-grained, for that repository only: Contents read, Pull requests + Issues read/write), `WANDB_API_KEY`; for a hosted model uncomment `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `DEEPSEEK_API_KEY` |

After editing `endpoint.toml` or `vllm.toml`: `bash scripts/setup.sh --configs`.

To use Claude or GPT instead of the local model: in `config/endpoint.toml` uncomment
`model = "anthropic/claude-sonnet-5-5"` (or `"openai/gpt-6-luna"`) and comment out the local one; in
`config/secrets.env` uncomment `ANTHROPIC_API_KEY=` (or `OPENAI_API_KEY=`) and paste your key; then
`bash scripts/setup.sh --configs` and `uv run python scripts/check.py`.

## Your own fork

Nothing in the code names a GitHub repository or a W&B account: pull requests go to the repository you cloned (or
the one in `[git] repository`) and runs to your W&B account, so a fork works as it is.

1. Fork the repository on GitHub (copying `main` only is enough) and clone your fork over SSH. The agents push
   their branches with your SSH key, so it must be on your GitHub account.
2. `config/secrets.env`: a fine-grained `GITHUB_TOKEN` for your fork only (Contents: Read-only, Pull requests:
   Read and write; Issues: Read and write for the labels, which needs Issues enabled in the fork's settings).
   Pull requests open in your fork, never in the original repository.
3. `config/autoresearch.toml`: your experiment name and `branch_prefix`; `[tracking] wandb_project = "d3a_demo"`
   logs to your own W&B account (or set `"<team>/<project>"`). To use another GitHub repository than the one you
   cloned, set `[git] repository = "<owner>/<name>"` (your token and SSH key need access to it).
4. `uv run python scripts/check.py` names the repository the pull requests go to and the W&B project.
5. To pick up later changes: `git remote add upstream https://github.com/harrisonzhu508/d3a_agentic_demo.git`
   once, then `git pull upstream main`.

`setup.sh` creates the data split and the locked test set on your machine; `checks/best.json`, the baseline every
experiment has to beat, comes with the repository.

## Running

```bash
bash scripts/run_pi.sh --experiment demo1 --experiments 5   # interactive TUI; --print for headless
bash scripts/run_pi.sh --model anthropic/claude-sonnet-5-5 --budget-seconds 1800 --no-push
bash scripts/run_codex.sh --experiment demo1 --experiments 5   # Codex instead of pi (same options, --model <codex model>)
bash scripts/parallel.sh 3 --experiment demo-par --experiments 3   # 3 agents at once, each in a git worktree
bash scripts/feedback.sh "Try an interaction between gyrA and parC next."   # talk to running agents
bash scripts/feedback.sh --stop                                            # finish this hypothesis, then stop
uv run python checks/status.py                                            # champion and last results
```

In the pi TUI: Enter steers the next step, Alt+Enter queues a follow-up, Escape pauses. `feedback.sh` messages reach
pi only (`--stop` and `--budget` work for every agent); in Codex, type into its TUI.

## What an experiment produces

- **GitHub**: `<prefix>/<experiment>/main` (the experiment branch, from `main`); one branch per hypothesis: kept
  ones with a PR into the experiment branch (the committed report is its description), the others kept as
  `<prefix>/<experiment>/discarded/<slug>`; and when the session ends, one PR from the experiment branch into
  `main` listing every hypothesis. Merging into `main` is always yours.
- **`results/<experiment>/`**: `results.tsv` (one row per hypothesis), `hooks.log` (every hook decision),
  `champion.json`, `session*.json`, `wandb_sync.log`, and a folder per hypothesis (`metrics.json`,
  `pointwise.csv`, `posterior.nc`, `review.png`, `report_draft.md`).
- **Weights & Biases** (`[tracking] wandb_project`): Weave → Agents → `pi-autoresearch`, one conversation per agent
  session with every model call and tool call; and one W&B run per scored hypothesis with its metrics and plots.

## Layout

```
demo/                         the agent's workspace (pi, Codex and Claude Code all start here)
  TASK.md AGENTS.md program.md  what the agent reads: task, rules, the loop, research directions
  src/                        the only code the agent may change (features, model, sampler)
  checks/                     the referee: evaluate (CV ELPD, gates, keep rule), smoke test, integrity, status
  hooks/                      what the agent may read, write and run; the stop hook that keeps the loop going
  skills/                     modelling, checking, biology, plots, evaluation + report, GitHub (see CATALOG.md)
  data/dev.csv                the agent's data; the locked test set is written to ~/.d3a-demo-private/
  scripts/                    setup, run_pi, parallel, vllm, feedback, check, render_configs, wandb_sync
dataset/                      humans only (outside the agent's workspace): source data, full table, DATASET.md
```

## Notes

- The hooks are a demo-grade guard (path and command patterns), not a security boundary: run agents in a
  container that mounts only `demo/` for anything serious.
- To change the harness (`checks/`, `hooks/`, `TASK.md`, `program.md`), commit to `main` before an experiment:
  evaluations refuse to score if these differ from `main`.
- Tests of the hooks: `uv run python hooks/tests/run_tests.py`.
- Codex: `scripts/run_codex.sh` starts it in `demo/`; it reads `AGENTS.md`, loads the skills from `.agents/skills` and
  the hooks from `.codex/hooks.json`. Like pi it runs without its sandbox (the agent commits, pushes and runs uv), with
  the hooks as the guard. Opening `demo/` in the Codex app also works: trust the project and approve the hooks in
  `/hooks` (again after any change to them). Codex does not load project hooks inside a git worktree, so
  `parallel.sh` stays pi-only. Scored hypotheses go to W&B; the Codex conversation itself does not (pi only).
- Claude Code: open `demo/` in the desktop app and ask it to "Read AGENTS.md and program.md, then run the program";
  hooks and skills load from `.claude/`.
