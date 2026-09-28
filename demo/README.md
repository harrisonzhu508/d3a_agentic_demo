# Autoresearch demo: an agent improves a censored Bayesian MIC model

An agent (pi, with Qwen3.8-27B or an OpenAI / DeepSeek model; or Claude Code) improves a NumPyro regression of
ciprofloxacin MIC on AMR genotype in *E. coli*, one hypothesis at a time. A fixed harness scores every change
(5-fold CV ELPD + sampler gates) and keeps it only if it beats the current best by more than 2 SE. Every
hypothesis becomes a git branch; kept ones become pull requests with a written report, and each experiment ends in
one pull request into `main`. Runs and every agent step are logged to Weights & Biases.

## Quick start

Needs Linux, `uv`, Node.js >= 22.19, `git` with an SSH key for GitHub, and (for a local model) a 48 GB NVIDIA GPU.

```bash
git clone git@github.com:harrisonzhu508/d3a_agentic_demo.git && cd d3a_agentic_demo/demo
bash scripts/setup.sh --vllm      # Python env, data split, pi, configs; vLLM + the model (--vllm is optional)
# edit config/*.toml and config/secrets.env (below), then:
bash scripts/vllm.sh start        # only for the local model
uv run python scripts/check.py    # model endpoint and keys
bash scripts/run_pi.sh --experiment demo1 --experiments 5
```

## Configuration (all files gitignored, created from the `.example` files; the agent cannot edit them)

| File | What |
|---|---|
| `config/autoresearch.toml` | experiment name and budget; `branch_prefix` (the only branches agents may push), `base_branch`, `push`, `auto_merge` (agent merges its own kept PRs into the experiment branch); `[evaluation] budget_seconds` (time limit per evaluation, 0 = none); `[tracking] wandb_project` |
| `config/endpoint.toml` | the agent's model, e.g. `local/qwen3.8-27b`, `dide2/qwen3.8-27b`, `openai/gpt-6-sol`, `deepseek/deepseek-v4-pro`; other OpenAI-compatible endpoints as `[endpoints.<name>]` |
| `config/vllm.toml` | the local vLLM server: model, GPU, port, context length |
| `config/secrets.env` | `GITHUB_TOKEN` (fine-grained, this repo: Contents read, Pull requests + Issues read/write), `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `WANDB_API_KEY` |

After editing `endpoint.toml` or `vllm.toml`: `bash scripts/setup.sh --configs`.

## Running

```bash
bash scripts/run_pi.sh --experiment demo1 --experiments 5   # interactive TUI; --print for headless
bash scripts/run_pi.sh --model openai/gpt-6-sol --budget-seconds 1800 --no-push
bash scripts/parallel.sh 3 --experiment demo-par --experiments 3   # 3 agents at once, each in a git worktree
bash scripts/feedback.sh "Try an interaction between gyrA and parC next."   # talk to running agents
bash scripts/feedback.sh --stop                                            # finish this hypothesis, then stop
uv run python checks/status.py                                            # champion and last results
```

In the TUI: Enter steers the next step, Alt+Enter queues a follow-up, Escape pauses.

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
demo/                         the agent's workspace (open this folder in Claude Code)
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
- Claude Code: open `demo/` in the desktop app and ask it to "Read AGENTS.md and program.md, then run the program";
  hooks and skills load from `.claude/`.
