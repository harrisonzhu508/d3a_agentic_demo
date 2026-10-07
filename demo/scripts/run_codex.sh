#!/usr/bin/env bash
# Start Codex on the autoresearch program, with the harness hooks (.codex/hooks.json) and the skills (.agents/skills).
#
#   bash scripts/run_codex.sh [options]
#     --experiment NAME    results/NAME/ and branches <prefix>/NAME/...   (else config/autoresearch.toml)
#     --experiments N      hypotheses in this session
#     --model ID           a Codex model (else Codex's default)
#     --budget-seconds N   time limit per evaluation, 0 = none
#     --config FILE        another settings file instead of config/autoresearch.toml
#     --no-push            keep branches and pull requests local
#     --print              headless (codex exec) instead of the interactive TUI
#     --prompt TEXT        another first message
#
# Uses your Codex login (codex login). Like pi, Codex runs without its sandbox and approvals (the agent has to commit,
# push and run uv), so the hooks are the guard; they are this repository's own, so they run without the one-time
# /hooks review. Scored hypotheses are synced to Weights & Biases; the Codex conversation itself is not (pi only).
set -euo pipefail
cd "$(dirname "$0")/.."

PRINT=0; MODEL=(); PROMPT="Read AGENTS.md and program.md, then run the program."
while [ $# -gt 0 ]; do
  case "$1" in
    --experiment) export D3A_EXPERIMENT="$2"; shift ;;
    --experiments) export D3A_MAX_EXPERIMENTS="$2"; shift ;;
    --model) MODEL=(-m "$2"); shift ;;
    --budget-seconds) export D3A_BUDGET_SECONDS="$2"; shift ;;
    --config) export D3A_CONFIG="$(realpath "$2")"; shift ;;
    --no-push) export D3A_NO_PUSH=1 ;;
    --print) PRINT=1 ;;
    --prompt) PROMPT="$2"; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done

export D3A_HARNESS=codex
args=("${MODEL[@]}" --dangerously-bypass-approvals-and-sandbox --dangerously-bypass-hook-trust -C "$PWD")

# W&B sync while Codex runs; at the end a last sync and the experiment PR
EXPDIR=$(.venv/bin/python -c 'from checks.settings import experiment_dir; print(experiment_dir())')
SYNC=(uv run --quiet --group tracking python scripts/wandb_sync.py)
if [ "${D3A_SYNC:-1}" = 1 ]; then
  "${SYNC[@]}" --watch 60 >> "$EXPDIR/wandb_sync.log" 2>&1 &
  WATCH=$!
  trap 'kill $WATCH 2>/dev/null || true; "${SYNC[@]}" >> "$EXPDIR/wandb_sync.log" 2>&1 || true
        .venv/bin/python skills/github-workflow/scripts/experiment_pr.py || true' EXIT
fi

echo "codex: ${MODEL[1]:-default model} | results: $EXPDIR"
if [ "$PRINT" = 1 ]; then
  codex exec "${args[@]}" "$PROMPT" < /dev/null     # exec also reads piped stdin: never wait on it
else
  codex "${args[@]}" "$PROMPT"
fi
