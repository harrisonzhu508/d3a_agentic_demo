#!/usr/bin/env bash
# Start pi on the autoresearch program, with the harness extension and the skills.
#
#   bash scripts/run_pi.sh [options]
#     --experiment NAME    results/NAME/ and branches <prefix>/NAME/...   (else config/autoresearch.toml)
#     --experiments N      hypotheses in this session
#     --model P/ID         e.g. local/qwen3.8-27b, dide2/qwen3.8-27b, openai/gpt-6-sol (else config/endpoint.toml)
#     --budget-seconds N   time limit per evaluation, 0 = none
#     --config FILE        another settings file instead of config/autoresearch.toml
#     --no-push            keep branches and pull requests local
#     --print              headless instead of the interactive TUI
#     --prompt TEXT        another first message
#     --list-models        the models pi can use
#
# While pi runs, the experiment is synced to Weights & Biases every minute (results/<experiment>/wandb_sync.log);
# at the end the experiment's pull request into the base branch is opened or updated.
set -euo pipefail
cd "$(dirname "$0")/.."

PRINT=0; LIST=0; MODEL=""; PROMPT="Read AGENTS.md and program.md, then run the program."
while [ $# -gt 0 ]; do
  case "$1" in
    --experiment) export D3A_EXPERIMENT="$2"; shift ;;
    --experiments) export D3A_MAX_EXPERIMENTS="$2"; shift ;;
    --model) MODEL="$2"; shift ;;
    --budget-seconds) export D3A_BUDGET_SECONDS="$2"; shift ;;
    --config) export D3A_CONFIG="$(realpath "$2")"; shift ;;
    --no-push) export D3A_NO_PUSH=1 ;;
    --print) PRINT=1 ;;
    --prompt) PROMPT="$2"; shift ;;
    --list-models) LIST=1 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done

PI=.tools/node_modules/.bin/pi
export PI_CODING_AGENT_DIR="${PI_CODING_AGENT_DIR:-$PWD/.pi-agent}" D3A_HARNESS=pi
# only the model keys reach pi (and so the agent); the GitHub and W&B tokens stay in the file
eval "$(grep -E '^(OPENAI|DEEPSEEK|DIDE2)_API_KEY=.' config/secrets.env | sed 's/^/export /' || true)"
[ "$LIST" = 1 ] && exec "$PI" --list-models
MODEL=${MODEL:-$(python3 -c 'import json, os; s = json.load(open(os.environ["PI_CODING_AGENT_DIR"] + "/settings.json")); print(s["defaultProvider"] + "/" + s["defaultModel"])')}

args=(--approve -e .pi/extensions/d3a-harness.ts --model "$MODEL")
for s in skills/*/; do args+=(--skill "$s"); done

# W&B sync while pi runs; at the end a last sync and the experiment PR (parallel.sh does these once for all)
EXPDIR=$(.venv/bin/python -c 'from checks.settings import experiment_dir; print(experiment_dir())')
SYNC=(uv run --quiet --group tracking python scripts/wandb_sync.py)
if [ "${D3A_SYNC:-1}" = 1 ]; then
  "${SYNC[@]}" --watch 60 >> "$EXPDIR/wandb_sync.log" 2>&1 &
  WATCH=$!
  trap 'kill $WATCH 2>/dev/null || true; "${SYNC[@]}" >> "$EXPDIR/wandb_sync.log" 2>&1 || true
        .venv/bin/python skills/github-workflow/scripts/experiment_pr.py || true' EXIT
fi

echo "pi: $MODEL | results: $EXPDIR"
if [ "$PRINT" = 1 ]; then
  "$PI" "${args[@]}" -p "$PROMPT" < /dev/null     # print mode also reads piped stdin: never wait on it
else
  "$PI" "${args[@]}" "$PROMPT"
fi
