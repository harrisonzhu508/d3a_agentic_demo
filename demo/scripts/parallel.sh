#!/usr/bin/env bash
# N agents at once on one experiment, each a headless pi session in its own git worktree (.worktrees/w1, ...).
# They share results/<experiment>/ (and so see each other's results.tsv), the champion and the experiment branch;
# each has its own budget, feedback file and branch names (w2-<slug>) and a research focus.
#
#   bash scripts/parallel.sh 3 --experiment demo-par --experiments 3    # run_pi.sh options are passed on
#   bash scripts/parallel.sh --clean                                     # remove the worktrees
set -euo pipefail
cd "$(dirname "$0")/.."
DEMO=$PWD; REL=$(git rev-parse --show-prefix); REL=${REL%/}
if [ "${1:-}" = --clean ]; then
  for wt in .worktrees/w*/; do git worktree remove --force "$wt"; done
  git worktree prune; exit
fi
N=$1; shift
args=()
while [ $# -gt 0 ]; do       # --experiment and --config apply here too; the rest goes to run_pi.sh
  case "$1" in
    --experiment) export D3A_EXPERIMENT="$2"; shift ;;
    --config) export D3A_CONFIG="$(realpath "$2")"; shift ;;
    *) args+=("$1") ;;
  esac
  shift
done
export D3A_RESULTS_DIR="$DEMO/results" PI_CODING_AGENT_DIR="$DEMO/.pi-agent"
FOCUS=("the likelihood and the censoring model" "priors and shrinkage" "features and interactions"
       "the sampler and the parameterisation")

BASE=$(.venv/bin/python -c "
import contextlib, sys; sys.path.insert(0, 'skills/github-workflow/scripts')
from gitlib import champion_branch, ensure_experiment_branch
with contextlib.redirect_stdout(sys.stderr): ensure_experiment_branch()
print(champion_branch())")
EXP=$(.venv/bin/python -c 'from checks.settings import experiment_dir; print(experiment_dir())')

pids=()
for i in $(seq 1 "$N"); do
  w=w$i; wt=$DEMO/.worktrees/$w
  if [ ! -d "$wt" ]; then     # a worktree linked to the shared environments, tools, configs and results
    git worktree add -q --detach "$wt" "$BASE"
    for f in .venv .tools results config/autoresearch.toml config/endpoint.toml config/vllm.toml config/secrets.env; do
      ln -sfn "$DEMO/$f" "$wt/$REL/$f"
    done
  fi
  focus=${FOCUS[$(( (i - 1) % ${#FOCUS[@]} ))]}
  prompt="Read AGENTS.md and program.md, then run the program. You are worker $w of $N agents working in parallel on
this experiment; everyone's results are in results.tsv, so check checks/status.py before each hypothesis and do not
repeat what another worker tried. Your focus: $focus."
  (cd "$wt/$REL" && D3A_WORKER=$w D3A_SYNC=0 bash scripts/run_pi.sh --print "${args[@]}" --prompt "$prompt") \
    > "$EXP/worker-$w.log" 2>&1 &
  pids+=($!)
  echo "$w: $focus (log $EXP/worker-$w.log)"
  sleep 20     # staggered, so the first branches and evaluations do not collide
done

SYNC=(uv run --quiet --group tracking python scripts/wandb_sync.py)
"${SYNC[@]}" --watch 60 >> "$EXP/wandb_sync.log" 2>&1 &
watch=$!
wait "${pids[@]}" || true
kill $watch || true
"${SYNC[@]}" >> "$EXP/wandb_sync.log" 2>&1 || true
.venv/bin/python skills/github-workflow/scripts/experiment_pr.py || true
.venv/bin/python checks/status.py --last $((N * 3))
