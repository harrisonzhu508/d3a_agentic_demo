#!/usr/bin/env bash
# One-time setup (safe to re-run). Needs uv, Node.js >= 22.19 and git.
#
#   bash scripts/setup.sh            # Python env, data split, pi, configs
#   bash scripts/setup.sh --vllm     # the same, plus vLLM and the model for a local server (scripts/vllm.sh)
#   bash scripts/setup.sh --configs  # only re-render pi's config after editing config/endpoint.toml or vllm.toml
set -euo pipefail
cd "$(dirname "$0")/.."

for f in autoresearch endpoint vllm; do [ -f "config/$f.toml" ] || cp "config/$f.example.toml" "config/$f.toml"; done
[ -f config/secrets.env ] || cp config/secrets.env.example config/secrets.env
if [ "${1:-}" != --configs ]; then
  uv sync
  [ -f "${D3A_PRIVATE_DIR:-$HOME/.d3a-demo-private}/test.csv" ] || uv run python checks/split.py
  [ -x .tools/node_modules/.bin/pi ] || npm install --prefix .tools --no-fund --no-audit @earendil-works/pi-coding-agent
fi
uv run python scripts/render_configs.py
if [ "${1:-}" = --vllm ]; then bash scripts/vllm.sh install && bash scripts/vllm.sh download; fi
echo "Next: fill in config/secrets.env, check with: uv run python scripts/check.py; run: bash scripts/run_pi.sh"
