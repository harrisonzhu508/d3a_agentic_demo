#!/usr/bin/env bash
# A local vLLM server for the agent, configured in config/vllm.toml (pi's "local" provider).
#
#   bash scripts/vllm.sh install    # once: vLLM into .tools/vllm (CUDA 12.9 build for NVIDIA drivers < 580)
#   bash scripts/vllm.sh download   # once: the model into the Hugging Face cache
#   bash scripts/vllm.sh start      # in the background; waits until it answers
#   bash scripts/vllm.sh stop | status | logs
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=.tools/vllm; LOG=.tools/vllm-server.log; PIDFILE=.tools/vllm-server.pid
[ -f config/vllm.toml ] || cp config/vllm.example.toml config/vllm.toml

# config/vllm.toml [server] as shell variables (MODEL, GPU, PORT, ...)
eval "$(uv run --quiet --no-project --python 3.12 python -c '
import shlex, tomllib
for k, v in tomllib.load(open("config/vllm.toml", "rb"))["server"].items():
    v = " ".join(map(str, v)) if isinstance(v, list) else str(v)
    print(k.upper() + "=" + shlex.quote(v))')"
URL="http://127.0.0.1:$PORT/v1"
[ -n "${HF_HOME:-}" ] && export HF_HOME
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"

case "${1:-status}" in
  install)
    cuda=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+')
    variant=$([ "$cuda" -ge 13 ] && echo cu130 || echo cu129)
    uv venv -q --allow-existing "$VENV" --python 3.12
    uv pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION" "transformers>=5.8" --index-strategy unsafe-best-match \
      --extra-index-url "https://wheels.vllm.ai/$VLLM_VERSION/$variant" --extra-index-url "https://download.pytorch.org/whl/$variant"
    ;;
  download) "$VENV/bin/hf" download "$MODEL" --exclude "*.md" ;;
  start)
    args=(serve "$MODEL" --served-model-name "$SERVED_NAME" --host "$HOST" --port "$PORT"
          --tensor-parallel-size "$TENSOR_PARALLEL" --max-model-len "$MAX_MODEL_LEN"
          --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-num-seqs "$MAX_NUM_SEQS"
          --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml)
    [ -n "$API_KEY" ] && args+=(--api-key "$API_KEY")
    # shellcheck disable=SC2206
    args+=($EXTRA_ARGS)
    nohup setsid "$VENV/bin/vllm" "${args[@]}" > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "starting vLLM on GPU $GPU (log $LOG); loading the model takes a few minutes"
    until curl -sf -m 5 "$URL/models" > /dev/null; do
      kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { tail -20 "$LOG"; exit 1; }
      sleep 5
    done
    echo "ready: $URL (model $SERVED_NAME)"
    ;;
  stop) kill -TERM -- "-$(cat "$PIDFILE")" && rm -f "$PIDFILE" && echo stopped ;;
  status) curl -sf -m 5 "$URL/models" > /dev/null && echo "up: $URL" || echo "not running"
          nvidia-smi --id="$GPU" --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader ;;
  logs) tail -f "$LOG" ;;
  *) sed -n '2,8p' "$0" ;;
esac
