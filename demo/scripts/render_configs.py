"""Render pi's project-local config (.pi-agent/models.json and settings.json) from config/endpoint.toml.

    uv run python scripts/render_configs.py
    uv run python scripts/render_configs.py --endpoint local    # print "base_url model api_key" for one endpoint

pi reads models.json only from its agent directory; scripts/run_pi.sh points PI_CODING_AGENT_DIR at .pi-agent/.
models.json holds the OpenAI-compatible endpoints: "local" (the vLLM server of scripts/vllm.sh, from
config/vllm.toml) and each [endpoints.<name>] of config/endpoint.toml. Anthropic, OpenAI and DeepSeek are built
into pi (keys come from config/secrets.env via scripts/run_pi.sh). settings.json sets the default model from [agent].
"""

import json
import sys
import tomllib
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
QWEN_COMPAT = {"supportsDeveloperRole": False, "supportsStore": False, "supportsReasoningEffort": False,
               "maxTokensField": "max_tokens", "thinkingFormat": "qwen-chat-template"}


def load(name: str) -> tuple[dict, Path]:
    f = DEMO / "config" / f"{name}.toml"
    if not f.exists():
        f = DEMO / "config" / f"{name}.example.toml"
    return tomllib.loads(f.read_text()), f


def endpoints() -> dict[str, dict]:
    """name -> {base_url, model, api_key, context_window, max_output_tokens} for every OpenAI-compatible endpoint."""
    s = load("vllm")[0]["server"]
    host = "127.0.0.1" if s["host"] == "0.0.0.0" else s["host"]
    out = {"local": {"base_url": f"http://{host}:{s['port']}/v1", "model": s["served_name"],
                     "api_key": s.get("api_key") or "EMPTY", "context_window": s["max_model_len"],
                     "max_output_tokens": s["max_output_tokens"]}}
    cfg = load("endpoint")[0]
    out.update(cfg.get("endpoints", {}))
    if "endpoint" in cfg:                                   # older single-endpoint layout
        e = cfg["endpoint"]
        out[e["provider"]] = e
    return out


def main() -> None:
    eps = endpoints()
    if "--endpoint" in sys.argv:
        e = eps[sys.argv[sys.argv.index("--endpoint") + 1]]
        print(e["base_url"], e["model"], e.get("api_key", "EMPTY"))
        return
    cfg, cfg_file = load("endpoint")
    agent = cfg.get("agent", {})
    provider, model = agent.get("model", "local/qwen3.8-27b").split("/", 1)
    thinking = agent.get("thinking", "off")

    models = {"providers": {name: {
        "name": f"{name} ({e['model']})", "baseUrl": e["base_url"], "api": "openai-completions",
        "apiKey": e.get("api_key", "EMPTY"), "compat": QWEN_COMPAT,
        "models": [{"id": e["model"], "name": e["model"], "reasoning": True, "input": ["text"],
                    "contextWindow": int(e["context_window"]), "maxTokens": int(e["max_output_tokens"]),
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}],
    } for name, e in eps.items()}}
    settings = {
        "defaultProvider": provider, "defaultModel": model, "defaultThinkingLevel": thinking,
        "defaultTools": ["read", "bash", "edit", "write", "grep", "find", "ls"],
        "defaultProjectTrust": "always",
        "compaction": {"enabled": True, "reserveTokens": 16384, "keepRecentTokens": 20000},
    }
    agent_dir = DEMO / ".pi-agent"
    agent_dir.mkdir(exist_ok=True)
    (agent_dir / "models.json").write_text(json.dumps(models, indent=2))
    (agent_dir / "settings.json").write_text(json.dumps(settings, indent=2))
    if provider in eps:
        where = f"at {eps[provider]['base_url']}"
    else:
        key = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
               "deepseek": "DEEPSEEK_API_KEY"}.get(provider, "its API key")
        where = f"(built into pi; needs {key} in config/secrets.env)"
    print(f"wrote .pi-agent/models.json ({', '.join(eps)}) and settings.json: agent model {provider}/{model} "
          f"{where}, thinking {thinking} (from {cfg_file.relative_to(DEMO)})")


if __name__ == "__main__":
    main()
