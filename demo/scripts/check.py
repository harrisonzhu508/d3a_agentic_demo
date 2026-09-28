"""Check the model endpoint and the keys in config/secrets.env (keys are never printed).

    uv run python scripts/check.py

- the agent's model ([agent] in config/endpoint.toml): if it is an OpenAI-compatible endpoint (local vLLM,
  myserver), one chat request that must return a tool call;
- GITHUB_TOKEN: can read the repository's branches and open pull requests (probed without creating anything);
- OPENAI_API_KEY, DEEPSEEK_API_KEY: list the models; WANDB_API_KEY: who it belongs to.
"""

import base64
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO / "scripts"))
from render_configs import endpoints, load  # noqa: E402


def call(url: str, auth: str = "", payload: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload else None,
                                 headers={"Authorization": auth, "Content-Type": "application/json",
                                          "Accept": "application/json", "User-Agent": "d3a-agentic-demo"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as err:
        return err.code, {}
    except OSError as err:
        return 0, {"error": str(err)}


def endpoint() -> str:
    provider, model = load("endpoint")[0].get("agent", {}).get("model", "local/qwen3.8-27b").split("/", 1)
    e = endpoints().get(provider)
    if not e:
        return f"{provider}/{model}: built into pi (see its key below)"
    tool = {"type": "function", "function": {"name": "bash", "description": "Run a shell command", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
    s, r = call(e["base_url"] + "/chat/completions", f"Bearer {e.get('api_key', 'EMPTY')}",
                {"model": e["model"], "max_tokens": 200, "tools": [tool], "chat_template_kwargs": {"enable_thinking": False},
                 "messages": [{"role": "user", "content": "List the files in the current directory."}]})
    calls = (r.get("choices") or [{}])[0].get("message", {}).get("tool_calls") if s == 200 else None
    return (f"ok: {provider}/{model} at {e['base_url']} made a tool call" if calls else
            f"FAIL {s or r.get('error')}: {provider}/{model} at {e['base_url']}"
            + (" (local server not running? bash scripts/vllm.sh start)" if provider == "local" else ""))


def github(token: str) -> str:
    url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=DEMO, capture_output=True, text=True).stdout
    repo = "https://api.github.com/repos/" + re.search(r"github\.com[:/](.+?)(?:\.git)?$", url.strip()).group(1)
    if call(repo + "/branches?per_page=1", f"Bearer {token}")[0] != 200:
        return "FAIL: the token cannot read this repository (give it access, and Contents: Read-only)"
    s, _ = call(repo + "/pulls", f"Bearer {token}", {"title": "probe", "head": "no-such-branch", "base": "main"})
    return "ok: can open pull requests" if s == 422 else f"FAIL {s}: cannot open pull requests (Pull requests: write)"


def models(url: str, token: str) -> str:
    s, r = call(url, f"Bearer {token}")
    return f"ok: {len(r.get('data', []))} models" if s == 200 else f"FAIL {s}"


def wandb(token: str) -> str:
    s, r = call("https://api.wandb.ai/graphql", "Basic " + base64.b64encode(f"api:{token}".encode()).decode(),
                {"query": "{ viewer { username } }"})
    v = (r.get("data") or {}).get("viewer") if s == 200 else None
    return f"ok: user {v['username']}" if v else f"FAIL {s}"


def main() -> None:
    f = DEMO / "config" / "secrets.env"
    keys = dict(re.findall(r"^([A-Z_]+)=(.+)$", f.read_text(), flags=re.M)) if f.exists() else {}
    print(f"{'model':17s} {endpoint()}")
    checks = {"GITHUB_TOKEN": github, "OPENAI_API_KEY": lambda k: models("https://api.openai.com/v1/models", k),
              "DEEPSEEK_API_KEY": lambda k: models("https://api.deepseek.com/models", k), "WANDB_API_KEY": wandb}
    for name, check in checks.items():
        print(f"{name:17s} {check(keys[name].strip()) if keys.get(name, '').strip() else 'not set'}")


if __name__ == "__main__":
    main()
