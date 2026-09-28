"""Send an experiment to Weights & Biases, so every run and every agent step can be audited there.

    uv run --group tracking python scripts/wandb_sync.py              # sync once (safe to repeat)
    uv run --group tracking python scripts/wandb_sync.py --watch 60   # keep syncing (run_pi.sh does this)
    uv run --group tracking python scripts/wandb_sync.py --dry-run    # show what would be sent

Project: [tracking] wandb_project in config/autoresearch.toml; key: WANDB_API_KEY in config/secrets.env.
- Weave Agents view (Conversation SDK): agent "pi-autoresearch"; each pi session of the experiment is a
  conversation; each model call is a turn with an LLM span (the messages that led to it, the reply, thinking,
  token usage) and one span per tool call (arguments, full result including hook blocks, timing). Original
  timestamps are kept, so the whole trajectory can be audited, one conversation per agent (and per worker).
- W&B runs: one per scored hypothesis (group = experiment): its hypothesis, skill, branch, commit, verdict and
  metrics, the review and comparison plots, metrics.json, pointwise.csv and the report; plus one summary run
  with results.tsv as a table.
Progress is kept in results/<experiment>/wandb_sync.json, so each item is sent once.
"""

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
from checks.settings import experiment_dir, settings  # noqa: E402

NS = uuid.UUID("3b8a7c0e-5f0d-4c3e-9a55-0d3a2026d3a0")   # deterministic ids for Weave calls
os.environ.setdefault("WEAVE_PRINT_CALL_LINK", "false")   # one link per call floods the sync log


def secret(key: str) -> str:
    if os.environ.get(key):
        return os.environ[key]
    f = DEMO / "config" / "secrets.env"
    for line in (f.read_text().splitlines() if f.exists() else []):
        m = re.match(rf"\s*{key}\s*=\s*(.+?)\s*$", line)
        if m:
            return m.group(1).strip("'\"")
    return ""


def ts(value) -> dt.datetime:
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(value / 1000, dt.timezone.utc)
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))


# ------------------------------------------------------------------ pi sessions -> turns
def session_files() -> list[Path]:
    agent_dir = Path(os.environ.get("PI_CODING_AGENT_DIR") or DEMO / ".pi-agent")
    return sorted(agent_dir.glob("sessions/*/*.jsonl"))


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content or [] if b.get("type") == "text")


def load_session(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    header = rows[0] if rows and rows[0].get("type") == "session" else {}
    entries = [r for r in rows if r.get("type") in ("message", "custom_message")]
    exp = None
    for r in entries:   # the session-start context names the experiment
        m = re.search(r"Experiment '([^']+)'", text_of(r.get("content")) or text_of((r.get("message") or {}).get("content")))
        if m:
            exp = m.group(1)
            break
    cwd = header.get("cwd", "")
    worker = re.search(r"/\.worktrees/(w\d+)/", cwd + "/")
    worker = worker.group(1) if worker else ""
    started = header.get("timestamp")
    name = f"{exp} · {worker or 'agent'} · {ts(started):%d %b %H:%M}" if started else f"{exp} · {worker or 'agent'}"
    return {"id": header.get("id", path.stem), "cwd": cwd, "experiment": exp, "worker": worker, "name": name,
            "started": started, "entries": entries, "path": path}


def turns(session: dict) -> list[dict]:
    """Model turns: the messages that led to each assistant reply, the reply, and its tool calls with results."""
    results, out, pending, system, last_ts = {}, [], [], None, session["started"]
    for r in session["entries"]:
        m = r.get("message") or {}
        if m.get("role") == "toolResult":
            results[m["toolCallId"]] = r
    for r in session["entries"]:
        m = r.get("message") or {}
        role = m.get("role") or r.get("type")
        if role == "system":
            system = m.get("sections") or m.get("content")
        elif role == "assistant":
            calls = [b for b in m.get("content", []) if b.get("type") == "toolCall"]
            out.append({"entry": r, "inputs": pending, "system": system if not out else None, "started": last_ts,
                        "calls": [(c, results.get(c["id"])) for c in calls]})
            pending, last_ts = [], r["timestamp"]
        elif role in ("user", "custom", "custom_message"):
            pending.append({"role": "user" if role == "user" else f"harness:{r.get('customType') or m.get('customType')}",
                            "text": text_of(m.get("content") if m else r.get("content")), "time": r["timestamp"]})
        if role in ("toolResult", "user", "custom", "custom_message"):
            last_ts = r["timestamp"]
    return out


def system_text(system) -> str:
    if isinstance(system, dict):
        return "\n\n".join(f"## {k}\n{v}" for k, v in system.items() if isinstance(v, str) and v)
    return str(system or "")


def send_conversation(session: dict, done: set, dry: bool) -> int:
    """One pi session as a Weave conversation (Agents view): each model call is a turn, with an LLM span and one
    span per tool call it made."""
    from weave.conversation import LLM, Message, Reasoning, TextPart, Tool, ToolCallPart, Usage, log_turn
    sent = 0
    for n, t in enumerate(turns(session), 1):
        a = t["entry"]
        if a["id"] in done or any(res is None for _, res in t["calls"]):
            continue           # already sent, or its tool results are not in yet
        m = a["message"]
        blocks = m.get("content", [])
        names = [c["name"] for c, _ in t["calls"]]
        if dry:
            print(f"  turn {n}: {', '.join(names) or 'reply'} ({len(t['inputs'])} new input messages)")
            done.add(a["id"]); sent += 1
            continue
        started, ended = ts(t["started"]), ts(a["timestamp"])
        inputs = [Message(role="user", content=(x["text"] if x["role"] == "user" else f"[{x['role']}] {x['text']}"))
                  for x in t["inputs"]]
        text = text_of(blocks)
        thinking = "\n".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
        output = [Message(role="assistant", parts=([TextPart(content=text)] if text else []) + [
            ToolCallPart(id=c["id"], name=c["name"], arguments=json.dumps(c.get("arguments"))) for c, _ in t["calls"]])]
        u = m.get("usage") or {}
        system = [system_text(t["system"])] if t["system"] else []
        model = f"{m.get('provider')}/{m.get('model')}"
        llm = LLM(model=m.get("model", ""), provider_name=m.get("provider", ""), response_id=m.get("responseId", ""),
                  usage=Usage(input_tokens=u.get("input", 0), output_tokens=u.get("output", 0),
                              total_tokens=u.get("totalTokens"), reasoning_tokens=u.get("reasoning", 0),
                              cache_read_input_tokens=u.get("cacheRead", 0)),
                  **({"reasoning": Reasoning(content=thinking)} if thinking else {}),
                  finish_reasons=[str(m.get("stopReason", ""))], system_instructions=system,
                  input_messages=inputs, output_messages=output, started_at=started, ended_at=ended)
        tools = []
        for c, res in t["calls"]:
            rm = res["message"]
            result = text_of(rm.get("content"))
            tools.append(Tool(name=c["name"], arguments=json.dumps(c.get("arguments")), tool_call_id=c["id"],
                              result=json.dumps(("[error] " if rm.get("isError") else "") + result),
                              started_at=ended, ended_at=ts(res["timestamp"])))
        log_turn(conversation_id=session["id"], conversation_name=session["name"], agent_name="pi-autoresearch",
                 model=model, messages=inputs, output_messages=output, system_instructions=system,
                 spans=[llm, *tools], started_at=started, ended_at=tools[-1].ended_at if tools else ended,
                 attributes={"d3a.experiment": session["experiment"], "d3a.worker": session["worker"],
                             "d3a.turn": n, "d3a.tools": ", ".join(names) or "reply"})
        done.add(a["id"])
        sent += 1
    return sent


def delete_legacy_calls(client) -> int:
    """Remove the flat pi.turn / pi.tool.* calls written by the first version of this script."""
    ids = [c.id for c in client.get_calls(columns=["id", "op_name"])
           if "/op/pi.turn:" in (c.op_name or "") or "/op/pi.tool." in (c.op_name or "")]
    for i in range(0, len(ids), 100):
        client.delete_calls(ids[i:i + 100])
    return len(ids)


# ------------------------------------------------------------------ hypotheses -> W&B runs
def settled(m: dict) -> bool:
    """Finished: a kept hypothesis is published, anything else is reverted."""
    if m.get("decision") in ("keep", "baseline"):
        return bool(m.get("pr_url"))
    return m.get("decision") is not None and bool(m.get("reverted"))


def ensure_comparison(run: Path) -> Path | None:
    png = run / "comparison.png"
    if not png.exists() and (run / "pointwise.csv").exists():
        subprocess.run([sys.executable, str(DEMO / "skills" / "mic-plots" / "scripts" / "compare.py"),
                        str(DEMO / "data" / "dev.csv"), str(run / "pointwise.csv"), str(run / "comparison")],
                       capture_output=True)
    return png if png.exists() else None


def send_run(entity: str, project: str, run: Path, m: dict, sessions: list[dict], dry: bool) -> str:
    exp = m.get("experiment") or run.parent.name
    when = ts(m["timestamp"]).replace(tzinfo=None)
    earlier = [s for s in sessions if s["started"] and ts(s["started"]).replace(tzinfo=None) <= when
               and (not m.get("worker") or s["cwd"].endswith(f"/{m['worker']}/demo"))]
    thread = max(earlier, key=lambda s: s["started"])["id"] if earlier else None   # the session that ran it
    config = {k: m.get(k) for k in ("hypothesis", "skill", "branch", "commit", "parent", "pr_base", "harness",
                                    "worker", "decision", "pr_url", "merged", "n_features", "latent_sites")}
    summary = {"elpd_cv": m.get("elpd_cv"), "elpd_cv_se": m.get("elpd_cv_se"), "within1": m.get("within1"),
               "coverage90": m.get("coverage90"), "parsimony": m.get("parsimony"), "runtime_s": m.get("runtime_s"),
               **{f"compare/{k}": v for k, v in (m.get("compare") or {}).items() if k != "reference"},
               **{f"gates/{k}": v for k, v in (m.get("gates") or {}).items()}}
    if dry:
        print(f"  run {exp}/{run.name}: {m.get('decision')} ΔELPD {summary.get('compare/elpd_diff')} (thread {thread})")
        return "dry-run"
    import wandb
    rid = re.sub(r"[^a-zA-Z0-9_-]", "-", f"{exp}-{run.name}")[:64]
    w = wandb.init(entity=entity, project=project, id=rid, resume="allow", name=f"{exp}/{run.name}", group=exp,
                   job_type="hypothesis", tags=[t for t in (m.get("decision"), m.get("skill"), m.get("worker")) if t],
                   config={**config, "experiment": exp, "weave_conversation": thread}, dir=str(experiment_dir()),
                   settings=wandb.Settings(silent=True))
    w.summary.update({k: v for k, v in summary.items() if v is not None})
    images = {}
    report_dir = DEMO / "reports" / exp / run.name
    for name in ("review", "comparison"):
        png = report_dir / f"{name}.png" if (report_dir / f"{name}.png").exists() else \
            (ensure_comparison(run) if name == "comparison" else run / "review.png")
        if png and png.exists():
            images[name] = wandb.Image(str(png))
    if images:
        w.log(images)
    report = report_dir / "report.md"
    if not report.exists() and m.get("report"):   # committed on the hypothesis branch (a parallel worker's worktree)
        here = subprocess.run(["git", "rev-parse", "--show-prefix"], cwd=DEMO, capture_output=True, text=True).stdout.strip()
        r = subprocess.run(["git", "show", f"{m['branch']}:{here}{m['report']}"], cwd=DEMO, capture_output=True, text=True)
        if r.returncode == 0:
            report = run / "report.md"
            report.write_text(r.stdout)
    for f in [run / "metrics.json", run / "pointwise.csv", report, run / "report_draft.md"]:
        if f.exists():
            w.save(str(f), base_path=str(f.parent), policy="now")
    url = w.url
    w.finish()
    return url


def send_summary(entity: str, project: str, exp_dir: Path, dry: bool) -> None:
    tsv = exp_dir / "results.tsv"
    if dry or not tsv.exists():
        return
    import csv
    import wandb
    rows = list(csv.DictReader(tsv.open(), delimiter="\t"))
    w = wandb.init(entity=entity, project=project, id=re.sub(r"[^a-zA-Z0-9_-]", "-", f"{exp_dir.name}-summary")[:64],
                   resume="allow", name=f"{exp_dir.name}/summary", group=exp_dir.name, job_type="summary",
                   dir=str(exp_dir), settings=wandb.Settings(silent=True))
    if rows:
        w.log({"results": wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows])})
    w.finish()


# ------------------------------------------------------------------ main
def sync(dry: bool) -> str:
    project = settings()["wandb_project"]
    if not project:
        return "tracking is off ([tracking] in config/autoresearch.toml)"
    if not dry and not secret("WANDB_API_KEY"):
        return "no WANDB_API_KEY in config/secrets.env"
    os.environ["WANDB_API_KEY"] = secret("WANDB_API_KEY")
    entity, name = project.split("/", 1)
    exp_dir = experiment_dir()
    state_file = exp_dir / "wandb_sync.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    state.setdefault("steps", {}), state.setdefault("runs", {})
    exp = settings()["experiment"]
    sessions = [s for s in map(load_session, session_files()) if s["experiment"] == exp]

    n_turns = 0
    client = None
    if not dry and ("turns" in state or any(any(t["entry"]["id"] not in state["steps"].get(s["id"], []) for t in turns(s))
                                          for s in sessions)):
        import weave
        client = weave.init(project)
        if "turns" in state:          # state of the first, flat format: clean it up once
            print(f"removed {delete_legacy_calls(client)} flat calls from the first sync format", flush=True)
            state.pop("turns")
    for s in sessions:
        done = set(state["steps"].get(s["id"], []))
        n_turns += send_conversation(s, done, dry)
        state["steps"][s["id"]] = sorted(done)
    if client is not None:
        from opentelemetry import trace
        trace.get_tracer_provider().force_flush()
        client.flush()

    n_runs = 0
    for run in sorted(p.parent for p in exp_dir.glob("*/metrics.json")):
        m = json.loads((run / "metrics.json").read_text())
        key = f"{run.name}:{m.get('decision')}:{m.get('pr_url')}"
        if run.name.startswith("explore-") or not settled(m) or state["runs"].get(run.name, {}).get("key") == key:
            continue
        state["runs"][run.name] = {"key": key, "url": send_run(entity, name, run, m, sessions, dry)}
        n_runs += 1
    if n_runs:
        send_summary(entity, name, exp_dir, dry)
    if not dry:
        state_file.write_text(json.dumps(state, indent=1))
    return (f"{'would send' if dry else 'sent'} {n_turns} agent turns from {len(sessions)} conversations and {n_runs} hypothesis "
            f"runs to https://wandb.ai/{project}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", type=int, default=0, help="repeat every N seconds until interrupted")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    while True:
        try:
            print(dt.datetime.now().strftime("%H:%M:%S"), sync(a.dry_run), flush=True)
        except Exception as err:   # a network hiccup must not stop the agent's run
            print(dt.datetime.now().strftime("%H:%M:%S"), f"sync failed: {type(err).__name__}: {err}", flush=True)
            if not a.watch:
                sys.exit(1)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
