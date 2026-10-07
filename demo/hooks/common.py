"""Shared code for the demo's hooks (standard library only; run with the project's .venv Python, 3.12).

Each hook reads one event, normalises it across harnesses, and exits 0 (allow) or 2 (block; message on
stderr, which Claude Code, Codex and the pi extension all pass back to the model).

Accepted payloads:
- Claude Code / Codex (JSON on stdin): {"hook_event_name", "tool_name", "tool_input": {...}, ...}
  Claude tools: Read/Edit/Write/MultiEdit (file_path), Bash (command), Grep/Glob (path, pattern).
  Codex tools: Bash (tool_input.command), apply_patch (tool_input.command holds the patch text).
- pi extension (--payload '<json>'): {"event", "tool": "read|bash|edit|write|grep|find|ls", "input": {...}}
"""

import datetime as dt
import json
import os
import re
import sys
from fnmatch import fnmatch
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
from checks.settings import (experiment_branch, experiment_dir, feedback_file, is_agent_branch,  # noqa: E402,F401
                             session_file, settings)

POLICY = json.loads((DEMO / "hooks" / "policy.json").read_text())

WRITE_TOOLS = {"edit", "write", "multiedit", "notebookedit", "apply_patch"}
READ_TOOLS = {"read", "grep", "glob", "find", "ls"}


def read_event(argv: list[str] | None = None) -> dict:
    argv = sys.argv[1:] if argv is None else argv
    harness = "unknown"
    if "--harness" in argv:
        harness = argv[argv.index("--harness") + 1]
    if "--payload" in argv:
        raw = argv[argv.index("--payload") + 1]
    else:
        raw = sys.stdin.read() if not sys.stdin.isatty() else "{}"
    try:
        ev = json.loads(raw or "{}")
    except json.JSONDecodeError:
        ev = {}
    return normalise(ev, harness)


def patch_paths(patch: str) -> list[str]:
    """File paths touched by a Codex apply_patch patch (*** Add/Update/Delete File: ..., *** Move to: ...)."""
    return [m.group(1).strip() for m in re.finditer(
        r"^\*\*\* (?:(?:Add|Update|Delete) File|Move to): (.+)$", patch, flags=re.M)]


def normalise(ev: dict, harness: str) -> dict:
    if "hook_event_name" in ev:                      # Claude Code or Codex
        tool = str(ev.get("tool_name", "")).lower()
        ti = ev.get("tool_input") or {}
        paths, command = [], ""
        if tool == "apply_patch":
            paths = patch_paths(ti.get("command", "") if isinstance(ti, dict) else str(ti))
        elif tool == "bash":
            command = ti.get("command", "")
        else:
            for key in ("file_path", "path", "notebook_path"):
                if ti.get(key):
                    paths.append(ti[key])
        return {"event": ev.get("hook_event_name", ""), "tool": tool, "paths": paths, "command": command,
                "harness": harness if harness != "unknown" else ("codex" if "turn_id" in ev else "claude"),
                "stop_hook_active": bool(ev.get("stop_hook_active")), "raw": ev}
    tool = str(ev.get("tool", "")).lower()           # pi extension
    inp = ev.get("input") or {}
    paths = [inp[k] for k in ("path", "file_path") if inp.get(k)]
    return {"event": ev.get("event", ""), "tool": tool, "paths": paths, "command": inp.get("command", ""),
            "harness": harness if harness != "unknown" else "pi", "stop_hook_active": False, "raw": ev}


def resolve(p: str) -> Path:
    """Absolute, normalised path (textually: a parallel worker's results/ is a symlink to the shared folder)."""
    q = Path(os.path.expanduser(p))
    return Path(os.path.normpath(q if q.is_absolute() else DEMO / q))


def under(p: Path, entry: str) -> bool:
    """p is entry or inside it; entries with * are glob patterns relative to demo/ (e.g. results/*/champion.json)."""
    if "*" in entry:
        return p.is_relative_to(DEMO) and fnmatch(p.relative_to(DEMO).as_posix(), entry)
    base = resolve(entry)
    return p == base or base in p.parents


def denied(paths: list[str], entries: list[str]) -> str | None:
    for raw in paths:
        p = resolve(raw)
        for e in entries:
            if under(p, e):
                return f"{raw} (protected: {e})"
    return None


def log(event: dict, decision: str, reason: str = "") -> None:
    """One JSON line per hook decision in results/<experiment>/hooks.log (tail -f it during the demo)."""
    rec = {"time": dt.datetime.now().isoformat(timespec="seconds"), "harness": event.get("harness"),
           **({"worker": settings()["worker"]} if settings()["worker"] else {}),
           "event": event.get("event"), "tool": event.get("tool"), "decision": decision, "reason": reason[:300],
           "target": (event.get("paths") or [event.get("command", "")[:120]])[0] if (event.get("paths") or event.get("command")) else ""}
    with (experiment_dir() / "hooks.log").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def block(event: dict, reason: str) -> None:
    log(event, "block", reason)
    print(reason, file=sys.stderr)
    sys.exit(2)


def allow(event: dict, note: str = "") -> None:
    log(event, "allow", note)
    sys.exit(0)
