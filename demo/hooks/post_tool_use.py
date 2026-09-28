"""PostToolUse: after an edit to src/*.py, compile it and run the contract smoke test; failures go back to the model."""

import json
import py_compile
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEMO, WRITE_TOOLS, allow, block, read_event, resolve, session_file, under  # noqa: E402

GIVE_UP_HINT = 3   # consecutive failed checks before the agent is told to abandon the hypothesis


def failures(ok: bool) -> int:
    """Count consecutive failed post-edit checks in this session (reset by a passing check)."""
    f = session_file()
    s = json.loads(f.read_text()) if f.exists() else {}
    s["edit_failures"] = 0 if ok else s.get("edit_failures", 0) + 1
    if f.exists():
        f.write_text(json.dumps(s, indent=2))
    return s["edit_failures"]


def fail(ev: dict, message: str) -> None:
    n = failures(False)
    if n >= GIVE_UP_HINT:
        message += (f"\n\nThis is failed check {n} in a row. Unless the fix is now obvious, abandon this hypothesis "
                    "and try a different one: uv run python checks/evaluate.py --name <slug> --abandon \"<why>\", "
                    "then uv run python skills/github-workflow/scripts/discard.py --name <slug>.")
    block(ev, message)


def python() -> list[str]:
    venv = DEMO / ".venv" / "bin" / "python"
    return [str(venv)] if venv.exists() else ["uv", "run", "python"]


def main() -> None:
    ev = read_event()
    edited = [resolve(p) for p in ev["paths"] if ev["tool"] in WRITE_TOOLS]
    src = [p for p in edited if under(p, "src") and p.suffix == ".py"]
    if not src:
        allow(ev)
    for p in src:
        try:
            py_compile.compile(str(p), doraise=True)
        except py_compile.PyCompileError as err:
            fail(ev, f"Post-edit check: {p.relative_to(DEMO)} does not compile:\n{err.msg}")
    try:
        r = subprocess.run(python() + ["checks/smoke.py"], cwd=DEMO, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        fail(ev, "Post-edit check: checks/smoke.py took more than 180 s; the model is probably far too slow.")
    if r.returncode != 0:
        fail(ev, "Post-edit check failed (checks/smoke.py). Fix this before committing:\n" + r.stdout[-2500:])
    failures(True)
    allow(ev, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "smoke OK")


if __name__ == "__main__":
    main()
