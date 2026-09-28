"""Stop: the agent may only finish when its work is complete and the experiment budget is used up.

This is what keeps the autoresearch loop going (Karpathy's "never stop"), with a budget: the number of
experiments per session (results/<experiment>/session.json; set in config/autoresearch.toml).
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEMO, allow, block, experiment_dir, read_event, session_file, settings  # noqa: E402

GIVE_UP_AFTER = 6   # identical consecutive blocks before we let it stop (avoids an endless loop)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=DEMO, capture_output=True, text=True).stdout.strip()


def load_session(harness: str) -> dict:
    f = session_file()
    if f.exists():
        return json.loads(f.read_text())
    return {"harness": harness, "max_experiments": settings()["max_experiments"], "experiments": []}


def reason_to_continue(s: dict) -> str | None:
    if git("status", "--porcelain", "--", "src"):
        return ("You have uncommitted changes in src/. Either commit them (git add src && git commit -m "
                "'hyp: <change> [skill: <name>]') and run checks/evaluate.py, or restore them (git restore src).")
    exps = s.get("experiments", [])
    if not exps and s.get("max_experiments", 3) > 0:
        return ("No experiment has been evaluated in this session yet. Follow program.md: new hypothesis branch, "
                "read the relevant skill, change src/, commit, then run checks/evaluate.py.")
    if not exps:
        return None
    name = exps[-1]
    mf = experiment_dir() / name / "metrics.json"
    m = json.loads(mf.read_text()) if mf.exists() else {}
    decision = m.get("decision")
    if decision == "keep" and not m.get("pr_url"):
        return (f"Experiment '{name}' was kept but not published. Write its report: "
                f"uv run python skills/mic-eval-harness/scripts/write_report.py --name {name} (fill the draft it "
                f"creates, run it again), then: uv run python skills/github-workflow/scripts/publish.py --name {name}")
    if decision not in ("keep", "baseline") and not m.get("reverted"):
        return (f"Experiment '{name}' was not kept ({decision}). Record it and go back to the champion: "
                f"python skills/github-workflow/scripts/discard.py --name {name}")
    done, budget = len(exps), s.get("max_experiments", 3)
    if done < budget:
        return (f"Experiment {done}/{budget} finished ({decision}). Start the next one: run "
                "python checks/status.py, pick ONE new hypothesis you have not tried, read the skill it needs, "
                "and follow program.md again.")
    return None


def main() -> None:
    ev = read_event()
    s = load_session(ev["harness"])
    reason = reason_to_continue(s)
    if reason is None:
        allow(ev, f"session complete: {len(s.get('experiments', []))} experiments")
    streak = s.get("stop_streak", {})
    count = streak.get("count", 0) + 1 if streak.get("reason") == reason else 1
    s["stop_streak"] = {"reason": reason, "count": count}
    session_file().write_text(json.dumps(s, indent=2))
    if count > GIVE_UP_AFTER:
        allow(ev, f"gave up after {GIVE_UP_AFTER} identical stop blocks: {reason[:120]}")
    block(ev, "Not done yet (autoresearch stop hook). " + reason)


if __name__ == "__main__":
    main()
