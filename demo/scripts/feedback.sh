#!/usr/bin/env bash
# Talk to running agents from another terminal (all sessions of the experiment, or one with --worker).
#
#   bash scripts/feedback.sh "Try an interaction between gyrA and parC next."
#   bash scripts/feedback.sh --stop            # finish the current hypothesis, then stop
#   bash scripts/feedback.sh --budget 5        # change the session's hypothesis budget
#   bash scripts/feedback.sh --worker w2 "..." # one parallel worker only
#
# In the pi TUI you can also type: Enter steers the next step, Alt+Enter queues a follow-up, Escape pauses.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python - "$@" <<'PY'
import json, subprocess, sys
from checks.settings import branch_namespace, experiment_branch, experiment_dir, feedback_file
exp, args = experiment_dir(), sys.argv[1:]
workers = [args[1]] if args[0] == "--worker" else None
args = args[2:] if workers else args
workers = workers or sorted(p.name[8:-5].lstrip(".") for p in exp.glob("session*.json")) or [""]
for w in workers:
    session = exp / (f"session.{w}.json" if w else "session.json")
    msg = " ".join(args)
    if args[0] in ("--stop", "--budget"):
        s = json.loads(session.read_text())
        done = len(s["experiments"])
        if args[0] == "--stop":   # a hypothesis in progress counts, so the stop hook makes the agent finish it
            branch = subprocess.run(["git", "-C", s["cwd"], "branch", "--show-current"], capture_output=True, text=True).stdout.strip()
            busy = branch.startswith(branch_namespace()) and branch != experiment_branch() \
                and branch.removeprefix(branch_namespace()) not in s["experiments"]
            s["max_experiments"] = done + busy
            msg = "Please finish the experiment you are on (evaluate it, then publish or discard it) and stop."
        else:
            s["max_experiments"] = int(args[1])
            msg = f"Your experiment budget for this session is now {s['max_experiments']}."
        session.write_text(json.dumps(s, indent=2))
    with open(feedback_file(w), "a") as fh:
        fh.write(msg + "\n")
    print(f"{w or 'agent'}: {msg}")
PY
