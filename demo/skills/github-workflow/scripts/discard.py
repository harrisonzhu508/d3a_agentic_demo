"""Discard a hypothesis: restore src/, go back to the champion, and archive the hypothesis branch as
<prefix>/<experiment>/discarded/<slug> (pushed, so every hypothesis stays visible on GitHub; a branch with no
commit of its own is simply deleted).

    uv run python skills/github-workflow/scripts/discard.py --name <name>
The result stays in results/<experiment>/ (results.tsv and <name>/), so the idea is recorded and not repeated.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gitlib import (branch_namespace, champion_branch, current_branch, experiment_branch, git,  # noqa: E402
                    is_agent_branch, metrics, settings)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    mf, m = metrics(args.name)
    if m.get("decision") == "keep":
        sys.exit(f"'{args.name}' was kept: publish it with publish.py instead.")
    branch = m.get("branch", "")
    if git("status", "--porcelain", "--", "src"):
        git("restore", "--staged", "--worktree", "--", "src")
    back = champion_branch()
    if settings()["worker"]:
        git("switch", "--detach", back)   # parallel agents: the champion may be checked out by another worker
    elif current_branch() != back:
        git("switch", back)
    note = ""
    if is_agent_branch(branch) and branch not in (back, experiment_branch()) \
            and git("rev-parse", "--verify", "--quiet", branch, check=False):
        if git("rev-list", "--count", f"{back}..{branch}", check=False) not in ("", "0"):
            ns = branch_namespace()
            archived = ns + "discarded/" + branch.removeprefix(ns)
            git("branch", "-M", branch, archived)
            if settings()["push"]:   # keep the record on GitHub; a network problem must not block the discard
                git("push", "-q", settings()["remote"], f"{archived}:refs/heads/{archived}", check=False)
            m["archived_branch"] = archived
            note = f"; branch kept as {archived}"
        else:
            git("branch", "-D", branch)
    m["reverted"] = True
    mf.write_text(json.dumps(m, indent=2))
    print(f"discarded '{args.name}' ({m.get('decision')}); back on {back}{note}")


if __name__ == "__main__":
    main()
