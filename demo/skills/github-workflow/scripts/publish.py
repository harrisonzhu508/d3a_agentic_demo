"""Publish a kept hypothesis: push its branch and open a pull request into the experiment branch (GitHub REST API).

    uv run python skills/github-workflow/scripts/publish.py --name <name> [--draft]

Needs the committed report from skills/mic-eval-harness/scripts/write_report.py; it becomes the pull-request
description (with the plots linked from the branch). The pull request goes into <prefix>/<experiment>/main, which
is pushed first if GitHub does not have it yet.

Needs a GitHub token with "Pull requests: write" (and "Issues: write" for the label) in GITHUB_TOKEN or
config/secrets.env. Without a token the branch is still pushed and the missing PR is recorded.
With git.push = false in config/autoresearch.toml (or D3A_NO_PUSH=1) nothing is pushed (rehearsals, tests).
With git.auto_merge = true the experiment branch is then fast-forwarded to the kept hypothesis (and pushed),
which merges the pull request; the next hypothesis starts from the experiment branch.
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gitlib import (DEMO, current_branch, ensure_experiment_branch, experiment_dir, git, git_ok,  # noqa: E402
                    is_agent_branch, metrics, owner_repo, push_remote, run_dir, secret, settings)

API = "https://api.github.com"


def request(method: str, url: str, token: str, payload: dict) -> dict:
    req = urllib.request.Request(url, method=method, data=json.dumps(payload).encode(), headers={
        "Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "d3a-autoresearch-demo"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as err:
        detail = err.read().decode()[:400]
        hint = ("\nThe GitHub token needs Contents: Read-only as well (check: uv run python scripts/check.py)."
                if "not all refs are readable" in detail else "")
        sys.exit(f"GitHub API {method} {url} failed: {err.code} {detail}{hint}")


def body(report: str, m: dict, owner: str, repo: str, report_dir: str) -> str:
    """The committed report, with its plots linked from the branch so they render in the pull request."""
    raw = f"https://github.com/{owner}/{repo}/blob/{m['branch']}/{report_dir}"
    text = re.sub(r"\]\(([\w.-]+\.png)\)", lambda g: f"]({raw}/{g.group(1)}?raw=true)", report)
    stacked = ""
    if m["parent"] != m["pr_base"]:
        pm = run_dir(m["parent"].rsplit("/", 1)[-1]) / "metrics.json"
        prev = json.loads(pm.read_text()).get("pr_url") if pm.exists() else None
        stacked = (f"> Builds on `{m['parent']}`" + (f" ({prev})" if prev else "") +
                   ", the previous kept hypothesis: merge that one first.\n\n")
    return (stacked + text + f"\n\nReport file: `{report_dir}/report.md` in this branch. "
            "_Opened by the autoresearch agent; a human decides whether to merge._")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--draft", action="store_true")
    args = ap.parse_args()
    mf, m = metrics(args.name)
    if m.get("decision") != "keep":
        sys.exit(f"'{args.name}' was not kept ({m.get('decision')}): use discard.py instead.")
    if current_branch() != m["branch"]:
        git("switch", m["branch"])
    if git("status", "--porcelain", "--", "src"):
        sys.exit("src/ has uncommitted changes: the kept result was for the committed code only.")
    if not is_agent_branch(m["branch"]):
        sys.exit(f"{m['branch']} is not under the agents' branch prefix ({settings()['branch_prefix']}/): not pushing.")
    report = m.get("report")
    if not report or not git("ls-files", "--", report):
        sys.exit(f"No committed report for '{args.name}': run "
                 f"uv run python skills/mic-eval-harness/scripts/write_report.py --name {args.name}")
    base = ensure_experiment_branch()
    if not settings()["push"]:
        m["pr_url"] = f"local-only (git.push = false; would open {m['branch']} -> {base})"
    else:
        remote = push_remote()
        if not git("ls-remote", "--heads", remote, base, check=False):
            git("push", remote, f"{base}:refs/heads/{base}")
        git("push", "-u", remote, m["branch"])
        token = secret("GITHUB_TOKEN")
        if not token:
            m["pr_url"] = f"none: pushed {m['branch']} but no GITHUB_TOKEN is configured (config/secrets.env)"
        else:
            owner, repo = owner_repo()
            prefix = git("rev-parse", "--show-prefix")
            report_dir = prefix + str(Path(report).parent)
            d = m["compare"]["elpd_diff"]
            title = f"{m['hypothesis'] or args.name}"[:90] + (f" (ΔELPD {d:+})" if d is not None else "")
            pr = request("POST", f"{API}/repos/{owner}/{repo}/pulls", token,
                         {"title": title, "head": m["branch"], "base": base, "draft": args.draft,
                          "body": body((DEMO / report).read_text(), m, owner, repo, report_dir)})
            m["pr_url"] = pr["html_url"]
            try:
                request("POST", f"{API}/repos/{owner}/{repo}/issues/{pr['number']}/labels", token,
                        {"labels": ["autoresearch", f"experiment:{m['experiment']}"]})
            except SystemExit:
                pass  # a missing label permission should not undo the PR
    if settings()["auto_merge"]:
        m["merged"] = merge(m, base)
    mf.write_text(json.dumps(m, indent=2))
    print(f"published {m['branch']} -> {base}: {m['pr_url']}"
          + (f"; merged into {base}" if m.get("merged") else ""))


def merge(m: dict, base: str) -> bool:
    """Fast-forward the experiment branch to the kept hypothesis (git.auto_merge = true)."""
    if not git_ok("merge-base", "--is-ancestor", base, m["branch"]):
        print(f"not merged: {base} has moved on since this branch started (merge the PR by hand)")
        return False
    if not git_ok("branch", "-f", base, m["branch"]):
        print(f"not merged: {base} is checked out in another worktree (merge the PR by hand)")
        return False
    if settings()["push"]:
        git("push", push_remote(), f"{base}:refs/heads/{base}")   # GitHub marks the PR as merged
    ch = experiment_dir() / "champion.json"
    if ch.exists():                        # new hypotheses now start from the experiment branch itself
        c = json.loads(ch.read_text())
        ch.write_text(json.dumps({**c, "branch": base, "merged_from": m["branch"]}, indent=2))
    return True


if __name__ == "__main__":
    main()
