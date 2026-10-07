"""Open or update the experiment's pull request: <prefix>/<experiment>/main into the base branch (e.g. main).

    uv run python skills/github-workflow/scripts/experiment_pr.py

This is where an experiment converges: the experiment branch collects the kept hypotheses (merged by you, or
by publish.py with git.auto_merge = true), and this pull request proposes the result for the base branch. Its
description lists every hypothesis tried (results.tsv), the kept ones with their pull requests, and the ELPD
from baseline to champion. run_pi.sh and parallel.sh run it when a session ends; it does nothing until the
experiment branch holds at least one merged improvement. Merging it into the base is always a human decision.
"""

import csv
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gitlib import (DEMO, experiment_branch, experiment_dir, git, owner_repo, push_remote, run_dir,  # noqa: E402
                    secret, settings)
from publish import API, request  # noqa: E402


def get(url: str, token: str) -> list | dict:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}",
                                               "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "d3a-autoresearch-demo"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or b"[]")


def body(exp: str) -> tuple[str, int]:
    rows = list(csv.DictReader((experiment_dir() / "results.tsv").open(), delimiter="\t")) \
        if (experiment_dir() / "results.tsv").exists() else []
    best = json.loads((DEMO / "checks" / "best.json").read_text()).get("elpd_cv") if (DEMO / "checks" / "best.json").exists() else None
    ch = experiment_dir() / "champion.json"
    champ = json.loads(ch.read_text()) if ch.exists() else {}
    lines, kept = [], 0
    for r in rows:
        mf = run_dir(r["name"]) / "metrics.json"
        m = json.loads(mf.read_text()) if mf.exists() else {}
        pr = m.get("pr_url") or ""
        link = f"[PR]({pr})" if pr.startswith("http") else ""
        kept += r["decision"] == "keep"
        diff = f"{r['elpd_diff']} ± {r['se_diff']}" if r.get("elpd_diff") else "–"
        lines.append(f"| {r['name']} | **{r['decision']}** | {diff} | {r.get('skill') or '-'} | {r['hypothesis'][:90]} | {link} |")
    project = settings()["wandb_project"]
    sync = experiment_dir() / "wandb_sync.json"
    if project and sync.exists():       # wandb_sync.py records it with its entity
        project = json.loads(sync.read_text()).get("project", project)
    text = (f"Autoresearch experiment `{exp}`: {len(rows)} hypotheses tried, {kept} kept.\n\n"
            f"- ELPD (5-fold CV on the development set): baseline {best} → champion {champ.get('elpd_cv', best)}"
            + (f" (`{champ.get('merged_from') or champ.get('branch')}`)" if champ else "") + "\n"
            + (f"- Runs and every agent step: https://wandb.ai/{project} (experiment `{exp}`)\n" if "/" in project else "")
            + "- Discarded hypotheses are kept as branches under `.../discarded/`.\n\n"
            "| Hypothesis | Verdict | ΔELPD ± SE | Skill | What | PR |\n|---|---|---|---|---|---|\n"
            + "\n".join(lines)
            + "\n\n_Opened by the autoresearch harness. Merging into the base branch is a human decision; run the locked "
              "test set (checks/final_test.py) first._")
    return text, kept


def main() -> None:
    s = settings()
    exp_branch, base = experiment_branch(), s["base_branch"]
    token = secret("GITHUB_TOKEN")
    if not s["push"] or not token:
        print("experiment PR: skipped (git.push is off or no GITHUB_TOKEN)")
        return
    remote = push_remote()
    git("fetch", "-q", remote, base, check=False)
    if not git("ls-remote", "--heads", remote, exp_branch, check=False):
        print(f"experiment PR: {exp_branch} is not on GitHub yet (no hypothesis published)")
        return
    git("fetch", "-q", remote, exp_branch, check=False)
    ahead = git("rev-list", "--count", f"{remote}/{base}..{remote}/{exp_branch}", check=False)
    if ahead in ("", "0"):
        print(f"experiment PR: {exp_branch} has nothing beyond {base} yet (merge a kept PR, or set git.auto_merge)")
        return
    owner, repo = owner_repo()
    text, kept = body(s["experiment"])
    champ = experiment_dir() / "champion.json"
    elpd = json.loads(champ.read_text()).get("elpd_cv") if champ.exists() else None
    title = f"[autoresearch] {s['experiment']}: {kept} kept hypothes{'i' if kept == 1 else 'e'}s" + (f", ELPD {elpd}" if elpd else "")
    open_prs = get(f"{API}/repos/{owner}/{repo}/pulls?state=open&head={owner}:{exp_branch}&base={base}", token)
    if open_prs:
        pr = request("PATCH", f"{API}/repos/{owner}/{repo}/pulls/{open_prs[0]['number']}", token, {"title": title, "body": text})
        print(f"experiment PR updated: {pr['html_url']}")
    else:
        pr = request("POST", f"{API}/repos/{owner}/{repo}/pulls", token,
                     {"title": title, "head": exp_branch, "base": base, "body": text})
        try:
            request("POST", f"{API}/repos/{owner}/{repo}/issues/{pr['number']}/labels", token,
                    {"labels": ["autoresearch", f"experiment:{s['experiment']}"]})
        except SystemExit:
            pass
        print(f"experiment PR opened: {pr['html_url']}")


if __name__ == "__main__":
    main()
