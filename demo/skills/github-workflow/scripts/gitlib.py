"""Small helpers shared by the github-workflow scripts (standard library only).

Branches, remote and experiment come from config/autoresearch.toml (see checks/settings.py).
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(DEMO))
from checks.settings import (branch_namespace, experiment_branch, experiment_dir, is_agent_branch,  # noqa: E402,F401
                             rel, run_dir, settings, slug)


def git(*args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=DEMO, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r.stdout.strip()


def git_ok(*args: str) -> bool:
    return subprocess.run(["git", *args], cwd=DEMO, capture_output=True).returncode == 0


def current_branch() -> str:
    return git("rev-parse", "--abbrev-ref", "HEAD")


def ensure_experiment_branch() -> str:
    """Create <prefix>/<experiment>/main from the base branch the first time an experiment is used."""
    main = experiment_branch()
    if not git("rev-parse", "--verify", "--quiet", main, check=False):
        git("branch", main, settings()["base_branch"])
        print(f"created the experiment branch {main} from {settings()['base_branch']}")
    return main


def champion_branch() -> str:
    """The best kept branch of this experiment so far, else the experiment branch."""
    f = experiment_dir() / "champion.json"
    if f.exists():
        b = json.loads(f.read_text())["branch"]
        if git("rev-parse", "--verify", "--quiet", b, check=False):
            return b
    return ensure_experiment_branch()


def metrics(name: str) -> tuple[Path, dict]:
    f = run_dir(name) / "metrics.json"
    if not f.exists():
        sys.exit(f"{rel(f)} not found: run checks/evaluate.py --name {name} first")
    return f, json.loads(f.read_text())


def secret(key: str) -> str | None:
    if os.environ.get(key):
        return os.environ[key]
    f = DEMO / "config" / "secrets.env"
    if f.exists():
        for line in f.read_text().splitlines():
            m = re.match(rf"\s*{key}\s*=\s*(.+?)\s*$", line)
            if m and not m.group(1).startswith("#"):
                return m.group(1).strip("'\"")
    return None


def owner_repo() -> tuple[str, str]:
    url = git("remote", "get-url", settings()["remote"])
    m = re.search(r"github\.com[:/](.+?)/(.+?)(?:\.git)?$", url)
    if not m:
        sys.exit(f"{settings()['remote']} is not a GitHub remote: {url}")
    return m.group(1), m.group(2)
