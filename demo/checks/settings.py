"""The user's autoresearch settings (config/autoresearch.toml), shared by the harness, the hooks and the skill scripts.

    uv run python checks/settings.py      # print the resolved settings

config/autoresearch.toml is per machine and gitignored (created from config/autoresearch.example.toml), so it
does not change when the agent switches branches. Environment overrides, set by scripts/run_pi.sh:
  D3A_CONFIG            another settings file instead of config/autoresearch.toml (run_pi.sh --config FILE)
  D3A_EXPERIMENT, D3A_MAX_EXPERIMENTS, D3A_NO_PUSH=1, D3A_BUDGET_SECONDS (run_pi.sh --budget-seconds N)
  D3A_WORKER            parallel agents (scripts/parallel.sh): own session, feedback and branch names per worker
  D3A_RESULTS_DIR       results/ shared by all parallel workers
"""

import json
import os
import re
import tomllib
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
CONFIG = Path(os.environ.get("D3A_CONFIG") or DEMO / "config" / "autoresearch.toml")
EXAMPLE = DEMO / "config" / "autoresearch.example.toml"
RESULTS = Path(os.environ.get("D3A_RESULTS_DIR") or DEMO / "results")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", text.lower()).strip("-.") or "default"


def settings() -> dict:
    if os.environ.get("D3A_CONFIG") and not CONFIG.exists():
        raise SystemExit(f"D3A_CONFIG={CONFIG} does not exist")
    cfg = tomllib.loads((CONFIG if CONFIG.exists() else EXAMPLE).read_text())
    exp, git, tracking = cfg["experiment"], cfg["git"], cfg.get("tracking", {})
    budget = os.environ.get("D3A_BUDGET_SECONDS", cfg.get("evaluation", {}).get("budget_seconds"))
    return {
        "experiment": slug(os.environ.get("D3A_EXPERIMENT") or exp["name"]),
        "max_experiments": int(os.environ.get("D3A_MAX_EXPERIMENTS") or exp.get("max_experiments", 3)),
        "branch_prefix": git["branch_prefix"].strip("/"),
        "base_branch": git["base_branch"],
        "remote": git.get("remote", "origin"),
        "push": bool(git.get("push", True)) and os.environ.get("D3A_NO_PUSH") != "1",
        "auto_merge": bool(git.get("auto_merge", False)),
        "wandb_project": tracking.get("wandb_project", "") if tracking.get("enabled", True) else "",
        "worker": slug(os.environ["D3A_WORKER"]) if os.environ.get("D3A_WORKER") else "",
        # time limit for one evaluation: None = the harness default (checks/harness.json), 0 = no limit
        "budget_seconds": None if budget in (None, "") else int(budget),
    }


def experiment_dir() -> Path:
    """results/<experiment>/: results.tsv, champion.json, session.json, hooks.log and one folder per run."""
    d = RESULTS / settings()["experiment"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def rel(path) -> str:
    """A path as seen from demo/: results paths always read results/<experiment>/..., which is right in the main
    checkout and in every parallel worker (whose results/ links to the shared folder)."""
    p = Path(path)
    if p.is_relative_to(RESULTS):
        return "results/" + p.relative_to(RESULTS).as_posix()
    return p.relative_to(DEMO).as_posix() if p.is_relative_to(DEMO) else str(p)


def session_file() -> Path:
    """This agent session's budget and progress (one file per parallel worker)."""
    w = settings()["worker"]
    return experiment_dir() / (f"session.{w}.json" if w else "session.json")


def feedback_file(worker: str | None = None) -> Path:
    """Where scripts/feedback.sh leaves messages for this agent (one file per parallel worker)."""
    w = settings()["worker"] if worker is None else worker
    return experiment_dir() / (f"feedback.{w}.md" if w else "feedback.md")


def run_dir(name: str) -> Path:
    return experiment_dir() / name


def branch_namespace() -> str:
    """Where new hypothesis branches go: <branch_prefix>/<experiment>/."""
    s = settings()
    return f"{s['branch_prefix']}/{s['experiment']}/"


def experiment_branch() -> str:
    """The experiment's integration branch, <branch_prefix>/<experiment>/main: created from the base branch, it
    receives the pull requests of kept hypotheses; only a human merges into it."""
    return branch_namespace() + "main"


def is_agent_branch(branch: str) -> bool:
    """Agents may commit to and push only branches under the configured prefix."""
    return branch.startswith(settings()["branch_prefix"] + "/")


if __name__ == "__main__":
    print(json.dumps({**settings(), "results": rel(experiment_dir()),
                      "experiment_branch": experiment_branch(), "new_branches": branch_namespace() + "<slug>"}, indent=2))
