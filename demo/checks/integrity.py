"""The referee must be the one a human committed: harness files have to match the base branch.

    uv run python checks/integrity.py      # exits 1 and lists the files that differ

There is no checksum file to maintain: git already knows what the harness should be. To change the harness,
commit the change to the base branch (humans only) before starting a campaign.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks.settings import DEMO, settings  # noqa: E402

PROTECTED = ["checks", "hooks", "data/dev.csv", "TASK.md", "program.md", "train.py",
             ":(exclude)checks/best.json"]   # best.json is the recorded baseline (evaluate.py --baseline)


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=DEMO, capture_output=True, text=True)


def verify() -> list[str]:
    base = settings()["base_branch"]
    diff = git("diff", "--name-only", base, "--", *PROTECTED)
    if diff.returncode != 0:
        return [f"cannot compare with the base branch '{base}': {diff.stderr.strip()}"]
    new = git("ls-files", "--others", "--exclude-standard", "--", *PROTECTED)
    return sorted(set(diff.stdout.split()) | set(new.stdout.split()))


if __name__ == "__main__":
    bad = verify()
    print(f"harness matches {settings()['base_branch']}" if not bad else "CHANGED: " + ", ".join(bad))
    sys.exit(1 if bad else 0)
