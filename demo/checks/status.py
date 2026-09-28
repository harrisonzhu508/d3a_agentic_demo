"""Where the research stands: the experiment, its current champion and the most recent hypotheses.

    uv run python checks/status.py [--last 10]
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks.settings import DEMO, branch_namespace, experiment_branch, experiment_dir, rel, settings  # noqa: E402


def summary(last: int = 10) -> str:
    s, exp = settings(), experiment_dir()
    lines = [f"Experiment '{s['experiment']}': results in {rel(exp)}/; hypothesis branches "
             f"{branch_namespace()}<slug> open pull requests into the experiment branch {experiment_branch()} "
             f"(agents commit and push only {s['branch_prefix']}/... hypothesis branches)."]
    ch = exp / "champion.json"
    best = DEMO / "checks" / "best.json"
    if ch.exists():
        c = json.loads(ch.read_text())
        lines.append(f"Champion: {c['branch']} ({c['name']}), ELPD(CV) {c['elpd_cv']}; new branches start from it.")
    elif best.exists():
        b = json.loads(best.read_text())
        lines.append(f"Champion: the baseline ({experiment_branch()}, from {s['base_branch']}), ELPD(CV) {b['elpd_cv']}.")
    else:
        lines.append("No baseline yet: run scripts/setup.sh.")
    res = exp / "results.tsv"
    if res.exists():
        rows = list(csv.DictReader(res.open(), delimiter="\t"))[-last:]
        if rows:
            lines.append(f"Last {len(rows)} hypotheses (decision | Δ ELPD ± SE | skill | hypothesis):")
            for r in rows:
                lines.append(f"- {r['decision']:>13} | {r['elpd_diff'] or '-':>7} ± {r['se_diff'] or '-':<6} | "
                             f"{r['skill'] or '-'} | {r['name']}: {r['hypothesis']}")
    else:
        lines.append("No hypotheses tested in this experiment yet.")
    return "\n".join(lines)


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--last") + 1]) if "--last" in sys.argv else 10
    print(summary(n))
