"""Write the pull-request report for a hypothesis: a fixed structure, your text, the harness's numbers and plots.

    uv run python skills/mic-eval-harness/scripts/write_report.py --name <slug>

1st run: creates results/<experiment>/<slug>/report_draft.md from references/report-template.md, with four
         sections to write: Results, Data, Method, Conclusion (guidance in each).
2nd run (after you filled the draft): checks it, draws the comparison plot, renders
         reports/<experiment>/<slug>/report.md with the verdict, metrics table, plots and code diff, and commits
         that folder on your hypothesis branch. Then run github-workflow's publish.py to open the pull request.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(DEMO))
from checks.settings import experiment_branch, is_agent_branch, rel, run_dir, settings  # noqa: E402

TEMPLATE = Path(__file__).resolve().parents[1] / "references" / "report-template.md"
PLOTS = DEMO / "skills" / "mic-plots" / "scripts"
SECTIONS = ["Results", "Data", "Method", "Conclusion"]
MAX_WORDS = 250
PAPER = (0xFA, 0xFA, 0xF0)


def git(*args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=DEMO, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r.stdout.strip()


def parse(draft: str) -> dict[str, str]:
    text = re.sub(r"<!--.*?-->", "", draft, flags=re.S)
    parts = re.split(r"^##\s+(\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def problems(sections: dict[str, str]) -> list[str]:
    out = []
    for s in SECTIONS:
        body = sections.get(s, "")
        words = len(body.split())
        if not body:
            out.append(f"'## {s}' is missing or still empty")
        elif words > MAX_WORDS:
            out.append(f"'## {s}' has {words} words; keep it under {MAX_WORDS}")
        elif re.search(r"\bTODO\b|\{name\}|<fill", body, flags=re.I):
            out.append(f"'## {s}' still contains a placeholder")
    extra = [s for s in sections if s not in SECTIONS]
    if extra:
        out.append(f"unexpected sections {extra}: use exactly {SECTIONS}")
    return out


def on_paper(src: Path, dst: Path) -> None:
    """Flatten a transparent plot onto the GPAP paper colour, so it reads in light and dark GitHub themes."""
    from PIL import Image
    img = Image.open(src).convert("RGBA")
    bg = Image.new("RGBA", img.size, PAPER + (255,))
    Image.alpha_composite(bg, img).convert("RGB").save(dst, optimize=True)


def fmt(v, spec: str = "", pct: bool = False) -> str:
    if v is None:
        return "–"
    return f"{v:.1%}" if pct else format(v, spec)


def render(m: dict, text: dict[str, str], diffstat: str, diff: str, n_iso: int, has_review: bool) -> str:
    c, g, ref = m["compare"], m["gates"], m["compare"].get("reference") or {}
    diff_txt = (f"{c['elpd_diff']:+.2f} ± {c['se_diff']:.2f}" if c.get("elpd_diff") is not None else "–")
    verdict = (f"**{m['decision']}**: ΔELPD {diff_txt} against {c['against']}, gates "
               f"{'pass' if g['passed'] else 'FAIL'} (rule: keep if ΔELPD > 2 SE and every gate passes)")
    rows = [
        ("ELPD (cross-validated)", f"{m['elpd_cv']:.2f} ± {m['elpd_cv_se']:.2f}", fmt(ref.get("elpd_cv"), ".2f"), diff_txt),
        ("Within ±1 dilution", fmt(m["within1"], pct=True), fmt(ref.get("within1"), pct=True), ""),
        ("90 % interval coverage", fmt(m["coverage90"], pct=True), fmt(ref.get("coverage90"), pct=True), ""),
        ("Non-null effects (parsimony)", fmt(m["parsimony"]), fmt(ref.get("parsimony")), ""),
        ("Divergences · max R-hat · min ESS",
         f"{g['divergences']} · {g['max_rhat']} · {min(g['min_ess_bulk'], g['min_ess_tail'])}", "", ""),
        ("Evaluation runtime", f"{m['runtime_s']} s", "", ""),
    ]
    table = "| Metric (5-fold CV, development set) | This branch | Reference | Difference |\n|---|---|---|---|\n"
    table += "".join(f"| {a} | {b} | {r} | {d} |\n" for a, b, r, d in rows)
    review = ("\n![Review plots: sampler traces, effects, posterior predictive check, calibration](review.png)\n"
              if has_review else "")
    return f"""# {m['hypothesis'] or m['name']}

| | |
|---|---|
| Verdict | {verdict} |
| Experiment | `{m['experiment']}`: branch `{m['branch']}` into `{m['pr_base']}`, evaluated at commit `{m['commit']}` |
| Evaluation | `data/dev.csv` ({n_iso} isolates, {m['n_features']} features), 5 fixed cross-validation folds; the locked test set is not used |
| Guided by | skill `{m['skill'] or '-'}`, agent harness `{m['harness']}` |

## Results

{table}
![Where on the MIC scale the ELPD changed, and per-isolate differences](comparison.png)

{text['Results']}
{review}
## Data

{text['Data']}

## Method

{text['Method']}

<details><summary>Code change: {diffstat or 'no change in src/'}</summary>

```diff
{diff}
```
</details>

## Conclusion

{text['Conclusion']}

---
<sub>Verdict, table and plots: `checks/evaluate.py` and the `mic-plots` skill. Text: the agent, in the fixed structure
of `skills/mic-eval-harness/references/report-template.md`. A human decides whether to merge.</sub>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    run = run_dir(args.name)
    mf = run / "metrics.json"
    if not mf.exists():
        sys.exit(f"{rel(mf)} not found: run checks/evaluate.py --name {args.name} first")
    m = json.loads(mf.read_text())
    draft = run / "report_draft.md"
    if not draft.exists():
        draft.write_text(TEMPLATE.read_text().replace("{name}", args.name))
        print(f"Created {rel(draft)}. Write the four sections (Results, Data, Method, Conclusion) "
              f"following the comments in it, then run this command again.")
        sys.exit(1)
    text = parse(draft.read_text())
    bad = problems(text)
    if bad:
        sys.exit(f"{rel(draft)} is not ready:\n- " + "\n- ".join(bad))

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != m["branch"]:
        sys.exit(f"You are on '{branch}', but '{args.name}' was evaluated on '{m['branch']}'. Switch back first.")
    if not is_agent_branch(branch) or branch == experiment_branch():
        sys.exit(f"Reports are committed on hypothesis branches only (not '{branch}').")
    if git("status", "--porcelain", "--", "src"):
        sys.exit("src/ has uncommitted changes: the report must describe the committed, evaluated code.")

    out = DEMO / "reports" / m["experiment"] / args.name
    out.mkdir(parents=True, exist_ok=True)
    label = "vs " + m["compare"]["against"].removeprefix("champion ").split(" (")[0]
    r = subprocess.run([sys.executable, str(PLOTS / "compare.py"), str(DEMO / "data" / "dev.csv"),
                        str(run / "pointwise.csv"), str(out / "comparison"), "--label", label],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"comparison plot failed:\n{r.stderr[-800:]}")
    (out / "comparison.svg").unlink(missing_ok=True)
    has_review = (run / "review.png").exists()
    if has_review:
        on_paper(run / "review.png", out / "review.png")

    base = git("merge-base", m["commit"], m["parent"], check=False) or f"{m['commit']}~1"
    diffstat = git("diff", "--shortstat", base, m["commit"], "--", "src")
    diff = git("diff", base, m["commit"], "--", "src").splitlines()
    diff = "\n".join(diff[:200] + (["... (diff truncated)"] if len(diff) > 200 else []))
    n_iso = sum(1 for _ in open(DEMO / "data" / "dev.csv")) - 1
    (out / "report.md").write_text(render(m, text, diffstat, diff, n_iso, has_review))

    out_rel = str(out.relative_to(DEMO))
    git("add", "--", out_rel)
    d = m["compare"].get("elpd_diff")
    git("commit", "-q", "-m", f"report: {args.name} ({m['decision']}" + (f", ΔELPD {d:+}" if d is not None else "") + ")",
        "--", out_rel)
    m["report"] = out_rel + "/report.md"
    mf.write_text(json.dumps(m, indent=2))
    print(f"wrote and committed {out_rel}/report.md (+ comparison.png{', review.png' if has_review else ''}) on {branch}. "
          f"Next: uv run python skills/github-workflow/scripts/publish.py --name {args.name}")


if __name__ == "__main__":
    main()
