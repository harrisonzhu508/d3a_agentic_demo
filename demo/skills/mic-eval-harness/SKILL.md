---
name: mic-eval-harness
description: Score a model branch with the demo's read-only evaluation harness (cross-validated censored log predictive density, within-one-dilution agreement, interval coverage, sampler gates), decide keep or revert against main, and write the pull-request report. Use after every experiment, and in CI when assessing a branch.
license: MIT
---

# Evaluating a branch

The harness in `checks/` is the referee. It is read-only for agents: you run it, you never edit it, and you
never read the held-out test split. The same skill is used by the coding agent (to check its own work) and by
the CI assessor (to judge the branch), so both apply the same rules.

## Procedure

1. Commit your change first (the harness scores committed code only; a dirty tree is recorded as invalid).
2. Run the harness on the development folds (fixed 5-fold CV folds, identical for every branch; about 30 s to
   a few minutes):
   `uv run python checks/evaluate.py --name <slug> --hypothesis "<one line>" --skill <skill>`
   The last line of output is a JSON verdict; details are in `results/<experiment>/<slug>/metrics.json`, the review
   plots in `results/<experiment>/<slug>/review.png`. The locked test set is never scored in the loop.
3. Read the output against `references/metrics.md`:
   - **gates** must all pass (0 divergences, R-hat < 1.01, ESS > 400, runtime within budget), otherwise the
     branch fails regardless of its scores;
   - **decide on ELPD**: keep only if `elpd_diff` versus the current champion exceeds 2 standard errors of the
     difference (the harness applies this rule and prints the decision);
   - use within-one-dilution agreement, 90 % coverage and the parsimony count as secondary evidence and tie-breakers.
4. Check plausibility with the `amr-genotype-phenotype` skill (signs, null controls, interactions).
5. If kept, write the pull-request report with the report tool:
   `uv run python skills/mic-eval-harness/scripts/write_report.py --name <slug>`
   - The first run creates `results/<experiment>/<slug>/report_draft.md` from `references/report-template.md`:
     four fixed sections, **Results, Data, Method, Conclusion**, each with guidance on what to write. Replace
     the guidance with your text (keep the headings; under 250 words each; harness numbers only).
   - Run it again: it checks the draft and renders `reports/<experiment>/<slug>/report.md` with the verdict,
     the metrics table against the reference, the comparison plot (where on the MIC scale the ELPD changed),
     the review plots and the code diff, in the GPAP style, and commits that folder on your branch.
   Then publish with the `github-workflow` skill; discard anything that was not kept. Check progress any time
   with `uv run python checks/status.py`.

## Rules (verification before claims)

- Quote numbers only from harness output produced in this run; never estimate or round up a result.
- A branch that did not run the harness has no result.
- Never change folds, metrics, seeds or data to make a branch look better; if you think the harness is wrong,
  say so in the report and leave it unchanged.
