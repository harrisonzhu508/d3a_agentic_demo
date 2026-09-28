# Skill catalog

Pick the skill for what you are about to do, **read its SKILL.md before acting**, and open a reference file
only when the SKILL.md points you to it. Name the skill you used in your commit message and PR.

| Skill | Use when | Start with | Key references |
|---|---|---|---|
| `censored-mic-regression` | changing the likelihood, priors or sampler in `src/model.py` / `src/sampler.py` | `skills/censored-mic-regression/SKILL.md` | `numpyro-primer.md`, `censored-likelihoods.md`, `sparse-priors.md`; example code `scripts/fit.py` |
| `bayesian-workflow` | deciding what to try next, checking a fit, comparing models | `skills/bayesian-workflow/SKILL.md` | `workflow.md`, `mcmc-diagnostics.md`, `model-comparison.md` |
| `amr-genotype-phenotype` | choosing features or interactions (`src/features.py`), judging whether an effect is plausible | `skills/amr-genotype-phenotype/SKILL.md` | `fluoroquinolone-resistance.md`, `mic-testing.md`, `amrfinderplus-output.md`, `related-work.md` |
| `mic-plots` | looking at a fit (traces, effects, predictive check, calibration) | `skills/mic-plots/SKILL.md` | `scripts/plots.py` (run automatically by the harness and `train.py`) |
| `mic-eval-harness` | scoring a committed change and deciding keep or discard; writing the pull-request report | `skills/mic-eval-harness/SKILL.md` | `metrics.md`, `report-template.md`; report tool `scripts/write_report.py` |
| `github-workflow` | starting a hypothesis branch; publishing (push + PR into `<prefix>/<experiment>/main`) or discarding | `skills/github-workflow/SKILL.md` | `scripts/new_branch.py`, `publish.py`, `discard.py` |

## Typical order in one experiment

1. `github-workflow` → start a branch.
2. `bayesian-workflow` → decide what to try from the last review plots and `checks/status.py`.
3. `censored-mic-regression` or `amr-genotype-phenotype` → read before editing `src/`.
4. `mic-eval-harness` → evaluate, read the verdict; if kept, write the report with `scripts/write_report.py`.
5. `github-workflow` → publish (PR into the experiment branch) or discard.
