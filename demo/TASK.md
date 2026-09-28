# Task: predict ciprofloxacin resistance levels from bacterial genomes

## Goal

Improve a Bayesian model that predicts, for each *Escherichia coli* isolate, its **minimum inhibitory
concentration (MIC) of ciprofloxacin** from its genotype. The model is judged on how well it predicts isolates
it has not seen.

## Data: `data/dev.csv` (the only data you may use)

One row per clinical isolate (558 isolates):

| Column | Meaning |
|---|---|
| `genome_id` | isolate identifier |
| `mic_raw` | the laboratory MIC as reported, in mg/L, on an agar-dilution series (for example `0.25` or `>4`) |
| `censor` | `=` for an on-grid value, `>` for "above the highest concentration tested" |
| `log2_lo`, `log2_hi` | the interval that contains the true log2 MIC: `(log2_lo, log2_hi]`; `-inf` / `inf` at the ends |
| `log2_mic` | a single number per isolate (`>x` replaced by `2x`); convenient for plots, never for the likelihood |
| 77 further columns | 0/1 presence of antimicrobial-resistance determinants called by AMRFinderPlus: acquired genes (for example `qnrS1`, `blaCTX-M-15`) and chromosomal point mutations (for example `gyrA_S83L`, `parC_S80I`) |

MICs are measured on a two-fold dilution grid, so every observation is an **interval**, and the lowest and
highest dilutions are open-ended. The data are real, public clinical isolates.

## What you may change

Only `src/features.py`, `src/model.py` and `src/sampler.py`, within the interface in `checks/contract.md`.
Everything else is read-only and guarded by hooks: `checks/` (the referee), `hooks/`, `data/`, `skills/`, this
file, `program.md` and the configuration files.

## Success

An experiment succeeds when **the harness says keep**:

`uv run python checks/evaluate.py --name <slug> --hypothesis "<one line>" --skill <skill>`

- **Primary**: cross-validated expected log predictive density (ELPD) of the observed MIC intervals on fixed
  development folds. Keep only if the improvement over the current champion exceeds 2 standard errors of the
  difference.
- **Gates** (all must pass): 0 divergent transitions, max R-hat < 1.01, bulk and tail ESS > 400, evaluation
  within the time budget, harness files unchanged.
- **Secondary**, reported for every run: share of isolates predicted within ±1 dilution, coverage of 90 %
  predictive intervals, number of non-null effects (parsimony), plausibility of the effects.

Simpler is better: a change that adds complexity for a gain within noise is not an improvement.

## Deliverables for every experiment

1. A commit on your hypothesis branch, `<prefix>/<experiment>/<slug>` (message names the skill that guided the
   change).
2. A row in `results/<experiment>/results.tsv` and a folder `results/<experiment>/<slug>/` (written by the harness).
3. If kept: a report in the fixed structure (Results, Data, Method, Conclusion, plus the harness's table and
   plots), committed as `reports/<experiment>/<slug>/report.md` by the `mic-eval-harness` report tool, and a pull
   request into the experiment branch `<prefix>/<experiment>/main` (`github-workflow` skill). If not kept: the
   branch is discarded with the same skill, so the idea is recorded but not repeated.

## Rules

- Work from `data/dev.csv`, this file, `program.md`, `checks/contract.md` and the skills. Do not look for other
  data, answers or published results for this dataset; the hooks block the obvious places.
- One hypothesis per experiment; change one thing at a time.
- Never edit the harness, never tune on anything the harness does not show you, and quote only numbers the
  harness produced.
