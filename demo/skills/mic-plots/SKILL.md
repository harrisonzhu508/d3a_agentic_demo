---
name: mic-plots
description: Make the standard review plots for a fitted MIC model (sampler traces, effect forest plot versus a naive fit, posterior predictive check on the dilution grid, calibration with the plus-or-minus one dilution band) in the GPAP slide style. Use after every model fit, and when preparing figures for a pull request or a talk.
license: MIT
---

# MIC review plots

Every branch that fits a model produces the same four panels, so a reviewer (human or LLM) can compare
branches at a glance. Run `scripts/plots.py` on the data table and the saved `posterior.nc`.

## What to look at, and what it means

- **(a) Traces**: chains should overlap and look like noise. Drifting or separated chains mean the fit is
  not usable, whatever the point estimates look like.
- **(b) Effects**: censored-model effects are usually larger than naive least squares when many MICs are
  right-censored (`>x` carries no upper bound). A determinant with no plausible mechanism for the drug (for a
  fluoroquinolone, a beta-lactamase gene) is a useful null control and should sit near zero.
- **(c) Predictive check**: simulated counts per dilution bin versus the observed counts. Passing R-hat and
  ESS is not enough: a model can sample perfectly and still misfit the middle of the grid.
- **(d) Calibration**: posterior-median prediction versus observed bin; points outside the green band are
  more than one dilution off (the standard "essential agreement" metric). Clusters of misses far from the
  diagonal point at genotype-phenotype discordance rather than a wrong prior.

## Rules

- Keep the style (IBM Plex, GPAP palette, transparent background) so figures drop straight into the slides.
- Label intervals with their level (89 % here) and axes with units (doublings of MIC, mg/L).
- Never smooth over the censored end bins: show `<=` and `>` bins explicitly.

## Run

`scripts/compare.py` draws where a branch gained or lost ELPD against its reference (by MIC bin and per isolate);
the report tool in `mic-eval-harness` uses it. Both are run for you; by hand, from `demo/`
(`checks/evaluate.py` and `train.py` already draw the review plots for every run):

```bash
uv run python skills/mic-plots/scripts/plots.py data/dev.csv results/<experiment>/<name>/posterior.nc \
    results/<experiment>/<name>/review
```
