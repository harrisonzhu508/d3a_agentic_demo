---
name: bayesian-workflow
description: Check, criticise and compare Bayesian models the way an applied statistician would (prior and posterior predictive checks, MCMC diagnostics, PSIS-LOO and cross-validation, model expansion). Use before trusting any fit, before claiming one model beats another, and when deciding what to try next.
license: MIT
---

# Bayesian workflow

A fit is not a result until it has passed three kinds of checks: the sampler worked, the model can
reproduce the data, and it predicts new data better than the alternative. Change one thing at a time and
record what you tried, including failures.

## Checklist for every fit

1. **Computation**: 0 divergences, max rank-normalised R-hat < 1.01, bulk and tail ESS > 400 per parameter
   of interest, no saturation of the tree depth. If any fails, fix the computation (reparameterise, rescale,
   tighter priors) before looking at estimates. See `references/mcmc-diagnostics.md`.
2. **Posterior predictive check**: simulate data from the posterior and compare with the observed data on
   the scale that matters (here: counts per dilution bin, and the censored end bins). Systematic misfit
   means the model is wrong even if the sampler is fine.
3. **Predictive performance**: estimate out-of-sample log predictive density (ELPD) with PSIS-LOO or K-fold
   CV, and compare models by the difference and its standard error. See `references/model-comparison.md`.
4. **Plausibility**: do effect sizes and signs make scientific sense? Use domain knowledge (the
   `amr-genotype-phenotype` skill) as a sanity check, not as a target to fit.

## How to decide what to try next

Follow the workflow in `references/workflow.md`: start simple, expand the model where the checks point
(misfit in a particular region, outliers, heterogeneity between groups), and prefer the simpler model when
predictive performance is within about two standard errors.

## Rules

- Never compare models on in-sample fit.
- Never tune anything on held-out test data; use the development folds only.
- Report negative results: a change that did not help is information for the next hypothesis.
