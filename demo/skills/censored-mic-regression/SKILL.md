---
name: censored-mic-regression
description: Fit and check interval-censored Bayesian regressions of log2 MIC (antibiotic minimum inhibitory concentration) on binary resistance determinants with NumPyro. Use when modelling MICs from a dilution series, including values like "<=0.008" or ">4", or when editing model.py or sampler.py in this demo.
license: MIT
---

# Censored MIC regression (NumPyro)

MICs are measured on a two-fold dilution grid, so each observation is an interval on the log2 scale,
not a number. Treat it that way.

## Rules

1. Encode every MIC as an interval `(log2_lo, log2_hi]`: an on-grid value `v` means `(log2 v - 1, log2 v]`;
   `>x` means `(log2 x, +inf)`; the lowest dilution tested is an upper bound only, `(-inf, log2 v]`.
   The data table already has `log2_lo` and `log2_hi` columns. Never replace censored values with `2x` or `x/2` inside
   the likelihood (that convention is only for comparing with published metrics).
2. The project's model lives in `src/model.py` (latent `log2 MIC ~ Normal(alpha + X beta, sigma)`,
   likelihood `P(lo < y <= hi)`, independent Normal priors on `beta`). `scripts/fit.py` is a standalone example
   with a regularised horseshoe on `beta`, useful as a template. Change one thing at a time, and keep the
   interface in `checks/contract.md` (`model`, `simulate`, `FEATURE_EFFECTS`).
3. Compute censored log-probabilities with the NaN-safe helper in `scripts/fit.py` (`log_interval_prob`).
   Masking `inf` bounds with `jnp.where` after the fact gives NaN gradients and a stuck sampler.
4. Sample with NUTS, 4 chains, `target_accept_prob >= 0.95`. Report divergences, max R-hat and min bulk ESS
   every time; do not interpret a fit with divergences > 0, R-hat > 1.01 or ESS < 400.
   Mind the cost: the harness fits 6 models within a 10-minute budget. A dense mass matrix over ~80 effects,
   `max_tree_depth` above 10 or `target_accept_prob` 0.99 can each multiply the run time; fix divergences with
   the parameterisation first (non-centred forms, see `references/sparse-priors.md`), and change one sampler
   setting at a time.
5. Save the posterior as ArviZ InferenceData with a `log_likelihood` group so PSIS-LOO and the evaluation
   harness can score it.

## Background (read when needed)

- `references/numpyro-primer.md`: NumPyro model code, NUTS settings, JAX pitfalls, saving to ArviZ.
- `references/censored-likelihoods.md`: interval and Tobit likelihoods for dilution-series data, and a
  ready-to-use censored Student-t likelihood (copy it rather than deriving the CDF).
- `references/sparse-priors.md`: horseshoe, regularised horseshoe and R2D2 priors, and their sampling geometry.
- For the biology of the determinants, use the `amr-genotype-phenotype` skill; for checking a fit, `bayesian-workflow`.

## Run

In this project: `uv run python train.py` (full fit + review plots, not a score) and `uv run python
checks/smoke.py` (contract check). `scripts/fit.py` runs standalone on any table with the same columns.
