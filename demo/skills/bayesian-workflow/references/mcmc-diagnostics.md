# MCMC diagnostics

## R-hat (potential scale reduction)

Compares between-chain and within-chain variance; values near 1 mean the chains agree. Use the rank-normalised,
split R-hat of Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021) *Bayesian Analysis* 16:667, which is what
ArviZ reports. Threshold: R-hat < 1.01 with at least 4 chains. An R-hat of `inf` or `nan` usually means the
chains never moved (for example NaN gradients).

## Effective sample size (ESS)

The number of independent draws with the same information as the autocorrelated chain. Report **bulk ESS**
(for means and medians) and **tail ESS** (for 5 % and 95 % quantiles and intervals). Rule of thumb: > 400 in
total for the quantities you report, and check the Monte Carlo standard error (MCSE) is small relative to the
precision you quote.

## Divergent transitions (HMC and NUTS)

A divergence means the numerical integrator could not follow the posterior's curvature, typically in a funnel
(a scale parameter near zero with many parameters depending on it). Even a few divergences can bias
estimates (Betancourt 2017, arXiv:1701.02434). Remedies, in order:
1. reparameterise (non-centred forms for hierarchical and shrinkage priors);
2. rescale predictors and the outcome so parameters are of order 1;
3. raise `target_accept_prob` (0.95 to 0.99);
4. reconsider the prior (for example a global scale with a very small prior mode).

## Other signals

- **Tree depth saturation**: many transitions hitting `max_tree_depth` means slow exploration, often from very
  different scales or strong correlations; consider `dense_mass=True` or rescaling.
- **Energy / E-BFMI** (Betancourt 2016): low values (< 0.3) indicate the momentum resampling is not exploring
  the energy distribution well, often with heavy tails.
- **Trace and rank plots**: chains should overlap and look like stationary noise; rank histograms should be flat.
- **Pairs plots**: look for funnels and strong correlations between scale and location parameters.

## With ArviZ 1.x

```python
import arviz as az
az.summary(idata, var_names=["beta", "sigma"])   # mean, sd, intervals, ess_bulk, ess_tail, r_hat, mcse
az.plot_trace(idata, var_names=["sigma"])         # or az.plot_rank(...)
```
