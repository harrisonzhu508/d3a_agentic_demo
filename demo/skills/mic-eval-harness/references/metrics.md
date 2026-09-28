# Harness metrics

All development metrics use the same fixed 5-fold split of the development isolates, stratified by MIC bin.

## Primary: cross-validated ELPD

`elpd_cv = sum over folds, sum over held-out isolates i of log p(obs_i | training folds)`, where `p(obs_i)`
is the posterior predictive probability of isolate i's observed MIC *interval* (averaged over posterior
draws: `log mean_s P(lo_i < y* <= hi_i | theta_s)`). Higher is better. The harness also reports the standard
error of the difference from `main`, computed from pointwise differences:
`se_diff = sqrt(n) * sd_i(elpd_i(branch) - elpd_i(main))`.

**Decision**: keep a branch if `elpd_diff > 2 * se_diff` and all gates pass. Within 2 standard errors,
prefer the simpler model (fewer non-null effects, no new model components).

## Secondary

- **Within ±1 dilution (essential agreement)**: posterior-median prediction versus observed bin, with
  `>x` counted as correct when the prediction is at or above the top tested dilution and the lowest bin
  counted as correct when the prediction is at or below it. Reported the conventional way as well, for
  comparison with the literature.
- **90 % interval coverage**: the fraction of held-out isolates whose observed interval overlaps the central
  90 % posterior predictive interval. Well calibrated is close to 0.90; much higher means intervals too wide.
- **Parsimony**: the number of determinants whose posterior effect exceeds ±0.5 doublings with at least 90 %
  probability, and `p_loo` from a full-data PSIS-LOO fit.

## Gates (any failure fails the branch)

| Gate | Threshold |
|---|---|
| divergent transitions | 0 |
| max rank-normalised R-hat | < 1.01 |
| min bulk and tail ESS | > 400 |
| wall-clock per fit | within the budget in `program.md` |
| harness files, data and folds | identical to the base branch (`checks/integrity.py`, git diff) |
