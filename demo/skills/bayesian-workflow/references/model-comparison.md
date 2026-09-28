# Comparing models by predictive performance

## ELPD

The expected log pointwise predictive density for new data, `elpd = sum_i log p(y_i_new | data)`, is the
default yardstick: it rewards calibrated predictive distributions, not just good point predictions. For
censored data, `p(y_i | ...)` is the probability of the observed interval (see the censored-likelihood reference).

## PSIS-LOO (Vehtari, Gelman & Gabry 2017, *Statistics and Computing* 27:1413)

Approximates leave-one-out cross-validation from a single fit by Pareto-smoothed importance sampling of the
pointwise log-likelihood. Needs the `log_likelihood` group in the InferenceData.

- `elpd_loo`: estimated ELPD (higher is better); `se`: its standard error.
- `p_loo`: effective number of parameters, a parsimony summary.
- **Pareto k** per observation: k < 0.7 is fine; k > 0.7 means the approximation is unreliable for that point
  (often influential or outlying observations); refit without it or use K-fold CV.

```python
import arviz as az
loo = az.loo(idata, pointwise=True)
az.compare({"baseline": idata0, "candidate": idata1})   # elpd_diff, dse (SE of the difference), weights
```

## Deciding

- Look at `elpd_diff` together with the standard error of the difference (`dse`). A difference smaller than
  about 2 to 4 times `dse` is not convincing; prefer the simpler model.
- With few observations or many influential points, prefer K-fold CV (for example 5 folds, stratified by the
  outcome bin) over PSIS-LOO. Use the **same folds** for every model you compare.
- Report secondary, interpretable metrics next to ELPD (for MICs: the share of predictions within one
  dilution, and the coverage of 90 % predictive intervals), but decide on ELPD.
- Pointwise ELPD differences show *where* a model is better (which isolates, which MIC range), which is
  often the best hint for the next change.
