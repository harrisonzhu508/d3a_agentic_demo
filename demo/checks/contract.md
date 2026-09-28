# Model contract (what `checks/` expects from `src/`)

The harness imports the three evolvable modules of the current branch. Keep these interfaces and you can change
anything inside them. `checks/smoke.py` checks the contract in under a minute and runs after every edit.

## `src/features.py`

```python
def build(df: pandas.DataFrame) -> tuple[numpy.ndarray, list[str]]
```
- Returns `X` (float, shape `(n_isolates, n_features)`, finite) and one name per column.
- Uses genotype columns only. The MIC columns (`mic_raw`, `censor`, `log2_lo`, `log2_hi`, `log2_mic`) must not
  influence `X` (the smoke test shuffles them and checks that `X` is unchanged).
- Must be a deterministic function of each row's genotype (the same features are used for every fold).

## `src/model.py`

```python
def model(X, lo=None, hi=None): ...
def simulate(samples: dict, X, key) -> array  # shape (draws, n)
FEATURE_EFFECTS: str | None
```
- `model` is a NumPyro model. When `lo` and `hi` (log2-MIC interval bounds, `-inf`/`+inf` allowed) are given, it
  adds the likelihood of each isolate's interval and records the pointwise log-likelihood as
  `numpyro.deterministic("log_lik", ...)` with shape `(n,)`. With `lo=None` it must still run (used to predict).
- `simulate` returns latent log2-MIC draws for the rows of `X` from posterior samples (latent sites plus the
  deterministic sites recomputed for `X`). It is used for agreement, coverage and the predictive check.
- `FEATURE_EFFECTS` names the per-feature effect site (shape `(n_features,)`) for the parsimony count, or `None`.

## `src/sampler.py`

```python
SETTINGS = {"num_warmup": int, "num_samples": int, "num_chains": int,
            "target_accept_prob": float, "max_tree_depth": int, "dense_mass": bool}
```
Used for the full development fit that the gates are computed on. The cross-validation folds use the harness's
fixed budget (`checks/harness.json`) but take `target_accept_prob`, `max_tree_depth` and `dense_mass` from here.

## What the harness computes

- **ELPD (CV)**: for each held-out isolate, `log mean_s exp(log_lik_s)` from a fit on the other four folds; summed.
- **Gates** on the full development fit: divergences, max R-hat, min bulk and tail ESS over the latent sites and
  `FEATURE_EFFECTS`; runtime within budget; harness files unchanged.
- **Decision**: keep if `elpd_diff > 2 * se_diff` against the current champion (or `checks/best.json`) and all
  gates pass; discard otherwise.
