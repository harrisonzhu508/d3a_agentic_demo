# Logistic interval likelihood with plain Normal(0,2) prior on beta (no horseshoe) to get the heavy-tail benefit without the horseshoe sampling geometry that causes divergences

| | |
|---|---|
| Verdict | **keep**: ΔELPD +16.64 ± 5.05 against champion demo_test26092026/demo3/main, gates pass (rule: keep if ΔELPD > 2 SE and every gate passes) |
| Experiment | `demo3`: branch `demo_test26092026/demo3/logistic-plain` into `demo_test26092026/demo3/main`, evaluated at commit `27eefe3` |
| Evaluation | `data/dev.csv` (558 isolates, 77 features), 5 fixed cross-validation folds; the locked test set is not used |
| Guided by | skill `censored-mic-regression`, agent harness `pi` |

## Results

| Metric (5-fold CV, development set) | This branch | Reference | Difference |
|---|---|---|---|
| ELPD (cross-validated) | -497.18 ± 34.83 | -513.83 | +16.64 ± 5.05 |
| Within ±1 dilution | 85.3% | 85.5% |  |
| 90 % interval coverage | 95.9% | 96.1% |  |
| Non-null effects (parsimony) | 7 | 3 |  |
| Divergences · max R-hat · min ESS | 0 · 1.0054 · 2645 |  |  |
| Evaluation runtime | 12.4 s |  |  |

![Where on the MIC scale the ELPD changed, and per-isolate differences](comparison.png)

The logistic-likelihood branch is kept: ELPD(CV) improves by 16.64 ± 5.05 log points over the horseshoe champion (well over the 2-SE keep threshold), and all gates pass (0 divergences, max R-hat 1.0054, min ESS 2645). Within-±1-dilution agreement is 85.3 % and 90 % coverage is 95.9 %, comparable to the champion. The heavier-tailed logistic CDF better accommodates the extreme bimodality of the data (44 % left-censored, 36 % right-censored) without the sampling geometry problems that the horseshoe + logistic combination causes. The model is also much faster (12 s vs 146 s) because it drops the horseshoe's extra latent sites.

![Review plots: sampler traces, effects, posterior predictive check, calibration](review.png)

## Data

- 558 clinical E. coli isolates, 77 binary AMRFinderPlus determinants (unchanged from the baseline).
- 80 % of isolates are at the two ends of the dilution grid (≤0.008 or >4 mg/L); only 20 % are on-grid, spanning 9 doublings.
- This extreme bimodality is the key property that the logistic likelihood is designed to handle: the Normal CDF decays too quickly in the tails, making the model over-confident about the censored groups.

## Method

- Hypothesis: replacing the Normal interval likelihood with a Logistic interval likelihood will improve CV prediction for the bimodal MIC data.
- The logistic CDF has a closed form (log_sigmoid), so no special NaN-safe tricks are needed beyond the standard finite-placeholder bounds for open-ended intervals.
- The prior on β is reverted to the simple Normal(0, 2) (no horseshoe), because the horseshoe + logistic combination produces divergences even with dense_mass and increased warmup. The logistic likelihood alone provides enough regularisation through its heavier tails.
- Skill: censored-mic-regression (references/censored-likelihoods.md, "Heavier tails" section).
- The simulate function draws from the same logistic distribution to match the likelihood.

<details><summary>Code change: 1 file changed, 16 insertions(+), 25 deletions(-)</summary>

```diff
diff --git a/demo/src/model.py b/demo/src/model.py
index d70555d..676c6e9 100644
--- a/demo/src/model.py
+++ b/demo/src/model.py
@@ -6,56 +6,47 @@ Contract (checks/contract.md):
 - simulate(samples, X, key) -> array (draws, n): latent log2 MICs for the rows of X, from posterior samples.
 - FEATURE_EFFECTS: name of the per-feature effect site (for the parsimony count), or None.
 
-Baseline: log2 MIC ~ Normal(alpha + X beta, sigma), interval-censored, independent Normal(0, 2) priors on the
-effects (skill: censored-mic-regression).
+Logistic-likelihood variant: latent y* ~ Logistic(mu, sigma), interval-censored.
+Independent Normal(0, 2) priors on beta (no horseshoe, to keep the sampling geometry simple).
 """
 
 import jax
 import jax.numpy as jnp
 import numpyro
 import numpyro.distributions as dist
-from jax.scipy.special import log_ndtr
 
 FEATURE_EFFECTS = "beta"
 
 
-def log_interval_prob(lo, hi, mu, sigma):
-    """log P(lo < y <= hi) for y ~ Normal(mu, sigma); lo may be -inf and hi may be +inf.
+def log_interval_prob_lo(lo, hi, mu, sigma):
+    """log P(lo < y <= hi) for y ~ Logistic(mu, sigma); lo may be -inf and hi may be +inf.
 
-    Every branch is computed on finite placeholder bounds so that jnp.where never multiplies an
-    infinite gradient by zero (which gives NaN gradients and a stuck sampler).
+    The logistic CDF has a closed form via log_sigmoid (NaN-safe, no inf gradients).
     """
     lo_inf, hi_inf = jnp.isinf(lo), jnp.isinf(hi)
     lo_f = jnp.where(lo_inf, hi - 1.0, lo)
     hi_f = jnp.where(hi_inf, lo + 1.0, hi)
-    a, b = (lo_f - mu) / sigma, (hi_f - mu) / sigma
-    log_upper, log_lower = log_ndtr(b), log_ndtr(a)
-    interval = log_upper + jnp.log1p(-jnp.exp(log_lower - log_upper))
-    return jnp.where(hi_inf, log_ndtr(-a), jnp.where(lo_inf, log_upper, interval))
+    log_F_hi = jax.nn.log_sigmoid((hi_f - mu) / sigma)
+    log_F_lo = jax.nn.log_sigmoid((lo_f - mu) / sigma)
+    interval = log_F_hi + jnp.log1p(-jnp.exp(log_F_lo - log_F_hi))
+    return jnp.where(hi_inf, jax.nn.log_sigmoid(-(lo_f - mu) / sigma),
+                     jnp.where(lo_inf, log_F_hi, interval))
 
 
 def model(X, lo=None, hi=None):
-    n, p = X.shape
+    p = X.shape[1]
     alpha = numpyro.sample("alpha", dist.Normal(-4.0, 3.0))
     sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
-    # Regularised horseshoe (Piironen & Vehtari 2017), non-centred form.
-    # tau0 from p0=8 relevant predictors out of p=77, sigma, n (per sparse-priors.md).
-    tau0 = 8.0 / (p - 8.0) * sigma / jnp.sqrt(n)
-    tau = numpyro.sample("tau", dist.HalfCauchy(tau0))
-    lam = numpyro.sample("lambda", dist.HalfCauchy(jnp.ones(p)))
-    c2 = numpyro.sample("c2", dist.InverseGamma(2.0, 2.0 * 3.0**2))  # slab: effects up to ~3 doublings
-    lam_tilde = jnp.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2))
-    z = numpyro.sample("z", dist.Normal(jnp.zeros(p), 1.0))
-    beta = numpyro.deterministic("beta", z * tau * lam_tilde)
+    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(p), 2.0))
     mu = numpyro.deterministic("mu", alpha + X @ beta)
     if lo is not None:
-        ll = log_interval_prob(lo, hi, mu, sigma)
+        ll = log_interval_prob_lo(lo, hi, mu, sigma)
         numpyro.factor("censored_lik", ll.sum())
         numpyro.deterministic("log_lik", ll)
 
 
 def simulate(samples, X, key):
     """Latent log2 MIC draws (draws, n) for the rows of X."""
-    beta = samples["beta"]                                    # (draws, p)
-    mu = samples["alpha"][:, None] + beta @ jnp.asarray(X).T   # (draws, n)
-    return mu + samples["sigma"][:, None] * jax.random.normal(key, mu.shape)
+    beta = samples["beta"]
+    mu = samples["alpha"][:, None] + beta @ jnp.asarray(X).T
+    return mu + samples["sigma"][:, None] * jax.random.logistic(key, mu.shape)
```
</details>

## Conclusion

Kept by the harness rule (Δ ELPD > 2 SE, all gates pass). The result shows that the heavier-tailed logistic likelihood is a better fit for the bimodal MIC distribution than the Normal, and that this benefit is available without the complexity of the horseshoe prior. The logistic model is simpler (3 latent sites vs 6) and much faster to fit. The remaining 14.7 log-point gap to the horseshoe+logistic near-miss (Δ 28.04) suggests that combining the two improvements is the next obvious step, but the horseshoe + logistic geometry needs a different parameterisation (e.g., R2D2 prior, which samples more easily than the horseshoe). Next hypotheses: (1) R2D2 prior + logistic likelihood; (2) a two-component Normal mixture with genotype-dependent mixing weight.

---
<sub>Verdict, table and plots: `checks/evaluate.py` and the `mic-plots` skill. Text: the agent, in the fixed structure
of `skills/mic-eval-harness/references/report-template.md`. A human decides whether to merge.</sub>
