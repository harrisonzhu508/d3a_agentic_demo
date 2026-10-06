"""EVOLVABLE: the NumPyro model.

Contract (checks/contract.md):
- model(X, lo=None, hi=None): when lo/hi are given, add the likelihood of each isolate's log2-MIC interval and
  record it pointwise as numpyro.deterministic("log_lik", ...) with shape (n,).
- simulate(samples, X, key) -> array (draws, n): latent log2 MICs for the rows of X, from posterior samples.
- FEATURE_EFFECTS: name of the per-feature effect site (for the parsimony count), or None.

Logistic-likelihood variant: latent y* ~ Logistic(mu, sigma), interval-censored.
Independent Normal(0, 2) priors on beta (no horseshoe, to keep the sampling geometry simple).
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

FEATURE_EFFECTS = "beta"


def log_interval_prob_lo(lo, hi, mu, sigma):
    """log P(lo < y <= hi) for y ~ Logistic(mu, sigma); lo may be -inf and hi may be +inf.

    The logistic CDF has a closed form via log_sigmoid (NaN-safe, no inf gradients).
    """
    lo_inf, hi_inf = jnp.isinf(lo), jnp.isinf(hi)
    lo_f = jnp.where(lo_inf, hi - 1.0, lo)
    hi_f = jnp.where(hi_inf, lo + 1.0, hi)
    log_F_hi = jax.nn.log_sigmoid((hi_f - mu) / sigma)
    log_F_lo = jax.nn.log_sigmoid((lo_f - mu) / sigma)
    interval = log_F_hi + jnp.log1p(-jnp.exp(log_F_lo - log_F_hi))
    return jnp.where(hi_inf, jax.nn.log_sigmoid(-(lo_f - mu) / sigma),
                     jnp.where(lo_inf, log_F_hi, interval))


def model(X, lo=None, hi=None):
    p = X.shape[1]
    alpha = numpyro.sample("alpha", dist.Normal(-4.0, 3.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(p), 2.0))
    mu = numpyro.deterministic("mu", alpha + X @ beta)
    if lo is not None:
        ll = log_interval_prob_lo(lo, hi, mu, sigma)
        numpyro.factor("censored_lik", ll.sum())
        numpyro.deterministic("log_lik", ll)


def simulate(samples, X, key):
    """Latent log2 MIC draws (draws, n) for the rows of X."""
    beta = samples["beta"]
    mu = samples["alpha"][:, None] + beta @ jnp.asarray(X).T
    return mu + samples["sigma"][:, None] * jax.random.logistic(key, mu.shape)
