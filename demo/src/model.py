"""EVOLVABLE: the NumPyro model.

Contract (checks/contract.md):
- model(X, lo=None, hi=None): when lo/hi are given, add the likelihood of each isolate's log2-MIC interval and
  record it pointwise as numpyro.deterministic("log_lik", ...) with shape (n,).
- simulate(samples, X, key) -> array (draws, n): latent log2 MICs for the rows of X, from posterior samples.
- FEATURE_EFFECTS: name of the per-feature effect site (for the parsimony count), or None.

Hypothesis: logistic latent errors instead of Normal, so the ~80 % of isolates censored at a plate limit are not
fitted by inflating the residual scale (skill: censored-mic-regression).
log2 MIC ~ Logistic(alpha + X beta, s), interval-censored, independent Normal(0, 2) priors on the effects.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import log_ndtr

FEATURE_EFFECTS = "beta"


def log_interval_prob(lo, hi, mu, sigma):
    """log P(lo < y <= hi) for y ~ Normal(mu, sigma); lo may be -inf and hi may be +inf.

    Every branch is computed on finite placeholder bounds so that jnp.where never multiplies an
    infinite gradient by zero (which gives NaN gradients and a stuck sampler).
    """
    lo_inf, hi_inf = jnp.isinf(lo), jnp.isinf(hi)
    lo_f = jnp.where(lo_inf, hi - 1.0, lo)
    hi_f = jnp.where(hi_inf, lo + 1.0, hi)
    a, b = (lo_f - mu) / sigma, (hi_f - mu) / sigma
    log_upper, log_lower = log_ndtr(b), log_ndtr(a)
    interval = log_upper + jnp.log1p(-jnp.exp(log_lower - log_upper))
    return jnp.where(hi_inf, log_ndtr(-a), jnp.where(lo_inf, log_upper, interval))


def logistic_log_interval_prob(lo, hi, mu, s):
    """log P(lo < y <= hi) for y ~ Logistic(mu, s); lo may be -inf and hi may be +inf.

    The logistic CDF has the closed form sigmoid(z) = 1/(1+exp(-z)), so its log is -log(1+exp(-z)) =
    -logaddexp(0,-z): no underflow in either tail, unlike a normal CDF difference when 80 % of the observations
    sit in one tail (which is the case here). The interval probability is sigmoid(b) - sigmoid(a) taken in log
    space, and every branch is evaluated on finite placeholder bounds so jnp.where cannot multiply an infinite
    gradient by zero (skill: censored-mic-regression, censored-likelihoods.md: "a logistic or Student-t latent
    in place of the normal ... via jax.nn.log_sigmoid").
    """
    lo_inf, hi_inf = jnp.isinf(lo), jnp.isinf(hi)
    lo_f = jnp.where(lo_inf, hi - 1.0, lo)                          # finite placeholders
    hi_f = jnp.where(hi_inf, lo + 1.0, hi)
    log_p_upper = -jnp.logaddexp(0.0, -(hi_f - mu) / s)             # log P(y <= hi)
    log_p_lower = -jnp.logaddexp(0.0, -(lo_f - mu) / s)             # log P(y <= lo)
    gap = jnp.clip(log_p_lower - log_p_upper, -80.0, 0.0)           # log P(y<=lo) - log P(y<=hi) <= 0
    interval = log_p_upper + jnp.log(-jnp.expm1(gap))
    right = -jnp.logaddexp(0.0, (lo_f - mu) / s)                    # log P(y > lo)
    return jnp.where(hi_inf, right, jnp.where(lo_inf, log_p_upper, interval))


def model(X, lo=None, hi=None):
    p = X.shape[1]
    alpha = numpyro.sample("alpha", dist.Normal(-4.0, 3.0))
    s = numpyro.sample("s", dist.HalfNormal(2.0))                   # logistic scale; sd = s*pi/sqrt(3)
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(p), 2.0))
    mu = numpyro.deterministic("mu", alpha + X @ beta)
    if lo is not None:
        ll = logistic_log_interval_prob(lo, hi, mu, s)
        numpyro.factor("censored_lik", ll.sum())
        numpyro.deterministic("log_lik", ll)


def simulate(samples, X, key):
    """Latent log2 MIC draws (draws, n) for the rows of X, from the same logistic error as the likelihood."""
    beta = samples["beta"]                                       # (draws, p)
    mu = samples["alpha"][:, None] + beta @ jnp.asarray(X).T   # (draws, n)
    u = jax.random.uniform(key, mu.shape, minval=1e-7, maxval=1.0 - 1e-7)
    return mu + samples["s"][:, None] * jnp.log(u / (1.0 - u))  # logistic inverse CDF
