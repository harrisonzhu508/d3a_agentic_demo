"""EVOLVABLE: the NumPyro model.

Contract (checks/contract.md):
- model(X, lo=None, hi=None): when lo/hi are given, add the likelihood of each isolate's log2-MIC interval and
  record it pointwise as numpyro.deterministic("log_lik", ...) with shape (n,).
- simulate(samples, X, key) -> array (draws, n): latent log2 MICs for the rows of X, from posterior samples.
- FEATURE_EFFECTS: name of the per-feature effect site (for the parsimony count), or None.

Baseline: log2 MIC ~ Normal(alpha + X beta, sigma), interval-censored, independent Normal(0, 2) priors on the
effects (skill: censored-mic-regression).
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import log_ndtr

FEATURE_EFFECTS = "beta"

# Set by src/features.py (only it sees the column names): boolean array, True for the determinants too rare to
# have their own unrestricted effect. None = treat every column as the champion does.
RARE_COLUMNS = None


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


def logistic_log_interval_prob(lo, hi, mu, scale):
    """log P(lo < y <= hi) for y ~ Logistic(mu, scale); lo may be -inf and hi may be +inf.

    Heavier tails than the normal (log-survival ~ -|z| instead of -z^2/2), for isolates whose genotype
    does not explain an extreme MIC. log_sigmoid keeps each tail accurate; the middle bin is written
    as log_sigmoid(za) - log_sigmoid(zb) (both arguments positive, so no cancellation). Finite
    placeholders everywhere so jnp.where never lets a gradient multiply an inf by zero.
    """
    lo_f = jnp.where(jnp.isneginf(lo), hi - 1.0, lo)
    hi_f = jnp.where(jnp.isposinf(hi), lo_f + 1.0, hi)
    a, b = jnp.broadcast_arrays(lo_f, hi_f)
    za, zb = (mu - a) / scale, (b - mu) / scale
    log_upper = jax.nn.log_sigmoid(zb)                          # log P(y <= hi)
    log_surv = jax.nn.log_sigmoid(za)                           # log P(y > lo)
    interval = jnp.where(za > 0.0,                              # subtract in the more accurate tail
                         log_surv + jnp.log1p(-jnp.exp(jax.nn.log_sigmoid(-zb) - log_surv)),
                         log_upper + jnp.log1p(-jnp.exp(jax.nn.log_sigmoid(za) - log_upper)))
    return jnp.where(jnp.isposinf(hi), log_surv, jnp.where(jnp.isneginf(lo), log_upper, interval))


def model(X, lo=None, hi=None):
    p = X.shape[1]
    alpha = numpyro.sample("alpha", dist.Normal(-4.0, 3.0))
    # Logistic errors with the same variance as the baseline HalfNormal(2) errors: s = sigma sqrt(3/pi)
    scale = numpyro.sample("scale", dist.HalfNormal(2.0 * jnp.sqrt(3.0 / jnp.pi)))
    # Champion: independent StudentT(4, 0, 2) on every effect, which beat Normal(0,2) by 28.9 +/- 7.3 ELPD and
    # is the only regularisation tried so far that both gains and samples. One change: the *rare* determinants.
    # 34 of the 77 columns are carried by at most 15 isolates; for those the data carry almost no information
    # about the effect, and a t's polynomial tail means the posterior is barely tighter than the prior (their
    # posterior sd is 2.7-2.8 against a prior sd of 2, and the prior tail then lets a coefficient reach tens of
    # doublings - gyrA_D87Y, five isolates, sits at +2.9 +/- 4.5, and the extreme mu values come from such
    # columns combining). A half-Cauchy scale mixture is the standard weakly informative version of a sparsity
    # prior where the sparsity level itself is unknown (Polson & Scott 2012; Gelman et al. BDA3 ch. 21 on
    # hierarchical scales being the robust choice over fixed ones), and writing it non-centred (beta = u * |w|,
    # w ~ t(4,0,2), u ~ HalfCauchy(1)) adds no discrete inclusion site - the thing that made the earlier
    # horseshoe attempts unsamplable here was a funnel in tau, and a half-Cauchy scale on a t-distributed effect
    # has no funnel at 0 because both factors stay put. The 43 common columns keep the champion's prior unchanged.
    rare = jnp.asarray(RARE_COLUMNS, dtype=bool) if RARE_COLUMNS is not None else jnp.zeros(p, dtype=bool)
    beta_common = numpyro.sample("beta_common", dist.StudentT(4.0, jnp.zeros(p), 2.0))
    beta_rare_raw = numpyro.sample("beta_rare_raw", dist.StudentT(4.0, jnp.zeros(p), 2.0))
    rare_scale = numpyro.sample("rare_scale", dist.HalfCauchy(jnp.ones(p)))
    beta = numpyro.deterministic("beta", jnp.where(rare, beta_rare_raw * rare_scale, beta_common))
    mu = numpyro.deterministic("mu", alpha + X @ beta)
    if lo is not None:
        ll = logistic_log_interval_prob(lo, hi, mu, scale)
        numpyro.factor("censored_lik", ll.sum())
        numpyro.deterministic("log_lik", ll)


def simulate(samples, X, key):
    """Latent log2 MIC draws (draws, n) for the rows of X (logistic errors, matching the likelihood)."""
    beta = samples["beta"]                                    # (draws, p)
    mu = samples["alpha"][:, None] + beta @ jnp.asarray(X).T   # (draws, n)
    u = jax.random.uniform(key, mu.shape, minval=jnp.finfo(jnp.float32).tiny)
    return mu + samples["scale"][:, None] * jnp.log(u / (1.0 - u))
