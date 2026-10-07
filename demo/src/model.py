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
    # The error scale of the interval-censored likelihood is the parameter the censored rows identify least
    # well - a right-censored isolate is happy with almost any mu once mu is past the top of the plate, so what
    # pins s down is only the shape of the intervals, and its posterior is long-right-tailed (champion:
    # posterior mean 6.8 with a 95% interval of 5.2-8.7, which is the whole width of a ten-doubling grid). A
    # half-normal prior truncates that tail gently rather than hard: it is the weakly informative default for a
    # scale (Gelman et al., BDA3 ch. 21), and its log density is quadratic in s, so it leaves the bulk of the
    # posterior - where the 354 on-grid isolates do constrain s - untouched while removing the part of the tail
    # the censored rows cannot speak to. The champion's own error *distribution* change (logistic errors) was
    # worth +340 ELPD, so getting the residual right is where the loss still lives: coverage is 82.6% against a
    # nominal 90%.
    scale = numpyro.sample("scale", dist.HalfNormal(2.0))
    # The champion needs very large effects on a few QRDR alleles (gyrA D87N ~ +11, parC S80I ~ +9 log2
    # units) - the MIC really does leave the plate for those isolates - so width 2 is already close to the
    # posterior of the alleles that matter. A Student-t(4, 0, 2) prior has that width in the middle but
    # leaves the tail open, so the alleles that matter are barely regularised while the ~70 with no
    # fluoroquinolone mechanism stay shrunk. (A half-normal scale mixture was tried first and diverged;
    # see discarded/prior-width-4: the same gain came only when the width itself was raised.)
    # Sixteenth variant. The best gain the loop has seen is +9.75 +/- 5.81 with coverage 88.0% (against the
    # champion's 82.6%), from a slab on the rare columns whose scale had a heavy left-hand mass near zero
    # (discarded/rare-allele-slab-prior) - it cost 245 divergent transitions. Every cheap reparameterisation of
    # that scale has now been tried and the ranking is informative: the *bounded* forms of the same prior lose the
    # gain (spike as a factor +1.12, t(3) +2.21, t(4,0,2) on a split site +2.40, t(3,0,2) soft-cut +2.63), and the
    # forms that keep it diverge (half-Cauchy scale 245; half-normal(6) scale 4; its quantile form 6; its
    # effect-scaled form 7; a dense mass matrix made it worse, R-hat 1.07). So the gain and the divergences are the
    # same feature - a scale that can sit near zero - and the question is whether the divergence is intrinsic to
    # the spike or only to the heavy tail of the half-Cauchy. HalfNormal(6) as the scale isolates that: same
    # spike at zero (a normal scale's density is flat there, so beta's density vanishes linearly rather than
    # diverging, which is halfway between the two), no tail beyond three scale units, and it samples. If the
    # half-normal scale recovers the ~10, the tail was the problem and this is a champion; if it recovers ~1.4 as
    # discarded/slab-halfnormal-scale did at width 1, then the gain genuinely needs the funnel, and the honest
    # conclusion is that this model's best posterior is one NUTS cannot draw here. Width 6 rather than 1 because
    # that is where the diverging variant sat (discarded/rare-slab-halfnormal-wide, +5.12 at width 6).
    rare = jnp.asarray(X, dtype=jnp.float32).sum(axis=0) <= 15.0
    beta_common = numpyro.sample("beta_common", dist.StudentT(4.0, jnp.zeros(p), 2.0))
    beta_rare_raw = numpyro.sample("beta_rare_raw", dist.Cauchy(jnp.zeros(p), 1.0))
    rare_scale = numpyro.sample("rare_scale", dist.HalfNormal(6.0 * jnp.ones(p)))
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
