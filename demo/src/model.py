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
    # Thirty-fifth experiment, continuing on the one parameter this session has found still worth ELPD. The error
    # scale has now been measured twice against the champion's HalfNormal(2), both times clean (0 divergences, R-hat
    # under 1.009, 13 s) and both times positive at the best keep-ratio this session has seen: a half-Cauchy at 4
    # doublings, half the width of the dilution series, gains +4.55 +/- 2.41; at the champion's own scale of 2 it
    # gains +4.06 +/- 2.28. That the scale matters is a structural fact about this likelihood, not a coincidence: a
    # right-censored row contributes 1 - Phi((hi - mu)/s), flat in mu once mu is past hi + 3s, so the 204 isolates
    # above the top of the plate - 56% of the cross-validated loss - carry information about s and almost none about
    # mu. That the *family* barely matters (4 vs 2 doublings of half-Cauchy differ by 0.5 ELPD out of a 2.4 SE, and
    # 4 vs 2 of half-normal differed by +0.60) says the gain is the Cauchy's tail rather than its location: the
    # posterior mean sits at 6.8 doublings with a 95% interval of 5.2-8.7, and a half-normal's log density is
    # quadratic in s, which truncates that interval from above. The remaining unknown is the one that decides
    # whether this axis has anything left: is the gain the heaviness of the tail or merely the absence of the
    # Gaussian penalty? Gamma(2, 0.5) has mean 4 - the centre of the measured posterior - a mode at 2, and a log
    # density that is linear-plus-log in s, so it decays like exp(-s/2): much *wider* than the half-normal at the
    # posterior mode (density ratio 2.7x at s = 6.8) but genuinely light above 12, where the half-Cauchy leaves the
    # door open. If the gain survives, what the censored rows wanted was room around the mode and the axis is done;
    # if it collapses, they wanted the upper tail, and a proper heavy-tailed family (a scale-mixture t, or an
    # inverse-gamma) is where the remaining density is.
    # Seventy-sixth experiment: the one property of the residual prior that every measurement on this axis has held
    # fixed while everything else moved. The kept fit carries InvGamma(2, 128) on the residual scale, and six runs
    # have now varied its two parameters: the rate by quarter-steps in both directions (120 costs 0.23 +/- 0.39, 160
    # costs 0.10 +/- 0.62), the rate fourfold across the session (192 and 256 gained +7.2 and +6.9 against the
    # previous champion and were stopped by their own SEs and gates), and the shape from 2 to 3 at constant mean
    # (+0.61 with one divergence). Read together they say the fit is flat in the prior's location and mildly
    # responsive to its curvature, and neither of those is the property the censored rows are actually arguing
    # about. An InvGamma(a, b) has a survival that falls as s^-a beyond its mode: a = 2 is an s^-5 tail on the scale
    # itself, and a = 2.5 at the same mean (b = a(a-1)*128 = 480, mean 128, mode 38.4 of the precision hence a scale
    # mode of 4.4 doublings) is an s^-6.5 tail - the same prior mass in the region the posterior occupies, markedly
    # less in the far region past 40 doublings where the current fit puts 5% of it and where the assay's own
    # repeatability is 4-8 doublings. Two independent reasons to expect a small gain and a smaller SE. The first is
    # this loop's own finding about where the paired SE lives: the pointwise held-out log-density difference between
    # a wide-s fit and the champion has sd 0.157 across the 98 isolates whose MIC was measured against 0.138 across
    # the 204 whose MIC was not, so the measured rows pay most of the bill per row, and they pay it in proportion to
    # how much density the prior lends to scales no dilution series can have observed - which is the far tail and not
    # the mode. The second is the disagreement between groups that the same measurement identified: a tail exponent
    # is the only knob that gives the censored rows their plateau without paying for it in the far region, because it
    # taxes the region rather than the location. If this comes back flat, the prior's shape has been shown not to
    # matter at all over an order of magnitude in its two parameters and an exponent, and the conclusion is the
    # strongest form of what the last six runs have been circling: the residual scale of a censored MIC regression
    # is not identified by these data, and any statement about assay repeatability that this model appears to make is
    # the analyst's and not the experiment's.
    rare = jnp.asarray(X, dtype=jnp.float32).sum(axis=0) <= 15.0
    beta_common = numpyro.sample("beta_common", dist.StudentT(4.0, jnp.zeros(p), 2.0))
    beta_rare = numpyro.sample("beta_rare", dist.StudentT(4.0, jnp.zeros(p), 8.0))
    numpyro.factor("rare_tail", -jnp.sum(jax.nn.softplus(jnp.abs(beta_rare) - 16.0)))
    beta = numpyro.deterministic("beta", jnp.where(rare, beta_rare, beta_common))
    scale_raw = numpyro.sample("scale_raw", dist.Gamma(2.5, 480.0))
    scale = numpyro.deterministic("scale", 1.0 / scale_raw)
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
