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
    # Twenty-eighth experiment, and the first to intervene on the pathology rather than on its symptoms. Everything
    # this loop has measured about the champion points at one direction of the posterior: gyrA_D87N sits at +38 and
    # glpT_E448K at -14 log2 units, a 52-doubling spread that cancels for every isolate inside the tested range and
    # explodes only where a genotype is extrapolated. The prior width that produces it (t(4, 0, 2), worth +28.9 over
    # Normal(0,2)) cannot be reduced without losing 30-35 ELPD, and every prior that redistributes it - horseshoe,
    # level-plus-contrast, gene-hierarchical, within-gene ridges, prevalence scaling, spike-and-slab on the rare
    # columns at nine different widths and shapes - either fails the sampler gates or lands inside the +/-2 ELPD noise
    # floor this loop measured for itself. So stop constraining the coefficient and remove the columns that carry it.
    # gyrA_S83L and parC_S80I are present in 187 and 187 isolates and absent in 371 and 371, and gyrA_D87N is
    # present in 181 of them: the D87N column is carried by a subset of the S83L carriers (it never occurs alone -
    # an isolate with both is a double mutant at codons 83 and 87 of the same gene, which is exactly the stepwise
    # path the mechanism describes). A column that is a sub-indicator of another, both of which are near-saturated,
    # is the design's degenerate direction: the sum of their effects is identified and the difference is not, and a
    # wide prior spends the difference on whatever the fold happens to want. Drop the single most redundant column
    # (gyrA_D87N, whose 181 carriers are all S83L carriers) and let S83L carry gyrase codon-83 status: the design
    # loses one of 77 columns, keeps the identified contrast, and the wild-type, single-mutant and double-mutant
    # classes remain separately representable because S83L and D87N differ in whether parC is co-carried - no: they
    # do not, so this is the experiment that tests whether the D87N column carried information at all, since after
    # removing it the model can only express codon-83 status, not which substitution.
    keep = jnp.array([j for j, n in enumerate(FEATURE_NAMES) if n != "gyrA_D87N"], dtype=jnp.int32)
    beta_kept = numpyro.sample("beta_kept", dist.StudentT(4.0, jnp.zeros(p - 1), 2.0))
    # Scatter the kept effects back into a length-p vector with zero on the dropped column, so the effect site keeps
    # the shape the contract asks for (n_features,) and the log_lik stays pointwise.
    beta = numpyro.deterministic("beta", jnp.zeros(p).at[keep].set(beta_kept))
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
