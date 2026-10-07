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

# Column names of the design matrix, set by src/features.py in build() (only it sees them). The prior needs them:
# the fluoroquinolone target-gene alleles are the one group of columns the dilution assay resolves.
FEATURE_NAMES: list[str] = []


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
    # Twenty-sixth experiment. The control (a prior identical to the champion's, assembled from two sites with an
    # inactive cut) scored +1.90 +/- 2.25 and the same construction twice more gave +2.40 and +2.79, so about +2 of
    # ELPD is the harness's own floor for any change to this model, and every gain in the +2 to +3.5 band on the
    # rare-column axis has to be read against that floor - which closes it: nothing measured on that axis exceeds
    # its own noise. What has *not* been done is to use the prior to say something about the assay's resolution,
    # which is what the model needs. The likelihood here cannot see differences below the dilution step, so a
    # determinant that changes the MIC by less than half a doubling is unmeasurable by construction: a twofold
    # change in the MIC of the isolates carrying it moves them by 1 log2 unit, and the plate reports at
    # half-doubling resolution, so its effect is buried in the rounding noise of the medium. That is the standard
    # reason a MIC model shrinks small effects hard and lets the target-gene steps run free (the clinical breakpoint
    # literature is explicit that a one-dilution MIC difference is not interpretable), and it is a constraint on
    # *magnitude*, not on which columns are in the design. So: a Cauchy(0, 0.5) prior - half a doubling, the
    # resolution of the test - on the 74 columns with no target-gene mechanism, and the champion's t(4, 0, 2) kept
    # exactly on the three QRDR columns whose effects the plate does resolve (gyrA_D87N, gyrA_S83L, parC_S80I: the
    # common alleles, 179-187 carriers, that carry the resistance steps). This is not shrinkage for its own sake -
    # a beta-lactamase gene has no route to a fluoroquinolone MIC except by co-selection, which the mechanism
    # reference treats as a confound to be removed rather than an effect to be fitted.
    qrdr_common = jnp.array([j for j, n in enumerate(FEATURE_NAMES)
                             if n in ("gyrA_D87N", "gyrA_S83L", "parC_S80I")], dtype=jnp.int32)
    beta_rare = numpyro.sample("beta_rare", dist.Cauchy(jnp.zeros(p), 0.5))
    beta_qrdr = numpyro.sample("beta_qrdr", dist.StudentT(4.0, jnp.zeros(p), 2.0))
    in_qrdr = jnp.zeros(p, dtype=bool).at[qrdr_common].set(True)
    beta = numpyro.deterministic("beta", jnp.where(in_qrdr, beta_qrdr, beta_rare))
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
