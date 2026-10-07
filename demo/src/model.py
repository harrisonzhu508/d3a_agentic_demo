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

# Feature metadata, set by src/features.py when it builds the design matrix (only features.py sees the column
# names). QRDR_ALLELE_GROUPS: one array of column indices per ciprofloxacin target gene, listing that gene's
# allele columns; None means no grouping penalty is applied.
QRDR_ALLELE_GROUPS = None


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
    # The champion needs very large effects on a few QRDR alleles (gyrA D87N ~ +11, parC S80I ~ +9 log2
    # units) - the MIC really does leave the plate for those isolates - so width 2 is already close to the
    # posterior of the alleles that matter. A Student-t(4, 0, 2) prior has that width in the middle but
    # leaves the tail open, so the alleles that matter are barely regularised while the ~70 with no
    # fluoroquinolone mechanism stay shrunk. (A half-normal scale mixture was tried first and diverged;
    # see discarded/prior-width-4: the same gain came only when the width itself was raised.)
    # The QRDR allele columns are alternatives at two codons of the same two targets - gyrA_D87N and parC_S80I
    # correlate at 0.95 - so the likelihood does not identify their individual effects, and the champion spends
    # that freedom on gyrA_D87N +38 against glpT_E448K -14 (a 52-doubling spread that cancels for every isolate
    # inside the tested range). Every prior that shrinks coefficients has already been rejected: halving the
    # width costs 35.3 ELPD, the horseshoe family gains 29 but cannot be sampled, a gene-level hierarchical
    # effect costs 2.6 with 70 divergences, a ridge on within-gene deviations costs 17.8. What has not been
    # penalised is the specific contrast the mechanism says is spurious: *differences between the alleles of one
    # gene*, with no penalty on the level of a gene and no penalty at all on the ~60 non-QRDR columns, whose
    # wide prior is what buys the +28.9. Penalty: sum over the 4 target genes, over their alleles j,
    # (beta_j - mean beta of that gene)^2 / (2 * 1^2) - one doubling, since gyr codon-83 and codon-87 substitutions
    # alter the same drug-binding pocket and differ by less than that (Hooper & Jacoby 2015, Table 1; Huseby et
    # al. 2017 on resistance mutations being small individual steps that combine). Centred on the group mean, so
    # the level of each gene, which is the identified part, is untouched and no prior mass moves the fit.
    beta = numpyro.sample("beta", dist.StudentT(4.0, jnp.zeros(p), 2.0))
    groups = QRDR_ALLELE_GROUPS
    if groups is not None:                        # list of column-index arrays, one per target gene
        pen = []
        for idx in groups:
            g = jnp.asarray(idx, dtype=jnp.int32)
            dev = beta[g] - beta[g].mean()
            pen.append(jnp.where(g.size > 1, jnp.sum(dev ** 2), 0.0))
        numpyro.factor("allele_ridge", -0.5 * jnp.sum(jnp.stack(pen)))
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
