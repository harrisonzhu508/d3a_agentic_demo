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
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import log_ndtr

FEATURE_EFFECTS = "beta"
GENE_GROUPS = None            # set by src/features.py: (gene index per column, number of genes)


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


def _gene_groups(names):
    """Gene/locus identity per column, from the AMRFinderPlus names (gyrA_S83L -> gyrA, blaCTX-M-15 -> blaCTX)."""
    import re
    def gene_of(n):
        return re.match(r"^[A-Za-z]+", n).group(0)
    groups = sorted({gene_of(n) for n in names})
    index = {g: i for i, g in enumerate(groups)}
    return np.array([index[gene_of(n)] for n in names]), len(groups)


def model(X, lo=None, hi=None):
    p = X.shape[1]
    alpha = numpyro.sample("alpha", dist.Normal(-4.0, 3.0))
    # Logistic errors with the same variance as the baseline HalfNormal(2) errors: s = sigma sqrt(3/pi)
    scale = numpyro.sample("scale", dist.HalfNormal(2.0 * jnp.sqrt(3.0 / jnp.pi)))
    # The champion's fit splits the freedom in the QRDR columns into gyrA_D87N +38 against glpT_E448K -14, and
    # every prior tried so far that shrinks coefficients either costs ELPD (width 1: -35.3) or cannot be sampled
    # (the scale mixtures). What is not in question is the *champion's* fit for the columns it identifies; what
    # is wrong is the difference between near-collinear columns. So penalise only differences, and only within
    # a gene: the four gyrA columns are alternative mutations of one target, which the mechanism can alter in
    # one step at each of two codons, so their fitted effects should sit within a couple of doublings of each
    # other, whereas gyrA versus glpT may differ by twenty (Hooper & Jacoby 2015). A ridge on within-gene
    # differences is a soft version of that, keeps 76 free effects for everything else, and adds no hierarchical
    # scale (which is what diverged in discarded/hierarchical-gene-family-effects, 70 divergences).
    if GENE_GROUPS is None:                       # features.py did not label the columns: no grouping
        beta = numpyro.sample("beta", dist.StudentT(4.0, jnp.zeros(p), 2.0))
        mu = numpyro.deterministic("mu", alpha + X @ beta)
        if lo is not None:
            ll = logistic_log_interval_prob(lo, hi, mu, scale)
            numpyro.factor("censored_lik", ll.sum())
            numpyro.deterministic("log_lik", ll)
        return
    gene, n_gene = GENE_GROUPS
    beta_raw = numpyro.sample("beta_raw", dist.StudentT(4.0, jnp.zeros(p), 2.0))
    # Ridge on within-gene deviations of the raw effects: a centred quadratic penalty (the centring by the group
    # mean shrinks differences rather than shifting the whole fit), with no extra site and so no extra funnel.
    # w = 2 doublings: the alleles of one gene may differ, but not by twenty.
    D = jnp.eye(n_gene)[gene]                       # (p, n_gene) indicator matrix
    gcnt = D.sum(axis=0)
    dev = beta_raw - ((beta_raw @ D) / jnp.maximum(gcnt, 1.0))[gene]
    within = jnp.sum(jnp.where(gcnt[gene] > 1, dev ** 2, 0.0)) / (2.0 * 2.0 ** 2)
    numpyro.factor("gene_ridge", -within)
    # One centred offset per gene (centred so it is not confounded with the intercept), moving all of a gene's
    # alleles together - the direction the data do identify, gyrA up and glpT down. Non-centred, sd 2 doublings.
    theta = 2.0 * numpyro.sample("theta_z", dist.Normal(jnp.zeros(n_gene), 1.0))
    beta = numpyro.deterministic("beta", beta_raw + (theta - theta.mean())[gene])
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
