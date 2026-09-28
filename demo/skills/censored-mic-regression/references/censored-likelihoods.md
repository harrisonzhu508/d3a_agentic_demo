# Censored likelihoods for dilution-series data

## Why MICs are intervals

A minimum inhibitory concentration (MIC) is read from a two-fold dilution series (..., 0.125, 0.25, 0.5, ...
mg/L): the MIC is the lowest concentration with no visible growth. If growth stops at 0.25 but not at 0.125,
the true inhibitory concentration lies somewhere in (0.125, 0.25]. On the log2 scale every reading is an
interval of width 1. Values beyond the tested range are one-sided:

| Reported | Meaning | log2 interval |
|---|---|---|
| `0.25` | stops between 0.125 and 0.25 | (log2 0.125, log2 0.25] = (-3, -2] |
| `>4` | still grows at the highest concentration | (2, +inf) |
| `<=0.008` or the lowest dilution tested | no growth even at the lowest concentration | (-inf, log2 0.008] |

Michael, Kelman & Pitesky (2020) *Animals* 10:1405 review this interval-censoring view of MIC data.

## The interval likelihood

With a latent continuous log2 MIC `y* ~ Normal(mu, sigma)`:

```
p(obs | mu, sigma) = Phi((hi - mu) / sigma) - Phi((lo - mu) / sigma)
```

- right-censored (`hi = +inf`): `1 - Phi((lo - mu)/sigma) = Phi(-(lo - mu)/sigma)`;
- left-censored (`lo = -inf`): `Phi((hi - mu)/sigma)`.

Compute everything on the log scale (`log_ndtr`), and for a proper interval use
`log Phi(b) + log1p(-exp(log Phi(a) - log Phi(b)))`. This is the Tobit model generalised to intervals.

## Common shortcuts and why they bias results

- Replacing `>x` by `2x` and `<=x` by `x/2`, then fitting ordinary least squares, is common in the MIC
  prediction literature, where it is used to compute "within one dilution" accuracy. Inside a likelihood it
  shrinks effects towards the middle of the grid when many values are censored, and it understates
  uncertainty at the plate limits.
- Treating the reported MIC as the exact value ignores the within-interval uncertainty (width 1 on log2),
  which matters for well-calibrated predictive intervals.

## Alternatives the agent may consider

- **Heavier tails**: a logistic or Student-t latent in place of the normal (the logistic CDF has a closed form
  via `jax.nn.log_sigmoid`), for data with occasional gross outliers.
- **Mixtures**: a two-component model in which a small fraction of observations follow a broad distribution
  that ignores the covariates. Mixtures can be hard to sample; start from a centred parameterisation of the
  mixing weight and check for label switching and divergences.
- **Ordinal models**: treat the dilution bins as ordered categories with a cumulative link (ordered logit or
  probit); equivalent in spirit, with cut points fixed at the dilution boundaries.
- **Measurement error**: repeat MIC testing typically agrees within one doubling dilution, so an extra noise
  term of that size is plausible.

## Ready-to-use: a censored Student-t likelihood in NumPyro

Do not derive the Student-t CDF by hand (incomplete-beta formulas are easy to get wrong and give NaN gradients
in float32): NumPyro's `dist.StudentT(nu).cdf` is exact and differentiable. This drop-in replacement for the
normal interval probability is checked against `scipy.stats.t` and has finite gradients for open-ended
intervals:

```python
import jax.numpy as jnp
import numpyro.distributions as dist


def t_log_interval_prob(lo, hi, mu, sigma, nu):
    """log P(lo < Y <= hi) for Y ~ StudentT(nu, mu, sigma); lo may be -inf and hi +inf. NaN-safe gradients."""
    t = dist.StudentT(nu)
    lo_inf, hi_inf = jnp.isneginf(lo), jnp.isposinf(hi)
    a = jnp.where(lo_inf, hi - 1.0, lo)            # finite placeholders, so no branch sees inf
    b = jnp.where(hi_inf, a + 1.0, hi)
    za, zb = (a - mu) / sigma, (b - mu) / sigma
    upper = jnp.log(t.cdf(-za))                    # P(Y > lo) when hi = +inf (symmetry)
    lower = jnp.log(t.cdf(zb))                     # P(Y <= hi) when lo = -inf
    tiny = jnp.finfo(za.dtype).tiny
    right = za > 0                                 # take the difference in the tail where it is precise
    mid = jnp.log(jnp.clip(jnp.where(right, t.cdf(-za) - t.cdf(-zb), t.cdf(zb) - t.cdf(za)), tiny))
    return jnp.where(hi_inf, upper, jnp.where(lo_inf, lower, mid))
```

Put a prior on the degrees of freedom (for example `nu ~ Gamma(2, 0.1)`, or fix it at 4 to 10 first), keep
`numpyro.deterministic("log_lik", ...)` with this function, and draw `simulate` from the same Student-t
(`mu + sigma * jax.random.t(key, nu, mu.shape)`) so the predictive checks match the likelihood.

## Further reading

- Jaspers, Lambert & Aerts (2016) *Ann Appl Stat* 10(2), doi:10.1214/16-AOAS918: a Bayesian model for
  interval-censored MIC distributions.
- Tobin (1958) *Econometrica* 26:24 (the Tobit model).
