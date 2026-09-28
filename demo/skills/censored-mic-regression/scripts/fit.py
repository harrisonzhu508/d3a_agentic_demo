"""Fit an interval-censored Bayesian regression of log2 MIC on binary determinants (NumPyro, NUTS).

    uv run --with numpyro --with jax --with arviz --with h5netcdf --with h5py --with pandas \
        demo/skills/censored-mic-regression/scripts/fit.py demo/data/dev.csv out/

Each isolate contributes P(lo < y* <= hi) with y* ~ Normal(alpha + x @ beta, sigma), where [lo, hi] is its
log2-MIC interval on the dilution grid (lo = -inf at the lowest dilution tested, hi = +inf for ">x").
beta has a regularised horseshoe prior (Piironen & Vehtari 2017): most determinants have no effect.
Writes posterior.nc (ArviZ InferenceData) and summary.csv to the output directory.
"""

import sys
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax.scipy.special import log_ndtr
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)
META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]


def log_interval_prob(lo, hi, mu, sigma):
    """log P(lo < y <= hi) for y ~ Normal(mu, sigma); lo may be -inf and hi may be +inf.

    Every branch is computed on finite placeholder bounds so that jnp.where never multiplies an
    infinite gradient by zero (which would give NaN gradients and a stuck sampler).
    """
    lo_inf, hi_inf = jnp.isinf(lo), jnp.isinf(hi)
    lo_f = jnp.where(lo_inf, hi - 1.0, lo)  # placeholders keep the interval branch valid
    hi_f = jnp.where(hi_inf, lo + 1.0, hi)
    a, b = (lo_f - mu) / sigma, (hi_f - mu) / sigma
    log_upper, log_lower = log_ndtr(b), log_ndtr(a)
    interval = log_upper + jnp.log1p(-jnp.exp(log_lower - log_upper))
    return jnp.where(hi_inf, log_ndtr(-a), jnp.where(lo_inf, log_upper, interval))


def model(X, lo=None, hi=None, expected_nonzero=8):
    n, p = X.shape
    alpha = numpyro.sample("alpha", dist.Normal(-4.0, 3.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
    tau0 = expected_nonzero / (p - expected_nonzero) * sigma / jnp.sqrt(n)
    tau = numpyro.sample("tau", dist.HalfCauchy(tau0))
    lam = numpyro.sample("lambda", dist.HalfCauchy(jnp.ones(p)))
    c2 = numpyro.sample("c2", dist.InverseGamma(2.0, 2.0 * 3.0**2))  # slab: effects up to ~3 doublings
    lam_tilde = jnp.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2))
    z = numpyro.sample("z", dist.Normal(jnp.zeros(p), 1.0))
    beta = numpyro.deterministic("beta", z * tau * lam_tilde)
    mu = numpyro.deterministic("mu", alpha + X @ beta)
    if lo is not None:
        ll = log_interval_prob(lo, hi, mu, sigma)
        numpyro.factor("censored_lik", ll.sum())
        numpyro.deterministic("log_lik", ll)


def main(csv: str, out: str) -> None:
    d = pd.read_csv(csv)
    feats = [c for c in d.columns if c not in META]
    X = jnp.asarray(d[feats].to_numpy(float))
    lo, hi = jnp.asarray(d["log2_lo"].to_numpy(float)), jnp.asarray(d["log2_hi"].to_numpy(float))
    mcmc = MCMC(NUTS(model, target_accept_prob=0.95), num_warmup=1000, num_samples=1000, num_chains=4)
    mcmc.run(jax.random.PRNGKey(0), X, lo, hi)
    idata = az.from_numpyro(mcmc, coords={"feature": feats, "isolate": d["genome_id"].tolist()},
                            dims={"beta": ["feature"], "lambda": ["feature"], "z": ["feature"],
                                  "mu": ["isolate"], "log_lik": ["isolate"]})
    idata["log_likelihood"] = idata.posterior.to_dataset()[["log_lik"]].rename({"log_lik": "y"})  # ArviZ 1.x DataTree
    Path(out).mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(Path(out) / "posterior.nc")
    summ = az.summary(idata, var_names=["alpha", "sigma", "tau", "beta"])
    summ.to_csv(Path(out) / "summary.csv")
    div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    print(summ.sort_values("mean", key=abs, ascending=False).head(12).to_string())
    print(f"divergences: {div}  max R-hat: {summ['r_hat'].max():.3f}  min bulk ESS: {summ['ess_bulk'].min():.0f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
