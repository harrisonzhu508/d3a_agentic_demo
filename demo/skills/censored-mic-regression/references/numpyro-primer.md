# NumPyro primer for this project

NumPyro is a probabilistic programming library on JAX: a model is a plain Python function, and inference
(NUTS here) is compiled with JAX. Docs: <https://num.pyro.ai>.

## A model is a function

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def model(X, y=None):
    n, p = X.shape
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 3.0))          # latent variable
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(p), 1.0))   # vector-valued: one site, shape (p,)
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    mu = numpyro.deterministic("mu", alpha + X @ beta)              # stored in the samples, not sampled
    with numpyro.plate("obs", n):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)          # observed site: y given
```

- `numpyro.sample(name, dist)` declares a random variable; with `obs=` it is conditioned on data.
- `numpyro.deterministic(name, value)` records a derived quantity in the posterior samples.
- `numpyro.factor(name, log_prob)` adds an arbitrary term to the log density. Use it for likelihoods that
  are not a standard distribution, such as the interval-censored likelihood in `scripts/fit.py`.
  Note: `factor` sites are not "observed", so ArviZ will not build a `log_likelihood` group automatically.
  Store the pointwise terms with `numpyro.deterministic` and add the group yourself (see `fit.py`).
- `numpyro.plate` marks conditionally independent dimensions (needed for subsampling and some reparameterisations).
- Names must be unique; they are the keys of the posterior dictionary.

## Inference with NUTS

```python
import jax
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)        # call before JAX does any work, to run chains in parallel on CPU
kernel = NUTS(model, target_accept_prob=0.95, max_tree_depth=10)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=4)
mcmc.run(jax.random.PRNGKey(0), X, y)
mcmc.print_summary()                    # mean, sd, quantiles, n_eff, r_hat per site
divergences = mcmc.get_extra_fields()["diverging"].sum()
```

- `target_accept_prob` 0.8 is the default; raise it (0.95 to 0.99) when you see divergences. That shrinks
  the step size; it helps, but a reparameterisation usually helps more.
- `dense_mass=True` learns a full mass matrix (useful for strongly correlated posteriors, costly in high dimension).
- Initialisation: `NUTS(model, init_strategy=numpyro.infer.init_to_median(num_samples=15))` is robust.
- Double precision: `numpyro.enable_x64()` at start-up if you see numerical trouble in tails (log CDFs).

## Reparameterisation (divergences, funnels)

Hierarchical scales and shrinkage priors create funnels. Use the non-centred form:
`z ~ Normal(0, 1)`, `beta = z * scale`, which is what `fit.py` does for the horseshoe. The same can be done
automatically with `numpyro.handlers.reparam(config={"beta": LocScaleReparam(centered=0)})`
(`from numpyro.infer.reparam import LocScaleReparam`).

## JAX pitfalls that break samplers

- **`jnp.where` does not stop gradients from the unused branch.** If the unused branch is `inf` or `nan`
  (for example `log_ndtr((inf - mu) / sigma)`), the gradient becomes `nan` and NUTS gets stuck (every
  transition diverges, R-hat is infinite). Replace the bad inputs with safe placeholders *before* the
  computation, then select (see `log_interval_prob` in `fit.py`).
- Use `jax.scipy.special.log_ndtr` for log normal CDFs, not `jnp.log(norm.cdf(x))`, which underflows.
- Shapes are static inside compiled code: avoid Python `if` on array values; use `jnp.where`.
- Random numbers need explicit keys (`jax.random.split`).

## Predictions and posterior predictive checks

```python
from numpyro.infer import Predictive
pred = Predictive(model, posterior_samples=mcmc.get_samples())(jax.random.PRNGKey(1), X_new)
```

For a censored model, simulate the latent `y* ~ Normal(mu, sigma)` and map it onto the dilution grid
(bin it) before comparing with observed MICs.

## Saving for ArviZ

```python
import arviz as az
idata = az.from_numpyro(mcmc, coords={"feature": names}, dims={"beta": ["feature"]})
```

ArviZ 1.x stores results as an xarray `DataTree`: add groups by assignment,
`idata["log_likelihood"] = dataset`, and write with `idata.to_netcdf(path)` (needs `h5netcdf` and `h5py`).
