"""Shared harness code (read-only for agents): data, fitting, held-out scoring, diagnostics.

Import this module before anything else imports JAX: it sets the host device count so NUTS chains run in parallel.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import importlib
import json
import subprocess
import sys
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.special import logsumexp
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
from checks.settings import (branch_namespace, experiment_branch, experiment_dir, run_dir,  # noqa: E402,F401
                             RESULTS, rel, session_file, settings, slug)

META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]


# ------------------------------------------------------------------ configuration and data
def config() -> dict:
    return json.loads((DEMO / "checks" / "harness.json").read_text())


def private_dir() -> Path:
    return Path(os.environ.get("D3A_PRIVATE_DIR", config()["private_dir_default"])).expanduser()


def load_dev() -> pd.DataFrame:
    return pd.read_csv(DEMO / "data" / "dev.csv")


def load_folds() -> dict[str, int]:
    return json.loads((DEMO / "checks" / "folds.json").read_text())


def intervals(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df["log2_lo"].to_numpy(float), df["log2_hi"].to_numpy(float)


def dilution_uppers(df: pd.DataFrame) -> np.ndarray:
    """Upper edges (log2) of the on-grid dilution bins; bin 0 = lowest (<=), last bin = '>' (censored)."""
    return np.sort(df.loc[df["censor"] == "=", "log2_hi"].unique())


def to_bin(y, uppers: np.ndarray) -> np.ndarray:
    return np.searchsorted(uppers, np.asarray(y), side="left")


def observed_bin(df: pd.DataFrame, uppers: np.ndarray) -> np.ndarray:
    hi = df["log2_hi"].where(df["censor"] == "=", np.inf).to_numpy(float)
    return to_bin(hi, uppers)


# ------------------------------------------------------------------ branch code
def import_branch():
    """Import the branch's evolvable modules (src/features.py, src/model.py, src/sampler.py) fresh."""
    mods = []
    for name in ("src.features", "src.model", "src.sampler"):
        if name in sys.modules:
            mods.append(importlib.reload(sys.modules[name]))
        else:
            mods.append(importlib.import_module(name))
    return tuple(mods)


def latent_sites(model_fn, X, lo, hi) -> list[str]:
    """Names of the unobserved sample sites (what NUTS samples; deterministic sites excluded)."""
    k = min(8, len(X))
    tr = handlers.trace(handlers.seed(model_fn, 0)).get_trace(jnp.asarray(X[:k]), jnp.asarray(lo[:k]), jnp.asarray(hi[:k]))
    return [n for n, s in tr.items() if s["type"] == "sample" and not s["is_observed"]]


# ------------------------------------------------------------------ fitting and scoring
def fit(model_fn, X, lo, hi, settings: dict, seed: int) -> MCMC:
    kernel = NUTS(model_fn, target_accept_prob=settings.get("target_accept_prob", 0.9),
                  max_tree_depth=settings.get("max_tree_depth", 10), dense_mass=settings.get("dense_mass", False))
    mcmc = MCMC(kernel, num_warmup=settings["num_warmup"], num_samples=settings["num_samples"],
                num_chains=settings["num_chains"], chain_method="parallel", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), jnp.asarray(X), jnp.asarray(lo), jnp.asarray(hi))
    return mcmc


def diagnostics(mcmc: MCMC, latent: list[str], effects: str | None) -> dict:
    """Divergences, max rank-normalised R-hat and min bulk/tail ESS over the latent sites and the effects site."""
    idata = az.from_numpyro(mcmc)
    names = [n for n in latent + ([effects] if effects else []) if n in idata.posterior.data_vars]
    summ = az.summary(idata, var_names=names)
    return {
        "divergences": int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
        "max_rhat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "min_ess_tail": float(summ["ess_tail"].min()),
    }


def heldout(model_mod, mcmc: MCMC, latent: list[str], X_te, lo_te, hi_te, seed: int) -> dict:
    """Pointwise held-out ELPD, plus latent draws for agreement and coverage."""
    samples = mcmc.get_samples()
    latent_only = {k: v for k, v in samples.items() if k in latent}   # drop deterministic sites (training shapes)
    pred = Predictive(model_mod.model, posterior_samples=latent_only, return_sites=["log_lik"])
    ll = pred(jax.random.PRNGKey(seed), jnp.asarray(X_te), jnp.asarray(lo_te), jnp.asarray(hi_te))["log_lik"]
    S = ll.shape[0]
    elpd_i = np.asarray(logsumexp(ll, axis=0) - jnp.log(S))
    # recompute the model's deterministic sites (effects, means) for the held-out rows, then simulate
    tr = handlers.trace(handlers.seed(model_mod.model, 0)).get_trace(jnp.asarray(X_te))
    det = [n for n, s in tr.items() if s["type"] == "deterministic"]
    full = Predictive(model_mod.model, posterior_samples=latent_only, return_sites=det)(
        jax.random.PRNGKey(seed + 1), jnp.asarray(X_te)) if det else {}
    draws = np.asarray(model_mod.simulate({**latent_only, **full}, X_te, jax.random.PRNGKey(seed + 2)))
    return {"elpd": elpd_i, "draws": draws}


def agreement_and_coverage(draws: np.ndarray, df_te: pd.DataFrame, uppers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per isolate: within ±1 dilution (median prediction vs observed bin) and 90 % interval overlap."""
    pred_bin = to_bin(np.median(draws, axis=0), uppers)
    within = np.abs(pred_bin - observed_bin(df_te, uppers)) <= 1
    q05, q95 = np.percentile(draws, [5, 95], axis=0)
    lo, hi = intervals(df_te)
    covered = (q95 > lo) & (q05 <= hi)
    return within, covered


def parsimony(mcmc: MCMC, effects: str | None, threshold: float, prob: float) -> int | None:
    if not effects:
        return None
    b = np.asarray(mcmc.get_samples()[effects])
    return int(((np.abs(b) > threshold).mean(axis=0) >= prob).sum())


def save_idata(mcmc: MCMC, path: Path, feature_names: list[str], isolate_ids: list[str]) -> None:
    samples = mcmc.get_samples()
    dims, p, n = {}, len(feature_names), len(isolate_ids)
    for k, v in samples.items():
        if v.ndim == 2 and v.shape[1] == p:
            dims[k] = ["feature"]
        elif v.ndim == 2 and v.shape[1] == n:
            dims[k] = ["isolate"]
    idata = az.from_numpyro(mcmc, coords={"feature": feature_names, "isolate": isolate_ids}, dims=dims)
    if "log_lik" in idata.posterior.data_vars:
        idata["log_likelihood"] = idata.posterior.to_dataset()[["log_lik"]].rename({"log_lik": "y"})
    path.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(path)


# ------------------------------------------------------------------ git and run state
def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=DEMO, capture_output=True, text=True).stdout.strip()


def git_state() -> dict:
    return {
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "src")),
    }


def champion() -> dict | None:
    f = experiment_dir() / "champion.json"
    return json.loads(f.read_text()) if f.exists() else None


def reference() -> tuple[str, str, dict[str, float], dict]:
    """What a new branch must beat: (label, parent branch, {genome_id: elpd}, summary metrics).

    The experiment's champion (its best kept branch) if there is one, else the baseline in checks/best.json,
    which is the code the experiment branch starts from."""
    ch = champion()
    if ch:
        f = ch["pointwise_file"]
        pw = pd.read_csv(RESULTS / f.removeprefix("results/") if f.startswith("results/") else DEMO / f)
        mf = run_dir(ch["name"]) / "metrics.json"
        m = json.loads(mf.read_text()) if mf.exists() else {"elpd_cv": ch["elpd_cv"]}
        summary = {k: m.get(k) for k in ("elpd_cv", "within1", "coverage90", "parsimony")}
        return f"champion {ch['branch']}", ch["branch"], dict(zip(pw["genome_id"], pw["elpd"])), summary
    best = DEMO / "checks" / "best.json"
    if best.exists():
        b = json.loads(best.read_text())
        summary = {k: b.get(k) for k in ("elpd_cv", "within1", "coverage90", "parsimony")}
        return "the baseline (checks/best.json)", experiment_branch(), b["pointwise"], summary
    return "none", experiment_branch(), {}, {}
