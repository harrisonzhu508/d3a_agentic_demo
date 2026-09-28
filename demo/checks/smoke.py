"""Fast contract check for the branch's src/ modules (under a minute). Exit 0 = OK, 1 = broken (with reasons).

    uv run python checks/smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import checks.lib as lib  # noqa: E402  (sets JAX flags before JAX is imported)

import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from checks.lib import import_branch, intervals, latent_sites, load_dev


def main() -> int:
    t0 = time.time()
    problems: list[str] = []
    try:
        features, model, sampler = import_branch()
    except Exception:
        print("FAIL: importing src/ raised:\n" + traceback.format_exc(limit=3))
        return 1

    df = load_dev().head(150)
    lo, hi = intervals(df)
    try:
        X, names = features.build(df)
        X = np.asarray(X, dtype=float)
        if X.shape != (len(df), len(names)):
            problems.append(f"features.build returned X of shape {X.shape} but {len(names)} names for {len(df)} rows")
        if not np.isfinite(X).all():
            problems.append("features.build returned non-finite values")
        shuffled = df.copy()
        for c in ["mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]:
            shuffled[c] = shuffled[c].sample(frac=1.0, random_state=0).to_numpy()
        if not np.array_equal(np.asarray(features.build(shuffled)[0], dtype=float), X):
            problems.append("features.build depends on the MIC columns (label leakage)")
    except Exception:
        print("FAIL: features.build raised:\n" + traceback.format_exc(limit=3))
        return 1

    keys = {"num_warmup", "num_samples", "num_chains", "target_accept_prob", "max_tree_depth", "dense_mass"}
    if not keys <= set(getattr(sampler, "SETTINGS", {})):
        problems.append(f"sampler.SETTINGS must contain {sorted(keys)}")
    if not hasattr(model, "simulate"):
        problems.append("model.simulate(samples, X, key) is missing")

    try:
        latent = latent_sites(model.model, X, lo, hi)
        mcmc = lib.fit(model.model, X, lo, hi, {"num_warmup": 150, "num_samples": 100, "num_chains": 1,
                                                  "target_accept_prob": 0.9}, seed=0)
        s = mcmc.get_samples()
        if "log_lik" not in s or s["log_lik"].shape[-1] != len(df):
            problems.append('model must record numpyro.deterministic("log_lik", ...) with shape (n,) when lo/hi are given')
        elif not np.isfinite(np.asarray(s["log_lik"])).all():
            problems.append("log_lik contains non-finite values (check the censored likelihood)")
        out = lib.heldout(model, mcmc, latent, X[:20], lo[:20], hi[:20], seed=1)
        if out["draws"].ndim != 2 or out["draws"].shape[1] != 20:
            problems.append(f"model.simulate must return (draws, n); got {out['draws'].shape}")
        elif not np.isfinite(out["draws"]).all():
            problems.append("model.simulate returned non-finite draws")
        eff = getattr(model, "FEATURE_EFFECTS", None)
        if eff and (eff not in s or s[eff].shape[-1] != X.shape[1]):
            problems.append(f"FEATURE_EFFECTS='{eff}' is not a site of shape (n_features,)")
    except Exception as err:
        print("FAIL: fitting a tiny model raised:\n" + explain(err))
        return 1

    if problems:
        print("FAIL:\n- " + "\n- ".join(problems))
        return 1
    print(f"OK: contract satisfied ({X.shape[1]} features, latent sites: {', '.join(latent)}; {time.time() - t0:.0f} s)")
    return 0


def explain(err: BaseException) -> str:
    """The error, the lines of src/ it came through, and the innermost frames (where it was raised)."""
    frames = traceback.extract_tb(err.__traceback__)
    ours = [f for f in frames if "/src/" in f.filename]
    out = [f"{type(err).__name__}: {str(err)[:1200]}"]
    if ours:
        out.append("in your code:")
        out += [f"  src/{Path(f.filename).name}:{f.lineno} in {f.name}: {(f.line or '').strip()}" for f in ours]
    out.append("innermost frames:")
    out += [f"  {Path(f.filename).name}:{f.lineno} in {f.name}: {(f.line or '').strip()}" for f in frames[-4:]]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
