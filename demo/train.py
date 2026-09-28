"""Fit the branch's model on all development isolates and draw the review plots (exploration; not a score).

    uv run python train.py [--name <run-name>] [--no-plots]

Uses src/features.py, src/model.py and src/sampler.py. Writes results/<experiment>/<name>/posterior.nc and
review.png and prints the sampler diagnostics. Scoring is done only by checks/evaluate.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks.lib as lib  # noqa: E402  (sets JAX flags before JAX is imported)

import argparse  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    features, model, sampler = lib.import_branch()
    df = lib.load_dev()
    X, names = features.build(df)
    X = np.asarray(X, dtype=float)
    lo, hi = lib.intervals(df)
    name = lib.slug(args.name or "explore-" + lib.git_state()["branch"].removeprefix(lib.branch_namespace()))
    out = lib.run_dir(name)
    t0 = time.time()
    latent = lib.latent_sites(model.model, X, lo, hi)
    mcmc = lib.fit(model.model, X, lo, hi, sampler.SETTINGS, seed=lib.config()["seed"])
    diag = lib.diagnostics(mcmc, latent, getattr(model, "FEATURE_EFFECTS", None))
    lib.save_idata(mcmc, out / "posterior.nc", names, df["genome_id"].tolist())
    print(f"fit {len(df)} isolates x {len(names)} features in {time.time() - t0:.0f} s: "
          f"divergences {diag['divergences']}, max R-hat {diag['max_rhat']:.3f}, "
          f"min ESS bulk/tail {diag['min_ess_bulk']:.0f}/{diag['min_ess_tail']:.0f} -> {out / 'posterior.nc'}")
    if not args.no_plots:
        plots = lib.DEMO / "skills" / "mic-plots" / "scripts" / "plots.py"
        subprocess.run([sys.executable, str(plots), str(lib.DEMO / "data" / "dev.csv"), str(out / "posterior.nc"),
                        str(out / "review")], check=False)


if __name__ == "__main__":
    main()
