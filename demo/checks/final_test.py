"""Humans only: score the current branch on the locked test set (never run by or shown to the agent).

    uv run python checks/final_test.py
Fits the branch's model on all development isolates and scores the held-out test isolates in
$D3A_PRIVATE_DIR/test.csv (ELPD, within ±1 dilution, 90 % coverage).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import checks.lib as lib  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    features, model, sampler = lib.import_branch()
    dev, test = lib.load_dev(), pd.read_csv(lib.private_dir() / "test.csv")
    Xd, _ = features.build(dev)
    Xt, _ = features.build(test)
    lod, hid = lib.intervals(dev)
    lot, hit = lib.intervals(test)
    latent = lib.latent_sites(model.model, np.asarray(Xd, float), lod, hid)
    mcmc = lib.fit(model.model, np.asarray(Xd, float), lod, hid, sampler.SETTINGS, seed=lib.config()["seed"])
    out = lib.heldout(model, mcmc, latent, np.asarray(Xt, float), lot, hit, seed=7)
    within, covered = lib.agreement_and_coverage(out["draws"], test, lib.dilution_uppers(dev))
    st = lib.git_state()
    print(f"{st['branch']}@{st['commit']} on the locked test set ({len(test)} isolates): "
          f"ELPD {out['elpd'].sum():.1f}, within ±1 dilution {within.mean():.1%}, 90% coverage {covered.mean():.1%}")


if __name__ == "__main__":
    main()
