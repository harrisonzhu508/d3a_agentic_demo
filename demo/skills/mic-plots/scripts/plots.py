"""Standard review plots for a fitted MIC model, in the GPAP style (for the human reviewer and the PR).

    uv run --with arviz --with h5netcdf --with h5py --with matplotlib --with pandas \
        demo/skills/mic-plots/scripts/plots.py demo/data/dev.csv out/posterior.nc out/review

Panels: (a) sampler traces, (b) censored-model effects vs a naive least-squares fit, (c) posterior
predictive check on the dilution grid, (d) calibration: predicted vs observed MIC with the ±1-dilution band.
Writes <prefix>.svg and <prefix>.png.
"""

import sys
from pathlib import Path

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[4]
FONTS = REPO / "slides" / "fonts"
INK, ACCENT, ACCENTLIGHT, SLATE, GOOD, BAD, PAPER = "#353B33", "#3D4592", "#B3B3F1", "#8EA0A2", "#52856D", "#8E3B2E", "#FAFAF0"
META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]


def style() -> None:
    for f in FONTS.glob("IBMPlexSans-*.otf"):
        font_manager.fontManager.addfont(str(f))
    mpl.rcParams.update({
        "font.family": "IBM Plex Sans", "font.size": 8, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.7, "svg.fonttype": "path",
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
    })


def dilution_bins(d: pd.DataFrame):
    """Upper edges of the observed dilution bins (log2), with -inf/+inf for the censored ends."""
    uppers = np.sort(d.loc[d["censor"] == "=", "log2_hi"].unique())
    labels = ["≤" + _fmt(uppers[0])] + [_fmt(u) for u in uppers[1:]] + [">" + _fmt(uppers[-1])]
    return uppers, labels


def _fmt(log2v: float) -> str:
    v = 2.0**log2v
    return f"{v:.3g}".lstrip("0") if v < 1 else f"{v:.0f}"


def to_bin(y, uppers):
    return np.searchsorted(uppers, y, side="left")  # 0 = lowest bin (<= lowest), len(uppers) = ">" bin


def main(csv: str, posterior: str, prefix: str) -> None:
    style()
    d = pd.read_csv(csv)
    feats = [c for c in d.columns if c not in META]
    idata = az.from_netcdf(posterior)
    post = idata.posterior
    beta = post["beta"].values.reshape(-1, len(feats))
    mu = post["mu"].values.reshape(-1, len(d))
    sigma = post["sigma"].values.reshape(-1)
    uppers, labels = dilution_bins(d)
    obs_bin = to_bin(d["log2_hi"].where(d["censor"] == "=", np.inf).to_numpy(), uppers)

    fig, ax = plt.subplots(1, 4, figsize=(10.24, 3.55), gridspec_kw={"width_ratios": [1, 1.25, 1.35, 1.1], "wspace": 0.55})

    # (a) traces
    a = ax[0]
    chains = post["beta"].sel(feature="gyrA_S83L").values
    for c, col in zip(chains, [ACCENT, ACCENTLIGHT, SLATE, GOOD]):
        a.plot(c, lw=0.5, color=col, alpha=0.9)
    a.set_title("(a) traces: gyrA S83L", loc="left", fontsize=8.5, fontweight="bold", pad=16)
    a.set_xlabel("draw"); a.set_ylabel("effect (doublings)")
    summ = az.summary(idata, var_names=["beta", "sigma"])
    a.text(0.02, 0.02, f"max R̂ {summ['r_hat'].max():.3f}\nmin ESS {summ['ess_bulk'].min():.0f}", transform=a.transAxes, fontsize=7)

    # (b) forest: censored posterior vs naive OLS on the common '>x -> 2x' convention
    b = ax[1]
    X = d[feats].to_numpy(float)
    ols = np.linalg.lstsq(np.c_[np.ones(len(d)), X], d["log2_mic"].to_numpy(), rcond=None)[0][1:]
    top = np.argsort(-np.abs(np.median(beta, 0)))[:8][::-1]
    lo, med, hi = np.percentile(beta[:, top], [5.5, 50, 94.5], axis=0)
    ys = np.arange(len(top))
    b.hlines(ys, lo, hi, color=ACCENT, lw=1.6)
    b.plot(med, ys, "o", color=ACCENT, ms=3.5, label="censored model")
    b.plot(ols[top], ys, "o", mfc="none", mec=INK, ms=4, mew=0.8, label="naive least squares")
    b.axvline(0, color=SLATE, lw=0.6)
    b.set_yticks(ys, [feats[i].replace("_", " ") for i in top], fontsize=7, fontstyle="italic")
    b.set_xlabel("effect on log$_2$ MIC (doublings), 89% interval")
    b.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, handletextpad=0.2, columnspacing=0.8)
    b.set_title("(b) effects", loc="left", fontsize=8.5, fontweight="bold", pad=16)

    # (c) posterior predictive check on the dilution grid
    c = ax[2]
    rng = np.random.default_rng(0)
    draws = rng.choice(len(sigma), 400, replace=False)
    ystar = mu[draws] + sigma[draws, None] * rng.standard_normal((len(draws), len(d)))
    nb = len(uppers) + 1
    pp = np.stack([np.bincount(to_bin(y, uppers), minlength=nb) for y in ystar])
    obs = np.bincount(obs_bin, minlength=nb)
    x = np.arange(nb)
    c.bar(x, obs, color=ACCENTLIGHT, width=0.8, label="observed")
    pl, pm, ph = np.percentile(pp, [5.5, 50, 94.5], axis=0)
    c.errorbar(x, pm, yerr=[pm - pl, ph - pm], fmt="o", color=INK, ms=2.5, lw=0.8, capsize=1.5, label="posterior predictive")
    c.set_xticks(x, labels, rotation=60, fontsize=6.5)
    c.set_xlabel("MIC (mg/L)"); c.set_ylabel("isolates")
    c.set_title("(c) predictive check", loc="left", fontsize=8.5, fontweight="bold", pad=16)
    c.legend(frameon=False, fontsize=6.5, loc="upper center")

    # (d) calibration: posterior median prediction vs observed bin, ±1 dilution band
    e = ax[3]
    pred_bin = to_bin(np.median(mu, 0), uppers)
    jitter = rng.uniform(-0.25, 0.25, (2, len(d)))
    hit = np.abs(pred_bin - obs_bin) <= 1
    e.fill_between([-0.5, nb - 0.5], [-1.5, nb - 2.5], [0.5, nb - 0.5], color=GOOD, alpha=0.15, lw=0)
    e.scatter(obs_bin + jitter[0], pred_bin + jitter[1], s=3, c=np.where(hit, ACCENT, BAD), lw=0)
    e.set_xticks(x[::2], labels[::2], rotation=60, fontsize=6.5); e.set_yticks(x[::2], labels[::2], fontsize=6.5)
    e.set_xlabel("observed MIC"); e.set_ylabel("predicted MIC")
    e.set_title(f"(d) {hit.mean():.0%} within ±1 dilution", loc="left", fontsize=8.5, fontweight="bold", pad=16)

    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(f"{prefix}.{ext}", bbox_inches="tight", dpi=300)
    print(f"wrote {prefix}.svg/.png; in-sample within ±1 dilution: {hit.mean():.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
