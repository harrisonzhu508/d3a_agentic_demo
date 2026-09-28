"""Where did a branch gain or lose predictive accuracy? Pointwise ELPD against the reference, in the GPAP style.

    uv run python skills/mic-plots/scripts/compare.py data/dev.csv results/<experiment>/<name>/pointwise.csv \
        results/<experiment>/<name>/comparison [--label "vs the champion"]

pointwise.csv is written by checks/evaluate.py: genome_id, elpd and, when there was a reference, elpd_ref.
Panels: (a) ELPD difference summed within each observed MIC bin, (b) per-isolate differences, sorted.
Writes <prefix>.png (opaque paper background, readable in pull requests) and <prefix>.svg.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plots import ACCENT, BAD, GOOD, INK, PAPER, SLATE, dilution_bins, style, to_bin  # noqa: E402


def main(csv: str, pointwise: str, prefix: str, label: str) -> None:
    style()
    d = pd.read_csv(csv)
    pw = pd.read_csv(pointwise).merge(d[["genome_id", "censor", "log2_hi"]], on="genome_id")
    uppers, labels = dilution_bins(d)
    obs = to_bin(pw["log2_hi"].where(pw["censor"] == "=", np.inf).to_numpy(), uppers)
    has_ref = "elpd_ref" in pw and pw["elpd_ref"].notna().all()
    delta = (pw["elpd"] - pw["elpd_ref"]).to_numpy() if has_ref else pw["elpd"].to_numpy()
    what = "Δ ELPD" if has_ref else "ELPD"

    nb = len(uppers) + 1
    x = np.arange(nb)
    per_bin = np.bincount(obs, weights=delta, minlength=nb)
    n_bin = np.bincount(obs, minlength=nb)
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.1), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.32})

    a = ax[0]
    a.bar(x, per_bin, width=0.75, color=[GOOD if v >= 0 else BAD for v in per_bin] if has_ref else ACCENT)
    a.axhline(0, color=SLATE, lw=0.6)
    for xi, (v, n) in enumerate(zip(per_bin, n_bin)):
        if n:
            a.text(xi, v, f"{n}", ha="center", va="bottom" if v >= 0 else "top", fontsize=5.5, color=SLATE)
    a.set_xticks(x, labels, rotation=60, fontsize=6.5)
    a.set_xlabel("observed MIC (mg/L); numbers = isolates in the bin")
    a.set_ylabel(f"{what}, summed over the bin")
    a.set_title("(a) where on the MIC scale it changed" if has_ref else "(a) ELPD by MIC bin",
                loc="left", fontsize=8.5, fontweight="bold", pad=10)

    b = ax[1]
    s = np.sort(delta)
    i = np.arange(len(s))
    if has_ref:
        b.fill_between(i, 0, s, where=s >= 0, color=GOOD, alpha=0.7, lw=0, interpolate=True)
        b.fill_between(i, 0, s, where=s < 0, color=BAD, alpha=0.7, lw=0, interpolate=True)
    else:
        b.plot(i, s, color=ACCENT, lw=1.2)
    b.axhline(0, color=SLATE, lw=0.6)
    total, se = delta.sum(), np.sqrt(len(delta)) * delta.std(ddof=1)
    note = (f"total {total:+.1f} ± {se:.1f} (SE)\n{(delta > 0).mean():.0%} of isolates improve" if has_ref
            else f"total {total:.1f} ± {se:.1f} (SE)")
    b.text(0.04, 0.95, note, transform=b.transAxes, va="top", fontsize=7, color=INK)
    b.set_xlabel("isolates, sorted")
    b.set_ylabel(f"{what} per isolate")
    b.set_title(f"(b) per isolate, {label}" if has_ref else "(b) per isolate", loc="left", fontsize=8.5,
                fontweight="bold", pad=10)

    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(f"{prefix}.{ext}", bbox_inches="tight", dpi=200, transparent=False, facecolor=PAPER)
    print(f"wrote {prefix}.png/.svg ({what} {total:+.1f} ± {se:.1f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv"), ap.add_argument("pointwise"), ap.add_argument("prefix")
    ap.add_argument("--label", default="vs the reference")
    a = ap.parse_args()
    main(a.csv, a.pointwise, a.prefix, a.label)
