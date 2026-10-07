"""EVOLVABLE: turn the data table into a design matrix.

Contract (checks/contract.md): build(df) -> (X, names), with X a float array of shape (n_isolates, n_features)
and one name per column. Use genotype columns only, never the MIC columns.
"""

import numpy as np
import pandas as pd

META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]


# Ninety-sixth experiment: one column added to a design that has been a bag of alleles for ninety-five runs. The
# model has always been linear in 77 binary determinants, which is the form a genome-wide scan would use and not the
# form the fluoroquinolone literature describes. That literature is explicit about ciprofloxacin resistance in
# E. coli accumulating: gyrA at codon 83 or 87 is the initiating step, parC at 80 or 84 is secondary and is nearly
# always found on a gyrA background, and the MIC rises with the number of these QRDR substitutions rather than with
# the presence of any one of them - one gyrA change moves the MIC two to four doublings, adding a parC change takes
# it past the clinical breakpoint, and the isolates above the top of a dilution series carry four or five. The
# development table says the same thing, and the fit on main has never been allowed to: the fraction of isolates
# whose MIC is above the plate is 0.02, 0.01, 0.01, 0.01, 0.08 and 0.24 at zero, one, two, three, four and five QRDR
# substitutions, a pattern that no set of twelve additive codon columns reproduces, because an additive model must
# must give each codon one coefficient shared across every background while these data put the effect on the
# combination. The column added here is that combination written as the biology states it: the count of target-site
# substitutions across gyrA, gyrB, parC and parE - this table carries at most one allele per codon, so a sum over
# those columns is a count of steps - together with a term for isolates carrying both a gyrase and a topoisomerase
# change, which is the epistatic step the mechanism reference names, since the second one costs more on a resistant
# gyrase than on a wild-type one. Both are centred near the development mean so the intercept keeps its meaning and
# the per-codon coefficients keep theirs; with the count present, a coefficient on it says each further substitution
# is worth so many doublings whatever the codon, and the codon terms become what they ought to be, deviations from
# that. Everything else is main's: the rare split, untouched because all four genes are common, the wide rare prior,
# the knee at 16, the inverse-gamma scale. This is the first change of the session to the mean function rather than
# to a prior, and the only one these data appear to ask for.
def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Binary columns as-is, plus the number of QRDR steps and the gyrase-by-topoisomerase pair."""
    names = [c for c in df.columns if c not in META]
    X = df[names].to_numpy(dtype=float)
    target = [c for c in names if c.split("_")[0] in ("gyrA", "gyrB", "parC", "parE")]
    gyr = [c for c in target if c.startswith("gyr")]
    par = [c for c in target if c.startswith("par")]
    steps = df[target].to_numpy(dtype=float).sum(axis=1)
    pair = ((df[gyr].to_numpy(dtype=float).sum(axis=1) > 0) & (df[par].to_numpy(dtype=float).sum(axis=1) > 0))
    X = np.column_stack([X, steps - steps.mean(), pair.astype(float) - 0.3])
    return X, names + ["qrdr_steps", "gyrase_x_topoisomerase"]
