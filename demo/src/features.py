"""EVOLVABLE: turn the data table into a design matrix.

Contract (checks/contract.md): build(df) -> (X, names), with X a float array of shape (n_isolates, n_features)
and one name per column. Use genotype columns only, never the MIC columns.

The resistance-determinant table is a set of presence/absence calls, and the calls for the ciprofloxacin targets
are not independent observations of seven separate things: they are ordered steps of one pathway. Resistance is
built up by successive mutations, first in gyrA (gyrase, the primary target at ciprofloxacin concentrations),
then in parC/parE (topoisomerase IV, the secondary target), and the order and the number of steps, not the
particular residue, is what tracks the MIC (Hooper & Jacoby 2015; Huseby et al. 2017; Bagel et al. 1999 -
topoIV mutants are only selected once gyrase is already altered). The data agree: the fraction of isolates above
the top of the plate goes 4%, 4%, 9%, 55%, 96%, 97% with the number of QRDR mutations carried, and no isolate
with 3 or fewer target mutations is above 4 mg/L except 6 of them, whereas 181 of the 204 right-censored
isolates carry 4 or 5.

So on top of the raw determinants, X carries the number of QRDR *steps* the isolate has - gyrA (counted once
however many codons), parC (once), parE (once) and the two PMQR mechanisms that act on the same pathway
(qnr, aac(6')-Ib-cr) - that is, a count of how far along the stepwise path the genotype is. A model with free
per-column effects has to rediscover that monotone path as a sum of seventeen near-collinear allele columns,
which is exactly where its coefficients become unidentified; the step count hands it the path directly.
"""

import re

import numpy as np
import pandas as pd

META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]


def steps(df: pd.DataFrame, names: list[str], cols: np.ndarray) -> np.ndarray:
    """How many QRDR steps the genotype has taken: gyrA, parC, parE, qnr, aac(6')-Ib-cr (each counted once)."""
    groups = {
        "gyrA": lambda n: n.startswith("gyrA") or n.startswith("gyrB"),
        "parC": lambda n: n.startswith("parC"),
        "parE": lambda n: n.startswith("parE"),
        "qnr": lambda n: n.startswith("qnr"),
        "aac6_cr": lambda n: "aac(6')-Ib-cr" in n,
    }
    out = np.zeros(len(df))
    for test in groups.values():
        idx = [j for j, n in enumerate(names) if test(n)]
        out += (cols[:, idx].sum(axis=1) > 0).astype(float)
    return out


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = [c for c in df.columns if c not in META]
    cols = df[names].to_numpy(dtype=float)
    qrd = steps(df, names, cols)
    # The step count enters twice, as the two shapes a monotone step path can take on the plate: how far along
    # it is (linear in the count, free to be as steep as the data want) and whether it has passed the step at
    # which the MIC leaves the scale (4+ steps: 88% of those isolates are `>4`, against 3% of the others).
    return np.column_stack([cols, qrd, (qrd >= 4).astype(float)]), names + ["qrd_steps", "qrd_past_plate"]
