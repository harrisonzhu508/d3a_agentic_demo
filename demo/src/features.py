"""EVOLVABLE: turn the data table into a design matrix.

Contract (checks/contract.md): build(df) -> (X, names), with X a float array of shape (n_isolates, n_features)
and one name per column. Use genotype columns only, never the MIC columns.

Mechanism score plus the individual QRDR alleles. Ciprofloxacin acts on gyrase (GyrA, primary target in E. coli)
and topoisomerase IV (ParC/ParE) and resistance accumulates stepwise along that path, so one additive score over
the mutated targets carries the dose-response (Hooper & Jacoby 2015; Huseby et al. 2017); PMQR genes add through
separate mechanisms - target protection (qnr), acetylation (aac(6')-Ib-cr5), efflux regulation (marR, oqxAB) -
also roughly additive on top (Strahilevitz et al. 2009). The individual codon changes keep their own columns
because they differ in how much they shift the MIC (gyrA D87N and parC S80I dominate the champion's fit), and
each acquired accessory gene keeps its own column as well: the champion's pointwise scores show that the
background carriage columns are not interchangeable noise, and with the score absorbing the dose-response the
per-gene columns no longer have to compete with each other for it.
"""

import numpy as np
import pandas as pd

META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]
QRDR = ("gyrA_", "gyrB_", "parC_", "parE_")


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = [c for c in df.columns if c not in META]
    X = df[names].to_numpy(dtype=float)

    # Number of distinct QRDR codons mutated, per target gene: an isolate carries at most one residue
    # change at a codon, so counting alleles would double-count alternatives at the same site.
    def codons(prefix: str) -> np.ndarray:
        positions: dict[str, list[int]] = {}
        for j, c in enumerate(names):
            if c.startswith(prefix):
                positions.setdefault(c[: c.rfind("_")], []).append(j)
        return sum((X[:, cols].sum(axis=1) > 0).astype(float) for cols in positions.values())

    gyrA, parC, parE = codons("gyrA_"), codons("parC_"), codons("parE_")
    qnr = np.zeros(len(df))
    for j, c in enumerate(names):
        if c.startswith("qnr"):
            qnr += X[:, j]
    score = (gyrA + parC + parE + qnr
             + X[:, names.index("aac(6')-Ib-cr5")]
             + X[:, names.index("marR_S3N")]
             + (X[:, [names.index(c) for c in ("oqxA", "oqxB") if c in names]].max(axis=1)
                if any(c in names for c in ("oqxA", "oqxB")) else 0.0))
    return np.column_stack([score, X]), ["mechanism_score"] + names
