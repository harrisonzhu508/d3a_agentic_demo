"""EVOLVABLE: turn the data table into a design matrix.

Contract (checks/contract.md): build(df) -> (X, names), with X a float array of shape (n_isolates, n_features)
and one name per column. Use genotype columns only, never the MIC columns.

One change on top of the raw determinant table: the QRDR alleles are replaced by the number of *targets* of the
ciprofloxacin pathway that the isolate's genotype has knocked out. The alleles at a codon (gyrA S83L, D87N, D87G,
D87Y) are alternatives at one site, they are carried in near-perfect combinations (gyrA D87N and parC S80I
correlate at 0.95), and the phenotype does not distinguish which residue changed: what matters is that gyrase is
altered, and, on that stepwise path, how many targets are altered (Hooper & Jacoby 2015; Huseby et al. 2017;
Bagel et al. 1999 for gyrase-primary/topoIV-secondary). Fitting one column per allele on such a design is what
makes the effects unidentifiable, so the pathway enters as counts and the individual alleles stay in only as the
rare extra columns that carry information beyond the count. Everything else - the acquired accessory genes, the
efflux regulators and the non-QRDR point mutations - is kept exactly as before, so nothing is thrown away.
"""

import numpy as np
import pandas as pd

META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]
QRDR = ("gyrA_", "gyrB_", "parC_", "parE_")


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = [c for c in df.columns if c not in META]
    X = df[names].to_numpy(dtype=float)

    def targets(prefixes: tuple[str, ...]) -> np.ndarray:
        """0/1: does this target gene carry any QRDR mutation (a target counts once, however many codons)."""
        cols = [j for j, c in enumerate(names) if c.startswith(prefixes)]
        return (X[:, cols].sum(axis=1) > 0).astype(float)

    gyrA = targets(("gyrA_", "gyrB_"))
    parC = targets(("parC_",))
    parE = targets(("parE_",))
    qrdr = [j for j, c in enumerate(names) if c.startswith(QRDR)]
    keep = np.delete(np.arange(len(names)), qrdr)
    return np.column_stack([gyrA, parC, parE, gyrA * parC, gyrA * parE, X[:, keep]]), \
        ["gyrA_mutated", "parC_mutated", "parE_mutated", "gyrA_x_parC", "gyrA_x_parE"] + [names[j] for j in keep]
