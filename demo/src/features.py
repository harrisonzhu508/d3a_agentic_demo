"""EVOLVABLE: turn the data table into a design matrix.

Contract (checks/contract.md): build(df) -> (X, names), with X a float array of shape (n_isolates, n_features)
and one name per column. Use genotype columns only, never the MIC columns.

Baseline features plus one mechanistic term: gyrase (gyrA) is the primary fluoroquinolone target in E. coli, so
topoisomerase IV mutations (parC/parE) act mainly on a gyrA-mutated background (Bagel et al. 1999). The added
column is the product of "any QRDR mutation in gyrA" and "any QRDR mutation in parC/parE".
"""

import numpy as np
import pandas as pd

META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Every binary determinant as-is, plus the gyrA x (parC or parE) epistasis term."""
    names = [c for c in df.columns if c not in META]
    X = df[names].to_numpy(dtype=float)

    def any_of(prefixes: tuple[str, ...]) -> np.ndarray:
        cols = [c for c in names if c.startswith(prefixes)]
        return (df[cols].to_numpy(dtype=float).sum(axis=1) > 0).astype(float) if cols else np.zeros(len(df))

    gyrA = any_of(("gyrA", "gyrB"))
    topo4 = any_of(("parC", "parE"))
    extra = np.column_stack([gyrA * topo4])
    return np.column_stack([X, extra]), names + ["gyrA_x_parC_parE"]
