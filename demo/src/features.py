"""EVOLVABLE: turn the data table into a design matrix.

Contract (checks/contract.md): build(df) -> (X, names), with X a float array of shape (n_isolates, n_features)
and one name per column. Use genotype columns only, never the MIC columns.
"""

import numpy as np
import pandas as pd

import src.model as model

META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Baseline: every binary determinant as-is, plus the column names for the model's prior.

    src/model.py needs the names because the prior is written by mechanism, not by column index: the fluoroquinolone
    target-gene alleles (gyrA, gyrB, parC, parE) are the only determinants with a direct route to the MIC of this
    drug, and their effects are the ones the dilution series can resolve (Hooper & Jacoby 2015).
    """
    names = [c for c in df.columns if c not in META]
    model.FEATURE_NAMES = list(names)
    return df[names].to_numpy(dtype=float), names
