"""EVOLVABLE: turn the data table into a design matrix.

Contract (checks/contract.md): build(df) -> (X, names), with X a float array of shape (n_isolates, n_features)
and one name per column. Use genotype columns only, never the MIC columns.
"""

import numpy as np
import pandas as pd

META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Baseline: every binary determinant as-is."""
    names = [c for c in df.columns if c not in META]
    return df[names].to_numpy(dtype=float), names
