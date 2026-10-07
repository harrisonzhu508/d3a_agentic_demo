"""EVOLVABLE: turn the data table into a design matrix.

Contract (checks/contract.md): build(df) -> (X, names), with X a float array of shape (n_isolates, n_features)
and one name per column. Use genotype columns only, never the MIC columns.
"""

import numpy as np
import pandas as pd

QRDR_LOCI = ("gyrA", "gyrB", "parC", "parE")     # the ciprofloxacin targets (Hooper & Jacoby 2015)

META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Baseline: every binary determinant as-is."""
    names = [c for c in df.columns if c not in META]
    class LabeledArray(np.ndarray):
        """float array that can hold attributes (a plain ndarray cannot) - carries the QRDR indicator below."""

    X = np.asarray(df[names].to_numpy(dtype=float), dtype=np.float32).view(LabeledArray)
    # src/model.py anchors the QRDR-wild-type isolates (no gyrA/gyrB/parC/parE mutation) on their own prior;
    # only features.py sees the determinant names, so the group indicator is computed here.
    qrdr = [j for j, n in enumerate(names) if n.split("_")[0].split("-")[0] in QRDR_LOCI]
    setattr(X, "qrdr_wild_type", (X[:, qrdr].sum(axis=1) == 0).astype("float32"))
    return X, names
