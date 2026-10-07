"""EVOLVABLE: turn the data table into a design matrix.

Contract (checks/contract.md): build(df) -> (X, names), with X a float array of shape (n_isolates, n_features)
and one name per column. Use genotype columns only, never the MIC columns.

The determinant names also carry the gene identity, which the model needs and cannot see: gyrA_S83L, gyrA_D87N,
gyrA_D87G and gyrA_D87Y are four ways of altering one target (gyrase, the primary ciprofloxacin target), yet the
four columns together carry a combined effect of +41 log2 units in the current champion. The gene index per
column is computed here and handed to src/model.py through model.GENE_GROUPS.
"""

import re

import numpy as np
import pandas as pd

import src.model as model

META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]


def gene_of(name: str) -> str:
    """Gene/locus identity in an AMRFinderPlus-style name: gyrA_S83L -> gyrA, blaCTX-M-15 -> blaCTX."""
    return re.match(r"^[A-Za-z]+", name).group(0)


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = [c for c in df.columns if c not in META]
    groups = sorted({gene_of(n) for n in names})
    index = {g: i for i, g in enumerate(groups)}
    model.GENE_GROUPS = (np.array([index[gene_of(n)] for n in names]), len(groups))
    return df[names].to_numpy(dtype=float), names
