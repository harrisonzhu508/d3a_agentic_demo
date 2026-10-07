"""EVOLVABLE: turn the data table into a design matrix.

Contract (checks/contract.md): build(df) -> (X, names), with X a float array of shape (n_isolates, n_features)
and one name per column. Use genotype columns only, never the MIC columns.

Baseline: every binary determinant as-is. In addition, X carries `gene_groups` - the index of the gene/locus
each column belongs to - so that the model can pool the alternative alleles of one gene (gyrA S83L / D87N /
D87G / D87Y are four ways of altering one target, and the four columns together currently carry a combined
effect of +41 log2 units). Columns are grouped by the part of the name before the last `_` for point mutations
(`gyrA_D87N` -> `gyrA`, `ampC_C-42T` -> `ampC`) and before the first `-` for acquired genes (`blaCTX-M-15` ->
`blaCTX`, `aac(6')-Ib-cr5` -> `aac(6')`), which is how the AMRFinderPlus names already encode gene identity.
"""

import numpy as np
import pandas as pd

META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]


class _LabeledArray(np.ndarray):
    """A float array that can carry the gene index (a plain ndarray cannot hold attributes)."""


def gene_of(name: str) -> str:
    """Gene/locus identity encoded in an AMRFinderPlus-style determinant name."""
    if "_" in name and not name.startswith(("bla", "aac", "aad", "ant(", "aph", "mph", "ars", "ble")):
        return name.rsplit("_", 1)[0]              # point mutation: locus before the allele/codon
    if "-" in name:
        return name.split("-", 1)[0]               # acquired gene: gene family before the allele number
    return name


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = [c for c in df.columns if c not in META]
    X = df[names].to_numpy(dtype=float)
    groups = sorted({gene_of(n) for n in names})
    index = {g: i for i, g in enumerate(groups)}
    X = np.asarray(X, dtype=np.float32)
    X = X.view(_LabeledArray)                      # carries the gene index for src/model.py
    X.gene_groups = [index[gene_of(n)] for n in names]
    return X, names
