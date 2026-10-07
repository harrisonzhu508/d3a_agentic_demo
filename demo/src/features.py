"""EVOLVABLE: turn the data table into a design matrix.

Contract (checks/contract.md): build(df) -> (X, names), with X a float array of shape (n_isolates, n_features)
and one name per column. Use genotype columns only, never the MIC columns.

Baseline: every binary determinant as-is, plus the gene grouping the model needs and cannot see for itself. The
fourteen QRDR columns are alternative mutations at two codons of gyrase (gyrA/gyrB) and three of topoisomerase IV
(parC/parE) - one mechanism each, and gyrA_D87N correlates with parC_S80I at 0.95 - so which allele of a gene
did it is not identified from the MIC, while which gene was altered is (Hooper & Jacoby 2015). src/model.py uses
that grouping to penalise differences between the alleles of one gene and nothing else; only features.py sees the
names, so the groups are computed here.
"""

import numpy as np
import pandas as pd

import src.model as model

META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]
TARGET_GENES = ("gyrA", "gyrB", "parC", "parE")      # the two ciprofloxacin targets


def locus_of(name: str) -> str:
    """Target gene of a QRDR column: gyrA_S83L -> gyrA ('' for anything that is not a target-gene mutation)."""
    return name.split("_")[0] if name.split("_")[0] in TARGET_GENES else ""


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = [c for c in df.columns if c not in META]
    model.QRDR_ALLELE_GROUPS = [np.array([j for j, n in enumerate(names) if locus_of(n) == g])
                                for g in TARGET_GENES]
    return df[names].to_numpy(dtype=float), names
