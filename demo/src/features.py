"""EVOLVABLE: turn the data table into a design matrix.

Contract (checks/contract.md): build(df) -> (X, names), with X a float array of shape (n_isolates, n_features)
and one name per column. Use genotype columns only, never the MIC columns.

Mechanism blocks instead of 77 separate columns. Ciprofloxacin acts on gyrase (GyrA, the primary target in
E. coli) and on topoisomerase IV (ParC/ParE), and resistance accumulates stepwise along that path: low level
with one QRDR change, high level once a second target joins in (Hooper & Jacoby 2015; Huseby et al. 2017;
Bagel et al. 1999 for gyrase-primary/topoIV-secondary). So the columns below are the number of mutated codons
per target (0-2) plus a column per *pathway*: a gyrA-only path, a topoIV-only path, and the both-targets path,
which is where the clinically high MICs live. PMQR genes work by other mechanisms - target protection (qnr),
drug acetylation (aac(6')-Ib-cr5), efflux regulation (oqxAB, marR) - and give low-level shifts on their own
(Strahilevitz et al. 2009), so each gets its own column.

Everything with no known route to fluoroquinolone resistance is dropped: the beta-lactamases,
aminoglycoside/sulphonamide/phenicol/tetracycline genes and the non-QRDR point mutations, which mark plasmid
co-carriage and lineage rather than susceptibility (they are the negative controls of the amr skill).
"""

import numpy as np
import pandas as pd

META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]

# PMQR / permeability mechanisms with a known route to reduced fluoroquinolone susceptibility.
QNR = ["qnrA1", "qnrB19", "qnrS1"]                       # target protection
AAC6CR = ["aac(6')-Ib-cr5"]                              # ciprofloxacin acetylation
EFFLUX = ["oqxA", "oqxB", "marR_S3N"]                    # efflux regulation (oqxAB, marR -> AcrAB-TolC)


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = ["gyrA_codons_mutated", "parC_codons_mutated", "parE_codons_mutated",
             "gyrA_only", "both_targets", "qnr", "aac6_ib_cr5", "efflux"]

    def codons_mutated(prefix: str) -> np.ndarray:
        """Number of distinct QRDR codons of this gene that carry a mutation (0, 1 or 2)."""
        cols = [c for c in df.columns if c not in META and c.startswith(prefix)]
        pos: dict[str, list[str]] = {}
        for c in cols:
            pos.setdefault(c[: c.rfind("_")], []).append(c)      # gyrA_S83L, gyrA_S83F -> 'gyrA_S83'
        out = np.zeros(len(df), dtype=float)
        for alleles in pos.values():
            out += (df[alleles].to_numpy(dtype=float).sum(axis=1) > 0).astype(float)
        return out

    def present(cols: list[str]) -> np.ndarray:
        cols = [c for c in cols if c in df.columns]
        return (df[cols].to_numpy(dtype=float).sum(axis=1) > 0).astype(float) if cols else np.zeros(len(df))

    gyrA = codons_mutated("gyrA")
    parC = codons_mutated("parC")
    parE = codons_mutated("parE")
    topoIV = ((parC + parE) > 0).astype(float)
    gy = (gyrA > 0).astype(float)
    X = np.column_stack([
        gyrA, parC, parE,
        gy * (1.0 - topoIV),                 # gyrase hit alone: low-level resistance
        gy * topoIV,                         # both targets hit: the high-level path
        present(QNR),
        present(AAC6CR),
        present(EFFLUX),
    ])
    return X, names
