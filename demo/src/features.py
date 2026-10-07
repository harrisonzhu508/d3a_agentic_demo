"""EVOLVABLE: turn the data table into a design matrix.

Contract (checks/contract.md): build(df) -> (X, names), with X a float array of shape (n_isolates, n_features)
and one name per column. Use genotype columns only, never the MIC columns.

The QRDR alleles are re-expressed as the two quantities the mechanism actually determines, and the rest of the
determinant table is left exactly as the champion has it.

Why only the QRDR. Stratifying each acquired gene by the number of target-gene mutations the isolate carries
removes essentially all of its apparent effect for the housekeeping and efflux columns (uhpT_E350Q: -1.5 log2
units within the one-mutation stratum; glpT_E448K: -1.0 within the zero-mutation stratum against a marginal
correlation of -0.11), but not for the acquired determinants (tet(A) +1.8, aph(6)-Id +1.3, blaCTX-M-15 +0.7
within the zero-mutation stratum) - so the acquired genes stay in as their own columns, and the target-gene
background, which is what they were partly standing in for, is now modelled directly.

Why these two quantities. Ciprofloxacin resistance is built up stepwise in the two type II topoisomerases:
gyrase (gyrA/gyrB) is the primary target and a single gyrase mutation shifts the MIC several fold, while
topoisomerase IV (parC/parE) is the secondary target and is only selected once gyrase is already altered, with
the two together taking the MIC off the scale (Hooper & Jacoby 2015; Huseby et al. 2017; Bagel et al. 1999).
That is what the plate shows: the number of QRDR mutations carried gives P(MIC > 4) = 4%, 4%, 9%, 55%, 96%,
97%, and no isolate with fewer than four target mutations exceeds 4 mg/L except 6 of 21. Which residue changed
is secondary - the three common alleles at gyrA codon 83 and 87 differ by less than a doubling once the target
is counted - so they collapse into gene-level indicators (gyrase altered, topoIV altered) plus the interaction
between them, which is the step the additive model cannot express: gyrA_mutated x parC_or_parE_mutated is 1 for
the 200 isolates whose MIC is at or above the top of the plate in 92% of cases and 0 for essentially all the
others. Keeping the remaining 60 columns unchanged means nothing the champion could express is lost; the seven
rare QRDR allele columns (3-5 isolates each) go, since the gene indicator already contains them.
"""

import numpy as np
import pandas as pd

META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]
GYRASE = ("gyrA_", "gyrB_")             # primary target of ciprofloxacin
TOPOIV = ("parC_", "parE_")             # secondary target, selected only after gyrase


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = [c for c in df.columns if c not in META]
    cols = df[names].to_numpy(dtype=float)

    def mutated(prefixes: tuple[str, ...]) -> np.ndarray:
        idx = [j for j, n in enumerate(names) if n.startswith(prefixes)]
        return (cols[:, idx].sum(axis=1) > 0).astype(float)

    gyrA, parC = mutated(GYRASE), mutated(TOPOIV)
    keep = [j for j, n in enumerate(names) if not n.startswith(GYRASE + TOPOIV)]
    return (np.column_stack([gyrA, parC, gyrA * parC, cols[:, keep]]),
            ["gyrA_mutated", "parC_parE_mutated", "gyrA_x_topoIV"] + [names[j] for j in keep])
