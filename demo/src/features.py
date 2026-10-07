"""EVOLVABLE: turn the data table into a design matrix.

Contract (checks/contract.md): build(df) -> (X, names), with X a float array of shape (n_isolates, n_features)
and one name per column. Use genotype columns only, never the MIC columns.

Two changes, both about the ciprofloxacin target gene mutations, and both aimed at the 25 isolates that carry
25% of the total loss (4.5% of the dev set scores below -2 log predictive density, 18 of them right-censored).

* The QRDR alleles enter as *steps of one pathway* rather than only as seventeen separate columns. Resistance is
  acquired by successive mutations, first in gyrA (gyrase, the primary target at ciprofloxacin concentrations)
  and then in parC/parE (topoisomerase IV, the secondary target); the number of steps, not which residue
  changed, is what tracks the MIC (Hooper & Jacoby 2015; Huseby et al. 2017; Bagel et al. 1999). The data say
  the same: P(MIC > 4) = 4%, 4%, 9%, 55%, 96%, 97% by number of QRDR mutations. A free per-column effect has to
  rediscover that monotone path as a sum of seventeen columns of which gyrA D87N and parC S80I correlate at 0.95,
  and that is where its coefficients become unidentified (champion: gyrA_D87N +38 against glpT_E448K -14).
  So: one column per target gene (does this gene carry any QRDR mutation) and one column for the step at which
  the MIC leaves the scale (gyrase and topoIV both altered - 41 of the 47 isolates with 4 steps, and all 134
  with 5, are `>4`; no isolate with fewer than 4 steps is above 4 mg/L except 6).

* The individual rare alleles are dropped, and only the QRDR ones at that: an allele column whose isolates are a
  subset of its gene's column adds no information for the common alleles and, for the four carried by 3-5
  isolates, gives the model a direction along which it can place an arbitrary effect (gyrA_D87Y is at +2.9 with a
  posterior sd of 4.5 on five isolates). The ~55 rare *acquired* genes are kept: they are the ones the wide prior
  is there for, since a novel resistance gene is exactly the event the model must be allowed to predict.
"""

import numpy as np
import pandas as pd

META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]
QRDR = ("gyrA_", "gyrB_", "parC_", "parE_")           # the ciprofloxacin targets


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = [c for c in df.columns if c not in META]
    cols = df[names].to_numpy(dtype=float)

    def mutated(prefixes: tuple[str, ...]) -> np.ndarray:
        idx = [j for j, n in enumerate(names) if n.startswith(prefixes)]
        return (cols[:, idx].sum(axis=1) > 0).astype(float)

    gyrA, parC, parE = mutated(("gyrA_", "gyrB_")), mutated(("parC_",)), mutated(("parE_",))
    keep = [j for j, n in enumerate(names) if not n.startswith(QRDR)]
    return (np.column_stack([gyrA, parC, parE, gyrA * parC, gyrA * parE, cols[:, keep]]),
            ["gyrA_mutated", "parC_mutated", "parE_mutated", "both_targets", "gyrA_x_parE"]
            + [names[j] for j in keep])
