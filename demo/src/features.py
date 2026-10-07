"""EVOLVABLE: turn the data table into a design matrix.

Contract (checks/contract.md): build(df) -> (X, names), with X a float array of shape (n_isolates, n_features)
and one name per column. Use genotype columns only, never the MIC columns.
"""

import numpy as np
import pandas as pd

META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]

# One hundred and thirty-fifth experiment: three columns taken out of the design for being the same measurement written
# three times, on the evidence of the plate rather than of the genotype. The additive model this loop has defended for a
# hundred campaigns gives every determinant its own slope, which is the right prior for a genotype table and the wrong
# one for a dilution series, because a MIC is the concentration at which the last surviving cell divides and the
# fluoroquinolone ladder climbs that concentration in a fixed order: a first-step target mutation, then the second
# target site, then the topoisomerase partner, then the acquired modifiers on top. Counting the five quinolone
# determinants in this table - gyrA_S83L, gyrA_D87N, parC_S80I, parC_E84V, parE_I529L - sorts the 558 isolates into
# groups whose measured MICs are not additive in the count at all. Zero of the five: 273 isolates, a median on-grid MIC
# of 0.18 mg/L and 81% of them below the bottom well. One: 69 isolates, a median of 0.25 and 28% below. Two: 24
# isolates, a median of 0.5 and 4% below. Three or more: 192 isolates, 96% of them above the top well and unreadable.
# So the map from count to MIC is a staircase of two visible steps and a third that leaves the plate, the strongest
# non-linearity in this data set, and an additive fit cannot represent it - which is what the run that added a step
# count feature learned at a cost of 43.9 ELPD, a true result about the wrong encoding, since a single linear
# coefficient on a count of n asserts n equal doublings where the plate shows roughly 0, 1, 3 and then infinity.
#
# This run adds nothing. It removes three columns, and the case for each is a containment relation among the isolates
# themselves rather than a guess about mechanism:
#
#   gyrA_D87N      191 carriers, of which exactly one lacks gyrA_S83L. Correlation with S83L: 0.86.
#   parC_E84V      136 carriers, of which zero lack parC_S80I. A strict subset, 24% of the plate.
#   parE_I529L     178 carriers, of which 136 carry all three of gyrA_S83L, gyrA_D87N and parC_S80I; the 26 others
#                  are the whole of this column's independent information, in a design that already has 74 others.
#
# What those three columns are doing inside the champion is not nothing: they are slopes standing in for two quantities
# - a first-step target mutation and a second target site - split across the design's two shrinkage regimes by
# prevalence, so that gyrA_S83L at 257 carriers is held to a prior median of about 1.5 doublings while gyrA_D87N at 191
# sits under a prior four times wider with a ceiling past sixteen, on the same biological event; and parC_E84V, which
# no isolate in this collection carries without parC_S80I, is entitled to a slope the data cannot attribute to it.
# Removing them leaves the fit everything the assay measured and takes away three places in which it can be wrong about
# the same molecule. The mutation count is left out on purpose: the staircase says three or more is off the scale, and
# that is already what the 204 isolates above the top well tell the likelihood directly, through their censored
# intervals, with no column needed to tell it which side of three they sit on.


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Every binary determinant, minus three that are contained in others."""
    names = [c for c in df.columns if c not in META]
    drop = {"gyrA_D87N", "parC_E84V", "parE_I529L"}
    names = [c for c in names if c not in drop]
    return df[names].to_numpy(dtype=float), names
