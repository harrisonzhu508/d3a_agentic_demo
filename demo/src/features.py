"""EVOLVABLE: turn the data table into a design matrix.

Contract (checks/contract.md): build(df) -> (X, names), with X a float array of shape (n_isolates, n_features)
and one name per column. Use genotype columns only, never the MIC columns.

Rare determinants are pooled, allele families are merged, and the overall resistance load is made explicit.
The AMRFinderPlus guidance is to pool rare features by family or mechanism rather than hope a prior copes with
them (amr-genotype-phenotype, references/amrfinderplus-output.md), and here 38 of the 77 columns are carried by
fewer than 20 isolates. Those columns are not innocent: the number of acquired determinants tracks how resistant
the isolate is (2 % of isolates with 2-5 determinants are `>4`, against 75 % of those with 10-20), so a rare
column can stand in for a load the common columns already describe and extrapolate wildly on an isolate with a
different mix. Three feature-side changes:

* allele families merged - a family-level call (`blaCTX-M`) and its alleles (`blaCTX-M-15`) are the same gene,
  just called at different assembly quality;
* determinants below a 5 % prevalence collapsed into one column per drug class;
* one `n_classes` column, the number of resistance classes the isolate carries, so that load is a fitted effect
  on its own instead of being borrowed by whichever rare gene happens to co-occur.
"""

import numpy as np
import pandas as pd

META = ["genome_id", "mic_raw", "censor", "log2_lo", "log2_hi", "log2_mic"]
PREVALENCE_FLOOR = 0.05        # below this share of isolates, a determinant cannot be estimated on its own

# (label, prefixes, must_not_match). Checked in order; the first match wins.
CLASSES = [
    ("aac6_cr", ["aac(6')-Ib-cr"], []),                                   # ciprofloxacin acetylation
    ("quinolone_target", ["gyrA", "gyrB", "parC", "parE", "qnr", "oqx", "qepA", "marR"], []),
    ("beta_lactam", ["bla", "ampC"], []),
    ("aminoglycoside", ["aac", "aadA", "ant(", "aph", "rmt", "sat2"], ["aac(6')-Ib-cr"]),
    ("folate", ["dfrA", "sul"], []),
    ("phenicol", ["cat", "cmlA", "floR"], []),
    ("tetracycline", ["tet"], []),
    ("macrolide", ["erm", "mph", "msr"], []),
    ("other", [""], []),
]


def drug_class(name: str) -> str:
    for label, prefixes, exclude in CLASSES:
        if any(name.startswith(p) for p in prefixes) and not any(name.startswith(e) for e in exclude):
            return label + "_rare"
    return "other_rare"


def allele_family(name: str, cols: list[str]) -> str:
    """A column is an allele of the longest other column that is a hyphen-prefix of it."""
    parents = [c for c in cols if c != name and name.startswith(c + "-")]
    return max(parents, key=len) if parents else name


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in df.columns if c not in META]
    n = len(df)
    value = {c: df[c].to_numpy(dtype=float) for c in cols}

    families: dict[str, list[str]] = {}
    for c in cols:
        families.setdefault(allele_family(c, cols), []).append(c)

    names, blocks, pooled = [], [], {}
    for fam, members in sorted(families.items()):
        present = (sum(value[m] for m in members) > 0).astype(float)
        if present.mean() >= PREVALENCE_FLOOR:
            names.append(fam)
            blocks.append(present)
        else:                                    # too rare to estimate: pool it with its drug class
            cls = drug_class(fam)
            pooled[cls] = present if cls not in pooled else np.maximum(pooled[cls], present)
    for cls, col in sorted(pooled.items()):
        names.append(cls)
        blocks.append(col)

    # Resistance load: number of drug classes, from the original calls, that the isolate carries.
    load = np.zeros(n)
    for label, prefixes, exclude in CLASSES:
        member = [c for c in cols if drug_class(c) == label + "_rare"]
        load += (sum(value[c] for c in member) > 0).astype(float)
    names.append("n_classes")
    return np.column_stack(blocks + [load]), names
