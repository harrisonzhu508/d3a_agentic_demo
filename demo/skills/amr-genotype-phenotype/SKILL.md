---
name: amr-genotype-phenotype
description: Background on antimicrobial resistance genetics and susceptibility testing for interpreting genotype-to-MIC models (fluoroquinolone mechanisms in E. coli, MIC testing and agreement metrics, AMRFinderPlus output, related MIC-prediction studies). Use when choosing features or interactions, when an effect looks biologically implausible, and when explaining results.
license: MIT
---

# AMR genotype to phenotype

Use this knowledge to generate hypotheses and to sanity-check results, not as a target to fit.

## References (read the one you need)

- `references/fluoroquinolone-resistance.md`: how resistance to ciprofloxacin arises (target mutations,
  plasmid-mediated genes, efflux), which combinations interact, and what a plausible model looks like.
- `references/mic-testing.md`: how MICs are measured, dilution grids, reproducibility, breakpoints, ECOFFs,
  and the agreement metrics used in the literature (essential agreement, VME, ME).
- `references/amrfinderplus-output.md`: what the annotation columns mean, how point mutations and gene
  alleles are named, and known pitfalls when turning calls into features.
- `references/related-work.md`: methods and lessons from published genotype-to-MIC studies on other datasets.

## Rules

- Prefer mechanisms over correlations: a feature with no known route to fluoroquinolone resistance that gets a
  large effect is more likely confounded (co-carried on a plasmid, or marking a lineage) than causal.
- Name determinants exactly as in the data table (`gyrA_S83L`, `qnrS1`, `aac(6')-Ib-cr5`).
- Do not look for, or use, published results for the dataset in `data/`; the harness judges your model.
