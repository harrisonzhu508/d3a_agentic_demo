# Fluoroquinolone resistance in *E. coli* (background)

Ciprofloxacin is a fluoroquinolone. Fluoroquinolones trap two type II topoisomerases on DNA: **DNA gyrase**
(subunits GyrA and GyrB) and **topoisomerase IV** (ParC and ParE). In *E. coli* and other Gram-negatives,
gyrase is the primary target and topoisomerase IV the secondary one.
Review: Hooper & Jacoby (2015) *Ann N Y Acad Sci* 1354:12, doi:10.1111/nyas.12830.

## Mechanisms

1. **Target mutations in the quinolone resistance-determining regions (QRDRs)**
   - `gyrA`: codons 83 and 87 (for example S83L, D87N, D87G, D87Y) are the classic first-step changes.
   - `parC`: codons 80 and 84 (for example S80I, S80R, E84V, E84K); `parE`: for example S458A.
   - Mutations accumulate stepwise under selection. A single `gyrA` change gives low-level resistance;
     clinically high-level resistance usually combines two `gyrA` changes with one or more `parC` changes.
     Huseby *et al.* (2017) *Mol Biol Evol* 34:1029, doi:10.1093/molbev/msx052, describe the common
     evolutionary paths in *E. coli*.
2. **Epistasis between targets**: because gyrase is the primary target, a `parC` mutation on a wild-type
   `gyrA` background has little phenotypic effect; it matters once gyrase is already resistant.
   Bagel *et al.* (1999) *Antimicrob Agents Chemother* 43:868, doi:10.1128/AAC.43.4.868. Implication for a
   linear model on log2 MIC: additivity is an assumption worth testing (interaction terms, or a
   "number of QRDR mutations" feature).
3. **Plasmid-mediated quinolone resistance (PMQR)**: `qnr` genes (families qnrA, qnrB, qnrS, qnrC, qnrD)
   protect the targets; `aac(6')-Ib-cr` acetylates ciprofloxacin and norfloxacin (the `-cr` variant, unlike
   plain `aac(6')-Ib`); efflux pumps `qepA` and `oqxAB`. On their own these usually give **low-level**
   increases (Strahilevitz *et al.* 2009 *Clin Microbiol Rev* 22:664, doi:10.1128/CMR.00016-09), but they add
   to target mutations and ease the selection of higher-level resistance.
4. **Chromosomal efflux and permeability**: mutations in regulators such as `marR`, `acrR` and `soxR`
   up-regulate the AcrAB-TolC efflux pump; porin loss reduces uptake. Effects are usually modest and
   often not called by AMR annotation tools.

## Things that look like effects but are not

- **Co-carriage**: plasmids carry many resistance genes together (beta-lactamases, aminoglycoside and
  sulphonamide genes, `qnr`, `aac(6')-Ib-cr`). A gene with no fluoroquinolone mechanism can correlate with
  high MICs because it travels with one that has. Beta-lactamase genes are natural negative controls.
- **Lineage (population structure)**: resistant clones (for example ST131 in *E. coli*) share many
  background variants. Features that mark a lineage can absorb its average MIC.
- **Genotype-phenotype discordance**: in most published datasets a few per cent of isolates are resistant
  without a known determinant (unknown mechanism, missed call, low assembly quality) or susceptible despite
  one (sample mix-up, loss of a plasmid before testing, annotation error). Models should tolerate these rather
  than bend all other estimates around them.

## What a plausible fitted model looks like (qualitatively)

- Positive effects for QRDR mutations and PMQR genes; `parC`/`parE` effects larger when `gyrA` is mutated.
- Near-zero effects for determinants with no fluoroquinolone mechanism.
- Wild-type isolates at the bottom of the tested range; multi-mutation isolates beyond its top.
