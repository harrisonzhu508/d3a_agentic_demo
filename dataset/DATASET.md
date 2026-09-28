# Demo dataset: ciprofloxacin MIC in *E. coli* from AMR genotype

> **For humans only.** This file holds published results and our own exploration of the data, and this folder
> holds the full table (including the locked test isolates). It lives outside `demo/`, the agent's workspace, and
> the demo hooks deny it; skills carry general knowledge only (see `demo/skills/README.md`). The agent sees only
> `demo/data/dev.csv`, written by `demo/checks/split.py`.

Chosen 2026-09-26 for the live autoresearch demo (slides 15–17).

## Source

- **Data:** Gerada A. *et al.* "Clinical *Escherichia coli* strains with whole genome sequencing data and
  antimicrobial susceptibility meta-data for machine learning", University of Liverpool Research Data Catalogue,
  <https://datacat.liverpool.ac.uk/id/eprint/3008>. **Licence: CC BY 4.0** (see `liverpool_ecoli/README_dataset.md`).
- **Paper (results to validate against):** Gerada A., Zhong Y. *et al.* (2026) *npj Antimicrobials and Resistance* 4:51,
  doi:10.1038/s44259-026-00217-4 (open access, PMC13530227).
- **Contents:** 762 clinical isolates (Liverpool, 2017–21), agar-dilution MICs for 10 drugs, SPAdes assemblies,
  AMRFinderPlus v3.11.14 calls (acquired genes + point mutations; the NCBI counterpart of ResFinder + PointFinder).
  Isolates were chosen by phenotype (rare resistance over-sampled), so prevalences are not epidemiological.

## What is in this repo

| File | What |
|---|---|
| `liverpool_ecoli/meta_data_tidy.txt` | AST table as published (sha256 `dd3ca8a8…`, identical to the catalogue download) |
| `liverpool_ecoli/annots.zip` | 762 AMRFinderPlus TSVs as published |
| `build_cip_table.py` | builds `cip_features.csv` from the two files above |
| `cip_features.csv` | 703 isolates × 77 binary determinants (carried by ≥ 5 isolates), plus `mic_raw`, `censor`, `log2_lo`, `log2_hi` (the censoring interval) and `log2_mic` (paper convention: `>x` → 2x) |

Ciprofloxacin MICs lie on 11 agar-dilution levels, 0.008 to `>4` mg/L. 256 isolates (36 %) are right-censored at `>4`,
and 306 (44 %) sit on the lowest dilution tested, which is an upper bound only. That makes this a genuinely censored
regression problem.

## Numbers to validate against

| Model | Within ±1 dilution | Pearson r | Where |
|---|---|---|---|
| XGBoost on AMRFinderPlus genes ("Annot."), 5-fold CV stratified by MIC | **83.6 %** (588/703) | **0.89** | paper, Table 3 |
| Ridge on all 77 features, 5-fold CV (5 seeds) | 83.7 % (sd 1.0) | 0.895 | reproduced here |
| OLS on 7 textbook determinants (gyrA S83L/D87N, parC S80I/E84V, parE S458A, qnrS1, aac(6')-Ib-cr), same CV | 90.1 % (sd 0.3) | 0.898 | reproduced here |

Naive OLS effect sizes on all 703 isolates, in doublings of MIC (95 % CI): gyrA S83L 3.33 (2.83–3.82), gyrA D87N
3.39 (2.39–4.40), parC S80I 2.16 (1.12–3.19), qnrS1 2.81 (1.99–3.62), blaTEM-1 −0.02 (−0.33–0.28, a null control).
The paper's feature ranking for CIP is parC S80I > gyrA D87N > gyrA S83L. For an independent cohort, Pataki *et al.*
(2020) *Sci Rep* 10:15026 report a 4-feature linear model on log2 MIC: gyrA 87 +4.65, gyrA 83 +3.95, parC 80 +1.74,
qnrS1 +4.21 (their feature matrix is not released, but the coefficients are a sanity check).

## Proposed evaluation harness (`demo/checks/`, read-only to the agent)

1. **Lock a test set**: a seeded, MIC-stratified 20 % hold-out (about 140 isolates) that the agent can never read
   (enforced by a PreToolUse deny rule). Scored once, at the end.
2. **Development score** on the other 80 %: fixed 5-fold CV folds (saved to a file, same for every branch):
   - primary: CV log predictive density under the interval-censored likelihood (the ELPD the agent optimises);
   - secondary: % within ±1 dilution and Pearson r, computed the paper's way, for comparison with 83.6 % / 0.89;
   - calibration: coverage of 90 % predictive intervals.
3. **Gates** (a branch fails if any is violated): R̂ < 1.01, bulk ESS > 400, 0 divergences, fit under a fixed time budget.
4. **Parsimony**: number of determinants whose posterior effect exceeds ±0.5 doublings with 90 % probability.
5. **Knowledge check** (reported, not optimised): the posterior should recover gyrA/parC/qnr effects, and blaTEM-1 ≈ 0.

What the agent can improve: censoring (interval likelihood instead of the `>x` → 2x convention), shrinkage
(horseshoe across 77 correlated determinants), epistasis (parC S80I matters mainly on a gyrA-mutant background;
Bagel *et al.* 1999), and feature engineering (gene families, allele grouping).

## Alternatives considered (not used)

- **Demczuk *et al.* (2020) *AAC* 64:e02005-19, *N. gonorrhoeae*** (PMC7038236): 1,280 training isolates with 27
  ready-made determinant columns and log2 MICs for 6 drugs, two external validation sets (1,095 Canada; 431 UK/USA with
  SRA accessions), and the full published regression output. A search agent reported refitting the published
  azithromycin and ciprofloxacin models exactly. Best for "reproduce a published model, then validate externally";
  but © Crown copyright with no CC licence, and a different organism from the slides.
- **Demczuk *et al.* (2022) *AAC* 66:e01370-21, *S. pneumoniae*** (CC BY 4.0): same design for 10 drugs, with explicitly
  censored MIC strings.
- **Ma *et al.* (2020) *Nat Commun* 11:5374** (github.com/gradlab/rplD-conditional-GWAS, MIT): 4,505 gonococci with
  azithromycin MICs, determinant covariates and published per-term coefficients; 65 country effects invite partial pooling.
- **Pataki *et al.* (2020) *Sci Rep*** (github.com/patbaa/AMR_ciprofloxacin, GPL-3.0): 704 *E. coli* with ciprofloxacin
  MICs and a train/test flag, but the feature matrix is not released (rebuilding needs the raw reads).
- **CRyPTIC *M. tuberculosis*** (PLoS Biol 2022; EBI FTP): 12,287 isolates with censored MICs for 13 drugs; the
  genotype tables are too large to prepare for a live demo.
