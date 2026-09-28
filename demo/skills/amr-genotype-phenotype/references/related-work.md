# Related genotype-to-MIC studies (other datasets)

Methods and lessons only. By design this file contains nothing about the dataset in `data/` or its paper.

| Study | Data | Approach | Lesson for us |
|---|---|---|---|
| Nguyen *et al.* 2018, *Sci Rep* 8:421 | 1,668 *K. pneumoniae*, 20 drugs | XGBoost on genome k-mers, 10-fold CV | k-mer models reach about 92 % within ±1 dilution; interpretability is limited |
| Nguyen *et al.* 2019, *J Clin Microbiol* 57:e01260-18 | 5,278 *Salmonella* (US NARMS), 15 drugs | XGBoost on k-mers | about 95 % within ±1 dilution; very major errors concentrate where resistant isolates are rare |
| Pataki *et al.* 2020, *Sci Rep* 10:15026 | 704 *E. coli*, ciprofloxacin, several countries | known QRDR mutations and PMQR genes vs genome-wide features; linear models and random forests; leave-one-country-out CV | a handful of known determinants explains most of the variation; simple linear models are competitive with machine learning |
| Eyre *et al.* 2017, *J Antimicrob Chemother* 72:1937 | 681 *N. gonorrhoeae*, 5 drugs | multivariable regression of log2 MIC on known determinants | regression on known mechanisms gives accurate, interpretable predictions (about 93 % within ±1 dilution) |
| Demczuk *et al.* 2020, *Antimicrob Agents Chemother* 64:e02005-19 | 1,280 *N. gonorrhoeae* + two external sets | linear regression equations per drug on determinant indicators, external validation | publish the full model; validate on external collections, where accuracy drops |
| Batisti Biffignandi *et al.* 2024, *Microb Genom* 10:001222 | 4,367 *K. pneumoniae*, 4 drugs | elastic net, random forest and linear mixed models, lineage-aware splits | handle left- and right-censoring explicitly; account for lineages (population structure) in splits and models |
| CRyPTIC Consortium 2022, *PLoS Biol* | 12,287 *M. tuberculosis*, 13 drugs | quantitative MICs on a plate with censored ranges | censored MICs at scale; every measurement carries a quality flag worth using |
| Jaspers, Lambert & Aerts 2016, *Ann Appl Stat* 10(2) | MIC distributions | Bayesian model for interval-censored MICs | treat MICs as intervals, not numbers |
| Dürr 2026, AutoStan, arXiv:2603.27766 | several datasets | a coding agent improves Stan models, keeping changes only if held-out predictive density improves and diagnostics pass | the same loop as this demo: fixed evaluation, keep or revert |

## Common threads

- Known mechanisms get you most of the way; genome-wide features add little for well-understood drugs.
- Censoring and population structure are the two main ways to fool yourself.
- Report how censored values were handled and how the data were split, or the numbers are not comparable.
