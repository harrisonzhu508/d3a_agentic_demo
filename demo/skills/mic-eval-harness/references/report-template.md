<!-- Report draft for "{name}". This is the fixed structure of every autoresearch pull request.
     Replace each guidance comment with your own text and keep the four headings as they are. The verdict, the
     metrics table, the plots and the code diff are added by write_report.py from the harness output, so do not
     copy them here. Quote only numbers that checks/evaluate.py printed. -->

## Results
<!-- 3 to 5 sentences; the metrics table and the plots are placed above your text. Interpret them: the decision and
     by how much (ELPD difference ± SE against the reference), whether the gates passed, what the secondary
     metrics say (within ±1 dilution, 90 % coverage, non-null effects), where on the MIC scale the change helped
     or hurt (comparison plot), and anything notable in the review plots (effects, predictive check, calibration). -->

## Data
<!-- 2 to 4 bullets. The data set, its size and the CV folds are already in the header; do not repeat them. The
     features the model saw (unchanged, or what src/features.py now adds, removes or transforms), and any
     property of the data that motivated or limits the change (censoring at the ends of the dilution range, rare
     determinants, co-occurring mutations). -->

## Method
<!-- 3 to 5 bullets. The hypothesis in one line. The exact change (likelihood, priors, features or sampler), with
     the key formula if it helps, for example `beta_j ~ Normal(0, tau * lambda_j)`. The skill and reference file
     that guided it. Anything you tried on this branch and dropped, and why. -->

## Conclusion
<!-- 2 to 4 sentences. Keep or discard by the harness rule, and why. What this says about the biology or the
     measurement process. Limitations. One or two concrete next hypotheses. -->
