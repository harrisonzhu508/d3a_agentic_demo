# The Bayesian workflow (summary)

Main source: Gelman, Vehtari, Simpson, Margossian, Carpenter, Yao, Kennedy, Gabry, Bürkner & Modrák (2020),
"Bayesian Workflow", arXiv:2011.01808. The older framing is Box's loop (Box 1980; Blei 2014): build a model,
compute the posterior, criticise the model, revise.

## Steps

1. **Start from a simple model you understand**, with priors you can defend. Write down what each parameter means.
2. **Prior predictive check**: simulate data from the priors alone. If the simulated data are absurd
   (MICs of 10^9 mg/L, all isolates at the plate limit), the priors are too wide or badly scaled.
3. **Fit to simulated data** (fake-data check): simulate from known parameter values, fit, and check you
   recover them. Simulation-based calibration (Talts et al. 2018, arXiv:1804.06788) does this systematically.
4. **Fit to real data** and check the computation (see `mcmc-diagnostics.md`).
5. **Posterior predictive checks**: compare replicated data with observed data using test quantities that
   matter for the question (bin counts, the fraction at the plate limits, residuals by subgroup).
6. **Evaluate prediction**: out-of-sample ELPD by PSIS-LOO or K-fold CV (see `model-comparison.md`).
7. **Modify the model** where the checks point, one change at a time:
   - misfit in the tails or a few extreme observations: heavier-tailed or mixture likelihood;
   - misfit for a subgroup: an interaction or a group-level effect;
   - unstable or implausible effects: stronger or better-structured priors;
   - computation problems: reparameterise before changing the model.
8. **Compare the models you built**, keep the record of all of them, and prefer the simplest model that is
   not clearly worse. Multiverse or stacking is better than pretending one model is the truth.

## Folk theorem of statistical computing

"When you have computational problems, often there's a problem with your model" (Gelman 2008). Divergences
and poor mixing are often signs of priors on the wrong scale, non-identified parameters or a mis-specified
likelihood, not just a sampler that needs more tuning.
