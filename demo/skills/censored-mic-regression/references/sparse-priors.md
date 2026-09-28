# Sparse priors for many binary determinants

Genotype tables have many columns (genes, alleles, point mutations), most of which do not affect a given
drug, and many of which co-occur (genes carried on the same plasmid, mutations in the same lineage).
Priors should let a few effects be large while shrinking the rest towards zero.

## Horseshoe (Carvalho, Polson & Scott 2010)

`beta_j ~ Normal(0, tau^2 lambda_j^2)`, `lambda_j ~ HalfCauchy(1)`, `tau ~ HalfCauchy(tau0)`.
The global scale `tau` sets overall sparsity; heavy-tailed local scales `lambda_j` let individual effects escape.

## Regularised horseshoe (Piironen & Vehtari 2017, *Electron J Stat* 11:5018)

Adds a slab so that large effects are regularised rather than unbounded:
`lambda_tilde_j^2 = c^2 lambda_j^2 / (c^2 + tau^2 lambda_j^2)`, with `c^2 ~ InvGamma(nu/2, nu s^2 / 2)`.

- Choose `tau0` from a prior guess `p0` of the number of relevant predictors:
  `tau0 = p0 / (p - p0) * sigma / sqrt(n)`.
- Choose the slab scale `s` from the largest effect you consider plausible on the outcome scale
  (here: how many doublings of MIC a single determinant could shift).
- Always use the non-centred form (`z_j ~ Normal(0,1)`, `beta_j = z_j * tau * lambda_tilde_j`).
- Divergences with horseshoes are common: raise `target_accept_prob`, check that `tau0` is not absurdly
  small, and look at pairs plots of `log tau` against a few `log lambda_j`.

## R2D2 (Zhang et al. 2022, *JASA* 117:862)

Places the prior on the proportion of variance explained (`R^2 ~ Beta(a, b)`) and splits it across predictors
with a Dirichlet. Easier to reason about ("how much of the MIC variation should genotype explain?") and often
samples more easily than the horseshoe.

## Other options

- Independent `Normal(0, s)` priors: simplest; fine when the number of predictors is modest relative to `n`.
- Grouped or hierarchical priors: pool related determinants (allele families, genes of the same class) through
  a shared scale.
- Interactions: add product terms only for pairs with a mechanistic reason (see the AMR references), and give
  them tighter priors than main effects.

## Checking a sparse model

- Prior predictive: simulate MICs from the prior; they should cover the plausible range (about 0.001 to 1000
  mg/L), not pile up at the extremes.
- Posterior: report effects with intervals; the number of effects whose interval excludes zero is a simple
  parsimony summary, as is the effective number of parameters `p_loo` from PSIS-LOO.
