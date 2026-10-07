"""EVOLVABLE: NUTS settings for the full development fit (the harness uses its own fixed budget for CV folds,
but takes target_accept_prob, max_tree_depth and dense_mass from here)."""

SETTINGS = {
    "num_warmup": 1000,
    "num_samples": 1000,
    "num_chains": 4,
    "target_accept_prob": 0.95,
    "max_tree_depth": 10,
    # The design matrix is strongly correlated by construction - the QRDR allele columns are alternative
    # mutations at the same two codons and gyrA D87N correlates with parC S80I at 0.95 - so the posterior
    # covariance of beta is far from diagonal and a diagonal mass matrix has to take many small steps across
    # it. dense_mass estimates the full covariance during warmup and removes that anisotropy, which is the
    # recommended fix when the geometry problem comes from correlated predictors rather than from the prior
    # (bayesian-workflow, references/mcmc-diagnostics.md: 'different scales or strong correlations; consider
    # dense_mass=True'). p = 77, so the O(p^3) adaptation costs little next to the likelihood evaluations.
    "dense_mass": True,
}
