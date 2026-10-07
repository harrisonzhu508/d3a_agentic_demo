"""EVOLVABLE: NUTS settings for the full development fit (the harness uses its own fixed budget for CV folds,
but takes target_accept_prob, max_tree_depth and dense_mass from here)."""

SETTINGS = {
    # Ninety-seventh experiment: the champion's priors, features and likelihood untouched, with one number in this
    # dictionary raised, to test whether the fit on main is converged or merely accepted. The loop's rule is that a
    # branch is scored only if it produces no divergent transitions, a maximum R-hat under 1.01 and a minimum ESS
    # above 400, and main passes all three narrowly - zero divergences, an R-hat of 1.0077 and an ESS of 718 against
    # a floor of 400. Every other measurement this session has made of that fit says it is close to something the
    # sampler finds hard. The identical priors with the rare-column knee at 14 give one divergence; with the knee
    # removed, one divergence; with a Normal(0, 8) in place of the t(4, 0, 8), one; with a t(4, 0, 12) and four times
    # the draws, two; with the whole rare block given to the common columns as well, nothing but a gain that does not
    # clear its own noise. That is a neighbourhood in which small perturbations of the same object tip the fit over a
    # geometric obstacle, and the obstacle has a name: the rare coefficients enter mu linearly, their scale is set by
    # a prior that is flat over a wide band around zero, and the residual scale multiplies the whole likelihood, so a
    # near-flat direction in beta_rare is a near-flat direction in the ratio beta/scale, which is exactly the funnel
    # Neal's paper made famous and exactly the geometry that a diagonal mass matrix cannot step along.
    # target_accept_prob is the one control that addresses a funnel without reparametrising it: it shortens every
    # step the integrator takes in exchange for a deeper tree, which is what a narrow curved valley needs, and it is
    # the setting this file owns. The loop has tried it five times, always combined with a prior change, and always
    # reported as a failed hypothesis when the divergence it bought back was attributed to the prior. This run
    # changes nothing but the setting. If main is converged at 0.95, then raising it holds the score at -142.29, the
    # ESS rises and the champion's diagnostics are real rather than lucky, which is the assurance this loop has been
    # implicitly assuming for ninety-six experiments and has never once tested on the model it kept. If the score
    # moves while the diagnostics stay clean, then main's reported ELPD is a partial-explore artefact of a step size
    # chosen for a different model - the baseline, which had neither the wide rare prior nor the heavy scale - and the
    # numbers of the last thirty runs have been comparing fits that each explored their own fraction of their own
    # posterior, which would be the most consequential finding of the session and would require re-scoring the
    # champion before any of it could be quoted.
    "num_warmup": 1000,
    "num_samples": 1000,
    "num_chains": 4,
    "target_accept_prob": 0.98,
    "max_tree_depth": 10,
    "dense_mass": False,
}
