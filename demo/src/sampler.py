"""EVOLVABLE: NUTS settings for the full development fit (the harness uses its own fixed budget for CV folds,
but takes target_accept_prob, max_tree_depth and dense_mass from here)."""

# One hundred and second experiment: the sampler, one notch more careful, on a model nobody has given it yet. The
# residual scale of this fit has now been measured under ten priors and it has answered 6.9, 9.2, 10.8, 10.8, 11.2,
# 15.6, 33.0 and 82.1 doublings, the ordering following the effort each prior makes to hold it down rather than any
# property of the isolates; and on every one of those runs the harness asked for four chains that agree to within one
# per cent, and got it, except when it did not. The gate this session has failed most often is not the divergence
# count - a single divergence has cost more branches than any loss - and the champion itself passes with an R-hat of
# 1.0077 against a limit of 1.01, which is a margin of three ten-thousandths on a parameter whose posterior spans a
# factor of four. The runs nearest to that limit were not the ones with the widest priors; they were the ones with
# the sharpest change in prior density somewhere in the region the fit occupied: the whole session's worth of scale
# experiments failed on R-hat at 1.0100 and 1.036, and the two attempts to help them, a dense mass matrix and a
# deeper target acceptance, produced an R-hat of 1.554 with an effective sample size of 18 and two divergences
# against main's zero. That is the behaviour of a funnel - a narrow curved valley which a steeper step and a matrix
# that mixes everything together traverse worse, not better - and the model has two of them, one in the ratio of the
# rare coefficients to the residual scale and one in the ratio of the intercept to the same scale.
# The two fixes the geometry actually admits are a reparametrised scale, which the interface will not let me write,
# and a smaller step, which target_accept_prob buys at the price of tree depth. What the loop has never done is apply
# the second one on its own. target_accept_prob has been changed five times this session and every time alongside a
# prior, so each of those runs is evidence about its hypothesis and none of them is evidence about the setting; the
# last time it was raised on the champion's own priors it cost -0.25 ELPD and two divergences, which is one reading
# of a step that becomes too small to explore in the depth allowed. A step that is too small and a step that is too
# large both show up as chains disagreeing about the funnel, and only one of the two is improved by going the other
# way, so the experiment is to move the single number downward and leave every prior exactly where main has it. If
# the R-hat improves toward 1.003 - the value the same model reached once at three thousand draws - and the score
# holds, then the champion's diagnostics sit on the edge of a gate for a reason the step size controls, the last
# thirty comparisons in the log have been scored on fits that differed in how much of their own posterior they saw,
# and the loop should re-score its own champion before quoting anything further. If it diverges or drifts, the
# 0.95 that the baseline shipped with is the right number for this likelihood and the session's gate failures are
# the priors' doing after all, which is what the report has been assuming and has never established.

SETTINGS = {
    "num_warmup": 1000,
    "num_samples": 1000,
    "num_chains": 4,
    "target_accept_prob": 0.93,
    "max_tree_depth": 10,
    "dense_mass": False,
}
