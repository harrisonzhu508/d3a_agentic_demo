"""EVOLVABLE: NUTS settings for the full development fit (the harness uses its own fixed budget for CV folds,
but takes target_accept_prob, max_tree_depth and dense_mass from here)."""

# Ninety-third experiment: the sampler, changed by two hundred draws, to measure the one quantity this loop has
# never separated from the hypotheses it tests. The harness keeps a change whose cross-validated ELPD gain exceeds
# twice its paired standard error, and the paired SE is sqrt(n) times the across-isolate sd of the pointwise
# held-out log-density difference - a number computed from the posterior a run happens to draw, not from the
# posterior a model has. Every comparison in this session therefore carries two sources of variation at once: the
# priors being compared, and the finite sample of draws that scores them. The loop's own history says the second
# source is not negligible. The champion was kept at +9.19 with an SE of 4.45, the widest margin of error of any
# change on main; the same priors with the rare-column knee pulled in to 12 scored +8.28 with an R-hat of 1.0122;
# and the one run that drew four times as many samples reproduced the champion's ELPD to the printed cent -
# -142.29, a difference of exactly 0.0 against an SE of 0.0 - while reaching an R-hat of 1.0018 against main's
# 1.0077. That is either a reassuring fact about 1000 draws on a fit this well-behaved or an artefact of a
# deterministic seed, and the difference between those two readings decides how every small number in this
# session's log should be quoted. The harness reads its iteration count from the fold budget but takes
# target_accept_prob, max_tree_depth and dense_mass from this dictionary, so a twenty-per-cent increase in warmup
# and draws is the smallest possible change to the machinery: no prior, no feature and no term of the likelihood
# moves. What it buys is a measurement of the harness's own resolution. If the score moves from -142.29 by more
# than a few tenths, a run's reported ELPD carries Monte-Carlo error of the size of the smallest effect this loop
# has ever kept, the two-SE rule is comparing two noisy quantities and calling the difference a result, and the
# report must stop quoting point estimates without saying so. If it moves by nothing again, then every discarded
# branch that came back with an SE under half an ELPD was rejected on its merits and not on its sampler, and the
# plateau this session keeps finding in the priors is a property of a censored likelihood rather than of chains
# that have not run long enough.
SETTINGS = {
    "num_warmup": 1200,
    "num_samples": 1200,
    "num_chains": 4,
    "target_accept_prob": 0.95,
    "max_tree_depth": 10,
    "dense_mass": False,
}
