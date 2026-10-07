"""EVOLVABLE: NUTS settings for the full development fit (the harness uses its own fixed budget for CV folds,
but takes target_accept_prob, max_tree_depth and dense_mass from here)."""

SETTINGS = {
    "num_warmup": 1000,
    "num_samples": 1000,
    "num_chains": 4,
    # One hundred and forty-third experiment: the run that tests whether this loop has been optimising the wrong
    # file, on the single setting the harness measures and no experiment of mine has ever changed except in passing.
    # The record is a straight line in one variable. Fourteen of this session's runs were refused by divergences, and
    # thirteen of them by between one and four transitions out of eight thousand - single events in single chains, at
    # fits whose R-hat and effective sample size were otherwise at or better than the champion's. Four of those fits
    # were re-run with the acceptance target lifted from 0.95 to 0.97 and in every case the divergence vanished and the
    # cross-validated score came back to within two hundredths of an ELPD of the value it had had with the divergence
    # present: 3.48 against 3.56 on the intercept lever, 4.79 against 4.72 on the pair, and twice with no comparable
    # prior run at all. A fifth setting was tried at 0.99 on the design's oldest refused gain and left the transition
    # where it was. So the loop concluded that acceptance is the lever, and has since used it as a safety catch. But
    # acceptance does not remove a divergence; it changes the step size that produces the trajectory along which a
    # divergence occurs, and a divergence at a step size is a trajectory that reached a region where the symplectic
    # integrator cannot resolve the curvature in the number of leapfrogs the tree is allowed to take. The setting that
    # bounds that is on the line below this one, and it has been at its library default for the entire experiment: a
    # tree of depth 10 is at most 1024 leapfrog steps, and NumPyro reports a trajectory that hits the limit as a
    # divergence, which means that on this design a single chain in a single iteration need only wander into the
    # correlated ridge between the level and seventy-seven binary slopes and start walking along it for the harness to
    # record a transition that no amount of acceptance will remove. It is also the only mechanism consistent with the
    # shape of this log's failures - gains of 3, 4, 5 and 18 ELPD refused by 1 to 3 transitions, and never by 100,
    # which is what a pathological posterior looks like and is what the non-centred parameterisation produced when this
    # loop tried it properly: 2093 divergences, an R-hat of 1.023 and a lost 34 ELPD. So: depth 12, four times the
    # budget, and an acceptance target of 0.96 rather than 0.97 - the midpoint of the interval the loop has measured as
    # safe, chosen so that if this run returns the champion's own score with a clean gate, the difference between it and
    # the champion is attributable to the tree rather than to a setting already known to move nothing. The prediction
    # for the score is exact and that is the point of the experiment: 0.00 ELPD, with the fourteen refused runs
    # converted into fourteen measurements the harness can use.
    "target_accept_prob": 0.96,
    "max_tree_depth": 12,
    "dense_mass": False,
}
