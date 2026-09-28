"""EVOLVABLE: NUTS settings for the full development fit (the harness uses its own fixed budget for CV folds,
but takes target_accept_prob, max_tree_depth and dense_mass from here)."""

SETTINGS = {
    "num_warmup": 1000,
    "num_samples": 1000,
    "num_chains": 4,
    "target_accept_prob": 0.95,
    "max_tree_depth": 10,
    "dense_mass": False,
}
