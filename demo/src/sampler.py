"""EVOLVABLE: NUTS settings for the full development fit (the harness uses its own fixed budget for CV folds,
but takes target_accept_prob, max_tree_depth and dense_mass from here)."""

SETTINGS = {
    "num_warmup": 6000,
    "num_samples": 6000,
    "num_chains": 4,
    "target_accept_prob": 0.99,
    "max_tree_depth": 10,
    "dense_mass": False,
}
