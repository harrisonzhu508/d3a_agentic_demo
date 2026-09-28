"""One-off (human, via scripts/setup.sh): split the 703 isolates into development and locked test sets.

- data/dev.csv: 80 %, the only data the agent sees;
- $D3A_PRIVATE_DIR/test.csv (default ~/.d3a-demo-private/test.csv): 20 %, outside the repo, scored only by
  checks/final_test.py;
- checks/folds.json: fixed 5-fold assignment of the development isolates.
Both splits are stratified by MIC bin and seeded, so they are reproducible.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEMO = Path(__file__).resolve().parents[1]
CFG = json.loads((DEMO / "checks" / "harness.json").read_text())


def stratified_assign(labels: pd.Series, n_groups: int, rng: np.random.Generator) -> np.ndarray:
    """Round-robin assignment within each label after shuffling: balanced groups per stratum."""
    out = np.empty(len(labels), dtype=int)
    for _, idx in labels.groupby(labels).groups.items():
        idx = rng.permutation(np.asarray(list(idx)))
        out[labels.index.get_indexer(idx)] = np.arange(len(idx)) % n_groups
    return out


def main() -> None:
    import os
    full = pd.read_csv(DEMO.parent / "dataset" / "cip_features.csv")   # outside the agent workspace
    rng = np.random.default_rng(CFG["seed"])
    n_test_groups = round(1 / CFG["test_fraction"])                       # 5 groups -> 1 is the test set
    group = stratified_assign(full["mic_raw"].reset_index(drop=True), n_test_groups, rng)
    test, dev = full[group == 0], full[group != 0].reset_index(drop=True)

    private = Path(os.environ.get("D3A_PRIVATE_DIR", CFG["private_dir_default"])).expanduser()
    private.mkdir(parents=True, exist_ok=True)
    test.to_csv(private / "test.csv", index=False)
    dev.to_csv(DEMO / "data" / "dev.csv", index=False)

    folds = stratified_assign(dev["mic_raw"], CFG["n_folds"], np.random.default_rng(CFG["seed"] + 1))
    (DEMO / "checks" / "folds.json").write_text(json.dumps(dict(zip(dev["genome_id"], map(int, folds))), indent=0))
    print(f"dev {len(dev)} isolates -> data/dev.csv; test {len(test)} -> {private / 'test.csv'}; "
          f"{CFG['n_folds']} folds -> checks/folds.json")


if __name__ == "__main__":
    main()
