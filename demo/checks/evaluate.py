"""Score the current branch: cross-validated censored ELPD, gates, keep or discard. Read-only for agents.

    uv run python checks/evaluate.py --name <slug> --hypothesis "<one line>" --skill <skill-name>

Fits the 5 fixed development folds and one full development fit in parallel processes, then compares the
pointwise held-out ELPD with the experiment's champion (results/<experiment>/champion.json) or, failing that,
the baseline in checks/best.json. Keep iff elpd_diff > 2 * se_diff and every gate passes. Writes
results/<experiment>/<name>/{metrics.json, pointwise.csv, posterior.nc, review.png}, appends a row to
results/<experiment>/results.tsv and prints a one-line JSON verdict last.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import checks.lib as lib  # noqa: E402  (sets JAX flags before JAX is imported)

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, wait  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from checks import integrity  # noqa: E402

RESULTS_COLUMNS = ["timestamp", "name", "branch", "commit", "parent", "harness", "skill", "hypothesis",
                   "elpd_cv", "elpd_diff", "se_diff", "within1", "coverage90", "parsimony", "divergences",
                   "max_rhat", "min_ess", "runtime_s", "decision"]


# ------------------------------------------------------------------ jobs (run in spawned processes)
def _setup():
    features, model, sampler = lib.import_branch()
    df = lib.load_dev()
    X, names = features.build(df)
    X = np.asarray(X, dtype=float)
    lo, hi = lib.intervals(df)
    return features, model, sampler, df, X, names, lo, hi


def fold_job(k: int, seed: int) -> dict:
    _, model, sampler, df, X, _, lo, hi = _setup()
    folds = lib.load_folds()
    f = df["genome_id"].map(folds).to_numpy()
    tr, te = f != k, f == k
    cfg = lib.config()
    settings = {**cfg["cv_sampler"],
                **{key: sampler.SETTINGS[key] for key in ("target_accept_prob", "max_tree_depth", "dense_mass")}}
    latent = lib.latent_sites(model.model, X, lo, hi)
    mcmc = lib.fit(model.model, X[tr], lo[tr], hi[tr], settings, seed=seed + k)
    out = lib.heldout(model, mcmc, latent, X[te], lo[te], hi[te], seed=seed + 100 + k)
    within, covered = lib.agreement_and_coverage(out["draws"], df[te], lib.dilution_uppers(df))
    return {"fold": k, "ids": df.loc[te, "genome_id"].tolist(), "elpd": out["elpd"].tolist(),
            "within": within.tolist(), "covered": covered.tolist(),
            "divergences": int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())}


def full_job(seed: int, out_dir: str) -> dict:
    _, model, sampler, df, X, names, lo, hi = _setup()
    cfg = lib.config()
    latent = lib.latent_sites(model.model, X, lo, hi)
    effects = getattr(model, "FEATURE_EFFECTS", None)
    t0 = time.time()
    mcmc = lib.fit(model.model, X, lo, hi, sampler.SETTINGS, seed=seed)
    diag = lib.diagnostics(mcmc, latent, effects)
    par = lib.parsimony(mcmc, effects, cfg["parsimony"]["threshold_doublings"], cfg["parsimony"]["probability"])
    lib.save_idata(mcmc, Path(out_dir) / "posterior.nc", names, df["genome_id"].tolist())
    return {**diag, "parsimony": par, "n_features": len(names), "latent_sites": latent,
            "fit_seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ main
def append_result(row: dict) -> None:
    path = lib.experiment_dir() / "results.tsv"
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULTS_COLUMNS, delimiter="\t", extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def update_session(name: str) -> None:
    f = lib.session_file()
    if f.exists():
        s = json.loads(f.read_text())
        s.setdefault("experiments", []).append(name)
        f.write_text(json.dumps(s, indent=2))


def verdict(payload: dict, code: int = 0) -> None:
    print(json.dumps(payload))
    sys.exit(code)


def record_failure(name: str, out_dir: Path, state: dict, ref_branch: str, args, decision: str, reason: str,
                   runtime: float) -> None:
    """Crashes and budget overruns are experiments too: log them so the loop (and the stop hook) sees them."""
    metrics = {"name": name, "experiment": lib.settings()["experiment"], **state, "parent": ref_branch,
               "pr_base": lib.experiment_branch(), "harness": os.environ.get("D3A_HARNESS", "unknown"),
               "hypothesis": args.hypothesis, "skill": args.skill, "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
               "runtime_s": runtime, "decision": decision, "reason": reason, "pr_url": None, "reverted": False}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    append_result(metrics)
    update_session(name)
    verdict({"decision": decision, "name": name, "reason": reason,
             "metrics": lib.rel(out_dir / "metrics.json")}, 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="short slug for this hypothesis (default: the branch name without <prefix>/<experiment>/)")
    ap.add_argument("--hypothesis", default="", help="one line: what you changed and why")
    ap.add_argument("--skill", default="", help="the skill that guided the change")
    ap.add_argument("--baseline", action="store_true", help="(setup only) record this run as checks/best.json")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--abandon", metavar="REASON", default="",
                    help="record the hypothesis as abandoned without fitting it (it could not be made to work)")
    args = ap.parse_args()

    cfg = lib.config()
    changed = integrity.verify()
    if changed:
        verdict({"decision": "refused", "reason": "harness files changed: " + ", ".join(changed)}, 3)

    state = lib.git_state()
    if args.abandon:   # recorded like any other experiment, so it counts and is not tried again
        name = lib.slug(args.name or state["branch"].removeprefix(lib.branch_namespace()))
        out_dir = lib.run_dir(name)
        done = out_dir / "metrics.json"
        if done.exists():
            verdict({"decision": "refused", "name": name,
                     "reason": f"'{name}' already has a verdict ({json.loads(done.read_text()).get('decision')}); "
                               "do not abandon it, discard it: skills/github-workflow/scripts/discard.py --name " + name}, 3)
        out_dir.mkdir(parents=True, exist_ok=True)
        record_failure(name, out_dir, state, lib.reference()[1], args, "abandoned", args.abandon, 0.0)
    name = lib.slug(args.name or state["branch"].removeprefix(lib.branch_namespace()))
    out_dir = lib.run_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_label, ref_branch, ref_pw, ref_summary = lib.reference()
    pr_base = lib.experiment_branch()
    print(f"evaluating '{name}' on {state['branch']}@{state['commit']}{' (DIRTY)' if state['dirty'] else ''}; "
          f"reference: {ref_label}", flush=True)

    t0 = time.time()
    seed = cfg["seed"]
    ctx = mp.get_context("spawn")
    ex = ProcessPoolExecutor(max_workers=cfg["n_folds"] + 1, mp_context=ctx)
    futs = [ex.submit(fold_job, k, seed) for k in range(cfg["n_folds"])] + [ex.submit(full_job, seed, str(out_dir))]
    budget = lib.settings()["budget_seconds"]
    budget = cfg["budget_seconds"] if budget is None else (budget or None)    # 0 in the config: no limit
    done, pending = wait(futs, timeout=budget)
    runtime = round(time.time() - t0, 1)
    if pending:
        for p in list(getattr(ex, "_processes", {}).values()):
            p.terminate()
        ex.shutdown(wait=False, cancel_futures=True)
        record_failure(name, out_dir, state, ref_branch, args, "fail-budget",
                       f"evaluation exceeded the {budget} s budget. If the idea is promising, it is a "
                       "near miss: discard this branch and retry the same model with lighter sampler settings "
                       "(num_warmup/num_samples 1000, max_tree_depth 10, dense_mass False)", runtime)
    try:
        results = [f.result() for f in futs]
    except Exception as err:  # the branch's code raised inside a job
        ex.shutdown(wait=False, cancel_futures=True)
        record_failure(name, out_dir, state, ref_branch, args, "crash", f"{type(err).__name__}: {err}"[:500], runtime)
    ex.shutdown()
    folds, full = results[:-1], results[-1]

    ids = sum((r["ids"] for r in folds), [])
    elpd_i = np.array(sum((r["elpd"] for r in folds), []))
    within = np.array(sum((r["within"] for r in folds), []))
    covered = np.array(sum((r["covered"] for r in folds), []))
    n = len(ids)
    pointwise = pd.DataFrame({"genome_id": ids, "elpd": elpd_i})
    if ref_pw:
        pointwise["elpd_ref"] = pointwise["genome_id"].map(ref_pw)
    pointwise.to_csv(out_dir / "pointwise.csv", index=False)

    g = cfg["gates"]
    gates = {
        "divergences": full["divergences"], "max_rhat": round(full["max_rhat"], 4),
        "min_ess_bulk": round(full["min_ess_bulk"]), "min_ess_tail": round(full["min_ess_tail"]),
        "cv_divergences": sum(r["divergences"] for r in folds), "runtime_ok": budget is None or runtime <= budget,
    }
    gates["passed"] = bool(gates["divergences"] <= g["max_divergences"] and gates["max_rhat"] < g["max_rhat"]
                           and min(gates["min_ess_bulk"], gates["min_ess_tail"]) > g["min_ess"] and gates["runtime_ok"])

    compare = {"against": ref_label, "elpd_diff": None, "se_diff": None, "reference": ref_summary}
    if ref_pw:
        common = [i for i in ids if i in ref_pw]
        d = pointwise.set_index("genome_id").loc[common, "elpd"].to_numpy() - np.array([ref_pw[i] for i in common])
        compare.update(elpd_diff=round(float(d.sum()), 2), se_diff=round(float(np.sqrt(len(d)) * d.std(ddof=1)), 2))

    k = cfg["keep_rule"]["se_multiplier"]
    if args.baseline:
        decision = "baseline"
    elif state["dirty"]:
        decision = "invalid-dirty"
    elif not gates["passed"]:
        decision = "fail-gates"
    elif compare["elpd_diff"] is None or compare["elpd_diff"] > k * compare["se_diff"]:
        decision = "keep"
    else:
        decision = "discard"

    metrics = {
        "name": name, "experiment": lib.settings()["experiment"], "worker": lib.settings()["worker"], **state,
        "parent": ref_branch, "pr_base": pr_base,
        "harness": os.environ.get("D3A_HARNESS", "unknown"), "hypothesis": args.hypothesis, "skill": args.skill,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "runtime_s": runtime,
        "elpd_cv": round(float(elpd_i.sum()), 2), "elpd_cv_se": round(float(np.sqrt(n) * elpd_i.std(ddof=1)), 2),
        "within1": round(float(within.mean()), 3), "coverage90": round(float(covered.mean()), 3),
        "parsimony": full["parsimony"], "n_features": full["n_features"], "latent_sites": full["latent_sites"],
        "gates": gates, "compare": compare, "decision": decision, "pr_url": None, "reverted": False,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    append_result({**metrics, "elpd_diff": compare["elpd_diff"], "se_diff": compare["se_diff"],
                   "divergences": gates["divergences"], "max_rhat": gates["max_rhat"],
                   "min_ess": min(gates["min_ess_bulk"], gates["min_ess_tail"])})
    update_session(name)

    if decision == "keep":
        (lib.experiment_dir() / "champion.json").write_text(json.dumps({
            "branch": state["branch"], "name": name, "commit": state["commit"], "elpd_cv": metrics["elpd_cv"],
            "pointwise_file": lib.rel(out_dir / "pointwise.csv"), "previous": ref_branch}, indent=2))
    if args.baseline:
        (lib.DEMO / "checks" / "best.json").write_text(json.dumps({
            "branch": state["branch"], "commit": state["commit"], "elpd_cv": metrics["elpd_cv"],
            **{k: metrics[k] for k in ("within1", "coverage90", "parsimony")},
            "pointwise": dict(zip(ids, map(float, elpd_i)))}, indent=1))

    if not args.no_plots:
        plots = lib.DEMO / "skills" / "mic-plots" / "scripts" / "plots.py"
        r = subprocess.run([sys.executable, str(plots), str(lib.DEMO / "data" / "dev.csv"),
                            str(out_dir / "posterior.nc"), str(out_dir / "review")], capture_output=True, text=True)
        metrics["plots"] = lib.rel(out_dir / "review.png") if r.returncode == 0 else None

    print(f"ELPD(CV) {metrics['elpd_cv']} ± {metrics['elpd_cv_se']} | vs {ref_label}: "
          f"Δ {compare['elpd_diff']} ± {compare['se_diff']} | within ±1 dilution {metrics['within1']:.1%} | "
          f"90% coverage {metrics['coverage90']:.1%} | gates {'pass' if gates['passed'] else 'FAIL'} "
          f"(div {gates['divergences']}, R-hat {gates['max_rhat']}, ESS {min(gates['min_ess_bulk'], gates['min_ess_tail'])}) "
          f"| {runtime} s")
    verdict({"decision": decision, "name": name, "elpd_cv": metrics["elpd_cv"], "elpd_diff": compare["elpd_diff"],
             "se_diff": compare["se_diff"], "gates_passed": gates["passed"], "pr_base": pr_base,
             "metrics": lib.rel(out_dir / "metrics.json")})


if __name__ == "__main__":
    main()
