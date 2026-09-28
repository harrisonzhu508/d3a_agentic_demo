"""Start a hypothesis branch <branch_prefix>/<experiment>/<slug> from the experiment's current champion.

    uv run python skills/github-workflow/scripts/new_branch.py <slug>
    uv run python skills/github-workflow/scripts/new_branch.py <slug> --from <earlier-slug>

--from starts from an earlier hypothesis instead (a kept one, or a discarded near miss kept under discarded/),
to combine ideas that each helped a little. The new branch is still scored against the champion.

The first call of an experiment also creates its experiment branch, <branch_prefix>/<experiment>/main, from the
base branch: every kept hypothesis opens a pull request into it.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gitlib import branch_namespace, champion_branch, git, settings, slug  # noqa: E402


def main() -> None:
    args = sys.argv[1:]
    start = None
    if "--from" in args:     # build on an earlier hypothesis (e.g. combine two near misses)
        i = args.index("--from")
        start, args = args[i + 1], args[:i] + args[i + 2:]
    if len(args) != 1:
        sys.exit(__doc__)
    name = re.sub(r"[^a-z0-9]+", "-", slug(args[0])).strip("-")[:40] or "hypothesis"
    if name in ("main", "discarded"):                          # reserved: the experiment branch, the archive
        name += "-hypothesis"
    worker = settings()["worker"]
    if worker and not name.startswith(worker + "-"):           # parallel agents: names never collide
        name = f"{worker}-{name}"
    if git("status", "--porcelain", "--", "src"):
        sys.exit("src/ has uncommitted changes: commit them or run `git restore src` first.")
    base = champion_branch()
    if start:
        ns = branch_namespace()
        cands = [ns + start, ns + "discarded/" + start, start]
        found = next((b for b in cands if git("rev-parse", "--verify", "--quiet", b, check=False)), None)
        if not found:
            sys.exit(f"no branch for '{start}' (tried {', '.join(cands)})")
        base = found
    ns = branch_namespace()
    branch, i = ns + name, 2
    while git("rev-parse", "--verify", "--quiet", branch, check=False):
        branch, i = f"{ns}{name}-{i}", i + 1
    git("switch", "-c", branch, base)      # straight from the champion (it may be checked out by another worker)
    print(f"on {branch} (from {base}); use --name {branch.removeprefix(ns)} with checks/evaluate.py")


if __name__ == "__main__":
    main()
