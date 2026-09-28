---
name: github-workflow
description: Git and GitHub steps for one autoresearch experiment - start a hypothesis branch from the current champion, commit with the right message, and after evaluation either publish (push and open a pull request with the report) or discard (record it and go back). Use at the start and end of every experiment, and whenever you would otherwise run git push or git branch yourself.
license: MIT
---

# Git and GitHub for autoresearch

One hypothesis = one branch = one commit (or a few small ones) = one evaluation = keep (PR) or discard.
The scripts below do the risky git steps for you; run them instead of hand-written git commands.

## Steps

1. **Start**: `uv run python skills/github-workflow/scripts/new_branch.py <short-slug>`
   Creates `<prefix>/<experiment>/<short-slug>` (prefix and experiment are set by the user in
   `config/autoresearch.toml`) from the current champion: the best kept branch of this experiment so far, else
   the experiment branch `<prefix>/<experiment>/main`. The first call of an experiment creates that experiment
   branch from the base branch; every kept hypothesis opens its pull request into it.
   Slugs are short, lowercase, hyphenated, and describe the change: `horseshoe-prior`, `gyra-parc-interaction`.
   `--from <earlier-slug>` starts from an earlier hypothesis instead (a kept one, or a discarded near miss), to
   combine changes that each helped a little; the result is still compared with the champion.
2. **Commit** after editing `src/` and passing the post-edit check:
   `git add src && git commit -m "hyp: <what changed, one line> [skill: <skill that guided it>]"`
   Commit before evaluating: the harness scores only committed code (a dirty tree is recorded as invalid).
3. **Evaluate** with the `mic-eval-harness` skill (`checks/evaluate.py`). Read the JSON verdict.
4. **Keep** (`"decision": "keep"`): write and commit the report with the `mic-eval-harness` report tool
   (`skills/mic-eval-harness/scripts/write_report.py --name <name>`, see that skill), then
   `uv run python skills/github-workflow/scripts/publish.py --name <name>`
   This pushes the branch (and the experiment branch, the first time), opens a pull request into
   `<prefix>/<experiment>/main` with the committed report as its description, labels it `autoresearch` and
   `experiment:<name>`, and records the PR URL in `results/<experiment>/<name>/metrics.json`. If the branch builds
   on an earlier kept hypothesis that is not merged yet, the description says so. If the user enabled
   `git.auto_merge`, publish.py also merges it into the experiment branch (you do nothing extra), and the next
   hypothesis starts from there.
5. **Discard** (anything else, including `abandoned`): `uv run python skills/github-workflow/scripts/discard.py --name <name>`
   Restores `src/`, switches back to the champion, and keeps the branch as `<prefix>/<experiment>/discarded/<name>`
   (pushed, so every hypothesis is visible on GitHub). The result stays in `results/<experiment>/`, so it is not
   lost and not repeated.
6. **Converge**: you do nothing. When the session ends, `experiment_pr.py` opens (or updates) one pull request
   from the experiment branch into the base branch, listing every hypothesis and the kept ones.

## Rules

- Commit to and push only hypothesis branches under the configured prefix (`checks/status.py` prints it);
  never the experiment branch (`.../main`, it changes only through reviewed pull requests), the base branch or
  `main`, never force-push, never rewrite history (the pre-tool-use hook blocks these anyway).
- Never edit files outside `src/` in a commit; never commit `results/` (it is gitignored).
- By hand, git is for looking (`status`, `log --oneline`, `diff -- src`, `show --stat`) and for `git add src` +
  `git commit`. No `reset`, `stash`, `switch`/`checkout` or `--amend`: if the branch is in a state the scripts
  cannot handle, stop and say so in your final message.
- The PR body is evidence for a human reviewer: numbers only from the harness output, plus what you tried and
  discarded on the way.
- If `publish.py` reports that no GitHub token is configured, the branch is still pushed; say so in your final
  message instead of retrying.
