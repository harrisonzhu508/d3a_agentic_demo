# program.md: the autoresearch loop

(After Karpathy's autoresearch: the human writes this file, the agent edits `src/`, the harness is fixed.)

## Setup (once per session)

- Read `AGENTS.md`, `TASK.md` and `skills/CATALOG.md`; run `uv run python checks/status.py`.
- The session budget is a number of experiments (shown at session start). The stop hook will not let you
  finish until it is used up and your last experiment is published or discarded.

## One experiment

1. **Branch**: `uv run python skills/github-workflow/scripts/new_branch.py <short-slug>`
2. **Choose one hypothesis.** Look at the champion's review plots (`results/<experiment>/<champion>/review.png`, or run
   `uv run python train.py` for a fresh fit) and at `checks/status.py`. Use the `bayesian-workflow` skill:
   what does the misfit or the diagnostics point to? Do not repeat an idea that `results/<experiment>/results.tsv` already records.
   Spread the search over the research directions below: after two hypotheses in a row in one direction, pick
   another one. A near miss is worth one follow-up that fixes what stopped it: a gain that failed a gate (fix
   the sampler or parameterisation), or an evaluation that ran out of time (the same model with lighter sampler
   settings). If two or more hypotheses each gained more than one SE but none cleared two, combine them: start
   from the best one with `new_branch.py <slug> --from <its-slug>` and add the other change.
3. **Read the skill** for that change (`censored-mic-regression` for the likelihood, priors or sampler;
   `amr-genotype-phenotype` for features and interactions) before touching code.
4. **Edit `src/`**: one focused change. Wait for the post-edit check to pass. If it keeps failing and the fix
   is not obvious (the hook says so after three failures), abandon the hypothesis instead of rewriting it again:
   `uv run python checks/evaluate.py --name <slug> --abandon "<why>"`, then discard it (step 7).
5. **Commit**: `git add src && git commit -m "hyp: <change> [skill: <name>]"`
6. **Evaluate**: `uv run python checks/evaluate.py --name <slug> --hypothesis "<one line>" --skill <name>`
   (budget: 10 minutes; most runs take well under 2).
7. **Decide** from the JSON verdict:
   - `keep`: write the report with `uv run python skills/mic-eval-harness/scripts/write_report.py --name <slug>`.
     The first run creates `results/<experiment>/<slug>/report_draft.md`; write its four sections (Results,
     Data, Method, Conclusion) as its comments ask, then run the command again: it adds the harness numbers,
     the plots and the code diff, and commits `reports/<experiment>/<slug>/`. Then
     `uv run python skills/github-workflow/scripts/publish.py --name <slug>` opens the pull request into the
     experiment branch `<prefix>/<experiment>/main`. The kept branch becomes the new champion; the next
     hypothesis starts from it.
   - anything else (including `abandoned`): `uv run python skills/github-workflow/scripts/discard.py --name <slug>`.
     The branch is kept as `<prefix>/<experiment>/discarded/<slug>`, so the attempt stays on record.
8. **Repeat** from step 1 with a new hypothesis.

## Research directions

Directions, not answers: which ones pay off for this dataset is what the experiments find out. Read the skill
named in brackets before trying one.

| Direction | Examples of hypotheses |
|---|---|
| Likelihood and censoring (`censored-mic-regression`) | heavier-tailed errors (Student-t); noise that differs between groups of isolates; a model on the dilution bins themselves (ordinal / cumulative link); a mixture of latent subpopulations |
| Priors and shrinkage (`censored-mic-regression`) | horseshoe / regularised horseshoe, R2D2, hierarchical priors that pool effects within gene families or mechanisms, priors on scale set from what is plausible for a doubling |
| Features and interactions (`amr-genotype-phenotype`) | pairwise interactions between target mutations, counts of mutations per target, grouping rare determinants, removing near-constant or redundant columns |
| Sampler and parameterisation (`bayesian-workflow`) | non-centred forms for hierarchical or sparse priors, `target_accept_prob`, dense mass matrix, tree depth, warm-up length (to fix divergences or low ESS, not to raise ELPD) |
| Model comparison and checks (`bayesian-workflow`) | read the review plots and the pointwise comparison to find where the champion misfits, then aim a hypothesis at that |

## Principles

- Never stop early: when an experiment ends, start the next one until the budget is used.
- If a change crashes, fix it quickly if the fix is obvious; otherwise discard and move on.
- Simplicity wins ties: a smaller model with the same ELPD is better.
- Diagnostics before estimates: an estimate from a fit with divergences or poor mixing means nothing.
