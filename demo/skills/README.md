# Skills for the demo agent

Agent Skills format (<https://agentskills.io/specification>): each folder has a `SKILL.md` with `name` and
`description` frontmatter; `scripts/` and `references/` are loaded only when needed. Link or copy the folders
into the harness's skill path (`.claude/skills/`, `.agents/skills/` or `.github/skills/`, depending on the harness).

| Skill | For | Contents | Used by |
|---|---|---|---|
| `censored-mic-regression` | model fitting | NumPyro interval-censored regression with a horseshoe (`scripts/fit.py`); primers on NumPyro, censored likelihoods, sparse priors | coding agent |
| `bayesian-workflow` | checking and comparing models | workflow, MCMC diagnostics, PSIS-LOO and CV | coding agent, CI assessor |
| `amr-genotype-phenotype` | data interpretation | fluoroquinolone biology, MIC testing and metrics, AMRFinderPlus output, related studies | coding agent, CI assessor |
| `mic-plots` | visualisation | traces, effects, predictive check, calibration (`scripts/plots.py`) | coding agent, CI |
| `mic-eval-harness` | evaluation | how to run `checks/`, decision rule, metrics, PR report template | coding agent, CI assessor |
| `github-workflow` | git and GitHub | start a branch from the champion, publish (push + PR) or discard (`scripts/`) | coding agent |

`CATALOG.md` is the index the agent reads first: which skill to open for which action.

`mic-eval-harness` and `github-workflow` call the read-only scripts in `checks/` and the git helpers in
`github-workflow/scripts/`; the other skills are self-contained.

## Off-the-shelf skills worth adding (pin a commit and read before installing)

| Skill | Repository | Licence | Why |
|---|---|---|---|
| `arviz-diagnostics`, `prior-elicitation` | github.com/pymc-labs/pymc-modeling | MIT | ArviZ 1.x diagnostics, LOO, prior predictive checks |
| `amr-surveillance` | github.com/GPTomics/bioSkills | MIT | running AMRFinderPlus / ResFinder / RGI on new genomes |
| `scientific-visualization` | github.com/K-Dense-AI/scientific-agent-skills | MIT | general figure quality rules |
| `verification-before-completion` | github.com/obra/superpowers | MIT | no success claims without evidence from a command run |
| `skill-creator` | github.com/anthropics/skills | Apache-2.0 | benchmark a skill against a no-skill baseline |

Third-party skills are code: public registries contain vulnerable and malicious skills (Snyk found security
flaws in 37 % of 3,984 public skills; Koi Security found 341 malicious skills on ClawHub, Feb 2026).

## No-leak policy

Skills carry general knowledge (biology, statistics, NumPyro, other papers' methods) and never the results
for the demo dataset: no published accuracy figures, feature rankings or model choices from its paper, and no
findings from our own exploration of it. Those live outside the agent's workspace (`dataset/` and `slides/` at
the repository root), and the PreToolUse hook denies them, together with the locked test split.
