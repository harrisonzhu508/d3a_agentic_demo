# D3A 2026 agentic autoresearch demo

Live demo for "AI-assisted workflows for statistics and machine learning" (D3A Deep Dive Workshop, 2026): an AI
agent runs an autoresearch loop on a censored Bayesian regression of antibiotic MICs, with skills, hooks, an
evaluation harness, git branches and pull requests, and Weights & Biases tracking.

- `demo/`: the agent's workspace and everything to run it; start with [demo/README.md](demo/README.md).
- `dataset/`: the source data (Liverpool *E. coli*, Gerada et al. 2026, CC BY 4.0) and the notes for humans.
  The agent never reads this folder.
