# LXP
LXP (Latent Exchange Protocol) is a machine-native communication layer for AI agents. Instead of passing text, agents exchange structured embeddings, belief states, and uncertainty directly in latent space—enabling higher bandwidth coordination, distributed reasoning, and post-linguistic multi-agent intelligence.

## Local Checks

Run the unit suite from the repo root with:

```bash
venv/bin/python -m pytest -q
```

Generated benchmark, training, sweep, and visualization outputs default to `outputs/`. Runtime smoke reports check for uncaught errors and explicit handoff/cache/decode statuses; phase gates remain for real evaluation runs.

## Latent Blame

`src/utils/latent_blame.py` contains the first validation harness for git-blame-style latent packet attribution. It records latent packets with sender/receiver/turn/tensor metadata, replays a run through caller-provided replay logic, applies ablation/noise/replacement interventions, ranks packets by causal impact, and emits a concise blame report.
