# LXP
LXP (Latent Exchange Protocol) is a machine-native communication layer for AI agents. Instead of passing text, agents exchange structured embeddings, belief states, and uncertainty directly in latent space—enabling higher bandwidth coordination, distributed reasoning, and post-linguistic multi-agent intelligence.

## Local Checks

Run the unit suite from the repo root with:

```bash
venv/bin/python -m pytest -q
```

Generated benchmark, training, sweep, and visualization outputs default to `outputs/`. Runtime smoke reports check for uncaught errors and explicit handoff/cache/decode statuses; phase gates remain for real evaluation runs.

Semantic smoke gates are the fastest end-to-end checks for latent handoff quality:

```bash
venv/bin/python benchmark_all.py --semantic-smoke
venv/bin/python benchmark_all.py --hetero-smoke
```

The default semantic smoke uses `Qwen/Qwen3.5-2B -> Qwen/Qwen3.5-0.8B` with a generated-reasoning latent handoff. The default hetero smoke uses `LGAI-EXAONE/EXAONE-4.0-1.2B -> Qwen/Qwen3.5-0.8B` with a 640-token sender budget, character-aligned generated trajectory adapter, and top-4 receiver embedding-manifold projection. Reports include overall latent accuracy, sender final-answer completion rate, and latent accuracy conditioned on Agent A producing the correct answer.

## Latent Blame

`src/utils/latent_blame.py` contains the first validation harness for git-blame-style latent packet attribution. It records latent packets with sender/receiver/turn/tensor metadata, replays a run through caller-provided replay logic, applies ablation/noise/replacement interventions, ranks packets by causal impact, and emits a concise blame report.

## Cross-Model Handoff

Cross-model cache transfer is only valid for compatible architectures. When sender KV transfer is not compatible, `handoff.receiver_context.mode: "auto"` runs Agent B on the prompt first, then appends the latent handoff step using Agent B's own prompt cache for both scalar and sequence latent prefixes. `handoff.latent_pooling` can be set to `"mean"`/`"prompt_mean"` to compress the full sender prompt into one latent packet instead of using only the last token state. Prompt-only raw latent methods remain diagnostic for semantic-answer tasks; the current MVP path uses generated sender hidden trajectories with the aligned generated-trajectory adapter.
