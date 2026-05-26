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

Direct raw-space generated adapters can be tested with:

```bash
venv/bin/python benchmark_all.py --hetero-smoke --generated-trajectory-adapter-input-space raw
```

For raw generated input space, the CLI enables a bounded top-k local residual adapter by default. The adapter fits a global trajectory map, stores a capped residual memory, and applies nearest-neighbor residual correction before receiver embedding-manifold projection. Generated trajectory source/target training rows are cached separately in `.cache/generated_trajectory_rows`, so residual and adapter hyperparameter sweeps can reuse the expensive sender/receiver trace collection. Prompt-only raw latent methods remain diagnostic for semantic-answer tasks; the current production MVP path is generated sender hidden trajectories with either aligned or local-residual raw generated-trajectory adapters.

## Latent Blame

`src/utils/latent_blame.py` contains the first validation harness for git-blame-style latent packet attribution. It records latent packets with sender/receiver/turn/tensor metadata, replays a run through caller-provided replay logic, applies ablation/noise/replacement interventions, ranks packets by causal impact, and emits a concise blame report.

## Cross-Model Handoff

Cross-model cache transfer is only valid for compatible architectures. When sender KV transfer is not compatible, `handoff.receiver_context.mode: "auto"` runs Agent B on the prompt first, then appends the latent handoff step using Agent B's own prompt cache for both scalar and sequence latent prefixes. `handoff.latent_pooling` can be set to `"mean"`/`"prompt_mean"` to compress the full sender prompt into one latent packet instead of using only the last token state.
