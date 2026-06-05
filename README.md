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
venv/bin/python -B benchmark_all.py --semantic-smoke
venv/bin/python -B benchmark_all.py --hetero-smoke
```

The default semantic smoke uses `Qwen/Qwen3.5-2B -> Qwen/Qwen3.5-0.8B` with a generated-reasoning latent handoff. The default hetero smoke uses `LGAI-EXAONE/EXAONE-4.0-1.2B -> Qwen/Qwen3.5-0.8B` with a 640-token sender budget, character-aligned generated trajectory adapter, and top-4 receiver embedding-manifold projection. Reports include overall latent accuracy, sender final-answer completion rate, and latent accuracy conditioned on Agent A producing the correct answer.

Direct raw-space generated adapters can be tested with:

```bash
venv/bin/python benchmark_all.py --hetero-smoke --generated-trajectory-adapter-input-space raw
```

For raw generated input space, the CLI enables a bounded top-k local residual adapter by default. The adapter fits a global trajectory map, stores a capped residual memory, and applies nearest-neighbor residual correction before receiver embedding-manifold projection. Generated sender traces are cached per prompt in `.cache/generated_trajectory_traces`, then source/target training rows are cached separately in `.cache/generated_trajectory_rows`, so interrupted first fits can resume and residual/adapter hyperparameter sweeps can reuse the expensive sender/receiver trace collection. Benchmark reports expose sender trace hit rate, training row cache hit rate, and adapter cache hit rate. Prompt-only raw latent methods remain diagnostic for semantic-answer tasks; the current production MVP path is generated sender hidden trajectories with either aligned or local-residual raw generated-trajectory adapters.

Adapter caches can be prepared without benchmark decoding:

```bash
venv/bin/python benchmark_all.py --hetero-smoke --prepare-generated-trajectory-adapter --generated-trajectory-adapter-input-space raw --report-output outputs/generated_adapter_prepare_report.json
```

Eval sender traces can also be warmed before a larger semantic gate:

```bash
venv/bin/python benchmark_all.py --hetero-smoke --sample-indices 0,1,2,3,4 --limit 5 --prepare-generated-trajectory-eval-traces --generated-trajectory-adapter-input-space raw --report-output outputs/generated_eval_trace_prepare_report.json
```

Benchmark reports include an `eval_manifest` with resolved dataset, split, sample
indices, method list, model pair, seed, smoke profile, hashed sample
fingerprints, a sample-content digest, and a stable manifest digest. Reports
also include a `transfer_comparison_report` and
`heterogeneous_transfer_report`, so token-context controls and cross-family
adapter/context readiness are tracked in the JSON report instead of inferred
manually from CSV files. Summary rows also include receiver-side input token
pressure (`mean_receiver_input_token_count`, `max_receiver_input_token_count`,
and `total_receiver_input_tokens`), and the transfer comparison report includes
receiver-token ratio and savings fields. This lets long-context runs test
whether latent handoff is actually compressing Agent B's context, not just
whether it decodes an answer. Transfer retention, latency, and hetero-readiness
thresholds can be hardened in `configs/main.yaml` under
`reporting.transfer_comparison` and `reporting.heterogeneous_transfer`. Write
the same lock file separately with
`--write-eval-manifest`, then replay it with `--eval-manifest` before any paid
GPU run:

```bash
venv/bin/python -B benchmark_all.py --hetero-smoke --sample-indices 0,1,2,3,4 --limit 5 --methods token_context_handoff,verified_token_context_handoff,sender_answer_text_handoff,generated_context_latent_handoff --generated-trajectory-adapter-input-space raw --enable-sender-revision --generated-trajectory-adapter-no-train-on-missing --write-eval-manifest outputs/locked_eval_manifest_5.json
venv/bin/python -B benchmark_all.py --eval-manifest outputs/locked_eval_manifest_5.json --generated-trajectory-adapter-input-space raw --enable-sender-revision --generated-trajectory-adapter-no-train-on-missing
```

The staged production validation runner assembles the same workflow in the
recommended order:

```bash
venv/bin/python -B scripts/run_production_validation.py
venv/bin/python -B scripts/run_production_validation.py --execute --profile local --replay
```

Use `--profile gpu` or `--profile scale` only after the local profile passes and
the report's semantic, comparison, and heterogeneous transfer sections are
interpretable.

### Long-Context And MPS Validation

The repo includes a deterministic local `long_context_handoff` dataset for
testing the claim that latent handoff can preserve the answer while avoiding a
large receiver-side text context. Each row contains a short question, the target
answer, and a frozen long sender trace. The benchmark runner feeds that trace
into token-context controls while latent methods use it as Agent A's reasoning
source, making receiver token pressure directly comparable.

Inspect the long-context Apple Silicon profile:

```bash
venv/bin/python -B scripts/run_production_validation.py --profile long_context_mps
```

Run it on a machine where `torch.backends.mps.is_available()` is `True`:

```bash
venv/bin/python -B scripts/run_production_validation.py --profile long_context_mps --execute --replay
```

This profile uses `--dataset long_context_handoff`, `--torch-dtype float32`,
`--device-map mps`, and short decode budgets appropriate for local iteration.
If MPS is unavailable, the benchmark now fails before loading weights instead
of silently falling back or producing misleading timings. The report summary
prints best latent accuracy, latency ratio, receiver-token ratio, and
receiver-token savings percentage.

For a bounded DigitalOcean pilot, inspect the planned commands locally first:

```bash
venv/bin/python -B scripts/do_gpu_pilot.py
```

On the GPU host, run the same pilot with `--execute`.

If cloud GPUs are unavailable, use the local Apple Silicon path in
[`docs/mac_mps_training.md`](docs/mac_mps_training.md). It supports a small MPS
Stage-II smoke first:

```bash
venv/bin/python scripts/mac_mps_stage2_smoke.py
venv/bin/python scripts/mac_mps_stage2_smoke.py --execute
```

Autonomous research loops should use the guardrails in
[`docs/autoresearch_readiness.md`](docs/autoresearch_readiness.md): keep eval
manifests/reporting/token controls locked, restrict the editable module set, and
promote only changes that improve a held-out locked manifest.

For larger generated-latent runs, add `--semantic-min-sender-accuracy 100` when you want the report to distinguish a sender reasoning miss from a latent-transfer miss.

## Latent Blame

`src/utils/latent_blame.py` contains the first validation harness for git-blame-style latent packet attribution. It records latent packets with sender/receiver/turn/tensor metadata, replays a run through caller-provided replay logic, applies ablation/noise/replacement interventions, ranks packets by causal impact, and emits a concise blame report.

## Cross-Model Handoff

Cross-model cache transfer is only valid for compatible architectures. When sender KV transfer is not compatible, `handoff.receiver_context.mode: "auto"` runs Agent B on the prompt first, then appends the latent handoff step using Agent B's own prompt cache for both scalar and sequence latent prefixes. `handoff.latent_pooling` can be set to `"mean"`/`"prompt_mean"` to compress the full sender prompt into one latent packet instead of using only the last token state.
