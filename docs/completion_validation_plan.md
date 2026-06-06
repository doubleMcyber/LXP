# LXP Completion Validation Plan

This plan describes the remaining work to move LXP from smoke-proven prototype
to a defensible benchmarked system.

## Current Validated State

Latest committed branch: `mac-mps-training`

Validated local reports:

- Semantic/candidate/token decode smoke:
  `/private/tmp/lxp_fix_semantic_candidate_token_e10/mac_mps_training_report.json`
- Raw free-decode smoke:
  `/private/tmp/lxp_fix_raw_isolated_e30/mac_mps_training_report.json`
- Locked heterogeneous context-vs-latent smoke:
  `/private/tmp/lxp_prod_validation_3_report.json`
- Locked heterogeneous context-vs-latent replay:
  `/private/tmp/lxp_prod_validation_3_replay_report.json`

What is now smoke-proven:

- Actor semantic bridge decode reaches `100%` on the 3-sample train-overfit smoke.
- Latent token decoder reaches `100%` on the semantic/candidate/token smoke.
- Latent candidate fallback/readout reaches `100%` and now has an explicit
  `latent_candidate_fallback_ready` report field.
- Raw actor free decode reaches `100%` when trained in the raw-isolated preset.
- Unit suite passes: `158 passed, 1 skipped`.
- Current unit suite passes after production validation reporting:
  `165 passed, 1 skipped`.
- The benchmark report schema now records receiver-side input-token pressure and
  transfer comparison token savings, so context-vs-latent runs can measure
  compression directly.
- The local `long_context_handoff` dataset provides deterministic frozen sender
  traces for long-horizon handoff validation without depending on an external
  dataset download.
- The production validation runner has MPS and long-context MPS profiles. These
  profiles use `python -B`, `torch_dtype=float32`, and explicit `device_map=mps`
  guards so Apple Silicon runs fail before weight loading if MPS is unavailable.
- The 3-sample locked hetero benchmark and replay both pass semantic,
  transfer-comparison, and heterogeneous-readiness gates with matching
  `sample_content_digest`.

Important interpretation:

- These are controlled smoke gates, not production generalization claims.
- The raw-free-decode objective and the auxiliary token-decoder objective are
  intentionally separated because training them together regressed raw decode.

## Bigger Dataset Validation Ladder

The recommended orchestrated path is now:

```bash
venv/bin/python -B scripts/run_production_validation.py
venv/bin/python -B scripts/run_production_validation.py --execute --profile local --replay
venv/bin/python -B scripts/run_production_validation.py --profile long_context_mps
venv/bin/python -B scripts/run_production_validation.py --execute --profile long_context_mps --replay
venv/bin/python -B scripts/run_production_validation.py --execute --profile gpu --replay
venv/bin/python -B scripts/run_production_validation.py --execute --profile scale --replay
```

The runner orders work as tests, generated sender-trace warm-up, generated
trajectory adapter preparation, locked token-context-vs-latent benchmark, and
optional manifest replay. It writes reports under
`outputs/production_validation` by default.

### 1. Local Sanity Gates

Run before every larger validation:

```bash
venv/bin/python -m pytest -q

venv/bin/python scripts/mac_mps_stage2_smoke.py \
  --execute \
  --allow-cpu-fallback \
  --eval-on-train \
  --epochs 10 \
  --smoke-samples 3 \
  --batch-size 3 \
  --compressed-steps 8 \
  --max-length 64 \
  --output-dir outputs/semantic_candidate_token_smoke

venv/bin/python scripts/mac_mps_stage2_smoke.py \
  --execute \
  --allow-cpu-fallback \
  --eval-on-train \
  --full-decode-eval \
  --epochs 30 \
  --smoke-samples 3 \
  --batch-size 3 \
  --compressed-steps 8 \
  --max-length 64 \
  --output-dir outputs/raw_decode_smoke
```

### 2. Small Real GSM8K Stage-II Gate

Use this to test whether the Stage-II path generalizes beyond smoke rows:

```bash
venv/bin/python run_training.py \
  runtime.device=mps \
  runtime.mps.fallback_to_cpu=true \
  runtime.mps.torch_dtype=float32 \
  device_map=none \
  agent_a_model=Qwen/Qwen3.5-0.8B \
  agent_b_model=Qwen/Qwen3.5-0.8B \
  training.data.mode=real \
  training.data.dataset_name=gsm8k \
  training.data.train_limit=32 \
  training.data.eval_limit=16 \
  training.data.batch_size=2 \
  training.reasoner_max_length=128 \
  training.actor_max_length=128 \
  training.compressed_steps=8 \
  training.num_epochs=10 \
  training.train_reasoner=false \
  training.latent_handoff_adapter.enabled=true \
  training.latent_answer_probe.enabled=true \
  training.lambda_answer_probe=20.0 \
  training.checkpointing.enabled=false \
  reporting.training.history_output=outputs/gsm8k_stage2_32x16_history.csv \
  reporting.training.report_output=outputs/gsm8k_stage2_32x16_report.json
```

Pass condition:

- Actor text baseline is non-degenerate.
- Latent probe/readout improves over initialization.
- Held-out extraction remains `100%`.
- No smoke/report gate fails because of parser or baseline degeneration.

### 3. Locked Multi-Agent Handoff Benchmark

This tests actual multi-agent handoff surfaces, not just Stage-II train-overfit:

```bash
venv/bin/python benchmark_all.py \
  --hetero-smoke \
  --sample-indices 0,1,2,3,4 \
  --limit 5 \
  --methods token_context_handoff,verified_token_context_handoff,sender_answer_text_handoff,generated_context_latent_handoff \
  --generated-trajectory-adapter-input-space raw \
  --enable-sender-revision \
  --generated-trajectory-adapter-no-train-on-missing \
  --report-output outputs/context_vs_latent_5_report.json \
  --samples-output outputs/context_vs_latent_5_samples.csv \
  --summary-output outputs/context_vs_latent_5_summary.csv \
  --write-eval-manifest outputs/context_vs_latent_5_manifest.json
```

Replay the exact same benchmark:

```bash
venv/bin/python benchmark_all.py \
  --eval-manifest outputs/context_vs_latent_5_manifest.json \
  --generated-trajectory-adapter-input-space raw \
  --enable-sender-revision \
  --generated-trajectory-adapter-no-train-on-missing \
  --report-output outputs/context_vs_latent_5_replay_report.json \
  --samples-output outputs/context_vs_latent_5_replay_samples.csv \
  --summary-output outputs/context_vs_latent_5_replay_summary.csv
```

Pass condition:

- `verified_token_context_handoff` and `sender_answer_text_handoff` establish
  the receiver can use compact handoff payloads.
- `generated_context_latent_handoff` should be compared against those controls.
- Latent failures should be categorized by provenance, not just counted as wrong.

### 4. Long-Context MPS Context-Vs-Latent Benchmark

Use this when cloud GPUs are unavailable and local Apple Silicon MPS is
available:

```bash
venv/bin/python -B scripts/run_production_validation.py \
  --execute \
  --profile long_context_mps \
  --replay
```

Pass condition:

- The locked manifest replay matches the original `sample_content_digest`.
- Token-context controls receive the frozen long sender trace.
- Latent methods retain interpretable accuracy while reducing
  `mean_receiver_input_token_count` versus `token_context_handoff`.
- The terminal summary reports latency ratio, receiver-token ratio, and
  receiver-token savings for the best latent method.

Interpretation:

- This is the first local proof target for "latent handoff can be shorter than
  pure token handoff on long horizons."
- A passing three-row profile is still a smoke proof. Increase
  `--eval-limit` and `--train-limit` one axis at a time before making a broad
  performance claim.

### 5. GPU Pilot

When GPU access is available, use the bounded pilot:

```bash
venv/bin/python scripts/do_gpu_pilot.py --execute --eval-limit 20 --train-limit 128
```

Scale only after the 20-row run is interpretable:

- `eval-limit=64`
- `train-limit=256`
- `train-limit=512`

## Model Pair Recommendations

### Best Same-Family Next Pair

Use:

```text
Qwen/Qwen3.5-2B -> Qwen/Qwen3.5-0.8B
```

Repo preflight result:

- `Predicted KV cache compatibility: True`
- matching 24-layer Qwen3.5 hybrid topology
- hidden-size warning remains: `2048 -> 1024`, so latent/input embedding handoff
  still needs alignment

This is the best next same-family thinker/executor pair because it keeps the
actor cheap while giving Agent A a stronger reasoning model.

### Fallback Same-Family Pair

Use:

```text
Qwen/Qwen3-1.7B -> Qwen/Qwen3-0.6B
```

Repo preflight result:

- `Predicted KV cache compatibility: True`
- matching 28-layer Qwen3 full-attention topology
- hidden-size warning remains: `2048 -> 1024`

Use this if Qwen3.5 hybrid cache behavior remains awkward.

### Avoid For Direct KV Transfer To Qwen3.5-0.8B

Avoid:

```text
Qwen/Qwen3.5-4B -> Qwen/Qwen3.5-0.8B
```

Repo preflight result:

- `Predicted KV cache compatibility: False`
- mismatches include 32 vs 24 layers, attention-head mismatch, KV-head mismatch,
  and linear-value-head mismatch

This pair can still be studied with generated trajectory adapters or trained
cross-family-style alignment, but not as a clean direct KV-transfer MVP.

## Does Model Size Matter?

For compatibility, parameter count is not the core rule. Cache topology is the
core rule:

- number of layers
- layer type/order
- attention heads
- KV heads
- head dimension
- hybrid linear-attention metadata
- vocabulary/tokenization when using token-candidate decoders

For memory and runtime, model size matters immediately.

Approximate weight-only memory:

```text
bf16/fp16 memory ~= 2 bytes * parameter_count
```

Practical upper limits:

- If the constraint is one model file under roughly `5 GB` in bf16/fp16, stay at
  or below about `2.5B` parameters.
- If both sender and actor must be loaded locally on MPS, the practical ceiling
  is lower. Treat `2B -> 0.8B` as the upper local pair to try first.
- `4B` bf16/fp16 weights alone are roughly `8 GB`, before the actor, KV cache,
  activations, tokenizer buffers, and PyTorch overhead.
- 4-bit quantization can make larger models appear under `5 GB`, but that does
  not automatically make them compatible or trainable in this repo. Quantized
  inference is a separate engineering path from Stage-II latent training.

## Completion Criteria

The project should be considered benchmark-complete when:

- same-family Stage-II real-mode GSM8K passes a held-out locked manifest,
- multi-agent token handoff controls are measured on the same locked rows,
- generated latent handoff is compared against those controls,
- the report includes token counts, latency, compression ratio, and failure
  provenance,
- at least three seeds are run for the final claim,
- heterogeneous transfer is evaluated only after same-family held-out behavior is
  stable.
