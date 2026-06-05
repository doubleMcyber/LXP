# LXP Repo Explainer

This document explains how the repo works at its current state. It is written for a reviewer or teammate who wants to understand what runs, what each major file does, what the latest result means, and what remains unfinished.

## 1. Project Goal

LXP stands for Latent Exchange Protocol. The project explores whether language-model agents can communicate through compact continuous latent states instead of passing long text transcripts.

The current working MVP is not arbitrary cross-model latent transfer. The current working MVP is a same-family Stage-II smoke:

- Sender/reasoner: `Qwen/Qwen3.5-0.8B`
- Receiver/actor: `Qwen/Qwen3.5-0.8B`
- Handoff length: `8` compressed latent steps
- Device path: Apple Silicon MPS with CPU fallback enabled
- Latest smoke result: `100%` raw actor free-decode exact match on `3` train-overfit samples

The strongest honest claim is: the repo now has a functioning latent-handoff training and diagnostic system, and the same-family raw actor decode interface works on a small controlled smoke. It does not yet prove broad held-out benchmark generalization or heterogeneous family transfer.

## 2. The Main Data Flow

At a high level, a run does this:

1. Load a prompt and expected answer.
2. Run Agent A on the prompt and collect hidden states.
3. Compress or pool the hidden trajectory into a short latent handoff.
4. Align/adapt that latent into the actor interface.
5. Inject the latent prefix into Agent B.
6. Decode or score the final answer.
7. Write diagnostics that separate baseline failure, latent-content failure, actor decode failure, parser failure, and degeneration.

The important point is that final answer accuracy is not the only metric. The code now records whether:

- The actor baseline can answer from text.
- The latent probe can read the answer from the latent state.
- Semantic readout recovers the answer.
- Raw actor decode emits a parseable answer.
- Raw actor decode is exact-match correct.
- Predictions are non-degenerate across samples.
- The model pair is cache/topology compatible.

## 3. Major Entry Points

### `benchmark_all.py`

This is the benchmark/smoke harness for comparing methods. It supports semantic smoke runs, hetero smoke runs, method filtering, manifests, generated trajectory adapters, and report output.

Useful commands from the README:

```bash
venv/bin/python benchmark_all.py --semantic-smoke
venv/bin/python benchmark_all.py --hetero-smoke
```

For locked comparison work, use an eval manifest:

```bash
venv/bin/python benchmark_all.py --hetero-smoke --sample-indices 0,1,2,3,4 --limit 5 --methods token_context_handoff,verified_token_context_handoff,sender_answer_text_handoff,generated_context_latent_handoff --generated-trajectory-adapter-input-space raw --enable-sender-revision --generated-trajectory-adapter-no-train-on-missing --write-eval-manifest outputs/locked_eval_manifest_5.json
```

Then replay the same manifest:

```bash
venv/bin/python benchmark_all.py --eval-manifest outputs/locked_eval_manifest_5.json --generated-trajectory-adapter-input-space raw --enable-sender-revision --generated-trajectory-adapter-no-train-on-missing
```

### `latent_pipeline.py`

This is the runtime latent handoff pipeline. It handles model loading, prompt formatting, hidden-state collection, latent trajectory construction, semantic alignment, handoff adapters, embedding-manifold projection, receiver context handling, and the hybrid pipeline execution.

Important responsibilities:

- Load Agent A and Agent B.
- Estimate task complexity and build latent trajectories.
- Collect hidden states from selected reasoning layers.
- Compute or load alignment states.
- Apply handoff adapters.
- Project generated latent states back toward the receiver embedding manifold when configured.
- Decide whether receiver prompt context is needed when sender KV transfer is incompatible.
- Run baseline/fallback paths when latent handoff is not viable.

This file is large because it contains the end-to-end inference path and many compatibility/configuration options.

### `run_training.py`

This is the training orchestration entry point. It reads config overrides, loads models and datasets, calls Stage-II training, evaluates, and writes reports.

The latest successful smoke was run through this file indirectly by `scripts/mac_mps_stage2_smoke.py`.

### `train_compressor.py`

This is the Stage-II training core. It contains the training config and the losses/modules used to make a latent handoff useful to the actor.

Important pieces:

- `CompressionTrainConfig`: central training configuration.
- `compress_latent_trajectory`: reduces hidden trajectories to a fixed number of latent steps.
- Answer NLL and first-token losses: encourage the latent-adapted actor to score the correct answer.
- Latent answer probe: checks whether answer information exists in the latent state.
- Latent logit steering: helps raw actor decode emit answer tokens from the latent prefix.
- Latent token decoder path: diagnostic/direct token decoder path, not the current winning path.
- Raw decode readiness gate: early-stops only when raw exact match, extraction, and non-degeneracy all pass.

Current winning path:

- Semantic bridge decoder disabled for the final raw smoke.
- Latent token decoder disabled for the final raw smoke.
- Raw actor decode required.
- Latent semantic readout can be used for decode length control.
- Actor answer tokens are generated through the raw actor path with learned steering.

### `scripts/mac_mps_stage2_smoke.py`

This is the local Apple Silicon runner. It builds a `run_training.py` command with safe defaults for MPS and optionally executes it.

The final successful command was:

```bash
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
  --output-dir /private/tmp/lxp_mps_raw_decode_length_stop_batch3_e30
```

Outputs:

- Training history CSV: `/private/tmp/lxp_mps_raw_decode_length_stop_batch3_e30/mac_mps_training_history.csv`
- Training report JSON: `/private/tmp/lxp_mps_raw_decode_length_stop_batch3_e30/mac_mps_training_report.json`
- Rendered report HTML: `/private/tmp/lxp_mps_raw_decode_length_stop_batch3_e30/mac_mps_training_report.html`

### `scripts/render_stage2_report.py`

This turns a training report JSON plus history CSV into an HTML report. It shows readiness badges, metric cards, bar charts, learning curves, diagnostics, and missing requirements.

### `preflight_model_pair.py` and `src/utils/model_compat.py`

These implement config-only compatibility checks. They inspect architecture metadata before full model weights are loaded.

The check looks at fields such as:

- Layer count
- Layer types
- Hidden size
- Attention heads
- KV heads
- Head dimension
- Vocab size
- Qwen hybrid/linear-attention metadata

Why this matters: some failures are not trainable. For example, if the sender has 24 layers and the receiver expects 30 cache layers, direct KV transfer is structurally invalid.

### `src/utils/benchmarking.py`

This contains report aggregation and smoke gates. It turns raw per-sample results into structured reports.

Important current gates:

- No uncaught errors.
- Required status fields are present.
- Compatibility-aware cache transfer checks.
- Training smoke readiness.
- Raw actor free decoder readiness:
  - raw decode exact match must be `100%`
  - raw answer extraction must be `100%`
  - unique predicted answer count must be greater than `1`

### `src/utils/lm_eval.py`

This handles LM evaluation helpers, including generation from prefix embeddings. A key recent fix added the ability to stop after steered answer tokens, preventing untrained tail generation from corrupting correct raw latent decodes.

### `src/models/handoff_adapter.py`

This contains adapter logic for fitting a mapping from sender latent states to receiver-compatible states. The broader generated-trajectory path can use global alignment plus local residual correction.

### `src/utils/latent_blame.py`

This is an early "git blame for latent packets" harness. It records latent packets, replays a run with interventions, and ranks packets by causal impact. This is a future-facing feature, but it is already represented in the repo.

### `src/data/loader.py`

This loads datasets and splits, including GSM8K-style and MATH-style data paths. It also supports smoke rows used by local tests and tiny training runs.

## 4. Latest Empirical Result

Latest source report:

```text
/private/tmp/lxp_mps_raw_decode_length_stop_batch3_e30/mac_mps_training_report.json
```

Key fields:

- `training_smoke_report.passed`: `true`
- `raw_actor_free_decoder_ready`: `true`
- `final_heldout_raw_decode_exact_match_accuracy`: `100.0`
- `final_heldout_raw_decode_answer_extraction_rate_percentage`: `100.0`
- `final_heldout_raw_decode_unique_predicted_answer_count`: `3.0`
- `final_heldout_latent_semantic_readout_accuracy`: `100.0`
- `final_heldout_latent_probe_accuracy`: `100.0`
- `final_heldout_actor_text_baseline_accuracy`: `100.0`
- `latent_training_ready`: `true`

Example final raw actor outputs:

| Target | Raw latent actor decode | Baseline actor decode |
| --- | --- | --- |
| `4` | `Final answer: 4` | `Final answer: 4` |
| `42` | `Final answer: 42` | `Final answer: 42` |
| `5` | `Final answer: 5` | `Final answer: 5` |

## 5. Why the Phase-II Gate Still Says False

The top-level Phase-II report has `passed: false`, but that does not mean the smoke failed. It means the production gate requirements were not met.

The missing requirements were:

- Training mode is not `real`.
- Observed seed count is `1`, below required `3`.
- Baseline accuracy retention cannot be computed because the production baseline comparison was not available.

So the correct interpretation is:

- Smoke/interface gate: passed.
- Production benchmark gate: not yet run.

## 6. What Changed During Triage

The project moved through several important failures:

1. Initial heterogeneous pair selection was too ambitious for direct KV transfer.
   Qwen3.5-0.8B and EXAONE-4.0-1.2B did not have matching cache/layer topology.

2. The smoke gate was too weak.
   Earlier reports could pass infrastructure checks while semantic accuracy was zero. The gate now checks extraction, exact match, and degeneration.

3. Actor baseline and answer extraction had to be separated.
   If the actor baseline is broken, latent transfer cannot be judged. If extraction is brittle, valid generations can be misreported.

4. Raw actor decode initially collapsed.
   The actor produced repeated or wrong answer tokens from latent prefixes.

5. The final raw path improved after weighted answer losses, first-token objectives, logit steering, length/stop control, and early stopping on a strict raw readiness gate.

## 7. Current Visual/Presentation Files

Use these for the final submission:

- `docs/final_demo.html`: high-impact visual demo page with animation and metrics.
- `docs/lxp_research_paper.html`: paper-style report with figures, charts, results, and limitations.
- `docs/final_video_script.md`: final video narration.
- `docs/repo_explainer.md`: this repo explainer.
- `docs/mac_mps_training.md`: local MPS training guidance.
- `docs/digitalocean_gpu_training.md`: cloud GPU pilot guidance.
- `docs/autoresearch_readiness.md`: guardrails for autonomous research loops.

## 8. How to Run the Repo

### Fast unit tests

```bash
venv/bin/python -m pytest -q
```

Latest known full suite from the final raw-decode work: `155 passed, 1 skipped`.

### Inspect the Mac MPS training command

```bash
venv/bin/python scripts/mac_mps_stage2_smoke.py
```

### Execute the Mac MPS raw decode smoke

```bash
venv/bin/python scripts/mac_mps_stage2_smoke.py --execute --allow-cpu-fallback --eval-on-train --full-decode-eval --epochs 30 --smoke-samples 3 --batch-size 3 --compressed-steps 8 --max-length 64 --output-dir outputs/mac_mps_raw_decode_smoke
```

### Run semantic smoke

```bash
venv/bin/python benchmark_all.py --semantic-smoke
```

### Run hetero smoke

```bash
venv/bin/python benchmark_all.py --hetero-smoke
```

### Preflight a model pair

```bash
venv/bin/python preflight_model_pair.py --agent-a-model Qwen/Qwen3.5-2B --agent-b-model Qwen/Qwen3.5-0.8B
```

## 9. How to Explain the Current MVP

Use this wording:

"The current MVP proves a controlled same-family latent handoff interface. It trains a compressed latent prefix from a Qwen3.5-0.8B sender into a Qwen3.5-0.8B actor, then verifies that the actor can emit exact final-answer text from that latent handoff on a three-sample smoke. The project also includes compatibility checks, diagnostics, and report gates that make failure modes visible."

Avoid this wording:

"The project solves latent communication between arbitrary LLMs."

Avoid this too:

"It beats chain-of-thought or token context transfer."

That benchmark has not been completed yet.

## 10. What Still Needs To Be Done

Highest-value next steps:

1. Run a locked held-out benchmark comparing:
   - pure token-context handoff
   - verified token-context handoff
   - sender-answer text handoff
   - generated-context latent handoff

2. Scale same-family training:
   - `Qwen/Qwen3.5-2B -> Qwen/Qwen3.5-0.8B`
   - multi-seed
   - held-out eval
   - baseline retention

3. Re-open heterogeneous transfer:
   - avoid direct KV transfer when topology is incompatible
   - use generated trajectory adapters, embedding-manifold projection, residual correction, or trained cross-family alignment
   - compare EXAONE/Qwen only after sender trace quality and same-family transfer are stable

4. Improve the latent blame tool:
   - use it to identify which latent packets drive answer correctness
   - add visual per-packet intervention charts
   - compare token context attribution against latent packet attribution

5. Turn smoke reports into a standard benchmark report:
   - locked manifest digest
   - fixed sample IDs
   - repeated seeds
   - timing/cost metrics
   - token count and compression ratio

## 11. Reviewer Summary

The repo is best understood as a research workbench for latent model-to-model communication. The final deliverable is not just the 100% smoke number; it is the system that made that number meaningful:

- compatibility preflight
- latent extraction and adaptation
- local Stage-II training
- answer and first-token objectives
- logit steering
- raw actor decode
- semantic probes
- extraction and degeneration checks
- HTML reports and visual presentation artifacts

That combination shows substantial implementation, iteration, evaluation, and honest limitation tracking.
