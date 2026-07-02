# LXP — Progress Report

_Last updated: 2026-07-02. Branch: `mac-mps-training`._

LXP (Latent Exchange Protocol) is a machine-native communication layer for AI
agents: instead of passing text between models, a **sender** model's continuous
reasoning latents are aligned into a **receiver** model's embedding space and
consumed directly as a short latent prefix. The bet is that this raises
inter-agent bandwidth and cuts latency versus re-serializing reasoning as text.

This document is a factual snapshot of **what is actually built and verified**
versus **what is still unproven, unfinished, or aspirational**. It is written to
be honest about the gap between the design docs (`MD Files/`) and the code.

> Scope note: numbers below come from committed reports (`outputs/`) and the
> validation history. Where a result carries an important caveat (leakage,
> channel-fidelity-vs-semantics, sample size), the caveat is stated inline —
> do not quote a headline number without it.

---

## 1. Architecture at a glance

```
Sender (Agent A)  ──►  latent trajectory  ──►  alignment / adapter  ──►  Receiver (Agent B)
  reasoning traces      (hidden states,          (ridge / Procrustes /       consumes latent
  or ODE-integrated      final-answer tail,        per-step ridge /            prefix, then
  continuous thoughts)   consensus latents)        soft-prompt decoder)        decodes answer
```

Two largely independent tracks share this spine:

- **Latent handoff / benchmarking track** (`benchmark_all.py`, `latent_pipeline.py`,
  `src/utils/alignment.py`) — the mature, most-exercised path. Fits an adapter
  from sender latents to receiver embeddings and measures answer fidelity, cost,
  and compression against text/token-context baselines.
- **Compression-training track** (`train_compressor.py`, `run_training.py`,
  `src/models/losses.py`, `hidden_state.py`) — trains reasoning/decoder heads
  against a frozen actor. Fully implemented, less conclusively validated.

---

## 2. What is implemented

### Core infrastructure (Design Phase 1 — done)
- `latent_pipeline.py` (2.7k LOC): the real engine. Loads sender/receiver,
  intercepts hidden states (bypassing `lm_head`), extracts and normalizes the
  KV cache, applies alignment, injects the aligned prefix into the receiver, and
  decodes. `initialize_hybrid_pipeline` / `run_hybrid_pipeline` are the entry
  points reused across scripts.
- `src/models/dynamics.py`: KV-cache extraction/normalization/compatibility
  checking, including modern cache objects (`DynamicCache`, hybrid/SSM caches),
  not just legacy tuples.
- `src/utils/model_compat.py` + `preflight_model_pair.py`: model-pair KV-cache
  compatibility preflight (architecture summary, SSM/linear-attention awareness).
- Cross-model handoff for incompatible caches: when KV transfer is invalid,
  `handoff.receiver_context.mode: auto` runs Agent B on the prompt first, then
  appends the latent step using B's own prompt cache.

### Geometric alignment (Design Phase 3 — done)
- `src/utils/alignment.py` (936 LOC) implements **Orthogonal Procrustes** via SVD
  (`_safe_svd`, `compute_cross_covariance`, `compute_orthogonal_mapping` = U·Vᵀ),
  **ridge** regression, and a **hybrid_affine** strategy, plus semantic-anchor
  selection, centering/bias, adaptive projection, distortion metrics, and
  bootstrap anchor-stability scoring.
- `per_step_ridge` (one ridge map per tail slot) and global `ridge` are both
  wired through `apply_alignment` and selected by task type (see §4).
- `src/models/handoff_adapter.py`: fits a sequence-level adapter from A's
  reasoning latents to B's input-embedding space, with top-k snapping onto the
  receiver embedding manifold.

### Continuous dynamics (Design Phase 4 — implemented, lightly exercised)
- `torchdiffeq` Neural ODE is real and wired: `dh/dt = f(h) − h`
  (`dynamics.py::TransformerBlockDynamics`) integrated with `odeint(method="rk4")`
  in `latent_pipeline.py`. Note the "500+ step" horizon claim from the design doc
  is **not** validated here — the ODE runs, but the long-horizon degradation
  study (Validation Gate 4) has not been executed.

### Compression training (Design Phase 2 — implemented)
- `src/models/losses.py`: `LatentCompressorLoss` = task cross-entropy +
  uncertainty-weighted preference KL + geometric cosine loss, plus an EMA-based
  `AdaptiveLossBalancer` — matches the L_task / L_pref / L_geom design.
- `src/models/hidden_state.py` (531 LOC): the trainable heads — `LatentHandoffAdapter`
  (identity-init residual), `LatentSoftPromptDecoder` (latents → K virtual tokens),
  `LatentSequenceDecoderHead`, `LatentTokenDecoderHead`, `LatentLogitSteeringHead`,
  `LatentAnswerProbe`.
- `train_compressor.py` (3k LOC) + `run_training.py` (1.7k LOC): Stage-II training
  with a frozen actor, curriculum over `identity`/`orthogonal`/`hybrid_affine`,
  Hydra config, W&B, MPS-aware. `sweep_compression.py` sweeps latent-step budgets.

### Mid-reasoning continuation / "latent bridge" (Phase 0 de-risk — implemented + certified)
- `scripts/train_latent_bridge.py`: trains a small bridge (input projection +
  soft-prompt decoder) that turns a **truncated** sender reasoning trace into
  virtual receiver tokens; loss is the frozen receiver's NLL of the continuation.
- `scripts/certify_latent_bridge.py`: guards the result against artifacts
  (split disjointness, truncation-marker leakage, answer-literal stratification).
- `--sender-reasoning-truncation-fraction` truncates sender reasoning strictly
  before the final-answer marker for both latent and text baselines.

### Datasets
- `src/data/loader.py`: **GSM8K**, **MATH Level-5** (`hendrycks/competition_math`),
  and a synthetic deterministic **`long_context_handoff`** benchmark (frozen long
  sender traces for testing receiver-token compression without an external download).

### Reporting, validation harness, diagnostics
- `src/utils/benchmarking.py` (2.4k LOC, `REPORT_SCHEMA_VERSION = 23`): structured
  JSON/CSV reports with eval manifests (hashed sample fingerprints + stable
  digest for locked replay), transfer-comparison and heterogeneous-readiness
  sections, and receiver-side token-pressure metrics.
- `scripts/run_production_validation.py`: staged driver with `local` / `mps` /
  `long_context_mps` / `gpu` / `scale` profiles; MPS profiles fail before weight
  load if MPS is unavailable (no silent CPU fallback).
- `scripts/do_gpu_pilot.py` (dry-run by default), `scripts/mac_mps_stage2_smoke.py`,
  `scripts/render_stage2_report.py` (HTML), `analyze_distance_accuracy.py`
  (distance↔accuracy calibration), `visualize_thoughts.py` (latent trajectory PNG).
- `src/utils/latent_blame.py`: git-blame-style causal attribution for latent
  packets — records packets, replays, applies ablate/noise/replace interventions,
  ranks by causal impact. Self-contained validation harness.

### Tests
- 201 test functions across 21 files; the suite is **model-free** (mock model
  names, no real weights) — it asserts report/dict structure and unit behavior,
  **not** end-to-end accuracy. Run: `venv/bin/python -m pytest -q`. Lint: `ruff check`.

---

## 3. What is proven (and the caveats that matter)

1. **Leak-free long-context channel fidelity: 100% (32/32).** Shipped config:
   `source_mode=final_answer_tail` (pure consensus latents) + `tail_tokens`
   targets + `per_step_ridge` + `train_limit=128`, at ~0.099s/handoff vs 40.9s
   token-context (413× latency win, 99.78% receiver-token savings).
   **Caveat:** this certifies *channel fidelity + cost* (12 latent slots vs a
   ~3500-token prefill), **not** independent semantic transfer — the readout-row
   alignment distances are coupled to the construction.

2. **Mid-reasoning latent continuation genuinely works (N=30, certified).**
   GSM8K, Qwen3.5-2B→2B, truncation 0.5: **latent bridge 70% > truncated-text
   60% > receiver-alone 50%.** Certification (`outputs/latent_bridge_untrained/
   certification.json`): split-disjoint, truncation-safe, and on the 25/30
   copy-proof rows (answer literal absent from the prefix) the bridge still leads
   (68% vs 60% text vs 48% alone). **The decisive twist:** the *untrained*
   ridge-warm-started bridge won; four epochs of behavioral NLL training
   *degraded* it to 40% (overfit to sender token style). **Caveat:** N=30,
   same-family pair, single truncation fraction.

3. **GSM8K generated-reasoning handoff: 100% (8/8)**, cross-model
   Qwen3.5-2B→0.8B bf16, via global `ridge` + `final_answer_tail_anchored`.
   Receiver-forward decode (readout disabled) also hits 100% at answer PPL ~1.65.

4. **The design law is empirically grounded:** fixed-template tails → `per_step_ridge`
   on consensus-only latents; diverse text → global `ridge` + token-anchored
   features (per-slot maps are underdetermined on diverse text and confidently
   emit training-set digits). Scaling over `train_limit` is a **coverage phase
   transition**, not a power law (0% until the payload alphabet is covered, then
   100%) — and readout similarity stays ≥0.967 even at 0% accuracy, so the
   similarity gate cannot catch out-of-coverage errors; the per-slot
   unique-target-count diagnostic can.

5. **MPS OOM root causes fixed** (retained device tensors, an unused sender KV
   cache, per-handoff fp32 embedding re-materialization). The previously-impossible
   64-row adapter prep now completes (~37.5 min, peak RSS 8.5 GB) and multi-sample
   sequential evals run flat.

---

## 4. What is unproven, unfinished, or open

1. **The benchmark-vs-trainer parity gap (the headline open problem).** The
   production benchmark latent path scores **0%** (answer PPL ~18) on the *same*
   truncated latents that the Phase-0 trainer continues from at 70%. The cause is
   layout/framing/precision, not the latents. `scripts/parity_harness.py` exists
   specifically to diff the two paths' first-token logits and localize the
   divergence — **this is not yet resolved.** The Phase-0 layout fixes landed in
   the production path, but full parity has not been demonstrated.

2. **Full-sequence latent transcoding of arbitrary reasoning does not work.**
   Transcoded full-trace latents score 0% (receiver emits degenerate loops);
   the transcoder reaches only ~0.64 nearest-embedding similarity on full traces
   (vs 0.99 on answer tails). Increasing density 4× (128 rows) did **not** close
   it — unlike every fixed-tail task. Likely real fix: receiver fine-tuning on
   latent-prefix consumption (the heads in `hidden_state.py` exist for this).

3. **Behavioral NLL bridge training currently hurts.** Kept for ablation only;
   default artifact path is literally `outputs/latent_bridge_untrained/`.
   Revisit with more data / regularization / LoRA-on-receiver.

4. **The "100% accuracy" in old `production_context_vs_latent_3*` reports was
   train-on-test leakage** (`train_split: "test"`). The honest leak-free config
   scored 0% on the pre-fix 64-row adapter (latents carried the digits but
   readout sat at 0.6–0.72, below the 0.80 gate). This is now understood and
   documented; treat any historical 100% not listed in §3 with suspicion.

5. **Cross-family transfer is unsettled.** Cross-family text continuation
   (EXAONE→Qwen) underperformed receiver-alone under the *old* instruction; it
   has not been retested under the validated continuation instruction. No
   cross-family *latent* continuation result exists.

6. **Design success criteria not yet met.** From `MD Files/`:
   - Phase-4 Gate ("500+ step trajectory without catastrophic drift"): ODE runs,
     but the long-horizon degradation study has not been executed.
   - Phase-3 Gate (Qwen→**LLaMA** heterogeneous transfer): not run; the tested
     hetero pair is EXAONE→Qwen, and it has not passed a continuation gate.
   - Phase-2 Gate ("compress 128 steps→16, retain ≥90–95% accuracy on MATH"):
     the loss and training loop exist, but this specific compression-retention
     result on MATH has not been produced.
   - Overall "outperform text MAS on MATH Level-5" and "≥4× latency reduction":
     latency wins are shown on GSM8K/long-context channel-fidelity tasks; the
     head-to-head accuracy win on MATH Level-5 is not demonstrated.

7. **GPU / scale profiles are dry-run only.** `do_gpu_pilot.py` and the
   `gpu`/`scale` validation profiles have been assembled and inspected but not
   executed on real cloud hardware.

---

## 5. Named in the design docs but absent from the code

The `MD Files/` design specs name a stack that is only partially realized:

| Component (design intent)            | Status in code |
|--------------------------------------|----------------|
| Orthogonal Procrustes / SVD alignment | **Implemented** (`src/utils/alignment.py`) |
| torchdiffeq Neural ODE dynamics       | **Implemented** (`dynamics.py`, `latent_pipeline.py`) |
| MATH Level-5 dataset                  | **Implemented** (`loader.py`) |
| DeepSpeed ZeRO-2 training backend     | **Absent** — prose only, not imported, not in `requirements.txt` |
| vLLM inference engine                 | **Absent in code** — listed in `requirements.txt` but never imported (HF generation is used) |
| ALFWorld embodied benchmark           | **Absent** — prose only; loaders cover math/gsm8k/long_context only |
| LLaMA-3.1-8B heterogeneous receiver   | **Absent** — tested pairs are Qwen↔Qwen and EXAONE→Qwen |
| Qwen2.5-**7B** base models            | Config targets Qwen3.5-2B/0.8B (Mac-MPS-sized), not the 7B/8B pair in the docs |

There are **no `TODO`/`FIXME`/`NotImplementedError`/stub markers** in the source;
incompleteness is signaled instead by `*_untrained` artifact paths and by the
dedicated `parity_harness.py` / `certify_latent_bridge.py` diagnostic scripts.

---

## 6. One-line status

The latent-handoff **channel** and its cost/compression advantage are built and
certified; **mid-reasoning latent continuation is proven at small N** with an
untrained linear bridge. The two biggest open items are (a) closing the
**benchmark-vs-trainer parity gap** so the production path reproduces the
trainer's 70%, and (b) making **arbitrary full-sequence reasoning** transfer —
likely via receiver-side fine-tuning rather than a wider linear transcoder.
