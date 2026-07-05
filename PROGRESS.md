# LXP — Progress Report

_Last updated: 2026-07-04 (parity gap closed 07-02, dose-response + long-context
re-certification 07-04 — see §3.6–3.9 and §4.1). Branch: `mac-mps-training`._

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

6. **RESOLVED 2026-07-02 — the production benchmark path now reproduces latent
   continuation.** `generated_context_latent_handoff` scores **100% (8/8)**,
   matching the trainer bridge eval's 8/8 on the same sample indices (trainer
   per-row records live in `outputs/latent_bridge_untrained/bridge_report.json`;
   note the 8/8 run itself predates commit `fc473cf` — it ran from the then-dirty
   working tree containing the same fixes). At N=32 (all cached validation rows,
   locked manifest `outputs/parity_fix/locked_continuation_32.json`):
   **latent 65.6% (21/32) > text-hybrid 56.2% (18/32) > receiver-alone 21.9% (7/32)**.
   Latent > alone is significant (McNemar p≈0.0005); latent > text was
   directional at N=32 (p≈0.375) and is **now significant at N=128 — see §3.6b**.
   On the 27 copy-proof rows (answer literal absent from the truncated
   sender text): latent 66.7% vs text 59.3% — no parroting confound. Leak-free
   128-row ridge adapter, truncation 0.5, Qwen3.5-2B→2B. Artifacts:
   `outputs/parity_fix/`.

   **§3.6b — N=128 (2026-07-04): the headline claim is statistically
   significant.** Same protocol, GSM8K validation rows 0–127 (96 freshly
   generated sender traces + 32 cached), locked manifest
   `outputs/parity_fix/locked_continuation_128.json`:

   | method | accuracy |
   |---|---|
   | `generated_context_latent_handoff` | **68.8% (88/128)** |
   | `text_text_hybrid` | 57.0% (73/128) |
   | `pure_text_cot` (receiver alone) | 14.1% (18/128) |

   Paired per-row: **20 latent-only wins vs 5 text-only → McNemar exact
   p = 0.0041**; latent vs alone p ≈ 2×10⁻¹⁸. Copy-proof stratum (answer
   literal absent from the truncated sender text, 106/128 rows): latent 68.9%
   vs text 57.5%. Latent is also faster per handoff than the text hybrid
   (14.4s vs 18.5s). Adapter train split disjoint from eval (train head vs
   validation tail), sender traces cached and marker-free.
   Report: `outputs/parity_fix/continuation128_report.json`.

7. **Truncation dose-response — latent leads at every measured fraction
   (2026-07-04, N=32 per point; curve complete).** Same Qwen3.5-2B→2B protocol
   as §3.6, audited path, per-fraction leak-free adapters:

   | truncation f | latent | text-hybrid | receiver-alone | latent lead | McNemar (latent vs text) |
   |---|---|---|---|---|---|
   | 0.25 | **75.0%** | 62.5% | 21.9% | +12.5 | p≈0.219 |
   | 0.50 | **65.6%** | 56.2% | 21.9% | +9.4 | p≈0.375 |
   | 0.75 | **78.1%** | 56.2% | 21.9% | +21.9 | **p≈0.016** (7 latent-only vs 0 text-only) |

   The latent channel leads text at all three fractions; at f=0.75 the lead is
   significant even at N=32 (latent strictly dominates — zero rows where text
   was right and latent wrong; copy-proof stratum 80.8% vs 57.7% over 26 rows).
   The N=128 run at f=0.5 (§3.6b) anchors the mid-fraction significance.
   Reports: `outputs/parity_fix/continuation32_f25_*`, `continuation32_f75_*`.

8. **Long-context regression check passed after the parity fixes (2026-07-04).**
   Full `long_context_mps` ladder re-run post fused-forward + `logits_to_keep`
   changes: all 4 methods 100%, latent 0.43s vs 299s token-context, 99.78%
   receiver-token savings, leakage ruled out, semantic/comparison/heterogeneous
   gates all pass (digest `54f91d4e…`).

9a. **Adapter density fixes cross-family (2026-07-05, N=32).** Rerunning §3.9's
   exact protocol with a 128-row cross-family adapter (was 32):
   **latent 75.0% (24/32) > text 62.5% (20/32)** — the tie is broken
   (7 latent-only vs 3 text-only discordants, p≈0.34, directional at this N;
   leakage ruled out). Cross-family latent moved from 59.4%→75.0% purely by
   quadrupling adapter training rows, now matching the same-family pattern.
   Report: `outputs/paths/xfam128_report.json`.

9b. **Drafter→finisher works: cheap latents lift a bigger finisher
   (2026-07-05, N=32).** Qwen3.5-**0.8B** drafter (truncated 0.5) →
   Qwen3.5-**2B** finisher, audited path, locked manifest, leak-free:
   **latent 68.8% (22/32) > same-drafter text 53.1% (17/32) > 2B-alone 21.9%**.
   Latent strictly dominates text (5-0 discordants, p=0.0625 — one row short
   of significance at N=32, zero reverse cases; copy-proof stratum 66.7% vs
   51.9%; latent vs alone p=0.0003). The 0.8B's latents bring the 2B to the
   same 68.8% the 2B→2B handoff achieves. Latency for this run includes
   sender cache-miss generation and is not comparable — rerun warm for the
   economics number. Report: `outputs/paths/drafter_finisher32_report.json`.

9. **Cross-family latent continuation carries computation; parity with text,
   not superiority (2026-07-04, N=32; superseded by §3.9a's density result).** EXAONE-4.0-1.2B → Qwen3.5-2B,
   truncation 0.5, audited path, locked manifest
   (`outputs/parity_fix/locked_xfam_32.json`, leakage ruled out):
   **latent 59.4% (19/32) ≈ text 62.5% (20/32)** (McNemar p=1.0; copy-proof
   stratum an exact tie 62.1%/62.1% over 29 rows) — both far above
   **receiver-alone 21.9%** (latent vs alone p=0.0018). Latent is ~40% faster
   than the text handoff (13.1s vs 21.5s/sample) at equal fidelity, with lower
   answer PPL (4.9 vs 7.4). The earlier N=8 run (87.5% vs 62.5%,
   `crossfamily_continuation_report.json`) was anecdote — superseded by this
   N=32 result. So: crossing the family boundary through a 32-row linear ridge
   preserves the computation (refuting the old "incompatibility" reading, which
   was the instruction artifact), but the latent *advantage over text* is so far
   a same-family result. Note the cross-family adapter has only 32 training
   rows vs 128 same-family — density is the obvious untested lever.
   Report: `outputs/parity_fix/xfam32_report.json`.

---

## 4. What is unproven, unfinished, or open

1. ~~**The benchmark-vs-trainer parity gap.**~~ **RESOLVED 2026-07-02** (see
   §3.6). Three stacked causes, none of them the latents or the adapter:
   (a) historical runs used `generated_latent_handoff` (latent-only — the
   receiver never saw the question); the continuation method is
   `generated_context_latent_handoff`; (b) the default post-latent suffix
   ("Repeat the final answer…") primed a guess — it now defaults to free
   generation whenever sender truncation is active; (c) chunked context-prefill
   (context forward, then latents with `past_key_values`) drifts numerically on
   the hybrid linear-attention/SSM cache, enough to deflect 256-token greedy
   decodes — the receiver-context prefix now runs one fused forward over
   `[context, latents]`, matching the trainer mechanics. The earlier
   "128 transcoder rows still 0/8" finding was this layout bug, not a density
   limit.

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

5. ~~**Cross-family transfer is unsettled.**~~ **RESOLVED 2026-07-02** (see
   §3.9): cross-family latent continuation (EXAONE→Qwen3.5-2B) carries the
   computation under the fixed layout — parity with text at N=32 (59.4% vs
   62.5%, tie), both far above receiver-alone, latent ~40% faster. Remaining:
   denser cross-family adapters (32→128 rows), more family pairs, the reverse
   (big-drafter→small-finisher) direction.

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

**The headline claim is proven at significance**: mid-reasoning latent handoff
beats text handoff — latent 68.8% > text 57.0% > alone 14.1% at N=128, McNemar
p = 0.0041, leak-free, locked manifest, through the audited production path
(§3.6b). The channel's cost/compression advantage is re-certified post-fixes
(§3.8), cross-family continuation works at N=8 (§3.9), and the truncation
dose-response is consistent with computation-state transfer (§3.7). Remaining:
the f=0.75 sweep point, cross-family at N≥32, and the never-run design-doc
gates (long-horizon ODE study, MATH Level-5 head-to-head, GPU-scale profiles).
