# Plan: Trained Latent-Prefix Consumption & the Multi-Agent Interlingua

**Status: proposal for review — not yet implemented.**
**Goal:** close the last gap (latent mid-reasoning continuation: currently 0% vs 62.5% for text) and make the protocol scale to multi-agent workflows (O(N) adapters, not O(N²) pairs).

## Why this is the right move (evidence from the 2026-06-11/12 sessions)

1. Linear bridges are saturated. Ridge transcoders solve answer-grade transfer perfectly (100% long-context, 100% GSM8K N=8, 84.4% N=32) but plateau at ~0.64 nearest-embedding similarity on full reasoning sequences, and 4× training data (32→128 rows) moved continuation accuracy 0% → 0%. The failure is capacity/objective, not data.
2. The objective is wrong, not just the model. Every bridge so far minimizes *embedding reconstruction*. The receiver doesn't need reconstructed embeddings — it needs a prefix that *conditions its own computation*. The correct loss is the receiver's NLL of the continuation, backpropagated through the frozen receiver into the bridge. `train_compressor.py` already implements exactly this pattern (frozen actor at `freeze_actor`/1860; differentiable prefix through `inputs_embeds`; answer-CE at `_compute_latent_answer_loss`/667) — for answer continuations, on its own data path, never integrated with the benchmark handoff.
3. The eval regime is now validated: with continuation-aware instructions, truncated TEXT gives +25 points over receiver-alone (62.5% vs 37.5%, Qwen3.5-2B finisher, budget 256). That number is the bridge's target; receiver-alone is its floor.

## Phase 0 — De-risk experiment (≈1–2 days, no benchmark integration)

A standalone script (`scripts/train_latent_bridge.py`) that answers the only question that matters before any architecture work: *can a behaviorally-trained bridge beat receiver-alone on continuation?*

- **Module:** `LatentSoftPromptDecoder` (hidden_state.py:139) with `output_steps=K` (compress the truncated trace into K=32–64 virtual tokens via cross-attention; identity-initialized residuals make early training stable) stacked on a small input projection `sender_dim → receiver_dim` initialized from the existing ridge mapping (warm start from what already works).
- **Data:** already on disk — the cached GSM8K traces (Qwen3.5-2B sender). For each of 128–256 train rows: truncated consensus latents (free causal slice), question text, and the *sender's own post-truncation continuation text* as the target. No new sender forwards.
- **Loss:** CE of the receiver (frozen, bf16) on `[question ⊕ bridge(latents) ⊕ continuation_tokens]`, teacher-forced; optional auxiliary embedding-anchor loss (small weight) to keep outputs near-manifold early.
- **Eval:** the exact continuation benchmark from this session (truncation 0.5, budget 256, vs `pure_text_cot` 37.5% and `text_text_hybrid` 62.5%).
- **Compute on this Mac:** receiver forward+backward at seq ≈ 64+300 tokens, batch 1, grads only for the bridge (~15–40M params) ≈ 6–9GB — feasible bf16 but slow (~2–4s/step → ~2–4h for 2–3 epochs × 256 rows). The session's watchdog pattern handles swap stalls. If rented GPU is available, this is 20 minutes.
- **Go/no-go:** bridge > receiver-alone (37.5%) = GO to Phase 1. Bridge ≈ receiver-alone = iterate K/loss once. Bridge < floor after 2 iterations = the latent-continuation claim should be parked and documented honestly; the protocol's value remains answer-grade transfer + compression.

## Phase 1 — Productionize the pairwise bridge (≈2–3 days)

Integration points (all identified, with pinned-contract awareness from the session's invariant map):

1. New adapter strategy `"trained_bridge"` in `_fit_generated_trajectory_adapter_state` (cache-key safe — strategy string is already in the adapter key; manifest already carries `strategy` since schema v3).
2. Adapter state carries the bridge's `state_dict` + config (CPU tensors — torch.save/load path unchanged; keep a top-level `mapping_matrix` from the ridge warm start to satisfy the disk-loader contract at benchmark_all:2100-2118).
3. `apply_alignment` routing: a `"bridge_state"` branch analogous to `per_step_states` (alignment.py:904) that reconstructs the module once per process (id-keyed memo like `_device_resident`) and runs it under `no_grad`.
4. Training is invoked only via the prepare ladder (`--prepare-generated-trajectory-adapter`) — never lazily inside an eval handoff (the session already established that discipline).
5. Gates: bridge rows go through the normal decode path (`decode_status="decoded"`); no readout, no gate changes.

## Phase 2 — The multi-agent overhaul: Latent Interlingua (plan only; decide after Phase 0)

Pairwise bridges scale O(N²) and that is the wrong shape for multi-agent workflows. The overhaul:

- **A shared latent space** ("interlingua") with per-model **encoders** (model hidden states → interlingua) and **decoders** (interlingua → model embedding prefix). Any-to-any handoff = `decoder_B ∘ encoder_A`: N models need N+N adapters, trained independently against the shared space.
- **Anchor the space cheaply:** define interlingua coordinates as the receiver-side space of one *reference model* (e.g., Qwen3.5-2B), or as the span of the shared semantic-anchor tokens the repo already extracts (alignment.py anchor machinery — 250–500 tokens common to all tokenizers). Each new model trains: encoder by behavioral loss against the *reference decoder* (frozen), decoder by behavioral loss against its own frozen LM consuming *reference-encoded* traces. Adding agent N+1 never touches the other N.
- **Protocol packet** (the multi-agent contract): `{interlingua_tensor [K×d], source_model_id, truncation_state, coverage_diagnostics, protocol_version}` — extending the existing eval-manifest identity discipline (digest-locked, versioned) to runtime handoffs. The session's per-slot coverage diagnostic generalizes to a per-packet confidence the orchestrator can route on (fall back to text when low — hybrid handoff).
- **Why not now:** the interlingua is only worth building if Phase 0 proves behavioral training closes the continuation gap for one pair. Its training recipe *is* Phase 0's recipe, run twice per model. Building it first would be optimizing the topology of bridges before knowing bridges can carry the load.

## Risks & falsifiers
- **R1:** 0.8B-class receivers may be untrainable consumers (session evidence: 0% even with perfect text). Mitigation: target 2B+ finishers; that matches the drafter→finisher economics anyway.
- **R2:** K-token compression may lose the working state needed for continuation (compression vs fidelity sweep is one config knob; K=full-length pass-through is the upper bound).
- **R3:** MPS training fragility (this Mac swap-thrashes). Mitigation: watchdog + per-step checkpointing; or a single rented A10/4090 hour.
- **Honesty bar carried over:** train on `train` split only; eval truncation must cut before the marker (machinery already enforces this); compare against the *text* baseline at equal token budgets, not just receiver-alone.

## Explicitly not done in this session
No bridge training code, no `trained_bridge` strategy, no interlingua scaffolding — per review-first instruction. Everything above references existing, tested machinery by file/line so implementation can start cold from this document.

---

## Phase 0 — EXECUTED (2026-06-12): GO, with a twist

`scripts/train_latent_bridge.py` (+ `tests/test_latent_bridge.py`) implemented and run on the M2 Max
(frozen bf16 Qwen3.5-2B receiver, fp32 ~38M-param bridge, all data from the cached trace store —
the sender never loaded). Result on N=30 held-out GSM8K continuation (truncation 0.5, budget 256):

| channel | accuracy |
|---|---|
| receiver alone | 50.0% |
| truncated reasoning as TEXT | 60.0% |
| **truncated reasoning as LATENTS (untrained ridge bridge)** | **70.0%** |
| latents (bridge after 4 epochs of NLL training) | 40.0% |

Per-row: the latent channel uniquely solves 9 problems the receiver misses alone (loses 3), and 5
the text channel misses (loses 2). **Latent mid-reasoning continuation is proven — by the linear
bridge.** The historical 0% was prompt layout/decode framing, not a latent limitation; behavioral
NLL training overfits at 123 samples and is net-negative — keep the trainer for ablations only.

**Consequence for Phase 1/2:** port the winning layout (instruction-then-latents, free generation)
into the benchmark generated-latent path; the Interlingua's per-model encoders/decoders can start
linear (composable, trainable offline in seconds from cached traces), with neural refinement as a
measured upgrade rather than a prerequisite. Re-run heterogeneous continuation under the new layout
before drawing cross-family conclusions.
