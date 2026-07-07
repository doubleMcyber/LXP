# Receiver LoRA Training Spec: consuming latent prefixes better than the untrained bridge

Status: implementation spec, 2026-07-04. Target branch: `mac-mps-training`.
Author: research design pass over PROGRESS.md §3.6b/§4.2/§4.3, `scripts/train_latent_bridge.py`,
`benchmark_all.py::_run_generated_latent_variant`, `src/utils/lm_eval.py::prepare_receiver_context_latent_prefix_state`,
and the Phase-0 failure record (`outputs/latent_bridge_untrained/`).

## 0. Goal and success criteria

Train LoRA adapters inside the RECEIVER (Qwen/Qwen3.5-2B) so it consumes ridge-projected
sender latent prefixes better than the frozen receiver does today, without touching the
latent channel itself and without changing text/alone behavior.

Bars, all through the audited production path (`benchmark_all.py`, truncation 0.5, GSM8K):

- Primary: on the locked 128-row manifest `outputs/parity_fix/locked_continuation_128.json`,
  `generated_context_latent_handoff` with the LoRA'd receiver beats the certified baseline
  88/128 (68.8%). Success = paired exact McNemar (LoRA vs baseline latent, per-row from the
  two samples CSVs) p < 0.05 with a positive net win count. Directional improvement without
  significance is reported as such, not claimed as a win.
- Non-degradation (hard gate): `pure_text_cot` and `text_text_hybrid` rows in the
  certification run are bit-identical to `outputs/parity_fix/continuation128_samples.csv`
  (guaranteed by construction when LoRA scope is `latent_only`; verified anyway).
- Stretch: replay `outputs/parity_fix/locked_xfam_32.json` (EXAONE-4.0-1.2B -> Qwen3.5-2B)
  with the same-family-trained LoRA to measure whether receiver-side consumption skill
  transfers across sender families (baseline: latent 19/32).

The latent channel (raw truncated consensus latents -> cached 128-row ridge adapter
`5c041b5521a4663a1a469410c7114df8753762df45f7b6fad66bb9ca2f03c02b`, file
`.cache/generated_trajectory_adapter/generated_trajectory_adapter_8959603b...b1070b207.pt`)
is FROZEN throughout. Only receiver-internal LoRA weights train.

## 1. Why Phase 0 failed, and why this design does not repeat it

Phase 0 (2026-06-12) trained the BRIDGE with NLL of the sender's teacher-forced
continuation through a frozen receiver. Result: 70% -> 40% after 4 epochs, with eval
accuracy diverging from train loss from epoch 0. Mechanism: the only trainable parameters
sat between the latents and the receiver, so the gradient reshaped the latent
representation itself toward predicting sender token style, corrupting the very signal
that made the untrained ridge projection work. The training target (sender surface form)
was also off-policy for the receiver's own decoding distribution.

This design differs on all three axes:

1. The latent representation cannot be corrupted. The ridge adapter and the latents are
   frozen inputs; gradients flow only into receiver-internal low-rank deltas that are
   exactly zero at init (LoRA up-projection zero-initialized). Step 0 is bit-identical to
   the certified 68.8% baseline, and the step-0 gate asserts this.
2. The teacher signal is on-policy and answer-anchored, not sender style. The primary
   objective is NLL of the receiver's OWN successful continuations (rejection-sampled
   rollouts that end in the verified correct answer, generated through the exact
   production decode path). Minimizing it cannot drag the receiver toward an alien token
   distribution; it can only re-weight probability toward continuations of latent prefixes
   that the receiver itself already produces and that end correctly.
3. The Phase-0 divergence signature is instrumented as a kill switch, and the
   instrumentation is validated first by deliberately re-running the known-bad objective
   (sender-NLL canary, capped at 1 epoch) and confirming the gates fire.

## 2. Objectives

Notation per training row i: `c_i` = chat-templated context token ids (question +
continuation instruction), `Z_i` = projected latent prefix (frozen), `y_i` = target
continuation token ids, length `T_i` (<= 256). Input embeddings
`E_i = [Emb(c_i); Z_i; Emb(y_i)]`, one fused forward, loss only on `y_i` positions:

```
L_i = (1 / sum_t w_t) * sum_{t=1..T_i} w_t * CE(logits at position P+t-1, y_{i,t})
```

where `P = len(c_i) + len(Z_i)` (so the logit predicting `y_{i,1}` comes from the last
latent position) and the forward uses `logits_to_keep = T_i + 1` (the continuation is the
sequence tail, so this bounds the fp32 logit tensor to ~257 x 248320 instead of full-seq).

Three objectives, run in this order:

- Objective C, sender-NLL canary (instrumentation validation, expected FAIL):
  `y_i` = sender's post-truncation continuation tokens (`continuation_ids` exactly as
  `scripts/train_latent_bridge.py::load_bridge_samples` builds them), `w_t = 1`.
  This is Phase 0's loss with LoRA instead of the bridge. Run capped at 1 epoch on the
  full pool. Purpose: confirm the dev gates detect degradation (Phase 0 degraded within
  1 epoch). If the gates do NOT fire and dev accuracy holds, that is itself a finding
  (receiver-side capacity fixes the style-overfit) and objective C's checkpoint enters
  the same selection process as A/B.
- Objective A, self-consistent continuation NLL (primary):
  `y_i` = the receiver's own verified rollout (section 4.3), `w_t = 1`.
- Objective B, answer-weighted variant (run only if A passes non-degradation but misses
  the improvement bar): same `y_i` as A, but `w_t = 4.0` for every token from the first
  token of the final `Final answer:` line onward (locate by decoding `y_i` and mapping the
  last `Final answer` match back to token offsets via incremental decode), `w_t = 1`
  elsewhere. This sharpens the answer-anchoring without changing the on-policy property.

No other objectives. No KL terms, no auxiliary heads.

## 3. LoRA architecture, placement, precision, memory

### 3.1 Hand-rolled, not peft

Decision: hand-rolled `LoRALinear` in a new file `src/models/receiver_lora.py`. Rationale
(one line each): peft is not in `requirements.txt` and the repo style is self-contained
trainable modules (`hidden_state.py` precedent); we need per-method enable/disable at
inference and bf16-base/fp32-delta mixed precision on MPS, which is ~100 lines by hand and
an untested fight with peft's dtype handling for the hybrid `qwen3_5` architecture.

### 3.2 Module contract (`src/models/receiver_lora.py`)

```python
class LoRALinear(nn.Module):
    # wraps an existing nn.Linear (kept frozen, bf16); adds fp32 A/B
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float): ...
    # forward: y = base(x) + enabled * (alpha/rank) * (drop(x.float()) @ A.T @ B.T).to(x.dtype)
    # A: [rank, in_features] init N(0, 0.02); B: [out_features, rank] init zeros
    # self.lora_enabled: bool, default True

RECEIVER_LORA_TARGET_SUFFIXES = (
    ".linear_attn.in_proj_qkv", ".linear_attn.out_proj",
    ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj",
)

def apply_receiver_lora(model, *, rank=16, alpha=32.0, dropout=0.05) -> dict[str, LoRALinear]
    # walk model.named_modules(); wrap every nn.Linear whose qualified name ends with a
    # target suffix AND contains ".layers." AND does not contain "visual"
    # (setattr on the parent). Returns {qualified_name: wrapper}. Raise if count != 60.

def receiver_lora_state_dict(model) -> dict[str, torch.Tensor]      # A/B tensors only
def load_receiver_lora(model, path, *, strict=True) -> dict          # apply_receiver_lora if absent, then load; returns metadata
def set_receiver_lora_enabled(model, enabled: bool) -> int           # returns modules toggled; raise if 0
@contextmanager
def receiver_lora_scope(model, enabled: bool): ...                   # toggle + guaranteed restore
def receiver_lora_file_sha256(path) -> str
```

Expected wrap count for Qwen3.5-2B (24 text layers: 18 linear_attention at all indices
except 3,7,11,15,19,23; 6 full_attention at those indices): 18*2 + 6*4 = 60 modules.
MLP (`gate_proj/up_proj/down_proj`), embeddings, `lm_head`, norms, and the GatedDeltaNet
`in_proj_z/in_proj_b/in_proj_a` gates stay frozen. Deviation note: token-mixer-only
targeting (no MLP) is a deliberate capacity cap because the training pool is only
~160 samples; the consumption problem lives in the token mixers that read the latent
positions.

The saved artifact (`torch.save`) is a dict:
`{"format": "receiver_lora_v1", "rank", "alpha", "dropout", "target_suffixes",
"base_model": "Qwen/Qwen3.5-2B", "state": {name: tensor fp32}, "train_args": {...},
"adapter_cache_key_digest": "<the frozen ridge digest>", "objective": "A|B|C",
"dev_gate_accuracy": float}`.

### 3.3 Hyperparameters (fixed defaults, no menu)

| item | value |
|---|---|
| rank / alpha / dropout | 16 / 32.0 / 0.05 |
| optimizer | AdamW, lr 1e-4, betas (0.9, 0.999), weight_decay 0.0 |
| schedule | cosine to 1e-5, linear warmup 6 optimizer steps |
| grad accumulation | 8 samples (batch size 1 per forward, MPS) |
| epochs | 3 max, early stop per section 5.2 |
| grad clip | 1.0 (LoRA params only) |
| seed | 7 (torch + random; rollout sampling seeded per row: seed*100003 + sample_index) |
| precision | model bf16 on MPS, LoRA A/B fp32, loss/logits fp32 via `logits_to_keep` |

Parameter count at rank 16 (hidden 2048; in_proj_qkv out 6144; q_proj out 4096; k/v out
512; o/out_proj 2048):
18*(2048+6144)*16 + 18*(2048+2048)*16 + 6*((2048+4096)+(2048+512)*2+(2048+2048))*16
= 2.36M + 1.18M + 1.47M = 5.0M fp32 params.

Memory arithmetic (32GB M2 Max): weights ~4.3GB bf16 (2.1B incl. tied embedding and the
unused vision tower); LoRA params+grads+Adam moments = 20+20+40 = 80MB; retained
activations for backward at seq ~800 across 24 layers ~1.0-1.5GB; logits capped by
`logits_to_keep=257` at 257*248320*4B = 255MB fp32; rollout/eval phases run under
`torch.no_grad` with KV cache. Peak estimate 6-8GB, comfortably inside the machine's
proven 8.5GB-RSS envelope for heavier phases. `torch.mps.empty_cache()` after each
checkpoint write, matching the bridge trainer.

## 4. Data pipeline

### 4.1 Sources and split policy (leakage rules)

- Training rows: GSM8K TRAIN split rows 0..223 (via
  `get_dataloader("gsm8k", limit=224, split="train", validation_size=256)`, same loader
  and `validation_size` as everything else).
- Dev gate rows: TRAIN split rows 224..255 (32 rows), never trained on, used for
  early-stop/model selection. The validation split is not touched before final
  certification (rows 0..127 of validation are the locked eval; keeping model selection
  on train-split rows keeps validation pristine).
- Final eval: the locked manifest only.

Trace inventory today: 264 Qwen/Qwen3.5-2B bf16 traces in
`.cache/generated_trajectory_traces` (train rows 0..127, validation rows 0..127, plus
extras). Required new traces: train rows 128..255 = 128 traces at ~40s/row ~= 85 min.
Warm command (flags copied from the certified run's `run_metadata.argv` so the trace
cache key matches; run detached per the ops note):

```bash
venv/bin/python benchmark_all.py --hetero-smoke \
  --agent-a-model Qwen/Qwen3.5-2B --agent-b-model Qwen/Qwen3.5-2B \
  --dataset gsm8k --split train \
  --sample-indices $(python3 -c "print(','.join(str(i) for i in range(128,256)))") \
  --limit 128 --enable-sender-revision --reasoner-max-new-tokens 640 \
  --torch-dtype bfloat16 --device-map mps \
  --prepare-generated-trajectory-eval-traces \
  --report-output outputs/receiver_lora/train_trace_warm_report.json
```

### 4.2 Sample construction (must byte-match the production fused forward)

Reuse `scripts/train_latent_bridge.py::load_bridge_samples` unchanged (it already yields,
per cached trace: `question`, `answer` via `_target_answer`, raw truncated `latents`
(fp32, `consensus[:, :cut, :]` with `cut` from `_truncate_reasoning_token_ids` at fraction
0.5), sender `continuation_ids`, `truncated_text`, `sample_index`).

Per sample, the trainer then builds the production prefix:

1. Context ids: `chat_prefix_ids(tokenizer, question, CONTINUATION_INSTRUCTION)` from
   `train_latent_bridge.py`. This is textually identical to
   `latent_pipeline._format_receiver_context_prompt` under active truncation followed by
   `tokenizer(context_text, add_special_tokens=False)`; a pinned unit test asserts the
   two token-id sequences are equal for a real GSM8K question (section 8, chunk 2).
2. Latent prefix: load the frozen ridge adapter payload once
   (`torch.load(".cache/generated_trajectory_adapter/generated_trajectory_adapter_8959603bbf6825e4dc4df3f373ca600ccd7e61425e0aa9c66d7f0adb1070b207.pt")`,
   assert `adapter_cache_key_digest == "5c041b5521a4663a1a469410c7114df8753762df45f7b6fad66bb9ca2f03c02b"`),
   then `Z_i = apply_alignment(latents.unsqueeze(0), adapter_payload).to(torch.bfloat16)`
   using `src/utils/alignment.apply_alignment` (the payload's pre/post adaptive-projection
   states ride along exactly as in `_run_generated_latent_variant` with
   `input_space="raw"`). No generic handoff adapter, no embedding manifold, no suffix
   text, matching the certified run's flags.
3. Training forward: `inputs_embeds = cat([emb(c_i).bf16, Z_i, emb(y_i).bf16], dim=1)`,
   `attention_mask` all ones, `use_cache=False`, `logits_to_keep=T_i+1`, manual CE as in
   section 2 (do not pass `labels=`; full-vocab full-seq logits would cost ~800*248320*4B).

Rows whose trace has `generated_reasoning_status != "complete"` are skipped (inherited
from `load_bridge_samples`); expect ~5-10% attrition.

### 4.3 Rollout generation (objectives A/B)

For each training row, produce a verified receiver continuation through the production
decode mechanics:

1. Build `prefix_state = prepare_receiver_context_latent_prefix_state(model=receiver,
   tokenizer=tok, context_text=<templated context string>, handoff_step=Z_i,
   kv_cache=None, suffix_text="", latent_position="after_context")` (the fused forward,
   `src/utils/lm_eval.py`).
2. Greedy decode: `greedy_decode_from_prefix(..., max_new_tokens=256, stop_regex=None)`.
   Score with `_predicted_answer("gsm8k", text)` + `_answers_match("gsm8k", pred,
   _target_answer(...))` (import from `benchmark_all`). LoRA disabled (round-0 rollouts
   come from the certified baseline receiver; iterated STaR rounds are a later lever,
   out of scope here).
3. If greedy is wrong: up to 4 sampled retries (temperature 0.7, top_p 0.95, seeded as in
   3.3) via a small sampling twin of `greedy_decode_from_prefix` (same prefix_state
   reset, `torch.multinomial` instead of argmax). Keep the first verified-correct rollout.
4. Persist per row to `.cache/receiver_lora_rollouts/rollout_<sha256(prompt)>.pt`:
   `{"prompt_sha", "sample_index", "split": "train", "token_ids", "decoded_text",
   "greedy_correct": bool, "attempt": int, "verified": bool, "config_digest":
   sha256(receiver_model, adapter_digest, truncation, max_new_tokens, temperature,
   seed)}`. Rows with no verified rollout after 4 tries are excluded from A/B training
   (logged). Kill-safe: the trainer's `--prepare-rollouts` mode resumes by skipping
   existing files.

Expected yield: ~210 complete-trace rows * ~75% acceptance ~= 155-170 training samples.
Wall-clock: greedy pass ~210 * 14.5s ~= 51 min; retries ~55 rows * <=4 * 14.5s <= 53 min
worst case. Truncate stored rollouts at 256 tokens.

### 4.4 Step budget and wall-clock

~160 samples, grad-accum 8 -> ~20 optimizer steps/epoch, <=60 total. Training forward+
backward at seq ~800 on MPS ~= 4-8s/sample -> 11-22 min/epoch. Full objective-A run incl.
gates: under 2 hours. Everything >15 min runs under the detached nohup watchdog pattern
(ops note in the parity memory) and is resumable (section 5.3).

## 5. Trainer: `scripts/train_receiver_lora.py`

Mirror the structure/CLI conventions of `scripts/train_latent_bridge.py`.

### 5.1 CLI (defaults shown; no other knobs)

```
--receiver-model Qwen/Qwen3.5-2B   --sender-model Qwen/Qwen3.5-2B
--trace-dtype bfloat16             --dataset gsm8k
--train-rows 0:224                 --dev-rows 224:256
--truncation-fraction 0.5          --validation-size 256
--adapter-cache-path .cache/generated_trajectory_adapter/generated_trajectory_adapter_8959603bbf6825e4dc4df3f373ca600ccd7e61425e0aa9c66d7f0adb1070b207.pt
--objective A                      # A | B | C
--rank 16 --alpha 32 --dropout 0.05 --lr 1e-4 --epochs 3 --grad-accum 8
--max-continuation-tokens 256      --max-new-tokens 256
--rollout-max-tries 4 --rollout-temperature 0.7 --rollout-top-p 0.95
--checkpoint-every 32              # samples
--device mps --seed 7
--output-dir outputs/receiver_lora/objective_a
--prepare-rollouts                 # generate/verify rollouts and exit
--eval-only                        # run the dev gate on the checkpoint and exit
--max-steps 0                      # debug cap
```

Objective C ignores rollouts and trains on sender `continuation_ids`; `--epochs` is
force-capped at 1 for C.

### 5.2 Gates, early stop, kill criteria (the Phase-0 detector)

Dev gate = the 32 dev rows, scored exactly like the benchmark (greedy, 256 tokens,
`_predicted_answer`/`_answers_match`), four variants:

- `latent_lora`: production fused-forward prefix, LoRA enabled. Re-measured every gate.
- `latent_base`: same, LoRA disabled. Measured once at step 0, then reused (constant).
- `text`: truncated-text hybrid layout (`TEXT_INSTRUCTION` + `truncated_text`, as in the
  bridge trainer's triplet). Measured once at step 0 with LoRA disabled (constant), and
  re-measured with LoRA ENABLED at each epoch end as the style-drift canary
  (`text_lora_canary`).
- `alone`: measured once at step 0 (constant, context only).

Gate schedule and rules:

1. Step-0 identity gate (before any optimizer step): `latent_lora` per-row results must
   be IDENTICAL to `latent_base` (zero-init B guarantees this; any mismatch is a plumbing
   bug, abort).
2. Mid-epoch gate every 10 optimizer steps: `latent_lora` accuracy on the dev rows plus
   dev-set mean training-objective NLL. Cost ~32 * 14.5s ~= 8 min.
3. Epoch-end gate: mid-epoch gate + `text_lora_canary` + copy-proof stratum accuracy
   (dev rows whose `truncated_text` does not contain the normalized answer literal,
   reusing `certify_latent_bridge.py`'s `_normalized_number` logic).
4. KILL (stop run, mark FAILED in report, keep checkpoints) if any of:
   - `latent_lora` correct-count < `latent_base` correct-count minus 2 at any gate after
     optimizer step 10;
   - `text_lora_canary` correct-count < step-0 `text` correct-count minus 2;
   - copy-proof `latent_lora` accuracy drops below its step-0 value while overall
     `latent_lora` accuracy rises (parroting signature);
   - dev NLL decreasing for 2 consecutive gates while `latent_lora` accuracy decreased
     over the same window (the exact Phase-0 divergence signature).
5. Early stop / selection: keep the checkpoint with the best `latent_lora` dev
   correct-count (ties -> earlier step); stop after 2 consecutive epoch-end gates without
   improvement. `best_lora.pt` is written on every new best.

All gate outcomes append to `outputs/receiver_lora/<run>/lora_report.json`
(`{"history": [...], "latest": {...}}`, same shape as `bridge_report.json`, plus per-row
records for later certification).

Run order: C first (validates that gates fire on the known-bad objective; ~35 min), then
A. B only per section 2's condition.

### 5.3 Resumability

Checkpoint every 32 samples and at every epoch end to `<output-dir>/lora_checkpoint.pt`:
`{lora state, optimizer, scheduler, epoch, global_step, best_dev_correct, rng states
(torch, random), args}`. On start, if the checkpoint exists, resume (same contract as the
bridge trainer). Rollouts are already per-row cache files. External kills lose at most 32
samples of progress.

## 6. Production integration

### 6.1 Config and flags

`configs/main.yaml`, new block:

```yaml
handoff:
  receiver_lora:
    path: null          # null = disabled
    scope: latent_only  # latent_only | all
```

`benchmark_all.py::main` new args, written into cfg before state init:

- `--receiver-lora-path PATH` -> `cfg.handoff.receiver_lora.path`
- `--receiver-lora-scope {latent_only,all}` (default latent_only)

Both are permitted alongside `--eval-manifest` (that is exactly the certification
pattern: locked rows, new receiver config). The resolved manifest written by
`--write-eval-manifest` gains an additive optional block
`"receiver_lora": {"path", "file_sha256", "scope", "rank", "alpha",
"target_module_count"}` (absent key = disabled; no schema-version bump needed since the
field is optional and old manifests replay unchanged). When a loaded manifest contains
the block and the CLI also passes `--receiver-lora-path`, they must agree on
`file_sha256`, else exit with an error.

### 6.2 Loading (latent_pipeline.py)

In `_get_pipeline_state`, immediately after `agent_b = load_agent(...)`:

```python
lora_cfg = getattr(getattr(cfg, "handoff", None), "receiver_lora", None)
lora_path = getattr(lora_cfg, "path", None)
if lora_path:
    from src.models.receiver_lora import load_receiver_lora, set_receiver_lora_enabled
    meta = load_receiver_lora(agent_b, lora_path)      # raises on 0 modules / bad file
    set_receiver_lora_enabled(agent_b, str(getattr(lora_cfg, "scope", "latent_only")) == "all")
    print(f"Receiver LoRA loaded: {lora_path} sha={meta['file_sha256'][:12]} modules={meta['module_count']}")
```

`_pipeline_state_key` must append `(str(lora_path or ""), str(scope))` so cached pipeline
state cannot be reused across different LoRA configs (update its type annotation).

### 6.3 Scoped activation (benchmark_all.py)

With scope `latent_only`, LoRA stays disabled globally and is enabled only around the
receiver decode of the latent method. In `_run_generated_latent_variant`, wrap the
`_decode_handoff(...)` call (the one at the current line ~5378) as:

```python
with _receiver_lora_scope_for_method(variant_cfg, agent_b):   # thin helper:
    decode_metrics = _decode_handoff(...)
```

where the helper returns `receiver_lora_scope(agent_b, True)` when
`cfg.handoff.receiver_lora.path` is set and scope is `latent_only`, else a nullcontext.
Everything else (`run_pure_text_cot`, `run_text_text_hybrid`, alignment-distance
computation, sender tracing) runs with LoRA disabled. `_alignment_distances` only uses
the embedding table, which LoRA never touches. `_decode_handoff` internally covers prefix
prep, greedy decode, and answer-PPL metrics, so the whole latent measurement is
consistently LoRA-on.

Per-row report fields added by `_run_generated_latent_variant`:
`"receiver_lora_applied": bool`, `"receiver_lora_sha": str|None` (so the samples CSV
proves which rows ran through the adapter).

## 7. Certification protocol

### 7.1 Final locked run

```bash
venv/bin/python benchmark_all.py \
  --eval-manifest outputs/parity_fix/locked_continuation_128.json \
  --receiver-lora-path outputs/receiver_lora/objective_a/best_lora.pt \
  --write-eval-manifest outputs/receiver_lora/certification/locked_continuation_128_lora.json \
  --report-output outputs/receiver_lora/certification/continuation128_lora_report.json \
  --samples-output outputs/receiver_lora/certification/continuation128_lora_samples.csv \
  --summary-output outputs/receiver_lora/certification/continuation128_lora_summary.csv
```

(~2h; detached watchdog. The manifest pins suite/dataset/split/indices/methods/models/
seed; the trace and adapter caches make the sender side a pure replay.)

### 7.2 `scripts/certify_receiver_lora.py` (modeled on `certify_latent_bridge.py`)

Checks, all written to `outputs/receiver_lora/certification/certification.json`:

1. Split disjointness: prompt sha256 of every training rollout row and dev row is absent
   from the locked manifest's `sample_fingerprints[].prompt_sha256`.
2. Truncation safety: no training/dev `truncated_text` contains "final answer"
   (case-insensitive).
3. Channel identity: the certification report's per-row
   `handoff_adapter_cache_key_digest` equals
   `5c041b5521a4663a1a469410c7114df8753762df45f7b6fad66bb9ca2f03c02b` (same frozen latent
   channel as the baseline), and every latent row has `receiver_lora_applied == True`
   with the expected file sha.
4. Non-degradation: for methods `pure_text_cot` and `text_text_hybrid`, per-row
   `correct` values in the new samples CSV are identical to
   `outputs/parity_fix/continuation128_samples.csv` (any diff means LoRA leaked out of
   scope; FAIL).
5. Paired significance: exact McNemar on `generated_context_latent_handoff` rows joined
   by `sample_index` across the two CSVs (b = baseline-only wins, c = LoRA-only wins,
   p = exact two-sided binomial(b+c, 0.5)). Also report the copy-proof stratum (answer
   literal absent from the truncated sender text) accuracies.
6. Objective-C audit: the report records the canary run's outcome (which gate fired at
   which step), demonstrating the detector works.

### 7.3 Cross-family stretch (no training)

```bash
venv/bin/python benchmark_all.py \
  --eval-manifest outputs/parity_fix/locked_xfam_32.json \
  --receiver-lora-path outputs/receiver_lora/objective_a/best_lora.pt \
  --report-output outputs/receiver_lora/certification/xfam32_lora_report.json \
  --samples-output outputs/receiver_lora/certification/xfam32_lora_samples.csv \
  --summary-output outputs/receiver_lora/certification/xfam32_lora_summary.csv
```

McNemar vs `outputs/parity_fix/xfam32_samples.csv` (baseline 19/32). Interpretation is
pre-registered: improvement = consumption skill is sender-agnostic; no change or
degradation = pair-specific training needed (EXAONE traces already exist for that
follow-up; out of scope here).

## 8. Risk register (silent failure modes and their checks)

1. Layout drift between trainer and production (the class of bug that produced 0% for
   three weeks). Check: pinned test `test_receiver_lora_layout_parity` builds, for one
   real cached train row, (a) the trainer's `inputs_embeds` prefix and (b)
   `prepare_receiver_context_latent_prefix_state`'s fused prefix from
   `_format_receiver_context_prompt` output, and asserts context token ids are equal and
   latent tensors allclose (atol 0 in bf16); certification check 3 pins the adapter
   digest at eval time.
2. Parroting reward hack: rejection sampling over-selects rows where the answer literal
   survives in the latent prefix; LoRA learns to copy, not compute. Check: copy-proof
   stratum tracked at every epoch-end gate with an explicit kill rule (5.2.4), and
   reported in certification check 5; additionally the rollout report logs the fraction
   of accepted rollouts from copy-visible rows so a skewed acceptance mix is visible
   before training starts.
3. LoRA silently not applied (or applied to the wrong rows) at eval, making "no change"
   results meaningless. Check: `apply_receiver_lora` raises unless exactly 60 modules
   wrap; `set_receiver_lora_enabled` raises on 0 toggles; per-row
   `receiver_lora_applied`/sha fields in the samples CSV; and the step-0 identity gate
   plus certification check 4 catch the converse failure (LoRA applied where it must
   not be).

(Phase-0-style divergence is not "silent" here by construction: section 5.2's gates plus
the objective-C canary run make it loud.)

## 9. Implementation plan (5 reviewable chunks)

1. LoRA module. New `src/models/receiver_lora.py` + `tests/test_receiver_lora.py`
   (model-free: tiny nn.Module with linears named to match the target suffixes; cover
   wrap count enforcement, zero-init identity forward, enable/disable/scope restore,
   save/load round-trip incl. sha, fp32 params under bf16 base).
   Verify: `venv/bin/python -m pytest tests/test_receiver_lora.py -q && ruff check src/models/receiver_lora.py`.
2. Trainer data path. `scripts/train_receiver_lora.py` with sample building (reusing
   `load_bridge_samples`), frozen-adapter loading + `apply_alignment` projection,
   `--prepare-rollouts` (cached, resumable, verified), and the layout-parity test
   `tests/test_receiver_lora_layout.py` (instruction-string equality vs
   `_format_receiver_context_prompt` is model-free; the fused-prefix comparison may be
   marked as requiring the local model cache and run manually).
   Verify: `venv/bin/python scripts/train_receiver_lora.py --prepare-rollouts --train-rows 0:4 --output-dir outputs/receiver_lora/smoke` completes, writes 4 rollout files, prints acceptance stats; rerun is a no-op (cache hits).
3. Training loop. Objectives A/B/C, gates, kill rules, checkpoints, resume, report JSON.
   Verify: `--train-rows 0:16 --dev-rows 224:228 --max-steps 24` micro-run produces a
   step-0 identity-pass line and gate history; `kill -9` mid-run then rerun resumes from
   the checkpoint step.
4. Benchmark integration. Config block, CLI flags, `_get_pipeline_state` loading +
   state-key extension, scoped activation in `_run_generated_latent_variant`, manifest
   block, per-row CSV fields, plus `tests/test_receiver_lora_benchmark.py` (manifest
   round-trip; scope helper returns nullcontext when disabled; state key changes with
   path). Verify: full pytest suite + an 8-row live run
   (`--sample-indices 0..7`, flags cloned from the certified argv,
   `--receiver-lora-path <zero-init lora>`): latent rows must decode bit-identically to
   the baseline 8-row artifacts (zero-init = identity), with
   `receiver_lora_applied=True` in the CSV.
5. Runs and certification. Warm 128 train traces (4.1), rollouts (4.3), objective C then
   A (then B if triggered), `scripts/certify_receiver_lora.py` + the locked N=128 run
   (7.1) and stretch xfam run (7.3).
   Verify: `outputs/receiver_lora/certification/certification.json` has every check
   `true` (or the run is honestly reported FAILED), and PROGRESS.md gets the outcome
   either way.

## 10. What is intentionally out of scope

Iterated STaR rounds (round-1 rollouts from the LoRA'd receiver), MLP-target LoRA,
pair-specific cross-family LoRA training, MATH Level-5, and any change to the sender,
the ridge adapter, or the locked manifests. Each is a named lever for the next cycle,
contingent on the primary result.
