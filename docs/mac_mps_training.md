# Mac MPS Training Path

This branch adds an explicit Apple Silicon MPS path for Stage-II smoke training.
Use it as a local sanity and iteration path while cloud GPUs are unavailable.
Do not treat it as a replacement for broad adapter scaling or long heterogeneous
training.

## What This Can Do

- Verify `run_training.py` works end-to-end on Apple Silicon.
- Run small smoke-stage Stage-II training with `Qwen/Qwen3.5-0.8B -> Qwen/Qwen3.5-0.8B`.
- Exercise the training loop, alignment context, losses, evaluation callback, and reports.
- Produce local artifacts in `outputs/mac_mps`.

## What This Probably Cannot Do

- Efficiently train the full hetero `EXAONE -> Qwen` path.
- Replace a real GPU for 128/256/512-row generated-trajectory adapter sweeps.
- Guarantee that MPS supports every model-specific operation used by remote-code models.

The code sets `PYTORCH_ENABLE_MPS_FALLBACK=1`, so unsupported MPS operations may
fall back to CPU. That is useful for correctness checks, but it can be slow.

## First Local Smoke

Inspect the command:

```bash
venv/bin/python scripts/mac_mps_stage2_smoke.py
```

Run it:

```bash
venv/bin/python scripts/mac_mps_stage2_smoke.py --execute
```

Expected behavior:

- It uses `runtime.device=mps`.
- It forces `runtime.mps.torch_dtype=float32`.
- It uses `device_map=none`, then moves models onto MPS explicitly.
- It disables checkpoints.
- It runs tiny smoke data: batch size `1`, `4` samples, max length `64`.
- It prints both `training_smoke_passed` and `phase2_gate_passed`.
- It prints answer extraction diagnostics, including direct decode extraction
  rate and candidate-NLL fallback rate.
- It prints degenerate-prediction and actor text-baseline diagnostics.
- The actor text baseline uses a few-shot final-answer prompt. If it is still
  degenerate, fix prompt construction or model choice before interpreting
  latent training quality.

Outputs:

- `outputs/mac_mps/mac_mps_training_history.csv`
- `outputs/mac_mps/mac_mps_training_report.json`

Interpretation:

- `training_smoke_passed=true` means the local training loop, alignment,
  evaluation, answer extraction, and reporting completed without structural
  failures.
- `decode_answer_extraction_rate` is the rate at which greedy continuation from
  the latent prefix produced a parseable answer. `candidate_fallback_rate` is
  the rate at which smoke evaluation had to score known held-out answer
  candidates by NLL instead. Fallback keeps smoke evaluation debuggable, but it
  is labeled separately so it is not confused with generation quality.
- `unique_predicted_answer_count=1` with low exact-match accuracy means the
  latent path collapsed to one repeated answer. The smoke gate now fails this
  case even when answer extraction is `100%`.
- `actor_text_baseline_accuracy` is the same actor answering from a normal text
  prompt. If this baseline is non-degenerate while the latent path collapses,
  the issue is latent injection or the Stage-II objective, not answer parsing.
- `actor_text_baseline_candidate_accuracy` scores known smoke answer candidates
  by NLL. If candidate accuracy is healthy while greedy baseline generation is
  degenerate, the model has answer likelihood signal but needs a better decode
  or prompting surface.
- `latent_training_ready=true` means the local smoke did not find structural
  extraction, degenerate decode, or actor-baseline blockers.
- The Stage-II loop includes a latent-prefix answer loss (`l_answer`) so the
  optimized task matches evaluation: latent prefix plus `Final answer:` should
  make the frozen actor assign high likelihood to the target answer.
- Structured training rows feed prompt-only text to the reasoner; answers are
  used as targets, not as reasoner input.
- `initial_eval_diagnostics` records the latent path before training. Compare it
  with final diagnostics: collapse before training points at injection/alignment;
  collapse after training points at the objective or optimization settings.
- `phase2_gate_passed=false` is expected for this smoke run because it is not
  real multi-seed training and does not provide a baseline retention score.
- `final_heldout_exact_match_accuracy=0` on the tiny smoke run is not by itself a
  blocker. It means the model did not learn a useful semantic handoff in a few
  tiny steps. Use the smoke for code correctness, not final quality.

## If The First Smoke Passes

Increase only one axis at a time:

```bash
venv/bin/python scripts/mac_mps_stage2_smoke.py \
  --execute \
  --smoke-samples 8
```

Then:

```bash
venv/bin/python scripts/mac_mps_stage2_smoke.py \
  --execute \
  --smoke-samples 8 \
  --max-length 96
```

Then try the normal thinker/executor pair:

```bash
venv/bin/python scripts/mac_mps_stage2_smoke.py \
  --execute \
  --agent-a-model Qwen/Qwen3.5-2B \
  --agent-b-model Qwen/Qwen3.5-0.8B \
  --smoke-samples 4 \
  --max-length 64
```

If that fails on memory, go back to `0.8B -> 0.8B` and use MPS for code-level
iteration only.

## Overfit Readiness Smoke

Before spending time on held-out quality, verify that Stage II can overfit a tiny
prompt-only smoke set:

```bash
venv/bin/python scripts/mac_mps_stage2_smoke.py \
  --execute \
  --eval-on-train \
  --epochs 3 \
  --smoke-samples 8
```

Use this as a readiness check. If actor text baseline is non-degenerate but
latent outputs still collapse on train-overfit eval, focus on latent injection,
answer loss weight, and optimization. If the actor text baseline itself is
degenerate, fix prompt/model construction first.

## Direct `run_training.py` Command

The script expands to this shape:

```bash
venv/bin/python run_training.py \
  runtime.device=mps \
  runtime.mps.fallback_to_cpu=false \
  runtime.mps.torch_dtype=float32 \
  device_map=none \
  agent_a_model=Qwen/Qwen3.5-0.8B \
  agent_b_model=Qwen/Qwen3.5-0.8B \
  training.data.mode=smoke \
  training.data.batch_size=1 \
  training.data.smoke_num_samples=4 \
  training.reasoner_max_length=64 \
  training.actor_max_length=64 \
  training.compressed_steps=8 \
  training.num_epochs=1 \
  training.checkpointing.enabled=false \
  reporting.training.history_output=outputs/mac_mps/mac_mps_training_history.csv \
  reporting.training.report_output=outputs/mac_mps/mac_mps_training_report.json
```

## Stop Conditions

Stop local MPS escalation and wait for rented GPU access if:

- MPS falls back to CPU heavily and runtime becomes impractical.
- `Qwen/Qwen3.5-2B -> Qwen/Qwen3.5-0.8B` does not fit.
- You need 128+ generated trajectory training rows.
- You need heterogeneous EXAONE/Qwen training with meaningful wall-clock speed.

## Next Best Work While GPUs Are Unavailable

Use MPS to harden code and objectives, not to chase final accuracy:

1. Run the MPS smoke and fix any real training-loop failures.
2. Add receiver-side latent-to-answer objective tests using tiny local models.
3. Improve provenance reports for training runs.
4. Keep benchmark/report/extraction code locked.
5. When cloud GPUs return, run the bounded DigitalOcean pilot from
   `docs/digitalocean_gpu_training.md`.
