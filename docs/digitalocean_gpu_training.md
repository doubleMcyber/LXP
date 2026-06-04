# DigitalOcean GPU Training Handoff

This handoff is for the current MVP path on branch `hetero-handoff-experiments`.
The local evidence says the sender can solve the hard rows, and the receiver can
copy a verified answer, but raw token context and generated-context latent
handoff still fail on the selected hard cases. The next GPU work should therefore
scale generated trajectory data and adapter validation before spending credits on
larger architecture changes.

## Current Evidence

Latest local hard-row smoke:

```bash
venv/bin/python benchmark_all.py \
  --hetero-smoke \
  --sample-indices 9,11,19 \
  --limit 3 \
  --methods token_context_handoff,verified_token_context_handoff,sender_answer_text_handoff,generated_context_latent_handoff \
  --generated-trajectory-adapter-input-space raw \
  --enable-sender-revision \
  --generated-trajectory-adapter-no-train-on-missing \
  --report-output /private/tmp/lxp_v9_verified_token_context_hard_report.json \
  --samples-output /private/tmp/lxp_v9_verified_token_context_hard_samples.csv \
  --summary-output /private/tmp/lxp_v9_verified_token_context_hard_summary.csv
```

Observed accuracy on those rows:

- `token_context_handoff`: `0%`
- `verified_token_context_handoff`: `100%`
- `sender_answer_text_handoff`: `100%`
- `generated_context_latent_handoff`: `0%`

Interpretation: the receiver can obey compact verified handoff payloads, even
when the sender trace is present. The failing path is latent readability and
evidence selection, not sender answer generation.

## Useful References

- DigitalOcean GPU Droplets: https://docs.digitalocean.com/products/gpu-droplets/
- DigitalOcean create a Droplet: https://docs.digitalocean.com/products/droplets/how-to/create/
- DigitalOcean SSH keys: https://docs.digitalocean.com/products/droplets/how-to/add-ssh-keys/
- DigitalOcean Volumes: https://docs.digitalocean.com/products/volumes/
- DigitalOcean `doctl`: https://docs.digitalocean.com/reference/doctl/
- PyTorch install selector: https://pytorch.org/get-started/locally/

## Recommended GPU Job

Use one GPU Droplet with enough VRAM for the active pair and cache storage for
generated traces. Prefer a DigitalOcean GPU image with drivers already installed.
Attach a Volume if you want to keep Hugging Face model weights and `.cache`
artifacts after destroying the Droplet.

The first paid run should not start with stage-2 compression training. It should
first produce a larger generated-trajectory adapter and evaluate whether the
existing adapter family scales beyond the local 20-sample result. If it still
plateaus below target accuracy, move to a receiver-side learned latent-to-answer
objective.

The repo includes a bounded pilot runner. Locally, inspect the exact command
sequence without running it:

```bash
venv/bin/python scripts/do_gpu_pilot.py
```

On the GPU host, run:

```bash
venv/bin/python scripts/do_gpu_pilot.py --execute
```

The pilot runs unit tests, warms eval sender traces, prepares one 128-row raw
generated-trajectory adapter, then evaluates locked rows against token and latent
controls with `--generated-trajectory-adapter-no-train-on-missing`.

## Setup

```bash
git clone https://github.com/doubleMcyber/LXP.git
cd LXP
git checkout hetero-handoff-experiments

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If the image does not already include a compatible CUDA PyTorch build, use the
PyTorch install selector above and install the CUDA wheel it recommends for the
Droplet image.

Recommended cache environment:

```bash
mkdir -p /mnt/lxp-cache/hf /mnt/lxp-cache/lxp
export HF_HOME=/mnt/lxp-cache/hf
export TRANSFORMERS_CACHE=/mnt/lxp-cache/hf
```

If `/mnt/lxp-cache` is not available, use a directory on the boot disk. A
mounted Volume is better for repeated runs because model downloads dominate cold
start time.

## Step 1: Unit And Preflight Gates

```bash
venv/bin/python -m pytest -q

venv/bin/python preflight_model_pair.py \
  --agent-a-model LGAI-EXAONE/EXAONE-4.0-1.2B \
  --agent-b-model Qwen/Qwen3.5-0.8B
```

Expected preflight result for the current hetero pair: incompatible KV cache
topology. That is acceptable for the generated-context latent MVP because this
path transfers receiver-context latent prefixes rather than assuming direct KV
cache compatibility.

## Step 2: Warm Eval Sender Traces

Start with 20 or 64 eval rows. Use explicit indices so local and GPU runs are
comparable.

```bash
mkdir -p outputs

venv/bin/python benchmark_all.py \
  --hetero-smoke \
  --sample-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
  --limit 20 \
  --methods generated_context_latent_handoff \
  --generated-trajectory-adapter-input-space raw \
  --enable-sender-revision \
  --prepare-generated-trajectory-eval-traces \
  --report-output outputs/do_eval_trace_warm_20.json \
  --write-eval-manifest outputs/do_eval_trace_manifest_20.json
```

This should populate `.cache/generated_trajectory_traces` and make later method
sweeps cheaper.

## Step 3: Build A Larger Generated-Trajectory Adapter

Start with 128 rows. If cache build succeeds and VRAM/time are acceptable, repeat
with 256 or 512.

```bash
venv/bin/python benchmark_all.py \
  --hetero-smoke \
  --methods generated_context_latent_handoff \
  --generated-trajectory-adapter-input-space raw \
  --generated-trajectory-adapter-train-limit 128 \
  --enable-sender-revision \
  --prepare-generated-trajectory-adapter \
  --report-output outputs/do_adapter_raw_128_prepare.json
```

Then try stronger settings:

```bash
venv/bin/python benchmark_all.py \
  --hetero-smoke \
  --methods generated_context_latent_handoff \
  --generated-trajectory-adapter-input-space raw \
  --generated-trajectory-adapter-train-limit 256 \
  --generated-trajectory-local-residual-top-k 16 \
  --generated-trajectory-local-residual-blend 1.0 \
  --generated-trajectory-local-residual-max-memory-rows 8192 \
  --enable-sender-revision \
  --prepare-generated-trajectory-adapter \
  --report-output outputs/do_adapter_raw_256_prepare.json
```

## Step 4: Evaluate Against Token Controls

```bash
venv/bin/python benchmark_all.py \
  --hetero-smoke \
  --sample-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
  --limit 20 \
  --methods token_context_handoff,verified_token_context_handoff,sender_answer_text_handoff,generated_context_latent_handoff \
  --generated-trajectory-adapter-input-space raw \
  --generated-trajectory-adapter-train-limit 128 \
  --enable-sender-revision \
  --generated-trajectory-adapter-no-train-on-missing \
  --report-output outputs/do_context_vs_latent_20_report.json \
  --samples-output outputs/do_context_vs_latent_20_samples.csv \
  --summary-output outputs/do_context_vs_latent_20_summary.csv \
  --write-eval-manifest outputs/do_context_vs_latent_manifest_20.json
```

Inspect the run:

```bash
jq '.semantic_smoke_report | {passed, method_accuracy_percentage, missing_requirements}' outputs/do_context_vs_latent_20_report.json
jq '.latent_provenance_report | {method_accuracy_percentage, failure_counts_by_class, cache_paths}' outputs/do_context_vs_latent_20_report.json
jq '.eval_manifest | {manifest_digest, sample_indices, methods}' outputs/do_context_vs_latent_20_report.json
```

To replay the exact same eval lock, use:

```bash
venv/bin/python benchmark_all.py \
  --eval-manifest outputs/do_context_vs_latent_manifest_20.json \
  --generated-trajectory-adapter-input-space raw \
  --enable-sender-revision \
  --generated-trajectory-adapter-no-train-on-missing \
  --report-output outputs/do_context_vs_latent_20_replay_report.json \
  --samples-output outputs/do_context_vs_latent_20_replay_samples.csv \
  --summary-output outputs/do_context_vs_latent_20_replay_summary.csv
```

Pass criteria for this stage:

- `sender_answer_text_handoff` is at or near `100%`.
- `verified_token_context_handoff` is at or near `100%`.
- `generated_context_latent_handoff` improves materially from the current local
  hard-row failure and from the previous 20-row `85%` result.
- Wrong latent rows are not dominated by `sender_wrong`; if they are, improve
  sender revision before touching latent training.

## Step 5: When To Run Stage-2 Training

Run stage-2 training only after the larger adapter evaluation shows the current
generated-trajectory adapter family has hit a real ceiling. A real ceiling means:

- sender and verified-token baselines are strong,
- latent failures are mostly `latent_receiver_gap_verified_answer_available`,
- increasing train rows from 128 to 256 or 512 does not improve accuracy, and
- answer perplexity or provenance suggests the receiver cannot decode the latent
  evidence, not just that the sender trace is wrong.

At that point, the next legitimate production path is a supervised receiver-side
objective. Train a component that maps sender generated hidden trajectories to a
receiver state optimized for the target final-answer line, then keep the current
token controls in the eval run so gains cannot be explained by answer leakage.

## Artifact Checklist

Save these files from each paid run:

- `outputs/*_report.json`
- `outputs/*_samples.csv`
- `outputs/*_summary.csv`
- `outputs/*manifest*.json`
- `.cache/generated_trajectory_adapter/*.pt`
- `.cache/generated_trajectory_rows/*.pt`
- `.cache/generated_trajectory_traces/*.pt` if you want exact reruns

Do not commit large cache artifacts to the repository. Copy them to a Volume or
download them separately.

## Current Next Decision

If the 128/256-row generated adapter reaches high accuracy, continue scaling and
add broader benchmark sets. If it stays below target while verified-token
baselines stay high, spend the GPU credits on the receiver-side learned
latent-to-answer objective rather than more prompt or suffix tweaks.
