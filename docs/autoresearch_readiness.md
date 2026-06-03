# Autoresearch Readiness

Use an autoresearch loop only after the local benchmark architecture is locked
enough that automated edits cannot improve the score by moving the goalposts.
For this repo, that means manifests, no-train-on-missing evals, provenance
reports, and token controls must already be in place.

Reference pattern: https://github.com/karpathy/autoresearch

## Current Recommendation

Do not point an autonomous research loop at the whole repo yet. The right first
use is a constrained worker that can edit one module family at a time and must
beat the locked manifest evaluation.

Good first scopes:

- receiver-side latent-to-answer objective
- generated trajectory adapter objective
- embedding-manifold projection policy
- local residual retrieval policy

Bad first scopes:

- benchmark report code
- answer extraction
- dataset loader behavior
- eval manifest code
- token-control baselines

Those areas define the measurement surface. Letting autonomous search edit them
creates benchmark leakage risk.

## Minimum Gate Before Autoresearch

Autoresearch can start when all of these are true:

- `venv/bin/python -m pytest -q` passes.
- A locked eval manifest exists and is replayable with `--eval-manifest`.
- The eval command uses `--generated-trajectory-adapter-no-train-on-missing`.
- `sender_answer_text_handoff` and `verified_token_context_handoff` are strong on
  the locked eval rows.
- `latent_provenance_report.failure_counts_by_class` mostly identifies latent
  receiver gaps, not sender failures.
- The editable file set is explicit before the run starts.

## Autoresearch Harness Shape

The harness should run one short experiment loop:

1. Start from a clean branch.
2. Restrict edits to one implementation area.
3. Run unit tests.
4. Run the locked eval manifest.
5. Reject changes that lower token-control accuracy or mutate eval/report code.
6. Keep changes only if latent accuracy improves and provenance still passes.

Example locked eval command:

```bash
venv/bin/python benchmark_all.py \
  --eval-manifest outputs/do_context_vs_latent_manifest_20.json \
  --generated-trajectory-adapter-input-space raw \
  --enable-sender-revision \
  --generated-trajectory-adapter-no-train-on-missing \
  --report-output outputs/autoresearch_eval_report.json \
  --samples-output outputs/autoresearch_eval_samples.csv \
  --summary-output outputs/autoresearch_eval_summary.csv
```

## Promotion Gate

Promote an autoresearch result only if:

- tests still pass,
- report/schema/provenance code was not edited,
- token controls did not regress,
- latent accuracy improves on a held-out locked manifest, and
- the change can be explained as a real architectural improvement.

If a change only improves the original manifest but fails a second held-out
manifest, treat it as overfit and discard it.
