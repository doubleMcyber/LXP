from __future__ import annotations

import csv
import json

from scripts.render_stage2_report import render_stage2_report


def test_render_stage2_report_includes_probe_and_diagnostics(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    history_path = tmp_path / "history.csv"
    output_path = tmp_path / "report.html"
    report_path.write_text(
        json.dumps(
            {
                "model_pair": "A -> B",
                "effective_device": "mps",
                "effective_torch_dtype": "float32",
                "passed": False,
                "missing_requirements": ["Need real mode."],
                "training_smoke_report": {
                    "latent_probe_ready": True,
                    "latent_sequence_decoder_ready": True,
                    "latent_training_ready": False,
                    "final_heldout_latent_probe_accuracy": 100.0,
                    "final_heldout_latent_sequence_decoder_sequence_accuracy": 100.0,
                    "final_heldout_exact_match_accuracy": 0.0,
                    "final_heldout_latent_candidate_accuracy": 66.6667,
                    "final_heldout_actor_text_baseline_accuracy": 33.3333,
                    "final_heldout_answer_perplexity": 7.5,
                    "heldout_eval_diagnostics": (
                        "target=4 | predicted=1 | candidate_predicted=4 | "
                        "probe_predicted=4 | baseline_predicted=4 | decoded=Final answer: 1"
                    ),
                    "missing_requirements": ["Decode collapsed."],
                },
            }
        ),
        encoding="utf-8",
    )
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "loss",
                "answer_probe_accuracy",
                "latent_sequence_decoder_sequence_accuracy",
                "answer_contrast_accuracy",
                "handoff_adapter_update_norm",
                "latent_answer_probe_update_norm",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "loss": "10.0",
                "answer_probe_accuracy": "100.0",
                "latent_sequence_decoder_sequence_accuracy": "100.0",
                "answer_contrast_accuracy": "50.0",
                "handoff_adapter_update_norm": "0.1",
                "latent_answer_probe_update_norm": "0.2",
            }
        )

    render_stage2_report(report_path, history_path, output_path)

    html = output_path.read_text(encoding="utf-8")
    assert "LXP Stage II Latent Transfer Report" in html
    assert "Latent Transfer Map" in html
    assert "Geometric Handoff" in html
    assert "Latent Probe" in html
    assert "Sequence Decoder" in html
    assert "100.00%" in html
    assert "<td>4</td>" in html
    assert "Decode collapsed." in html
