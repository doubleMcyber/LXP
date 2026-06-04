from __future__ import annotations

import argparse

from scripts.do_gpu_pilot import build_commands


def test_do_gpu_pilot_builds_bounded_sequence() -> None:
    args = argparse.Namespace(
        python="venv/bin/python",
        output_dir="outputs",
        eval_limit=20,
        train_limit=128,
    )

    commands = build_commands(args)

    assert len(commands) == 4
    assert commands[0] == ["venv/bin/python", "-m", "pytest", "-q"]
    assert "--prepare-generated-trajectory-eval-traces" in commands[1]
    assert "--prepare-generated-trajectory-adapter" in commands[2]
    assert "--generated-trajectory-adapter-no-train-on-missing" in commands[3]
    assert "--write-eval-manifest" in commands[3]
    assert "token_context_handoff,verified_token_context_handoff,sender_answer_text_handoff,generated_context_latent_handoff" in commands[3]
