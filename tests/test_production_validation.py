from __future__ import annotations

from argparse import Namespace

from scripts.run_production_validation import build_commands


def test_production_validation_builds_locked_eval_and_replay_commands() -> None:
    args = Namespace(
        python="venv/bin/python",
        output_dir="outputs/prod",
        eval_limit=5,
        train_limit=32,
        generated_trajectory_adapter_input_space="raw",
        include_tests=True,
        prepare=True,
        replay=True,
    )

    commands = build_commands(args)

    assert commands[0] == ["venv/bin/python", "-m", "pytest", "-q"]
    assert any("--prepare-generated-trajectory-eval-traces" in command for command in commands)
    assert any("--prepare-generated-trajectory-adapter" in command for command in commands)
    benchmark_command = commands[-2]
    replay_command = commands[-1]
    assert "--generated-trajectory-adapter-no-train-on-missing" in benchmark_command
    assert "--write-eval-manifest" in benchmark_command
    assert "--eval-manifest" in replay_command
    assert "outputs/prod/production_context_vs_latent_5_manifest.json" in replay_command
