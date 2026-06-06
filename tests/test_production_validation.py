from __future__ import annotations

from argparse import Namespace

from scripts.run_production_validation import PROFILE_DEFAULTS, build_commands


def test_production_validation_builds_locked_eval_and_replay_commands() -> None:
    args = Namespace(
        python="venv/bin/python",
        python_flags="-B",
        output_dir="outputs/prod",
        dataset="gsm8k",
        eval_limit=5,
        train_limit=32,
        max_new_tokens=None,
        reasoner_max_new_tokens=None,
        torch_dtype=None,
        device_map=None,
        generated_trajectory_adapter_input_space="raw",
        include_tests=True,
        prepare=True,
        replay=True,
    )

    commands = build_commands(args)

    assert commands[0] == ["venv/bin/python", "-B", "-m", "pytest", "-q"]
    assert any("--prepare-generated-trajectory-eval-traces" in command for command in commands)
    assert any("--prepare-generated-trajectory-adapter" in command for command in commands)
    benchmark_command = commands[-2]
    replay_command = commands[-1]
    assert "--generated-trajectory-adapter-no-train-on-missing" in benchmark_command
    assert "--write-eval-manifest" in benchmark_command
    assert "--eval-manifest" in replay_command
    assert "outputs/prod/production_context_vs_latent_5_manifest.json" in replay_command


def test_production_validation_long_context_profile_uses_local_dataset() -> None:
    defaults = PROFILE_DEFAULTS["long_context_mps"]
    args = Namespace(
        python="venv/bin/python",
        python_flags="-B",
        output_dir="outputs/prod",
        dataset=defaults["dataset"],
        eval_limit=defaults["eval_limit"],
        train_limit=defaults["train_limit"],
        max_new_tokens=defaults["max_new_tokens"],
        reasoner_max_new_tokens=defaults["reasoner_max_new_tokens"],
        torch_dtype=defaults["torch_dtype"],
        device_map=defaults["device_map"],
        generated_trajectory_adapter_input_space="raw",
        include_tests=False,
        prepare=False,
        replay=True,
    )

    benchmark_command, replay_command = build_commands(args)

    assert "long_context_handoff" in benchmark_command
    assert "--torch-dtype" in benchmark_command
    assert "float32" in benchmark_command
    assert "--device-map" in benchmark_command
    assert "mps" in benchmark_command
    assert "--max-new-tokens" in benchmark_command
    assert "--reasoner-max-new-tokens" in benchmark_command
    assert "outputs/prod/production_context_vs_latent_3_manifest.json" in replay_command
