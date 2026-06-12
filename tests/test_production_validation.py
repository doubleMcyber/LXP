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
        methods=None,
        generated_trajectory_adapter_input_space="raw",
        generated_trajectory_adapter_train_split=None,
        generated_trajectory_adapter_source_mode=None,
        generated_trajectory_adapter_source_tail_tokens=None,
        generated_trajectory_adapter_target_mode=None,
        generated_trajectory_adapter_target_alignment=None,
        generated_trajectory_local_residual_temperature=None,
        generated_trajectory_semantic_memory_enabled=False,
        generated_trajectory_semantic_memory_min_similarity=None,
        generated_trajectory_semantic_memory_max_entries=None,
        generated_trajectory_token_readout_enabled=False,
        generated_trajectory_token_readout_min_similarity=None,
        include_tests=True,
        prepare=True,
        replay=True,
    )

    commands = build_commands(args)

    assert commands[0] == ["venv/bin/python", "-B", "-m", "pytest", "-q"]
    assert any("--prepare-generated-trajectory-eval-traces" in command for command in commands)
    prepare_adapter_command = next(
        command for command in commands if "--prepare-generated-trajectory-adapter" in command
    )
    assert "--generated-trajectory-adapter-train-on-missing" in prepare_adapter_command
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
        methods=defaults["methods"],
        generated_trajectory_adapter_input_space="raw",
        generated_trajectory_adapter_train_split=defaults[
            "generated_trajectory_adapter_train_split"
        ],
        generated_trajectory_adapter_source_mode=defaults[
            "generated_trajectory_adapter_source_mode"
        ],
        generated_trajectory_adapter_source_tail_tokens=defaults[
            "generated_trajectory_adapter_source_tail_tokens"
        ],
        generated_trajectory_adapter_target_mode=defaults[
            "generated_trajectory_adapter_target_mode"
        ],
        generated_trajectory_adapter_target_alignment=defaults[
            "generated_trajectory_adapter_target_alignment"
        ],
        generated_trajectory_adapter_strategy=defaults[
            "generated_trajectory_adapter_strategy"
        ],
        generated_trajectory_local_residual_enabled=defaults[
            "generated_trajectory_local_residual_enabled"
        ],
        generated_trajectory_local_residual_temperature=defaults[
            "generated_trajectory_local_residual_temperature"
        ],
        generated_trajectory_semantic_memory_enabled=defaults[
            "generated_trajectory_semantic_memory_enabled"
        ],
        generated_trajectory_semantic_memory_min_similarity=defaults[
            "generated_trajectory_semantic_memory_min_similarity"
        ],
        generated_trajectory_semantic_memory_max_entries=defaults[
            "generated_trajectory_semantic_memory_max_entries"
        ],
        generated_trajectory_token_readout_enabled=defaults[
            "generated_trajectory_token_readout_enabled"
        ],
        generated_trajectory_token_readout_min_similarity=defaults[
            "generated_trajectory_token_readout_min_similarity"
        ],
        include_tests=False,
        prepare=False,
        replay=True,
    )

    benchmark_command, replay_command = build_commands(args)
    benchmark_text = " ".join(benchmark_command)
    replay_text = " ".join(replay_command)

    assert "long_context_handoff" in benchmark_command
    assert "--torch-dtype" in benchmark_command
    assert "float32" in benchmark_command
    assert "--device-map" in benchmark_command
    assert "mps" in benchmark_command
    assert "--max-new-tokens" in benchmark_command
    assert "--reasoner-max-new-tokens" in benchmark_command
    assert "--methods" in benchmark_command
    assert "generated_latent_handoff" in benchmark_text
    assert "generated_context_latent_handoff" not in benchmark_text
    assert "--generated-trajectory-adapter-train-split" in benchmark_command
    assert "train" in benchmark_command
    assert "--generated-trajectory-adapter-source-mode" in benchmark_command
    assert "final_answer_tail" in benchmark_command
    assert "--generated-trajectory-adapter-target-mode" in benchmark_command
    assert "final_answer_line" in benchmark_command
    assert "--generated-trajectory-adapter-target-alignment" in benchmark_command
    assert "tail_tokens" in benchmark_command
    assert "--generated-trajectory-adapter-strategy" in benchmark_command
    assert "per_step_ridge" in benchmark_command
    assert "--disable-generated-trajectory-local-residual" in benchmark_command
    assert "--generated-trajectory-local-residual-temperature" in benchmark_command
    assert "0.05" in benchmark_command
    assert "--enable-generated-trajectory-semantic-memory" not in benchmark_command
    assert "--generated-trajectory-semantic-memory-min-similarity" in benchmark_command
    assert "0.98" in benchmark_command
    assert "--generated-trajectory-semantic-memory-max-entries" in benchmark_command
    assert "2048" in benchmark_command
    assert "--enable-generated-trajectory-token-readout" in benchmark_command
    assert "--generated-trajectory-token-readout-min-similarity" in benchmark_command
    assert "0.8" in benchmark_command
    assert "outputs/prod/production_context_vs_latent_8_manifest.json" in replay_command
    assert "--generated-trajectory-adapter-train-limit" in replay_command
    assert "128" in replay_command
    assert "--generated-trajectory-adapter-train-split" in replay_command
    assert "final_answer_tail" in replay_text
    assert "final_answer_line" in replay_text
    assert "per_step_ridge" in replay_text
