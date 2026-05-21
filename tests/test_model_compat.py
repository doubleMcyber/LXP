from __future__ import annotations

from omegaconf import OmegaConf

from src.utils.model_compat import (
    evaluate_model_pair_compatibility,
    summarize_model_config,
)


def _qwen35_config(*, hidden_size: int):
    return OmegaConf.create(
        {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": hidden_size,
                "num_hidden_layers": 24,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                ]
                * 6,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "vocab_size": 248320,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_num_value_heads": 16,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4,
            },
        }
    )


def _qwen3_config(*, hidden_size: int):
    return OmegaConf.create(
        {
            "model_type": "qwen3",
            "hidden_size": hidden_size,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
        }
    )


def _exaone_config():
    return OmegaConf.create(
        {
            "model_type": "exaone4",
            "hidden_size": 2048,
            "num_hidden_layers": 30,
            "layer_types": ["full_attention"] * 30,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 102400,
        }
    )


def test_qwen35_to_exaone_preflight_reports_layer_mismatch() -> None:
    sender = summarize_model_config(_qwen35_config(hidden_size=1024), model_id="Qwen/Qwen3.5-0.8B")
    receiver = summarize_model_config(_exaone_config(), model_id="LGAI-EXAONE/EXAONE-4.0-1.2B")

    report = evaluate_model_pair_compatibility(sender, receiver)

    assert report.kv_cache_compatible is False
    assert report.status == "unsupported_architecture_mismatch"
    assert "num_hidden_layers_mismatch" in report.reason


def test_qwen35_2b_to_qwen35_08b_preflight_reports_compatible_cache_topology() -> None:
    sender = summarize_model_config(_qwen35_config(hidden_size=2048), model_id="Qwen/Qwen3.5-2B")
    receiver = summarize_model_config(_qwen35_config(hidden_size=1024), model_id="Qwen/Qwen3.5-0.8B")

    report = evaluate_model_pair_compatibility(sender, receiver)

    assert report.kv_cache_compatible is True
    assert report.status == "predicted_compatible"
    assert report.reason == "matching_cache_topology"
    assert report.warnings


def test_qwen3_17b_to_qwen3_06b_preflight_reports_compatible_cache_topology() -> None:
    sender = summarize_model_config(_qwen3_config(hidden_size=2048), model_id="Qwen/Qwen3-1.7B")
    receiver = summarize_model_config(_qwen3_config(hidden_size=1024), model_id="Qwen/Qwen3-0.6B")

    report = evaluate_model_pair_compatibility(sender, receiver)

    assert report.kv_cache_compatible is True
    assert report.status == "predicted_compatible"
    assert report.reason == "matching_cache_topology"
