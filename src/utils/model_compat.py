from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf
from transformers import AutoConfig


_CACHE_TOPOLOGY_FIELDS: tuple[str, ...] = (
    "num_hidden_layers",
    "layer_types",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "linear_num_key_heads",
    "linear_key_head_dim",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "full_attention_interval",
)


@dataclass(frozen=True)
class ModelArchitectureSummary:
    model_id: str
    model_type: str
    num_hidden_layers: Optional[int]
    layer_types: tuple[str, ...]
    hidden_size: Optional[int]
    num_attention_heads: Optional[int]
    num_key_value_heads: Optional[int]
    head_dim: Optional[int]
    vocab_size: Optional[int]
    linear_num_key_heads: Optional[int]
    linear_key_head_dim: Optional[int]
    linear_num_value_heads: Optional[int]
    linear_value_head_dim: Optional[int]
    linear_conv_kernel_dim: Optional[int]
    full_attention_interval: Optional[int]


@dataclass(frozen=True)
class ModelPairCompatibility:
    agent_a: ModelArchitectureSummary
    agent_b: ModelArchitectureSummary
    kv_cache_compatible: bool
    status: str
    reason: str
    mismatches: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cfg_get(config: Any, field: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(field, default)
    return getattr(config, field, default)


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_layer_types(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        return ()
    return tuple(str(item) for item in value)


def _text_config(config: Any) -> Any:
    text_config = _cfg_get(config, "text_config")
    return text_config if text_config is not None else config


def summarize_model_config(config: Any, *, model_id: str) -> ModelArchitectureSummary:
    text_config = _text_config(config)
    layer_types = _as_layer_types(_cfg_get(text_config, "layer_types"))
    num_hidden_layers = _as_int(_cfg_get(text_config, "num_hidden_layers"))
    if num_hidden_layers is None and layer_types:
        num_hidden_layers = len(layer_types)

    hidden_size = _as_int(_cfg_get(text_config, "hidden_size"))
    num_attention_heads = _as_int(_cfg_get(text_config, "num_attention_heads"))
    num_key_value_heads = _as_int(_cfg_get(text_config, "num_key_value_heads"))
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads

    head_dim = _as_int(_cfg_get(text_config, "head_dim"))
    if head_dim is None and hidden_size is not None and num_attention_heads:
        head_dim = hidden_size // num_attention_heads

    model_type = str(_cfg_get(text_config, "model_type", _cfg_get(config, "model_type", "")))
    return ModelArchitectureSummary(
        model_id=str(model_id),
        model_type=model_type,
        num_hidden_layers=num_hidden_layers,
        layer_types=layer_types,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        vocab_size=_as_int(_cfg_get(text_config, "vocab_size")),
        linear_num_key_heads=_as_int(_cfg_get(text_config, "linear_num_key_heads")),
        linear_key_head_dim=_as_int(_cfg_get(text_config, "linear_key_head_dim")),
        linear_num_value_heads=_as_int(_cfg_get(text_config, "linear_num_value_heads")),
        linear_value_head_dim=_as_int(_cfg_get(text_config, "linear_value_head_dim")),
        linear_conv_kernel_dim=_as_int(_cfg_get(text_config, "linear_conv_kernel_dim")),
        full_attention_interval=_as_int(_cfg_get(text_config, "full_attention_interval")),
    )


def load_model_architecture_summary(model_id: str) -> ModelArchitectureSummary:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return summarize_model_config(config, model_id=model_id)


def _field_value(summary: ModelArchitectureSummary, field: str) -> Any:
    return getattr(summary, field)


def evaluate_model_pair_compatibility(
    agent_a: ModelArchitectureSummary,
    agent_b: ModelArchitectureSummary,
) -> ModelPairCompatibility:
    mismatches: list[str] = []
    warnings: list[str] = []

    for field in _CACHE_TOPOLOGY_FIELDS:
        left = _field_value(agent_a, field)
        right = _field_value(agent_b, field)
        if field == "layer_types":
            if (left or right) and left != right:
                mismatches.append(
                    "layer_types_mismatch: "
                    f"agent_a={_layer_type_summary(left)}, "
                    f"agent_b={_layer_type_summary(right)}"
                )
            continue
        if left is None or right is None:
            continue
        if left != right:
            mismatches.append(f"{field}_mismatch: agent_a={left}, agent_b={right}")

    if agent_a.hidden_size is not None and agent_b.hidden_size is not None:
        if agent_a.hidden_size != agent_b.hidden_size:
            warnings.append(
                "hidden_size differs; latent/input-embedding handoff still requires alignment "
                f"({agent_a.hidden_size} -> {agent_b.hidden_size})"
            )

    if mismatches:
        return ModelPairCompatibility(
            agent_a=agent_a,
            agent_b=agent_b,
            kv_cache_compatible=False,
            status="unsupported_architecture_mismatch",
            reason="; ".join(mismatches),
            mismatches=tuple(mismatches),
            warnings=tuple(warnings),
        )

    missing_required = [
        field
        for field in ("num_hidden_layers", "num_key_value_heads", "head_dim")
        if _field_value(agent_a, field) is None or _field_value(agent_b, field) is None
    ]
    if missing_required:
        return ModelPairCompatibility(
            agent_a=agent_a,
            agent_b=agent_b,
            kv_cache_compatible=False,
            status="unknown_config_metadata",
            reason="missing required cache metadata: " + ",".join(missing_required),
            mismatches=(),
            warnings=tuple(warnings),
        )

    return ModelPairCompatibility(
        agent_a=agent_a,
        agent_b=agent_b,
        kv_cache_compatible=True,
        status="predicted_compatible",
        reason="matching_cache_topology",
        mismatches=(),
        warnings=tuple(warnings),
    )


def load_model_pair_compatibility(agent_a_model: str, agent_b_model: str) -> ModelPairCompatibility:
    agent_a = load_model_architecture_summary(agent_a_model)
    agent_b = load_model_architecture_summary(agent_b_model)
    return evaluate_model_pair_compatibility(agent_a, agent_b)


def _layer_type_summary(layer_types: Sequence[str]) -> str:
    if not layer_types:
        return "not provided"
    counts = Counter(layer_types)
    ordered_names = []
    for name in layer_types:
        if name not in ordered_names:
            ordered_names.append(name)
    return ", ".join(f"{name}x{counts[name]}" for name in ordered_names)


def _format_model_summary(label: str, summary: ModelArchitectureSummary) -> list[str]:
    return [
        f"{label}: {summary.model_id}",
        f"  model_type: {summary.model_type or 'unknown'}",
        f"  layers: {summary.num_hidden_layers}",
        f"  layer_types: {_layer_type_summary(summary.layer_types)}",
        f"  hidden_size: {summary.hidden_size}",
        f"  attention_heads: {summary.num_attention_heads}",
        f"  kv_heads: {summary.num_key_value_heads}",
        f"  head_dim: {summary.head_dim}",
        f"  vocab_size: {summary.vocab_size}",
        (
            "  linear_attention: "
            f"qk_heads={summary.linear_num_key_heads}, "
            f"qk_dim={summary.linear_key_head_dim}, "
            f"v_heads={summary.linear_num_value_heads}, "
            f"v_dim={summary.linear_value_head_dim}, "
            f"conv_kernel={summary.linear_conv_kernel_dim}, "
            f"full_attention_interval={summary.full_attention_interval}"
        ),
    ]


def format_model_pair_preflight(report: ModelPairCompatibility) -> str:
    lines = ["Model pair preflight"]
    lines.extend(_format_model_summary("Agent A", report.agent_a))
    lines.extend(_format_model_summary("Agent B", report.agent_b))
    lines.extend(
        [
            f"Predicted KV cache compatibility: {report.kv_cache_compatible}",
            f"Status: {report.status}",
            f"Reason: {report.reason}",
        ]
    )
    if report.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in report.warnings)
    return "\n".join(lines)


def _load_agent_ids_from_config(config_path: Path) -> tuple[str, str]:
    cfg = OmegaConf.load(config_path)
    return str(cfg.agent_a_model), str(cfg.agent_b_model)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Preflight model-pair cache compatibility using configs only."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--agent-a-model", default=None)
    parser.add_argument("--agent-b-model", default=None)
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    args = parser.parse_args(argv)

    config_agent_a, config_agent_b = _load_agent_ids_from_config(args.config)
    agent_a_model = str(args.agent_a_model or config_agent_a)
    agent_b_model = str(args.agent_b_model or config_agent_b)
    report = load_model_pair_compatibility(agent_a_model, agent_b_model)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return
    print(format_model_pair_preflight(report))


if __name__ == "__main__":
    main()
