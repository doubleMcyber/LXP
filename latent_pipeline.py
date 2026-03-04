from __future__ import annotations

import inspect
import time
from types import MethodType
from typing import Any, Optional

import torch
from torch import nn
from torchdiffeq import odeint_adjoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from procrustes_alignment import apply_orthogonal_mapping, compute_orthogonal_mapping

AGENT_A_MODEL_NAME = "Qwen/Qwen3.5-0.8B"
AGENT_B_MODEL_NAME = "Qwen/Qwen3.5-0.8B"
DEFAULT_PROMPT = "Explain the concept of entropy"
LATENT_STEPS = 10
MAX_NEW_TOKENS = 50
SIMULATED_CONTINUOUS_STEPS = 50

_PIPELINE_STATE: Optional[dict[str, Any]] = None


def load_agent(model_name: str) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


def _normalize_kv_cache(past_key_values: Any) -> tuple:
    if past_key_values is None:
        return tuple()
    if hasattr(past_key_values, "to_legacy_cache"):
        return tuple(past_key_values.to_legacy_cache())
    return tuple(past_key_values)


def _build_position_ids(attention_mask: torch.Tensor) -> torch.LongTensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.clamp_min_(0)


def _kv_cache_seq_len(kv_cache: tuple) -> int:
    if not kv_cache:
        return 0
    first_layer = kv_cache[0]
    key_tensor = first_layer[0]
    return int(key_tensor.shape[-2])


def _move_kv_cache_to_device(kv_cache: tuple, device: torch.device) -> tuple:
    moved_layers = []
    for layer_cache in kv_cache:
        moved_layer = tuple(
            tensor.to(device) if torch.is_tensor(tensor) else tensor
            for tensor in layer_cache
        )
        moved_layers.append(moved_layer)
    return tuple(moved_layers)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class TransformerBlockDynamics(nn.Module):
    def __init__(self, transformer_block: nn.Module) -> None:
        super().__init__()
        self.transformer_block = transformer_block
        self._accepted_args = set(inspect.signature(transformer_block.forward).parameters.keys())
        self._attention_mask: Optional[torch.Tensor] = None
        self._position_ids: Optional[torch.LongTensor] = None
        self._cache_position: Optional[torch.LongTensor] = None

    def set_context(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> None:
        self._attention_mask = attention_mask
        self._position_ids = position_ids
        self._cache_position = cache_position

    def _extract_hidden_states(self, block_output: Any) -> torch.Tensor:
        if isinstance(block_output, tuple):
            return block_output[0]
        if hasattr(block_output, "hidden_states"):
            return block_output.hidden_states
        if hasattr(block_output, "last_hidden_state"):
            return block_output.last_hidden_state
        if torch.is_tensor(block_output):
            return block_output
        raise TypeError("Unsupported transformer block output type for ODE dynamics")

    def forward(self, t: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        del t
        kwargs: dict[str, Any] = {}
        accepts_hidden_kw = "hidden_states" in self._accepted_args
        if accepts_hidden_kw:
            kwargs["hidden_states"] = hidden_states
        if "attention_mask" in self._accepted_args and self._attention_mask is not None:
            kwargs["attention_mask"] = self._attention_mask
        if "position_ids" in self._accepted_args and self._position_ids is not None:
            kwargs["position_ids"] = self._position_ids
        if "cache_position" in self._accepted_args and self._cache_position is not None:
            kwargs["cache_position"] = self._cache_position
        if "past_key_value" in self._accepted_args:
            kwargs["past_key_value"] = None
        if "past_key_values" in self._accepted_args:
            kwargs["past_key_values"] = None
        if "use_cache" in self._accepted_args:
            kwargs["use_cache"] = False
        if "output_attentions" in self._accepted_args:
            kwargs["output_attentions"] = False
        if "output_hidden_states" in self._accepted_args:
            kwargs["output_hidden_states"] = False

        if accepts_hidden_kw:
            block_output = self.transformer_block(**kwargs)
        else:
            block_output = self.transformer_block(hidden_states, **kwargs)
        transformed = self._extract_hidden_states(block_output)
        return transformed - hidden_states


def attach_latent_forward(agent_model: AutoModelForCausalLM) -> None:
    def latent_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, tuple]:
        del return_dict
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = model_outputs.last_hidden_state
        past_key_values = _normalize_kv_cache(model_outputs.past_key_values)
        return hidden_states, past_key_values

    agent_model.forward = MethodType(latent_forward, agent_model)


def _get_pipeline_state() -> dict[str, Any]:
    global _PIPELINE_STATE
    if _PIPELINE_STATE is not None:
        return _PIPELINE_STATE

    tokenizer_a = AutoTokenizer.from_pretrained(AGENT_A_MODEL_NAME)
    tokenizer_b = AutoTokenizer.from_pretrained(AGENT_B_MODEL_NAME)
    agent_a = load_agent(AGENT_A_MODEL_NAME)
    agent_b = load_agent(AGENT_B_MODEL_NAME)
    attach_latent_forward(agent_a)

    _PIPELINE_STATE = {
        "tokenizer_a": tokenizer_a,
        "tokenizer_b": tokenizer_b,
        "agent_a": agent_a,
        "agent_b": agent_b,
    }
    return _PIPELINE_STATE


def initialize_hybrid_pipeline() -> None:
    _get_pipeline_state()


def run_hybrid_pipeline(prompt: str = DEFAULT_PROMPT) -> dict[str, Any]:
    state = _get_pipeline_state()
    tokenizer_a = state["tokenizer_a"]
    tokenizer_b = state["tokenizer_b"]
    agent_a = state["agent_a"]
    agent_b = state["agent_b"]

    encoded = tokenizer_a(prompt, return_tensors="pt")
    agent_a_device = next(agent_a.parameters()).device
    input_ids_a = encoded["input_ids"].to(agent_a_device)
    attention_mask_a = encoded["attention_mask"].to(agent_a_device)
    position_ids_a = _build_position_ids(attention_mask_a)

    with torch.no_grad():
        hidden_states, kv_cache_a = agent_a(
            input_ids=input_ids_a,
            attention_mask=attention_mask_a,
            position_ids=position_ids_a,
            use_cache=True,
        )

    agent_b_device = next(agent_b.parameters()).device
    encoded_b = tokenizer_b(prompt, return_tensors="pt")
    input_ids_b = encoded_b["input_ids"].to(agent_b_device)
    attention_mask_b_anchor = encoded_b["attention_mask"].to(agent_b_device)
    position_ids_b_anchor = _build_position_ids(attention_mask_b_anchor)

    with torch.no_grad():
        agent_b_anchor_hidden = agent_b.model(
            input_ids=input_ids_b,
            attention_mask=attention_mask_b_anchor,
            position_ids=position_ids_b_anchor,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

    procrustes_q = compute_orthogonal_mapping(
        hidden_states, agent_b_anchor_hidden
    )
    procrustes_q = procrustes_q.to(device=hidden_states.device, dtype=hidden_states.dtype)
    current_latent_step = hidden_states[:, -1:, :]
    continuous_position_ids = position_ids_a[:, -1:] + 1
    dynamics = TransformerBlockDynamics(agent_a.model.layers[0])
    dynamics.set_context(position_ids=continuous_position_ids)

    time_space = torch.linspace(
        0,
        1,
        LATENT_STEPS,
        device=current_latent_step.device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        continuous_trajectory = odeint_adjoint(
            dynamics,
            current_latent_step,
            time_space,
            method="rk4",
        )
    current_latent_step = continuous_trajectory[-1]

    simulated_time_space = torch.linspace(
        0,
        1,
        SIMULATED_CONTINUOUS_STEPS,
        device=current_latent_step.device,
        dtype=torch.float32,
    )
    _sync_if_cuda(current_latent_step.device)
    integration_start = time.perf_counter()
    with torch.no_grad():
        _ = odeint_adjoint(
            dynamics,
            current_latent_step,
            simulated_time_space,
            method="rk4",
        )
    _sync_if_cuda(current_latent_step.device)
    integration_duration = time.perf_counter() - integration_start

    agent_b_embed_dtype = agent_b.get_input_embeddings().weight.dtype
    handoff_step = apply_orthogonal_mapping(current_latent_step, procrustes_q)
    handoff_step = handoff_step.to(
        device=agent_b_device,
        dtype=agent_b_embed_dtype,
    )
    kv_cache_b = _move_kv_cache_to_device(kv_cache_a, agent_b_device)
    attention_mask_b = torch.ones(
        (handoff_step.shape[0], _kv_cache_seq_len(kv_cache_b) + 1),
        dtype=torch.long,
        device=agent_b_device,
    )
    position_ids_b = _build_position_ids(attention_mask_b)[:, -1:]

    with torch.no_grad():
        outputs_b = agent_b(
            inputs_embeds=handoff_step,
            past_key_values=kv_cache_b,
            attention_mask=attention_mask_b,
            position_ids=position_ids_b,
            use_cache=True,
            return_dict=True,
        )

    generated_token_ids: list[int] = []
    eos_token_id = tokenizer_b.eos_token_id

    for _ in range(MAX_NEW_TOKENS):
        next_token = torch.argmax(outputs_b.logits[:, -1, :], dim=-1)
        next_token_id = int(next_token.item())
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        generated_token_ids.append(next_token_id)

        kv_cache_b = _normalize_kv_cache(outputs_b.past_key_values)
        attention_mask_b = torch.cat(
            [
                attention_mask_b,
                torch.ones(
                    (attention_mask_b.shape[0], 1),
                    dtype=attention_mask_b.dtype,
                    device=attention_mask_b.device,
                ),
            ],
            dim=1,
        )
        position_ids_b = _build_position_ids(attention_mask_b)[:, -1:]

        with torch.no_grad():
            outputs_b = agent_b(
                input_ids=next_token.unsqueeze(-1),
                past_key_values=kv_cache_b,
                attention_mask=attention_mask_b,
                position_ids=position_ids_b,
                use_cache=True,
                return_dict=True,
            )

    decoded_text = tokenizer_b.decode(generated_token_ids, skip_special_tokens=True)
    return {
        "decoded_text": decoded_text,
        "final_latent_shape": tuple(current_latent_step.shape),
        "final_latent_dtype": str(current_latent_step.dtype),
        "procrustes_q_shape": tuple(procrustes_q.shape),
        "kv_cache_length": len(kv_cache_a),
        "continuous_integration_50_steps_seconds": integration_duration,
    }


def main() -> None:
    outputs = run_hybrid_pipeline(DEFAULT_PROMPT)
    print(f"Final latent step shape: {outputs['final_latent_shape']}")
    print(f"Final latent step dtype: {outputs['final_latent_dtype']}")
    print(f"Procrustes Q shape: {outputs['procrustes_q_shape']}")
    print(f"KV cache tuple length: {outputs['kv_cache_length']}")
    print(
        "Continuous integration time (50 steps): "
        f"{outputs['continuous_integration_50_steps_seconds']:.4f} seconds"
    )
    print(f"Agent B decoded text: {outputs['decoded_text']}")


if __name__ == "__main__":
    main()
