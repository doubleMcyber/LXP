from __future__ import annotations

from types import MethodType
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
PROMPT = "Explain the concept of entropy"
RIDGE_LAMBDA = 1e-4
LATENT_STEPS = 10
MAX_NEW_TOKENS = 50


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


def compute_alignment_matrix(
    agent_model: AutoModelForCausalLM, ridge_lambda: float = RIDGE_LAMBDA
) -> torch.Tensor:
    # Stage I ridge closed-form:
    # W_a = (W_out^T W_out + lambda I)^(-1) W_out^T W_in
    w_out = agent_model.get_output_embeddings().weight.detach().to(torch.float32)
    w_in = agent_model.get_input_embeddings().weight.detach().to(torch.float32)
    hidden_dim = w_out.shape[1]
    identity = torch.eye(hidden_dim, device=w_out.device, dtype=torch.float32)
    w_out_t = w_out.transpose(0, 1)
    return torch.linalg.inv(w_out_t @ w_out + ridge_lambda * identity) @ (w_out_t @ w_in)


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


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    agent_a = load_agent(MODEL_NAME)
    agent_b = load_agent(MODEL_NAME)

    attach_latent_forward(agent_a)

    encoded = tokenizer(PROMPT, return_tensors="pt")
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

    alignment_matrix = compute_alignment_matrix(agent_a).to(
        device=hidden_states.device, dtype=hidden_states.dtype
    )
    current_aligned_step = hidden_states[:, -1:, :] @ alignment_matrix

    for _ in range(LATENT_STEPS):
        attention_mask_a = torch.cat(
            [
                attention_mask_a,
                torch.ones(
                    (attention_mask_a.shape[0], 1),
                    dtype=attention_mask_a.dtype,
                    device=attention_mask_a.device,
                ),
            ],
            dim=1,
        )
        position_ids_a = _build_position_ids(attention_mask_a)[:, -1:]

        with torch.no_grad():
            hidden_states, kv_cache_a = agent_a(
                inputs_embeds=current_aligned_step,
                past_key_values=kv_cache_a,
                attention_mask=attention_mask_a,
                position_ids=position_ids_a,
                use_cache=True,
            )
        current_aligned_step = hidden_states[:, -1:, :] @ alignment_matrix

    print(f"Final latent step shape: {tuple(current_aligned_step.shape)}")
    print(f"Final latent step dtype: {current_aligned_step.dtype}")
    print(f"KV cache tuple length: {len(kv_cache_a)}")

    agent_b_device = next(agent_b.parameters()).device
    agent_b_embed_dtype = agent_b.get_input_embeddings().weight.dtype
    handoff_step = current_aligned_step.to(
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
    eos_token_id = tokenizer.eos_token_id

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

    decoded_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    print(f"Agent B decoded text: {decoded_text}")


if __name__ == "__main__":
    main()
