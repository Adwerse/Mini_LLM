"""Text generation utilities for SimpleLLM."""

from __future__ import annotations

from typing import Any

import torch


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _encode_prompt(tokenizer: Any, prompt: str, device: torch.device) -> torch.Tensor:
    encoded = tokenizer.encode(prompt, return_tensors="pt")
    if not torch.is_tensor(encoded):
        encoded = torch.tensor(encoded, dtype=torch.long)
    if encoded.dim() == 1:
        encoded = encoded.unsqueeze(0)
    if encoded.shape[1] == 0:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("Tokenizer produced no tokens and has no eos_token_id")
        encoded = torch.tensor([[eos_token_id]], dtype=torch.long)
    return encoded.to(device)


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: torch.device | str | None = None,
) -> str:
    """Generate text from a prompt using temperature and nucleus sampling."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the range (0, 1]")

    device = torch.device(device) if device is not None else _model_device(model)
    model.eval()

    tokens = _encode_prompt(tokenizer, prompt, device)
    prompt_limit = int(getattr(getattr(model, "args", None), "max_seq_len", tokens.shape[1]))
    rope_limit = int(getattr(getattr(model, "freqs_cis", None), "shape", [prompt_limit])[0])
    prompt_limit = max(1, min(prompt_limit, rope_limit))

    if tokens.shape[1] > prompt_limit:
        tokens = tokens[:, -prompt_limit:]

    logits, kv_caches = model(tokens, use_cache=True)
    curr_logit = logits[:, -1, :]
    result = tokens[0].tolist()

    available_new_tokens = max(0, rope_limit - tokens.shape[1])
    steps = min(max_new_tokens, available_new_tokens)

    for _ in range(steps):
        probabilities = torch.softmax(curr_logit / (temperature + 1e-10), dim=-1)

        sorted_p, sorted_i = torch.sort(probabilities, descending=True)
        cumulative_p = torch.cumsum(sorted_p, dim=-1)

        mask = cumulative_p - sorted_p > top_p
        sorted_p[mask] = 0.0
        sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)

        next_token_idx = torch.multinomial(sorted_p, 1)
        next_token = torch.gather(sorted_i, -1, next_token_idx).item()

        result.append(next_token)
        if next_token == getattr(tokenizer, "eos_token_id", None):
            break

        next_input = torch.tensor([[next_token]], device=device)
        logits, kv_caches = model(next_input, use_cache=True, kv_caches=kv_caches)
        curr_logit = logits[:, -1, :]

    return _decode(tokenizer, result)
