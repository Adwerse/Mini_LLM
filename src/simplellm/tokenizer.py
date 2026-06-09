"""Tokenizer helpers for SimpleLLM."""

from __future__ import annotations


def build_tokenizer(name: str = "gpt2"):
    """Build the Hugging Face tokenizer used by the original notebook."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, model_max_length=1e9)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
