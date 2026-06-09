"""Small inference wrapper for the SimpleLLM engine."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from .checkpoint import load_checkpoint
from .config import ModelConfig
from .generation import generate as generate_text
from .model import ModernLLM
from .tokenizer import build_tokenizer


class SimpleLLMInference:
    """Load tokenizer, model, optional checkpoint, and expose generation."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config: ModelConfig | Mapping | None = None,
        tokenizer_name: str = "gpt2",
        device: str | torch.device | None = None,
        strict: bool = True,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else None
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = (
            load_checkpoint(checkpoint_path, map_location=self.device)
            if checkpoint_path is not None
            else None
        )
        self.checkpoint_loaded = checkpoint is not None

        if checkpoint is not None and checkpoint.tokenizer_name and tokenizer_name == "gpt2":
            tokenizer_name = checkpoint.tokenizer_name
        self.tokenizer_name = tokenizer_name
        self.tokenizer = build_tokenizer(tokenizer_name)

        if config is None:
            self.config = checkpoint.config if checkpoint and checkpoint.config else ModelConfig()
        elif isinstance(config, ModelConfig):
            self.config = config
        else:
            self.config = ModelConfig.from_dict(config)

        tokenizer_vocab_size = getattr(self.tokenizer, "vocab_size", None)
        if tokenizer_vocab_size is not None:
            self.config.vocab_size = int(tokenizer_vocab_size)

        self.model = ModernLLM(self.config).to(self.device)
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint.model_state_dict, strict=strict)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> str:
        return generate_text(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=self.device,
        )
