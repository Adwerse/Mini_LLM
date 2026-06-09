"""Configuration objects for the SimpleLLM engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


@dataclass
class ModelConfig:
    """Architecture configuration matching the original notebook defaults."""

    dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    vocab_size: int = 50257
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        valid_fields = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in valid_fields}
        return cls(**filtered)
