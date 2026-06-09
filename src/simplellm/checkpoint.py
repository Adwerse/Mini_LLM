"""Checkpoint helpers for old and new SimpleLLM checkpoint formats."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import ModelConfig


@dataclass
class CheckpointLoadResult:
    format: str
    model_state_dict: Mapping[str, Any]
    config: ModelConfig | None = None
    config_dict: dict[str, Any] | None = None
    tokenizer_name: str | None = None
    step: int | None = None
    optimizer_state_dict: Mapping[str, Any] | None = None
    scheduler_state_dict: Mapping[str, Any] | None = None
    scaler_state_dict: Mapping[str, Any] | None = None
    extra: Mapping[str, Any] | None = None
    raw: Any | None = None


def _config_to_dict(config: ModelConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, ModelConfig):
        return config.to_dict()
    if is_dataclass(config):
        return asdict(config)
    return dict(config)


def _load_torch_file(path: str | Path, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _is_raw_state_dict(obj: Any) -> bool:
    return isinstance(obj, Mapping) and bool(obj) and all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in obj.items()
    )


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: ModelConfig | Mapping[str, Any],
    tokenizer_name: str = "gpt2",
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    step: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Save a structured checkpoint for future training/inference code."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format": "simplellm_checkpoint",
        "version": 1,
        "model_state_dict": model.state_dict(),
        "config": _config_to_dict(config),
        "tokenizer_name": tokenizer_name,
        "step": step,
        "extra": dict(extra or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()

    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> CheckpointLoadResult:
    """Load old raw state_dict checkpoints or new structured checkpoints."""

    raw = _load_torch_file(path, map_location=map_location)

    if isinstance(raw, Mapping) and "model_state_dict" in raw:
        config_dict = raw.get("config")
        config = ModelConfig.from_dict(config_dict) if isinstance(config_dict, Mapping) else None
        result = CheckpointLoadResult(
            format="structured",
            model_state_dict=raw["model_state_dict"],
            config=config,
            config_dict=dict(config_dict) if isinstance(config_dict, Mapping) else None,
            tokenizer_name=raw.get("tokenizer_name"),
            step=raw.get("step"),
            optimizer_state_dict=raw.get("optimizer_state_dict"),
            scheduler_state_dict=raw.get("scheduler_state_dict"),
            scaler_state_dict=raw.get("scaler_state_dict"),
            extra=raw.get("extra"),
            raw=raw,
        )
    elif _is_raw_state_dict(raw):
        result = CheckpointLoadResult(
            format="state_dict",
            model_state_dict=raw,
            raw=raw,
        )
    else:
        raise ValueError(f"Unsupported checkpoint format: {path}")

    if model is not None:
        model.load_state_dict(result.model_state_dict, strict=strict)

    return result
