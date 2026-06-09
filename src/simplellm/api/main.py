"""FastAPI routes for serving the educational SimpleLLM engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, status

from simplellm.config import ModelConfig
from simplellm.inference import SimpleLLMInference

from .schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
)


MODEL_TYPE = "Mini Transformer Language Model"
MAX_NEW_TOKENS_LIMIT = 256


class ApiState:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        tokenizer_name: str = "gpt2",
        device: str | torch.device | None = None,
        strict: bool = True,
        inference_service: Any | None = None,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.tokenizer_name = tokenizer_name
        self.device = str(device) if device is not None else None
        self.strict = strict
        self.inference_service = inference_service

    def get_inference(self, require_checkpoint: bool = True) -> Any | None:
        if self.inference_service is not None:
            return self.inference_service

        if self.checkpoint_path is None:
            if require_checkpoint:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        "Checkpoint is required for generation. Start the API with "
                        "--checkpoint or set SIMPLELLM_CHECKPOINT."
                    ),
                )
            return None

        checkpoint_file = Path(self.checkpoint_path)
        if not checkpoint_file.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Checkpoint not found: {checkpoint_file}",
            )

        try:
            self.inference_service = SimpleLLMInference(
                checkpoint_path=checkpoint_file,
                tokenizer_name=self.tokenizer_name,
                device=self.device,
                strict=self.strict,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Checkpoint not found: {checkpoint_file}",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model checkpoint: {exc}",
            ) from exc

        return self.inference_service


def _format_parameter_count(service: Any | None) -> str:
    if service is None or not hasattr(service, "model"):
        return "not loaded"
    total = sum(parameter.numel() for parameter in service.model.parameters())
    if total >= 1_000_000:
        return f"{total / 1_000_000:.1f}M"
    if total >= 1_000:
        return f"{total / 1_000:.1f}K"
    return str(total)


def _service_config(service: Any | None) -> ModelConfig:
    config = getattr(service, "config", None)
    return config if isinstance(config, ModelConfig) else ModelConfig()


def _service_device(service: Any | None, fallback: str | None = None) -> str:
    device = getattr(service, "device", None)
    if device is not None:
        return str(device)
    if fallback is not None:
        return fallback
    return "cuda" if torch.cuda.is_available() else "cpu"


def _service_tokenizer_name(service: Any | None, fallback: str) -> str:
    return str(getattr(service, "tokenizer_name", fallback))


def _checkpoint_loaded(service: Any | None) -> bool:
    return bool(getattr(service, "checkpoint_loaded", False))


def _checkpoint_path(service: Any | None, fallback: str | None) -> str | None:
    return getattr(service, "checkpoint_path", None) or fallback


def _prompt_token_count(service: Any, prompt: str) -> int:
    tokenizer = getattr(service, "tokenizer", None)
    if tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference service has no tokenizer.",
        )

    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    if hasattr(encoded, "numel"):
        return int(encoded.numel())
    return len(encoded)


def _validate_generation_request(request: GenerateRequest, service: Any) -> None:
    if not request.prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="prompt must not be empty",
        )

    config = _service_config(service)
    max_seq_len = int(config.max_seq_len)

    if request.max_new_tokens > MAX_NEW_TOKENS_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"max_new_tokens must be <= {MAX_NEW_TOKENS_LIMIT}",
        )
    if request.max_new_tokens > max_seq_len:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"max_new_tokens cannot exceed model max_seq_len ({max_seq_len})",
        )

    prompt_tokens = _prompt_token_count(service, request.prompt)
    if prompt_tokens == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="prompt produced no tokens",
        )
    if prompt_tokens + request.max_new_tokens > max_seq_len:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "context length exceeded: prompt tokens plus max_new_tokens "
                f"must be <= {max_seq_len}"
            ),
        )


def create_app(
    checkpoint_path: str | Path | None = None,
    tokenizer_name: str = "gpt2",
    device: str | torch.device | None = None,
    strict: bool = True,
    inference_service: Any | None = None,
) -> FastAPI:
    api_state = ApiState(
        checkpoint_path=checkpoint_path,
        tokenizer_name=tokenizer_name,
        device=device,
        strict=strict,
        inference_service=inference_service,
    )

    api = FastAPI(
        title="SimpleLLM API",
        description="HTTP API for an educational Mini Transformer Language Model.",
        version="0.1.0",
    )
    api.state.simplellm = api_state

    @api.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @api.get("/model/info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        service = api_state.get_inference(require_checkpoint=False)
        config = _service_config(service)
        return ModelInfoResponse(
            model_type=MODEL_TYPE,
            device=_service_device(service, api_state.device),
            checkpoint_loaded=_checkpoint_loaded(service),
            parameters=_format_parameter_count(service),
            max_seq_len=int(config.max_seq_len),
            tokenizer=_service_tokenizer_name(service, api_state.tokenizer_name),
        )

    @api.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        service = api_state.get_inference(require_checkpoint=True)
        if not _checkpoint_loaded(service):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Checkpoint is required for generation.",
            )

        _validate_generation_request(request, service)

        try:
            generated_text = service.generate(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        return GenerateResponse(
            generated_text=generated_text,
            model_info={
                "device": _service_device(service, api_state.device),
                "checkpoint": _checkpoint_path(service, api_state.checkpoint_path),
            },
        )

    return api


app = create_app(
    checkpoint_path=os.getenv("SIMPLELLM_CHECKPOINT"),
    tokenizer_name=os.getenv("SIMPLELLM_TOKENIZER", "gpt2"),
    device=os.getenv("SIMPLELLM_DEVICE"),
)
