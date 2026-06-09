"""Request and response schemas for the SimpleLLM API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_type: str
    device: str
    checkpoint_loaded: bool
    parameters: str
    max_seq_len: int
    tokenizer: str


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt text to continue.")
    max_new_tokens: int = Field(default=50, ge=1, le=256)
    temperature: float = Field(default=0.8, gt=0.0, le=5.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)


class GenerateModelInfo(BaseModel):
    device: str
    checkpoint: str | None


class GenerateResponse(BaseModel):
    generated_text: str
    model_info: GenerateModelInfo
