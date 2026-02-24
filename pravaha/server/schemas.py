"""OpenAI-compatible API schemas for Pravāha Server.

Pydantic models mirroring the OpenAI API specification for:
- Completions (/v1/completions)
- Chat Completions (/v1/chat/completions)
- Model listing (/v1/models)
- Error responses
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Common
# ─────────────────────────────────────────────

class UsageInfo(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ErrorResponse(BaseModel):
    """Standardized error format."""
    object: str = "error"
    message: str
    type: str = "invalid_request_error"
    code: Optional[str] = None


# ─────────────────────────────────────────────
# Completions API
# ─────────────────────────────────────────────

class CompletionRequest(BaseModel):
    """POST /v1/completions request body."""
    model: str
    prompt: Union[str, list[str]]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0)
    stream: bool = False
    stop: Optional[Union[str, list[str]]] = None
    n: int = Field(default=1, ge=1, le=1)  # Only n=1 supported for now

    # Pravāha-specific extensions (like vLLM's extra_body)
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class CompletionChoice(BaseModel):
    """A single completion choice."""
    index: int = 0
    text: str = ""
    finish_reason: Optional[Literal["stop", "length"]] = None
    logprobs: Optional[Any] = None


class CompletionResponse(BaseModel):
    """Non-streaming completion response."""
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:24]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


class CompletionStreamChunkChoice(BaseModel):
    """A single choice within a streaming chunk."""
    index: int = 0
    text: str = ""
    finish_reason: Optional[Literal["stop", "length"]] = None
    logprobs: Optional[Any] = None


class CompletionStreamChunk(BaseModel):
    """SSE streaming chunk for completions."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionStreamChunkChoice]


# ─────────────────────────────────────────────
# Chat Completions API
# ─────────────────────────────────────────────

class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions request body."""
    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0)
    stream: bool = False
    stop: Optional[Union[str, list[str]]] = None
    n: int = Field(default=1, ge=1, le=1)

    # Extensions
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class ChatCompletionChoice(BaseModel):
    """A single chat completion choice."""
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    """Non-streaming chat completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class ChatCompletionStreamDelta(BaseModel):
    """Delta content in a streaming chat chunk."""
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    """A single choice within a streaming chat chunk."""
    index: int = 0
    delta: ChatCompletionStreamDelta
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamChunk(BaseModel):
    """SSE streaming chunk for chat completions."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]


# ─────────────────────────────────────────────
# Models API
# ─────────────────────────────────────────────

class ModelPermission(BaseModel):
    """Model permission object."""
    id: str = "modelperm-pravaha"
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = False
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class ModelCard(BaseModel):
    """A single model entry."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "pravaha"
    permission: list[ModelPermission] = Field(default_factory=lambda: [ModelPermission()])


class ModelList(BaseModel):
    """Response for GET /v1/models."""
    object: str = "list"
    data: list[ModelCard]
