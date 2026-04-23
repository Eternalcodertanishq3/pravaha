"""Serving Schemas — OpenAI-compatible API request/response models."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: str | list[str] = ""
    max_tokens: int = Field(default=256, ge=1, le=32768)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0)
    stream: bool = False
    session_id: str | None = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: Literal["stop", "length"] | None = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:24]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[CompletionChoice] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str | list[Any] = ""


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int = Field(default=256, ge=1, le=32768)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0)
    stream: bool = False
    session_id: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage = Field(default_factory=lambda: ChatMessage(role="assistant", content=""))
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "pravaha"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)
