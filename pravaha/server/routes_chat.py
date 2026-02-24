"""Chat Completions API route — POST /v1/chat/completions.

Implements both streaming (SSE) and non-streaming chat completion,
compatible with the OpenAI Chat Completions API specification.
Applies chat templates when available from the tokenizer.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse

from pravaha.decoder.sampling import SamplingParams
from pravaha.server.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChunk,
    ChatCompletionStreamChoice,
    ChatCompletionStreamDelta,
    ChatMessage,
    UsageInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


def _to_sampling_params(req: ChatCompletionRequest) -> SamplingParams:
    """Convert OpenAI-style chat request fields to internal SamplingParams."""
    return SamplingParams(
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        max_new_tokens=req.max_tokens,
        repetition_penalty=req.repetition_penalty,
    )


def _apply_chat_template(engine, messages: list[ChatMessage]) -> str:
    """Convert chat messages to a single prompt string.

    Uses the tokenizer's chat_template if available, otherwise falls
    back to a simple concatenation format.
    """
    tokenizer = engine.tokenizer

    # Try HuggingFace chat template first
    try:
        message_dicts = [{"role": m.role, "content": m.content} for m in messages]
        prompt = tokenizer.tokenizer.apply_chat_template(
            message_dicts,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt
    except Exception:
        pass

    # Fallback: simple concatenation
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}")
    parts.append("Assistant:")
    return "\n".join(parts)


@router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """Create a chat completion (streaming or non-streaming)."""
    engine = raw_request.app.state.engine

    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    prompt = _apply_chat_template(engine, request.messages)
    params = _to_sampling_params(request)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model_name = request.model

    if request.stream:
        return StreamingResponse(
            _stream_chat_completion(engine, prompt, params, completion_id, created, model_name, raw_request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await _non_streaming_chat_completion(engine, prompt, params, completion_id, created, model_name)


async def _stream_chat_completion(
    engine,
    prompt: str,
    params: SamplingParams,
    completion_id: str,
    created: int,
    model_name: str,
    raw_request: Request,
) -> AsyncGenerator[str, None]:
    """Yield SSE chunks with delta objects as tokens are generated."""
    # First chunk: send the role
    first_chunk = ChatCompletionStreamChunk(
        id=completion_id,
        created=created,
        model=model_name,
        choices=[ChatCompletionStreamChoice(
            delta=ChatCompletionStreamDelta(role="assistant", content=""),
        )],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    try:
        async for token_text in engine.generate(prompt, params):
            if await raw_request.is_disconnected():
                logger.info(f"Client disconnected mid-stream: {completion_id}")
                break

            chunk = ChatCompletionStreamChunk(
                id=completion_id,
                created=created,
                model=model_name,
                choices=[ChatCompletionStreamChoice(
                    delta=ChatCompletionStreamDelta(content=token_text),
                )],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Final chunk with finish_reason
        final_chunk = ChatCompletionStreamChunk(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[ChatCompletionStreamChoice(
                delta=ChatCompletionStreamDelta(),
                finish_reason="stop",
            )],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Chat streaming error: {e}", exc_info=True)
        error_data = json.dumps({"error": {"message": str(e), "type": "server_error"}})
        yield f"data: {error_data}\n\n"


async def _non_streaming_chat_completion(
    engine,
    prompt: str,
    params: SamplingParams,
    completion_id: str,
    created: int,
    model_name: str,
) -> ChatCompletionResponse:
    """Collect all tokens and return a single chat response."""
    generated_text = ""
    token_count = 0

    async for token_text in engine.generate(prompt, params):
        generated_text += token_text
        token_count += 1

    prompt_tokens = len(prompt.split())  # Approximate

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=model_name,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=generated_text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=token_count,
            total_tokens=prompt_tokens + token_count,
        ),
    )
