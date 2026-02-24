"""Completions API route — POST /v1/completions.

Implements both streaming (SSE) and non-streaming text completion,
compatible with the OpenAI Completions API specification.
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
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamChunk,
    CompletionStreamChunkChoice,
    UsageInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Completions"])


def _to_sampling_params(req: CompletionRequest) -> SamplingParams:
    """Convert OpenAI-style request fields to internal SamplingParams."""
    return SamplingParams(
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        max_new_tokens=req.max_tokens,
        repetition_penalty=req.repetition_penalty,
    )


@router.post("/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Create a text completion (streaming or non-streaming)."""
    engine = raw_request.app.state.engine

    # Normalize prompt to string
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    params = _to_sampling_params(request)

    # Unique completion ID for this request
    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model_name = request.model

    if request.stream:
        return StreamingResponse(
            _stream_completion(engine, prompt, params, completion_id, created, model_name, raw_request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
    else:
        return await _non_streaming_completion(engine, prompt, params, completion_id, created, model_name)


async def _stream_completion(
    engine,
    prompt: str,
    params: SamplingParams,
    completion_id: str,
    created: int,
    model_name: str,
    raw_request: Request,
) -> AsyncGenerator[str, None]:
    """Yield SSE chunks as tokens are generated."""
    try:
        async for token_text in engine.generate(prompt, params):
            # Check if client disconnected
            if await raw_request.is_disconnected():
                logger.info(f"Client disconnected mid-stream: {completion_id}")
                break

            chunk = CompletionStreamChunk(
                id=completion_id,
                created=created,
                model=model_name,
                choices=[CompletionStreamChunkChoice(text=token_text)],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Send the final chunk with finish_reason
        final_chunk = CompletionStreamChunk(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[CompletionStreamChunkChoice(text="", finish_reason="stop")],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        error_data = json.dumps({"error": {"message": str(e), "type": "server_error"}})
        yield f"data: {error_data}\n\n"


async def _non_streaming_completion(
    engine,
    prompt: str,
    params: SamplingParams,
    completion_id: str,
    created: int,
    model_name: str,
) -> CompletionResponse:
    """Collect all tokens and return a single response."""
    generated_text = ""
    token_count = 0

    async for token_text in engine.generate(prompt, params):
        generated_text += token_text
        token_count += 1

    # Estimate prompt tokens (rough — tokenizer access would be more precise)
    prompt_tokens = len(prompt.split())  # Approximate

    return CompletionResponse(
        id=completion_id,
        created=created,
        model=model_name,
        choices=[
            CompletionChoice(
                text=generated_text,
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=token_count,
            total_tokens=prompt_tokens + token_count,
        ),
    )
