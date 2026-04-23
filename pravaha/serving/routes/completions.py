"""Completions API route — POST /v1/completions."""

from __future__ import annotations
import asyncio
import json
import time
import uuid

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from pravaha.decoder.sampling import SamplingParams
from pravaha.serving.schemas import CompletionChoice, CompletionRequest, CompletionResponse, UsageInfo

router = APIRouter(tags=["Completions"])


@router.post("/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    engine = raw_request.app.state.engine
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    params = SamplingParams(temperature=request.temperature, top_k=request.top_k, top_p=request.top_p, max_new_tokens=request.max_tokens, repetition_penalty=request.repetition_penalty)
    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if request.stream:
        async def stream():
            try:
                async for token in engine.generate(prompt, params, session_id=request.session_id):
                    if await raw_request.is_disconnected():
                        break
                    chunk = {"id": completion_id, "object": "text_completion", "created": created, "model": request.model, "choices": [{"text": token, "index": 0}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield f'data: {json.dumps({"id": completion_id, "object": "text_completion", "created": created, "model": request.model, "choices": [{"text": "", "finish_reason": "stop", "index": 0}]})}\n\n'
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f'data: {json.dumps({"error": {"message": str(e)}})}\n\n'
        return StreamingResponse(stream(), media_type="text/event-stream")

    text = ""
    tokens = 0
    async for token in engine.generate(prompt, params, session_id=request.session_id):
        text += token
        tokens += 1
    prompt_tokens = len(engine.tokenizer.encode(prompt))
    return CompletionResponse(id=completion_id, created=created, model=request.model, choices=[CompletionChoice(text=text, finish_reason="stop")], usage=UsageInfo(prompt_tokens=prompt_tokens, completion_tokens=tokens, total_tokens=prompt_tokens + tokens))
