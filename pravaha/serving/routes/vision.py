"""Vision route — POST /v1/vision/complete."""

from __future__ import annotations
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import Optional

router = APIRouter(tags=["Vision"])


class VisionContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[dict] = None


class VisionMessage(BaseModel):
    role: str
    content: list[VisionContent] | str


class VisionRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model: str = "llava-1.5-7b"
    messages: list[VisionMessage]
    stream: bool = False
    max_tokens: int = 512


@router.post("/vision/complete")
async def vision_complete(request: VisionRequest, raw_request: Request):
    """Process a multimodal vision request."""
    engine = raw_request.app.state.engine
    text_parts, image_data = [], None

    for msg in request.messages:
        if isinstance(msg.content, str):
            text_parts.append(msg.content)
        elif isinstance(msg.content, list):
            for c in msg.content:
                if c.type == "text" and c.text:
                    text_parts.append(c.text)
                elif c.type == "image_url" and c.image_url:
                    image_data = c.image_url.get("url", "")

    prompt = "\n".join(text_parts)
    if image_data:
        prompt = f"[IMAGE: {image_data[:50]}...]\n{prompt}"

    from pravaha.decoder.sampling import SamplingParams
    params = SamplingParams(max_new_tokens=request.max_tokens)
    output = ""
    async for token in engine.generate(prompt, params):
        output += token

    return {"model": request.model, "content": output, "has_image": image_data is not None}
