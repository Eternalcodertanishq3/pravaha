"""Vision Engine — Unified vision-language model interface."""

from __future__ import annotations
import logging
from typing import Any, AsyncGenerator, Optional
from pravaha.vision.detector import VisionDetector
from pravaha.vision.preprocessor import VisionPreprocessor

logger = logging.getLogger(__name__)


class VisionEngine:
    """Process multimodal inputs through vision-language models."""

    def __init__(self) -> None:
        self.detector = VisionDetector()
        self.preprocessor = VisionPreprocessor()
        self._vlm = None

    async def generate(self, messages: list[dict], engine: object, params: object) -> AsyncGenerator[str, None]:
        images = self.detector.detect(messages)
        text_parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part["text"])

        prompt = "\n".join(text_parts)
        if images:
            prompt = f"[Image attached: {len(images)} image(s)]\n{prompt}"

        async for token in engine.generate(prompt, params):  # type: ignore
            yield token
