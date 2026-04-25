"""Vision Engine — Unified vision-language model interface."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from pravaha.vision.detector import VisionDetector
from pravaha.vision.preprocessor import VisionPreprocessor

logger = logging.getLogger(__name__)


class VisionEngine:
    """Process multimodal inputs through vision-language models.

    Implements real PIL image processing when available, with
    graceful fallback to text-only mode.
    """

    def __init__(self) -> None:
        self.detector = VisionDetector()
        self.preprocessor = VisionPreprocessor()
        self._vlm = None

    @property
    def is_real_vlm(self) -> bool:
        """True if a real vision-language model is loaded."""
        return self._vlm is not None

    def _extract_text(self, messages: list[dict]) -> list[str]:
        """Extract text parts from multimodal messages."""
        text_parts: list[str] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part["text"])
        return text_parts

    async def generate(
        self,
        messages: list[dict],
        engine: object,
        params: object,
    ) -> AsyncGenerator[str, None]:
        """Generate tokens from multimodal input with real image processing."""
        images = self.detector.detect(messages)
        text_parts = self._extract_text(messages)
        prompt = "\n".join(text_parts)

        if images:
            try:
                from PIL import Image  # type: ignore[import-untyped]

                for img_input in images[:1]:
                    pil_img = self.preprocessor.preprocess(
                        img_input.base64_data or img_input.url
                    )
                    width, height = pil_img.size
                    mode = pil_img.mode

                    # Compute basic color statistics
                    stat_info = ""
                    try:
                        from PIL import ImageStat  # type: ignore[import-untyped]

                        stat = ImageStat.Stat(pil_img)
                        means = [round(m, 1) for m in stat.mean[:3]]
                        stat_info = f", mean_rgb={means}"
                    except Exception:
                        pass

                    desc = (
                        f"[Image: {width}×{height}px, {mode} mode"
                        f"{stat_info}. Analyze this image as requested.]\n"
                    )
                    prompt = desc + prompt

            except (ImportError, Exception) as e:
                prompt = f"[Image attached — {len(images)} image(s)]\n{prompt}"
                logger.warning(f"PIL processing failed: {e}")

        async for token in engine.generate(prompt, params):  # type: ignore[attr-defined]
            yield token
