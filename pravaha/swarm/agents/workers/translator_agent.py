"""Translator Agent — Language translation with code preservation."""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class TranslatorAgent(BaseAgent):
    """Translate text while preserving code blocks and detecting source language."""

    role = "translator"
    priority = 4
    max_tokens = 1536
    temperature = 0.3

    # Unicode script ranges for language detection
    SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
        "chinese": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
        "japanese": [(0x3040, 0x309F), (0x30A0, 0x30FF)],
        "korean": [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
        "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F)],
        "devanagari": [(0x0900, 0x097F)],
        "cyrillic": [(0x0400, 0x04FF)],
        "thai": [(0x0E00, 0x0E7F)],
        "hebrew": [(0x0590, 0x05FF)],
    }

    system_prompt = """You are a professional translator.

    Translation rules:
    1. Translate the content to the target language specified in the task
    2. If no target language is specified, translate to English
    3. Preserve ALL code blocks (```...```) completely unchanged
    4. Preserve variable names, function names, and API names unchanged
    5. Preserve formatting, markdown, and structure exactly
    6. Use natural, fluent phrasing — not word-for-word translation
    7. Preserve technical terms that have no good translation
       (add a translator's note: [TN: term means ...])
    8. Maintain the same level of formality as the source
    9. If content is already in the target language, note it and return as-is
    10. Add [SOURCE: detected_language] at the beginning of output
    """

    @classmethod
    def _detect_source_language(cls, text: str) -> str:
        """Detect source language from Unicode character ranges."""
        # Count characters in each script
        script_counts: dict[str, int] = {s: 0 for s in cls.SCRIPT_RANGES}

        for char in text[:500]:  # Sample first 500 chars
            cp = ord(char)
            for script, ranges in cls.SCRIPT_RANGES.items():
                for start, end in ranges:
                    if start <= cp <= end:
                        script_counts[script] += 1
                        break

        # Find dominant non-Latin script
        if script_counts:
            dominant = max(script_counts, key=script_counts.get)  # type: ignore[arg-type]
            if script_counts[dominant] > 5:
                return dominant

        # Check if mostly ASCII (likely English or Latin-script language)
        ascii_chars = sum(1 for c in text[:500] if ord(c) < 128)
        if ascii_chars / max(len(text[:500]), 1) > 0.9:
            return "english"

        return "unknown"

    async def run(
        self, task: str, context: SharedContext, engine: Any,
    ) -> AgentOutput:
        t0 = time.time()
        content = context.output or task

        # Detect source language
        source_lang = self._detect_source_language(content)

        # Extract and preserve code blocks
        code_blocks: list[str] = []
        preserved = content
        fence_pattern = re.compile(r"```.*?```", re.DOTALL)
        for i, match in enumerate(fence_pattern.finditer(content)):
            code_blocks.append(match.group())
            preserved = preserved.replace(
                match.group(), f"__CODE_BLOCK_{i}__",
            )

        prompt = self.build_prompt(
            f"[Detected source language: {source_lang}]\n\n"
            f"Translate this text. Code blocks marked __CODE_BLOCK_N__ "
            f"must be kept EXACTLY as-is:\n\n{preserved[:3000]}",
            context,
        )
        output = await self._generate(prompt, engine)

        # Restore code blocks
        for i, block in enumerate(code_blocks):
            output = output.replace(f"__CODE_BLOCK_{i}__", block)

        translator_notes = output.count("[TN:")

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=0.85 if source_lang != "unknown" else 0.6,
            metadata={
                "source_lang_detected": source_lang,
                "code_blocks_preserved": len(code_blocks),
                "translator_notes_added": translator_notes,
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"translation", "general"}
