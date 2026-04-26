"""Summarizer Agent — Extractive TF-IDF + LLM abstractive fallback."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Any

from pravaha.swarm.agents.base_agent import AgentOutput, BaseAgent, SharedContext


class SummarizerAgent(BaseAgent):
    """Summarize text using extractive TF-IDF (no LLM) with abstractive fallback."""

    role = "summarizer"
    priority = 4
    max_tokens = 512
    temperature = 0.3

    system_prompt = """You are a summarization specialist.

    Create a concise summary that:
    1. Preserves ALL key facts, numbers, and conclusions
    2. Removes filler, repetition, and tangential detail
    3. Maintains the original structure and logical flow
    4. Uses the same technical vocabulary as the source
    5. Is 20-30% the length of the original
    6. Opens with the single most important takeaway
    7. Ends with action items or open questions if any exist
    8. Never introduces new information not in the source
    9. Preserves code snippets and technical terms verbatim
    10. Uses bullet points for lists of 3+ items
    """

    @staticmethod
    def _extractive_summary(text: str, target_sentences: int = 5) -> str:
        """Pure-Python extractive summary using TF-IDF sentence scoring."""
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(sentences) <= target_sentences:
            return text

        # Tokenize
        def tokenize(s: str) -> list[str]:
            return re.findall(r"\b[a-zA-Z]{2,}\b", s.lower())

        # Compute document frequency
        doc_freq: Counter[str] = Counter()
        sent_tokens: list[list[str]] = []
        for sent in sentences:
            tokens = tokenize(sent)
            sent_tokens.append(tokens)
            for word in set(tokens):
                doc_freq[word] += 1

        num_docs = len(sentences)

        # Score each sentence by TF-IDF
        scores: list[float] = []
        for tokens in sent_tokens:
            if not tokens:
                scores.append(0.0)
                continue
            tf = Counter(tokens)
            score = 0.0
            for word, count in tf.items():
                tf_val = count / len(tokens)
                idf_val = math.log(num_docs / (1 + doc_freq.get(word, 0)))
                score += tf_val * idf_val
            scores.append(score)

        # Select top N sentences, maintaining original order
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True,
        )[:target_sentences]
        selected = sorted(ranked_indices)

        return " ".join(sentences[i] for i in selected)

    async def run(
        self, task: str, context: SharedContext, engine: Any,
    ) -> AgentOutput:
        t0 = time.time()
        content = context.output or task
        input_words = len(content.split())
        extractive_used = False

        # Phase 1: Try extractive summary (no LLM cost)
        target_sentences = max(3, input_words // 50)
        try:
            output = self._extractive_summary(content, target_sentences)
            extractive_used = True
        except Exception:
            output = ""

        # Phase 2: Fall back to LLM abstractive if extractive is poor
        output_words = len(output.split())
        compression = output_words / max(1, input_words)

        if not extractive_used or compression > 0.7 or output_words < 10:
            prompt = self.build_prompt(
                f"Summarize this to 20-30% of its length "
                f"({input_words} words → target {input_words // 4} words):\n\n"
                f"{content[:3000]}",
                context,
            )
            output = await self._generate(prompt, engine)
            extractive_used = False
            output_words = len(output.split())
            compression = output_words / max(1, input_words)

        context.output = output

        sentences_kept = len(re.split(r"(?<=[.!?])\s+", output.strip()))

        duration = (time.time() - t0) * 1000
        self._total_duration_ms += duration

        return AgentOutput(
            role=self.role,
            output=output,
            tokens_used=self._total_tokens,
            duration_ms=duration,
            confidence=0.85 if 0.15 <= compression <= 0.4 else 0.6,
            metadata={
                "extractive_used": extractive_used,
                "sentences_kept": sentences_kept,
                "compression_ratio": round(compression, 2),
                "input_words": input_words,
                "output_words": output_words,
            },
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"writing", "research", "analysis", "general"}
