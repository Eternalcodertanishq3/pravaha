"""Semantic Context Compression — Smart context windowing.

Feature E: When context exceeds max_seq_len, don't truncate — compress
semantically. Keep the START (system prompt/instructions) and END (recent
conversation), summarize the middle. No context is truly "lost."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in a conversation.

    Attributes:
        role: Message role (system, user, assistant).
        content: Message text content.
        token_count: Pre-computed token count for this message.
    """

    role: str
    content: str
    token_count: int = 0


@dataclass
class CompressionResult:
    """Result of context compression.

    Attributes:
        messages: The compressed message list.
        original_tokens: Token count before compression.
        compressed_tokens: Token count after compression.
        summary_inserted: Whether a summary was inserted.
        messages_summarized: Number of messages that were summarized.
    """

    messages: list[Message]
    original_tokens: int
    compressed_tokens: int
    summary_inserted: bool
    messages_summarized: int


class SemanticContextCompressor:
    """Smart context compression that preserves meaning.

    Strategy:
    1. Keep first N tokens (system prompt + context setup) — START
    2. Keep last M tokens (recent conversation) — END
    3. Summarize the middle section to P tokens — MIDDLE
    4. Result: full context fits within max_seq_len

    The START preserves instructions. The END preserves recent context
    (most relevant for the next response). The MIDDLE is compressed
    but not lost — it's summarized. This is dramatically better than
    naive truncation which simply cuts off old messages.

    When used with the swarm, the SummarizerAgent handles compression.
    In standalone mode, a simpler extraction-based compression is used.
    """

    def __init__(
        self,
        tokenizer: Optional[object] = None,
    ) -> None:
        """Initialize the compressor.

        Args:
            tokenizer: Tokenizer for counting tokens. If None, uses word-based
                approximation (1 word ≈ 1.3 tokens).
        """
        self._tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text.

        Returns:
            Token count.
        """
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text))  # type: ignore[union-attr]
            except Exception:
                pass
        # Fallback: rough word-based estimate
        return int(len(text.split()) * 1.3)

    def compress(
        self,
        messages: list[Message],
        max_tokens: int,
        keep_start: int = 512,
        keep_end: int = 1024,
    ) -> CompressionResult:
        """Compress a message list to fit within max_tokens.

        Preserves the start (system prompt / initial context) and end
        (recent messages), summarizing the middle portion.

        Args:
            messages: Full conversation history.
            max_tokens: Target maximum token count.
            keep_start: Tokens to preserve from the beginning.
            keep_end: Tokens to preserve from the end.

        Returns:
            CompressionResult with the compressed messages.
        """
        # Count total tokens
        for msg in messages:
            if msg.token_count == 0:
                msg.token_count = self.count_tokens(msg.content)

        total_tokens = sum(m.token_count for m in messages)

        # No compression needed
        if total_tokens <= max_tokens:
            return CompressionResult(
                messages=messages,
                original_tokens=total_tokens,
                compressed_tokens=total_tokens,
                summary_inserted=False,
                messages_summarized=0,
            )

        logger.info(
            f"Context compression needed: {total_tokens} tokens → {max_tokens} max"
        )

        # Identify start, middle, and end segments
        start_msgs: list[Message] = []
        end_msgs: list[Message] = []
        middle_msgs: list[Message] = []

        # Phase 1: Collect start messages up to keep_start tokens
        start_tokens = 0
        start_idx = 0
        for i, msg in enumerate(messages):
            if start_tokens + msg.token_count <= keep_start:
                start_msgs.append(msg)
                start_tokens += msg.token_count
                start_idx = i + 1
            else:
                break

        # Phase 2: Collect end messages up to keep_end tokens (from the back)
        end_tokens = 0
        end_idx = len(messages)
        for i in range(len(messages) - 1, start_idx - 1, -1):
            msg = messages[i]
            if end_tokens + msg.token_count <= keep_end:
                end_msgs.insert(0, msg)
                end_tokens += msg.token_count
                end_idx = i
            else:
                break

        # Phase 3: Middle messages are everything between start and end
        middle_msgs = messages[start_idx:end_idx]
        middle_tokens = sum(m.token_count for m in middle_msgs)

        if not middle_msgs:
            # Nothing to compress — just truncate end if needed
            result_msgs = start_msgs + end_msgs
            return CompressionResult(
                messages=result_msgs,
                original_tokens=total_tokens,
                compressed_tokens=start_tokens + end_tokens,
                summary_inserted=False,
                messages_summarized=0,
            )

        # Phase 4: Summarize the middle
        budget = max_tokens - start_tokens - end_tokens
        summary_text = self._summarize_messages(middle_msgs, budget)
        summary_tokens = self.count_tokens(summary_text)

        summary_msg = Message(
            role="system",
            content=f"[Compressed context — {len(middle_msgs)} messages summarized:]\n{summary_text}",
            token_count=summary_tokens,
        )

        result_msgs = start_msgs + [summary_msg] + end_msgs
        compressed_total = start_tokens + summary_tokens + end_tokens

        logger.info(
            f"Context compressed: {total_tokens} → {compressed_total} tokens "
            f"({len(middle_msgs)} messages summarized)"
        )

        return CompressionResult(
            messages=result_msgs,
            original_tokens=total_tokens,
            compressed_tokens=compressed_total,
            summary_inserted=True,
            messages_summarized=len(middle_msgs),
        )

    def _summarize_messages(
        self,
        messages: list[Message],
        budget_tokens: int,
    ) -> str:
        """Summarize a list of messages to fit within a token budget.

        In standalone mode, uses extraction: picks the most important
        sentences from each message. With a swarm engine, the SummarizerAgent
        would be used for higher-quality summaries.

        Args:
            messages: Messages to summarize.
            budget_tokens: Target token count for the summary.

        Returns:
            Summary text fitting within the budget.
        """
        # Extract key points from each message
        key_points: list[str] = []
        for msg in messages:
            role_prefix = f"[{msg.role}]"
            sentences = msg.content.replace("\n", " ").split(". ")

            if len(sentences) <= 2:
                # Short message — keep as-is
                key_points.append(f"{role_prefix} {msg.content.strip()}")
            else:
                # Extract first and last sentence (usually most informative)
                first = sentences[0].strip()
                last = sentences[-1].strip()
                key_points.append(f"{role_prefix} {first}. [...] {last}.")

        # Join and truncate to budget
        summary = "\n".join(key_points)
        summary_tokens = self.count_tokens(summary)

        # If still over budget, progressively truncate
        while summary_tokens > budget_tokens and key_points:
            # Remove the second-to-last point (keep first and last)
            if len(key_points) > 2:
                key_points.pop(-2)
            else:
                key_points.pop(0)
            summary = "\n".join(key_points)
            summary_tokens = self.count_tokens(summary)

        return summary
