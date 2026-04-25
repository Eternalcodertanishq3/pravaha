"""Pravaha Tokenizer — Wrapper around HuggingFace AutoTokenizer."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class PravahaTokenizer:
    """Unified tokenizer wrapping HuggingFace AutoTokenizer."""

    def __init__(self, model_path: str) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=True)

    def decode(self, token_ids: list[int]) -> str:
        from typing import cast
        return cast(str, self._tokenizer.decode(token_ids, skip_special_tokens=True))

    def decode_token(self, token_id: int) -> str:
        from typing import cast
        return cast(str, self._tokenizer.decode([token_id], skip_special_tokens=False))

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        return self._tokenizer.pad_token_id

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size
