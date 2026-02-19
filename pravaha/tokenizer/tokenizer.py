"""Tokenizer and preprocessing layer.

Wraps HuggingFace tokenizers with inference-specific utilities:
batch encoding with padding, streaming decode, and special token management.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class PravahaTokenizer:
    """HuggingFace tokenizer wrapper optimized for inference.

    Provides clean APIs for encoding prompts, decoding generated tokens
    (including incremental/streaming decode), and batch operations.
    """

    def __init__(self, model_path: str, trust_remote_code: bool = False):
        """Initialize tokenizer from a HuggingFace model.

        Args:
            model_path: HuggingFace model name or local directory.
            trust_remote_code: Whether to trust remote code for custom tokenizers.
        """
        logger.info(f"Loading tokenizer: {model_path}")

        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        # Ensure pad token is set (many models don't have one by default)
        if self._tokenizer.pad_token is None:
            if self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                logger.info(
                    f"Set pad_token to eos_token: '{self._tokenizer.eos_token}' "
                    f"(id={self._tokenizer.eos_token_id})"
                )
            else:
                raise ValueError(
                    "Tokenizer has no pad_token or eos_token. "
                    "Cannot proceed without a padding token."
                )

        logger.info(
            f"Tokenizer loaded: vocab_size={self.vocab_size}, "
            f"eos_id={self.eos_token_id}, pad_id={self.pad_token_id}"
        )

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Tokenize a single text string to token IDs.

        Args:
            text: Input text to tokenize.
            add_special_tokens: Whether to add BOS/EOS tokens.

        Returns:
            List of token IDs.
        """
        return self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )

    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode (list or tensor).
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            Decoded text string.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to its text representation.

        Useful for streaming: decode one token at a time.

        Args:
            token_id: Single token ID.

        Returns:
            Text for this token (may be a subword piece).
        """
        return self._tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
        )

    def batch_encode(
        self,
        texts: list[str],
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch encode multiple texts with padding.

        Args:
            texts: List of input texts.
            max_length: Maximum sequence length (None = longest in batch).
            add_special_tokens: Whether to add special tokens.

        Returns:
            Tuple of (input_ids, attention_mask) as LongTensors.
        """
        encoding = self._tokenizer(
            texts,
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        return encoding["input_ids"], encoding["attention_mask"]

    @property
    def eos_token_id(self) -> int:
        """End-of-sequence token ID."""
        return self._tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return self._tokenizer.pad_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        """Beginning-of-sequence token ID (None if not defined)."""
        return self._tokenizer.bos_token_id

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self._tokenizer.vocab_size

    @property
    def underlying(self) -> PreTrainedTokenizerBase:
        """Access the underlying HuggingFace tokenizer."""
        return self._tokenizer
