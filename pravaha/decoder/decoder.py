"""Autoregressive decoder engine.

Implements the core generation loop: prefill the prompt through the model,
then iteratively decode one token at a time using KV-cache. Supports two
cache modes:
  - HuggingFace cache (Phase 1): Model manages its own past_key_values
  - NaiveKVCache (Phase 2+): We manage pre-allocated cache tensors

This is the single-request baseline — continuous batching is added in Phase 3.
"""

from __future__ import annotations

import logging
import time
from typing import Generator, Optional

import torch
import torch.nn as nn

from pravaha.decoder.sampling import Sampler, SamplingParams
from pravaha.kv_cache.naive_cache import NaiveKVCache
from pravaha.tokenizer.tokenizer import PravahaTokenizer

logger = logging.getLogger(__name__)


class DecoderEngine:
    """Single-request autoregressive decoder.

    Generates tokens one at a time, yielding each decoded token for streaming.
    Supports both HuggingFace's built-in cache and our NaiveKVCache.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PravahaTokenizer,
        sampler: Optional[Sampler] = None,
        device: str = "cuda",
        kv_cache: Optional[NaiveKVCache] = None,
    ):
        """Initialize the decoder engine.

        Args:
            model: Loaded transformer model (in eval mode).
            tokenizer: Tokenizer for encoding/decoding.
            sampler: Token sampler. Defaults to a new Sampler instance.
            device: Device the model is on.
            kv_cache: Optional NaiveKVCache. If None, uses HF's cache.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler or Sampler()
        self.device = device
        self.kv_cache = kv_cache

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        params: Optional[SamplingParams] = None,
    ) -> Generator[str, None, None]:
        """Generate tokens autoregressively, yielding each token as a string.

        This is a streaming generator — each yielded value is one decoded token.
        The caller can collect all tokens or stream them to a client.

        Args:
            prompt: Input text prompt.
            params: Sampling parameters. Defaults to SamplingParams().

        Yields:
            Decoded token strings, one at a time.
        """
        if params is None:
            params = SamplingParams()

        # 1. Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        prompt_len = len(input_ids)

        logger.debug(f"Prompt length: {prompt_len} tokens")

        # 2. Reset cache if using NaiveKVCache
        if self.kv_cache is not None:
            self.kv_cache.clear()

        # 3. Prefill: forward pass on full prompt
        t_prefill_start = time.perf_counter()
        logits, past_key_values = self._prefill(input_tensor)
        t_prefill_end = time.perf_counter()

        logger.debug(
            f"Prefill: {t_prefill_end - t_prefill_start:.3f}s "
            f"({prompt_len} tokens)"
        )

        # If using NaiveKVCache, capture the prefill output into our cache
        if self.kv_cache is not None:
            self.kv_cache.update_from_hf_past_key_values(
                past_key_values, num_new_tokens=prompt_len
            )
            # From now on, use our cache
            past_key_values = self.kv_cache.to_hf_past_key_values()

        # 4. Decode loop
        generated_ids: list[int] = []
        all_stop_ids = {self.tokenizer.eos_token_id} | set(params.stop_token_ids)

        t_decode_start = time.perf_counter()

        for step in range(params.max_new_tokens):
            # Sample next token from last logits
            generated_tensor = (
                torch.tensor(generated_ids, dtype=torch.long, device=self.device)
                if generated_ids
                else None
            )
            next_token_id = self.sampler.sample(
                logits, params, generated_ids=generated_tensor
            )

            token_id = next_token_id.item()
            generated_ids.append(token_id)

            # Check stopping criteria
            if token_id in all_stop_ids:
                logger.debug(
                    f"Stopped at step {step + 1}: "
                    f"token_id={token_id} (stop token)"
                )
                break

            # Yield the decoded token text
            token_text = self.tokenizer.decode_token(token_id)
            yield token_text

            # Forward pass with single new token
            next_input = torch.tensor(
                [[token_id]], dtype=torch.long, device=self.device
            )
            logits, past_key_values = self._decode_step(
                next_input, past_key_values
            )

            # If using NaiveKVCache, capture the new token into our cache
            if self.kv_cache is not None:
                self.kv_cache.update_from_hf_past_key_values(
                    past_key_values, num_new_tokens=1
                )
                past_key_values = self.kv_cache.to_hf_past_key_values()

        t_decode_end = time.perf_counter()
        decode_time = t_decode_end - t_decode_start
        tokens_per_sec = len(generated_ids) / decode_time if decode_time > 0 else 0

        cache_mode = "naive" if self.kv_cache is not None else "hf"
        logger.info(
            f"Generation complete: {len(generated_ids)} tokens in "
            f"{decode_time:.3f}s ({tokens_per_sec:.1f} tok/s) "
            f"[cache={cache_mode}]"
        )

        # Log cache stats if using NaiveKVCache
        if self.kv_cache is not None:
            mem = self.kv_cache.memory_usage_bytes()
            logger.debug(
                f"KV-cache: {self.kv_cache.get_seq_len()} tokens cached, "
                f"{mem['kv_cache_used_mb']:.1f} MB used / "
                f"{mem['kv_cache_allocated_mb']:.1f} MB allocated"
            )

    @torch.inference_mode()
    def generate_text(
        self,
        prompt: str,
        params: Optional[SamplingParams] = None,
    ) -> str:
        """Generate and return the full response as a single string.

        Non-streaming convenience method.

        Args:
            prompt: Input text prompt.
            params: Sampling parameters.

        Returns:
            Complete generated text.
        """
        return "".join(self.generate(prompt, params))

    def _prefill(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple]:
        """Run model forward on the full prompt.

        Args:
            input_ids: Shape (1, seq_len) — the tokenized prompt.

        Returns:
            Tuple of (last_token_logits, past_key_values).
            last_token_logits shape: (vocab_size,)
        """
        outputs = self.model(
            input_ids=input_ids,
            use_cache=True,
        )
        # Extract logits for the last position only
        last_logits = outputs.logits[:, -1, :]  # (1, vocab_size)
        last_logits = last_logits.squeeze(0)  # (vocab_size,)
        return last_logits, outputs.past_key_values

    def _decode_step(
        self,
        token_id: torch.Tensor,
        past_key_values: tuple,
    ) -> tuple[torch.Tensor, tuple]:
        """Single decode step with KV-cache reuse.

        Args:
            token_id: Shape (1, 1) — the last generated token.
            past_key_values: Cached key-value states from previous steps.

        Returns:
            Tuple of (next_logits, updated_past_key_values).
            next_logits shape: (vocab_size,)
        """
        outputs = self.model(
            input_ids=token_id,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_logits = outputs.logits[:, -1, :].squeeze(0)  # (vocab_size,)
        return next_logits, outputs.past_key_values
