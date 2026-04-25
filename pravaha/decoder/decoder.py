"""Autoregressive decoder engine (Phase 3+).

# Phase 3: Continuous batching decoder with paged attention.
# Fix 2: Added complete single-request generate() method.

Implements the core generation loop: prefill the prompt through the model,
then iteratively decode one token at a time using KV-cache. Supports both
batched scheduling (step_prefill/step_decode) and standalone generation.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Generator

import torch
import torch.nn as nn

from pravaha.decoder.sampling import Sampler, SamplingParams
from pravaha.memory.paged_cache import PagedKVCache
from pravaha.tokenizer.tokenizer import PravahaTokenizer

logger = logging.getLogger(__name__)


class DecoderEngine:
    """Autoregressive decoder for continuous batching (Phase 3).

    Executes forward passes for batches of sequences.
    Supports Disjoint Execution Strategy:
      - Batch Prefill (new requests)
      - Batch Decode (running requests)
      - Single-request generate() for standalone use

    Bug Fix 2: Added the generate() method that was missing from
    the original implementation, enabling standalone single-request
    generation without the full scheduler pipeline.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PravahaTokenizer,
        sampler: Sampler | None = None,
        device: str = "cuda",
        kv_cache: PagedKVCache | None = None,
    ) -> None:
        """Initialize the decoder engine.

        Args:
            model: Loaded transformer model (in eval mode).
            tokenizer: Tokenizer for encoding/decoding.
            sampler: Token sampler. Defaults to a new Sampler instance.
            device: Device the model is on.
            kv_cache: PagedKVCache for Phase 3+ batching.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler or Sampler()
        self.device = device
        self.kv_cache = kv_cache

        if self.kv_cache is None:
            raise ValueError(
                "PagedKVCache is required for the decoder engine. "
                "Phase 4 paged attention is the minimum supported mode."
            )

    def generate(
        self,
        prompt: str,
        params: SamplingParams,
    ) -> Generator[str, None, None]:
        """Generate tokens for a single request (standalone mode).

        Fix 2: This method was missing from the original implementation.
        It provides a simple single-request generation API that handles
        block allocation, prefill, and autoregressive decoding internally.

        Args:
            prompt: Input text prompt.
            params: Sampling parameters.

        Yields:
            Generated token text, one token at a time.
        """
        assert self.kv_cache is not None, "KV cache is required for generation"

        input_ids = self.tokenizer.encode(prompt)
        request_id = str(uuid.uuid4())

        # Allocate KV cache blocks for this request
        # Need enough blocks for prompt + max_new_tokens
        blocks_needed = (len(input_ids) // self.kv_cache.block_size) + 2
        block_table = self.kv_cache.allocate_blocks(blocks_needed)

        try:
            # Phase 1: Prefill — process the entire prompt in one forward pass
            next_tokens = self.step_prefill([input_ids], [request_id], [block_table])
            yield self.tokenizer.decode_token(next_tokens[0])

            # Phase 2: Autoregressive decode — one token at a time
            context_len = len(input_ids)
            for _ in range(params.max_new_tokens - 1):
                # Allocate additional blocks if needed for the growing sequence
                needed = (
                    context_len + 1 + self.kv_cache.block_size - 1
                ) // self.kv_cache.block_size
                while len(block_table) < needed:
                    new_blocks = self.kv_cache.allocate_blocks(1)
                    block_table.extend(new_blocks)

                next_tokens = self.step_decode(
                    next_tokens, [request_id], [block_table], [context_len]
                )
                token_id = next_tokens[0]

                # Check for end of sequence
                if token_id == self.tokenizer.eos_token_id:
                    break

                yield self.tokenizer.decode_token(token_id)
                context_len += 1

        finally:
            # Always free KV cache blocks when done
            self.kv_cache.free_blocks(block_table)

    @torch.inference_mode()
    def step_prefill(
        self,
        input_ids_list: list[list[int]],
        request_ids: list[str],
        block_tables: list[list[int]],
    ) -> list[int]:
        """Perform a batched prefill forward pass for new requests (Paged).

        Args:
            input_ids_list: List of tokenized prompts.
            request_ids: Unique identifier for each request.
            block_tables: Physical block IDs assigned by scheduler.

        Returns:
            List of generated next-token IDs (one per request).
        """
        batch_size = len(input_ids_list)
        assert batch_size == len(block_tables)

        # Pad prompts to the longest in the current batch
        max_prompt_len = max(len(p) for p in input_ids_list)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        padded_inputs = torch.full(
            (batch_size, max_prompt_len),
            pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_prompt_len),
            dtype=torch.long,
            device=self.device,
        )

        last_token_indices: list[int] = []
        for i, prompt_ids in enumerate(input_ids_list):
            seq_len = len(prompt_ids)
            padded_inputs[i, :seq_len] = torch.tensor(
                prompt_ids, dtype=torch.long, device=self.device
            )
            attention_mask[i, :seq_len] = 1
            last_token_indices.append(seq_len - 1)

        # Model forward pass
        logger.debug(f"Prefill step: batch_size={batch_size}, max_len={max_prompt_len}")
        outputs = self.model(
            input_ids=padded_inputs,
            attention_mask=attention_mask,
            use_cache=True,
        )

        # Extract logits at the last valid prompt token per sequence
        all_logits = outputs.logits
        last_valid_logits = []
        for i, last_idx in enumerate(last_token_indices):
            last_valid_logits.append(all_logits[i, last_idx, :])

        stacked_logits = torch.stack(last_valid_logits)

        assert self.kv_cache is not None

        # Store KV states in physical blocks
        self.kv_cache.update_from_hf_past_key_values(
            outputs.past_key_values,
            num_new_tokens=max_prompt_len,
            request_ids=request_ids,
            block_tables=block_tables,
            slot_offsets=[0] * batch_size,
        )

        # Sample next tokens
        next_tokens = []
        for i in range(batch_size):
            single_logits = stacked_logits[i]
            next_id = self.sampler.sample(single_logits, SamplingParams())
            next_tokens.append(int(next_id.item()))

        return next_tokens

    @torch.inference_mode()
    def step_decode(
        self,
        token_ids: list[int],
        request_ids: list[str],
        block_tables: list[list[int]],
        context_lens: list[int],
    ) -> list[int]:
        """Perform a batched decode step (Paged).

        Args:
            token_ids: Last generated token ID for each request.
            request_ids: Unique identifier for each request.
            block_tables: Physical block IDs.
            context_lens: Current number of tokens ALREADY computed in KV-cache.

        Returns:
            List of generated next-token IDs.
        """
        batch_size = len(token_ids)
        assert batch_size == len(request_ids)

        # One new token per sequence
        input_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(1)

        assert self.kv_cache is not None

        # Retrieve padded KV-cache from physical blocks
        past_key_values = self.kv_cache.to_hf_past_key_values(block_tables, context_lens)

        # Attention mask covering full sequence (past + new token)
        max_seq_len = max(context_lens)
        total_len = max_seq_len + 1
        attention_mask = torch.zeros((batch_size, total_len), dtype=torch.long, device=self.device)
        for i, clen in enumerate(context_lens):
            attention_mask[i, : clen + 1] = 1

        # Model forward pass
        outputs = self.model(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # Extract logits for the new position
        stacked_logits = outputs.logits[:, -1, :]

        # Store new KV states
        self.kv_cache.update_from_hf_past_key_values(
            outputs.past_key_values,
            num_new_tokens=1,
            request_ids=request_ids,
            block_tables=block_tables,
            slot_offsets=context_lens,
        )

        # Sample next tokens
        next_tokens = []
        for i in range(batch_size):
            single_logits = stacked_logits[i]
            next_id = self.sampler.sample(single_logits, SamplingParams())
            next_tokens.append(int(next_id.item()))

        return next_tokens
