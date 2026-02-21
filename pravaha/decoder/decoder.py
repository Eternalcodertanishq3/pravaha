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
from pravaha.kv_cache.paged_cache import PagedKVCache
from pravaha.tokenizer.tokenizer import PravahaTokenizer

logger = logging.getLogger(__name__)


class DecoderEngine:
    """Autoregressive decoder for continuous batching (Phase 3).

    Executes forward passes for batches of sequences.
    Supports Disjoint Execution Strategy:
      - Batch Prefill (new requests)
      - Batch Decode (running requests)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PravahaTokenizer,
        sampler: Optional[Sampler] = None,
        device: str = "cuda",
        kv_cache: Optional[PagedKVCache] = None,
    ):
        """Initialize the decoder engine.

        Args:
            model: Loaded transformer model (in eval mode).
            tokenizer: Tokenizer for encoding/decoding.
            sampler: Token sampler. Defaults to a new Sampler instance.
            device: Device the model is on.
            kv_cache: Required NaiveKVCache for Phase 3 batching.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler or Sampler()
        self.device = device
        self.kv_cache = kv_cache

        if self.kv_cache is None:
            raise ValueError("Phase 4 Paged Attention requires PagedKVCache.")

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

        # 1. Pad prompts to the longest in the current batch
        max_prompt_len = max(len(p) for p in input_ids_list)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # Shape: (batch_size, max_prompt_len)
        padded_inputs = torch.full((batch_size, max_prompt_len), pad_token_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((batch_size, max_prompt_len), dtype=torch.long, device=self.device)

        # We also need to track the *true* last token index to extract logits
        # because the model outputs logits for all positions (including padding).
        last_token_indices = []

        for i, prompt_ids in enumerate(input_ids_list):
            seq_len = len(prompt_ids)
            padded_inputs[i, :seq_len] = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
            attention_mask[i, :seq_len] = 1
            last_token_indices.append(seq_len - 1)

        # 3. Model Forward Pass
        logger.debug(f"Prefill step: batch_size={batch_size}, max_len={max_prompt_len}")
        outputs = self.model(
            input_ids=padded_inputs,
            attention_mask=attention_mask,
            use_cache=True,
        )

        # 4. Extract logits exactly at the last *valid* prompt token
        # logits shape: (batch_size, max_prompt_len, vocab_size)
        all_logits = outputs.logits
        last_valid_logits = []
        for i, last_idx in enumerate(last_token_indices):
            last_valid_logits.append(all_logits[i, last_idx, :])
        
        # Shape: (batch_size, vocab_size)
        stacked_logits = torch.stack(last_valid_logits)

        # 5. Capture the prefill output into our physical blocks
        self.kv_cache.update_from_hf_past_key_values(
            outputs.past_key_values,
            num_new_tokens=max_prompt_len,
            request_ids=request_ids,
            block_tables=block_tables,
            slot_offsets=[0] * batch_size  # Prefill starts at pos 0
        )

        # 6. Sample next tokens (currently uses uniform SamplingParams for simplicity)
        # In a full implementation, we'd pass per-request SamplingParams
        next_tokens = []
        for i in range(batch_size):
            # Extract 1D logits for single sequence
            single_logits = stacked_logits[i]
            # No generated_ids history available yet at prefill boundary
            next_id = self.sampler.sample(single_logits, SamplingParams())
            next_tokens.append(next_id.item())

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

        # Shape: (batch_size, 1) - One new token per sequence
        input_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(1)

        # 1. Retrieve padded KV-cache for active blocks
        # This performs a gather from the physical pool to a contiguous HF-compatible batch
        past_key_values = self.kv_cache.to_hf_past_key_values(block_tables, context_lens)

        # 2. Construct 2D attention mask covering the FULL sequence length (past tokens + 1 new token)
        max_seq_len = max(context_lens)
        total_len = max_seq_len + 1
        
        attention_mask = torch.zeros((batch_size, total_len), dtype=torch.long, device=self.device)
        for i, clen in enumerate(context_lens):
            attention_mask[i, :clen + 1] = 1

        # 3. Model Forward Pass
        outputs = self.model(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # 4. Extract logits (only 1 sequence dimension, so take the last one)
        # logits shape: (batch_size, 1, vocab_size)
        stacked_logits = outputs.logits[:, -1, :]

        # 5. Store the new token KV-cache in physical blocks
        self.kv_cache.update_from_hf_past_key_values(
            outputs.past_key_values,
            num_new_tokens=1,
            request_ids=request_ids,
            block_tables=block_tables,
            slot_offsets=context_lens,
        )

        # 6. Sample next tokens
        next_tokens = []
        for i in range(batch_size):
            single_logits = stacked_logits[i]
            # Simplified sampler (lacks full generated history context)
            next_id = self.sampler.sample(single_logits, SamplingParams())
            next_tokens.append(next_id.item())

        return next_tokens
