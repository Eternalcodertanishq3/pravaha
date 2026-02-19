"""Sampling pipeline for token generation.

Implements temperature scaling, top-k filtering, top-p (nucleus) sampling,
and greedy decoding. Designed as a composable pipeline that transforms
raw logits into sampled token IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    """Parameters controlling the token sampling strategy.

    Attributes:
        temperature: Scaling factor for logits. Lower = more deterministic.
            0.0 = greedy decoding.
        top_k: Keep only the top-k highest probability tokens. 0 = disabled.
        top_p: Nucleus sampling — keep smallest set of tokens whose cumulative
            probability exceeds top_p. 1.0 = disabled.
        max_new_tokens: Maximum number of new tokens to generate.
        repetition_penalty: Penalty for repeating tokens. 1.0 = no penalty.
        stop_token_ids: Additional token IDs that trigger generation stop.
    """

    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    max_new_tokens: int = 256
    repetition_penalty: float = 1.0
    stop_token_ids: list[int] = field(default_factory=list)

    @property
    def is_greedy(self) -> bool:
        """Whether this is greedy decoding (temperature = 0)."""
        return self.temperature == 0.0


class Sampler:
    """Token sampling pipeline: logits → temperature → penalties → top-k → top-p → sample.

    Usage:
        sampler = Sampler()
        params = SamplingParams(temperature=0.8, top_k=40, top_p=0.9)
        next_token = sampler.sample(logits, params)
    """

    def sample(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        generated_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample the next token from logits.

        Args:
            logits: Raw model output logits, shape (batch_size, vocab_size)
                or (vocab_size,).
            params: Sampling configuration.
            generated_ids: Previously generated token IDs for repetition penalty.
                Shape (batch_size, seq_len) or (seq_len,).

        Returns:
            Sampled token IDs, shape (batch_size,) or scalar.
        """
        # Ensure 2D: (batch_size, vocab_size)
        squeeze = False
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            squeeze = True
            if generated_ids is not None and generated_ids.dim() == 1:
                generated_ids = generated_ids.unsqueeze(0)

        # 1. Apply repetition penalty
        if params.repetition_penalty != 1.0 and generated_ids is not None:
            logits = self._apply_repetition_penalty(
                logits, generated_ids, params.repetition_penalty
            )

        # 2. Greedy decoding
        if params.is_greedy:
            token_ids = logits.argmax(dim=-1)
            return token_ids.squeeze(0) if squeeze else token_ids

        # 3. Temperature scaling
        if params.temperature != 1.0:
            logits = logits / params.temperature

        # 4. Top-k filtering
        if params.top_k > 0:
            logits = self._top_k_filter(logits, params.top_k)

        # 5. Top-p (nucleus) filtering
        if params.top_p < 1.0:
            logits = self._top_p_filter(logits, params.top_p)

        # 6. Convert to probabilities and sample
        probs = F.softmax(logits, dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return token_ids.squeeze(0) if squeeze else token_ids

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Penalize tokens that have already been generated.

        For each token in generated_ids, divide its logit by penalty if positive,
        or multiply by penalty if negative. This discourages repetition.
        """
        logits = logits.clone()
        for batch_idx in range(logits.size(0)):
            prev_tokens = generated_ids[batch_idx].unique()
            for token_id in prev_tokens:
                if token_id < 0:
                    continue
                if logits[batch_idx, token_id] > 0:
                    logits[batch_idx, token_id] /= penalty
                else:
                    logits[batch_idx, token_id] *= penalty
        return logits

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Keep only the top-k logits, set the rest to -inf.

        Args:
            logits: Shape (batch_size, vocab_size).
            k: Number of top logits to keep.

        Returns:
            Filtered logits with non-top-k values set to -inf.
        """
        k = min(k, logits.size(-1))
        top_k_values, _ = torch.topk(logits, k, dim=-1)
        threshold = top_k_values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < threshold, float("-inf"))
        return logits

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
        """Nucleus sampling: keep the smallest set of tokens with cumulative prob >= p.

        Args:
            logits: Shape (batch_size, vocab_size).
            p: Cumulative probability threshold.

        Returns:
            Filtered logits with low-probability tokens set to -inf.
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Find where cumulative probability exceeds threshold
        # Shift right so the first token above threshold is kept
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
        sorted_logits[sorted_mask] = float("-inf")

        # Scatter back to original ordering
        logits = torch.zeros_like(logits).scatter_(
            dim=-1, index=sorted_indices, src=sorted_logits
        )
        return logits
