from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


def _ensure_divisibility(numerator: int, denominator: int, name: str) -> None:
    if numerator % denominator != 0:
        raise ValueError(f"{name} ({numerator}) must be divisible by denominator ({denominator})")


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.

    The weight matrix is partitioned column-wise. Thus, each rank holds a slice
    of the weight matrix corresponding to `out_features // world_size`.
    The forward pass computes the local matmul, then optionally performs an `all_gather`.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = True,
        gather_output: bool = True,
        process_group: dist.ProcessGroup | None = None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.gather_output = gather_output
        self.process_group = process_group

        _ensure_divisibility(out_features, world_size, "out_features")
        self.output_size_per_partition = out_features // world_size

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
        else:
            self.register_parameter('bias', None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if getattr(self, 'bias', None) is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output_parallel = torch.nn.functional.linear(input_, self.weight, self.bias)
        if self.gather_output and self.world_size > 1 and dist.is_initialized():
            outputs = [torch.empty_like(output_parallel) for _ in range(self.world_size)]
            dist.all_gather(outputs, output_parallel, group=self.process_group)
            output = torch.cat(outputs, dim=-1)
            return output
        return output_parallel


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.

    The input and weight matrices are partitioned row-wise (features dimension).
    Each rank computes a partial output, and then an `all_reduce` is performed to sum the results.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        process_group: dist.ProcessGroup | None = None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.input_is_parallel = input_is_parallel
        self.process_group = process_group

        _ensure_divisibility(in_features, world_size, "in_features")
        self.input_size_per_partition = in_features // world_size

        self.weight = nn.Parameter(torch.empty(out_features, self.input_size_per_partition))
        if bias and (rank == 0):
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if getattr(self, 'bias', None) is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel and self.world_size > 1:
            # Assumes input_ is gathered; we need to partition it
            input_ = torch.chunk(input_, self.world_size, dim=-1)[self.rank]

        output_parallel = torch.nn.functional.linear(input_, self.weight)

        if self.world_size > 1 and dist.is_initialized():
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM, group=self.process_group)

        if getattr(self, 'bias', None) is not None:
            output_parallel = output_parallel + self.bias

        return output_parallel


class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer partitioned across the vocabulary dimension.

    Each rank holds a portion of the embedding table. Tokens outside the local
    vocabulary range are masked before lookup, and results are all-reduced.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        world_size: int,
        rank: int,
        process_group: dist.ProcessGroup | None = None
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.world_size = world_size
        self.rank = rank
        self.process_group = process_group

        _ensure_divisibility(num_embeddings, world_size, "num_embeddings")
        self.vocab_size_per_partition = num_embeddings // world_size

        self.vocab_start_index = rank * self.vocab_size_per_partition
        self.vocab_end_index = self.vocab_start_index + self.vocab_size_per_partition

        self.weight = nn.Parameter(torch.empty(self.vocab_size_per_partition, embedding_dim))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.weight)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size > 1:
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0

            output_parallel = torch.nn.functional.embedding(masked_input, self.weight)
            output_parallel[input_mask, :] = 0.0

            if dist.is_initialized():
                dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM, group=self.process_group)
            return output_parallel
        else:
            return torch.nn.functional.embedding(input_, self.weight)


class ParallelAttention(nn.Module):
    """
    Parallel multi-head attention.

    Partitions the heads across TP ranks using ColumnParallelLinear for the QKV projection
    and RowParallelLinear for the output projection.
    """
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        world_size: int,
        rank: int,
        process_group: dist.ProcessGroup | None = None
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.world_size = world_size
        self.rank = rank
        self.process_group = process_group

        _ensure_divisibility(num_heads, world_size, "num_heads")
        self.num_heads_per_partition = num_heads // world_size
        self.hidden_size_per_partition = self.num_heads_per_partition * head_dim

        # QKV uses column parallelism to split heads across ranks without needing all-reduce immediately
        self.qkv_proj = ColumnParallelLinear(
            in_features=num_heads * head_dim,
            out_features=3 * num_heads * head_dim,
            world_size=world_size,
            rank=rank,
            bias=True,
            gather_output=False,
            process_group=process_group
        )

        # Out projection uses row parallelism since the input is parallelized across heads
        self.out_proj = RowParallelLinear(
            in_features=num_heads * head_dim,
            out_features=num_heads * head_dim,
            world_size=world_size,
            rank=rank,
            bias=True,
            input_is_parallel=True,
            process_group=process_group
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # qkv projection: [batch, seq_len, 3 * hidden_size_per_partition]
        qkv = self.qkv_proj(hidden_states)

        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)

        # Simplified scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)

        # Output projection
        output = self.out_proj(context)
        return output


class TensorParallelWrapper(nn.Module):
    """
    Wraps an existing model to use Tensor Parallelism.

    Provides utility methods to shard standard model layers into their parallel equivalents.
    """
    def __init__(self, model: nn.Module, tp_config: dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.tp_config = tp_config
        self.world_size = tp_config.get("world_size", 1)
        self.rank = tp_config.get("rank", 0)
        self.process_group = tp_config.get("process_group", None)

    def shard_model(self) -> None:
        """
        Recursively replaces standard layers (e.g., nn.Linear, nn.Embedding) with parallel layers.
        Implementation should iterate over `self.model.named_modules()` and apply replacements.
        """
        logger.info(f"Sharding model for rank {self.rank}/{self.world_size}")
        # Note: Actual sharding logic would replace modules dynamically
        pass

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)
