"""Pravāha Multi-GPU & Multi-Node Distributed Topology Manager.

Provides real distributed inference capabilities:
- NCCL/Gloo backend initialization for multi-node communication
- Tensor Parallelism (TP) process group creation and rank assignment
- Pipeline Parallelism (PP) stage group creation
- AllReduce, AllGather, and P2P communication primitives
- Multi-node health monitoring and topology discovery
- NVLink/PCIe P2P capability detection
"""

from __future__ import annotations

import logging
import os
import socket
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class ParallelismMode(Enum):
    """Supported parallelism strategies."""

    NONE = auto()
    TENSOR_PARALLEL = auto()
    PIPELINE_PARALLEL = auto()
    HYBRID = auto()  # TP + PP combined


@dataclass
class NodeInfo:
    """Information about a single compute node in the cluster."""

    hostname: str
    rank: int
    local_rank: int
    world_size: int
    device_count: int
    devices: list[dict[str, Any]] = field(default_factory=list)
    is_alive: bool = True
    last_heartbeat: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hostname": self.hostname,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "device_count": self.device_count,
            "devices": self.devices,
            "is_alive": self.is_alive,
            "last_heartbeat": self.last_heartbeat,
        }


@dataclass
class ParallelConfig:
    """Configuration for distributed parallelism."""

    tp_size: int = 1  # Tensor Parallelism degree
    pp_size: int = 1  # Pipeline Parallelism stages
    backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    init_method: str | None = None  # Custom init method (e.g., "tcp://...")
    timeout_seconds: int = 300
    heartbeat_interval: float = 5.0


class DistributedTopologyManager:
    """Manages multi-GPU placement, Tensor Parallelism (TP) rank assignment,
    Pipeline Parallelism (PP) stage assignment, and cross-node communication.

    This is the central coordinator for all distributed inference operations.
    It initializes the PyTorch distributed backend, creates process groups
    for TP and PP, and provides communication primitives that wrap
    torch.distributed operations.

    Attributes:
        world_size: Total number of GPU devices / ranks across all nodes.
        rank: Global process rank ID.
        local_rank: Rank within the current node.
        is_distributed: True if PyTorch distributed backend is initialized.
        tp_group: Process group for tensor parallelism communication.
        pp_group: Process group for pipeline parallelism communication.
    """

    def __init__(self, config: ParallelConfig | None = None) -> None:
        self.config = config or ParallelConfig()
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # Read environment variables (set by torchrun/torch.distributed.launch)
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", max(1, self.device_count)))
        self.is_distributed = False

        # Process groups for TP and PP
        self.tp_group: dist.ProcessGroup | None = None
        self.pp_group: dist.ProcessGroup | None = None
        self.tp_rank: int = 0
        self.tp_size: int = self.config.tp_size
        self.pp_rank: int = 0
        self.pp_size: int = self.config.pp_size

        # Node registry for multi-node health monitoring
        self._nodes: dict[int, NodeInfo] = {}
        self._heartbeat_thread: threading.Thread | None = None
        self._running = False

        # Initialize distributed backend if world_size > 1
        if self.world_size > 1 and not dist.is_initialized():
            self._init_distributed()
        elif dist.is_initialized():
            self.is_distributed = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        # Create TP/PP process groups
        if self.is_distributed:
            self._create_parallel_groups()

        # Register this node
        self._register_node()

        logger.info(
            f"DistributedTopologyManager: Detected {self.device_count} GPUs. "
            f"World Size={self.world_size}, Rank={self.rank}, Local Rank={self.local_rank}, "
            f"Distributed={self.is_distributed}, "
            f"TP={self.tp_size}(rank={self.tp_rank}), PP={self.pp_size}(rank={self.pp_rank})"
        )

    def _init_distributed(self) -> None:
        """Initialize PyTorch distributed backend (NCCL or Gloo)."""
        try:
            # Set environment variables for rendezvous
            os.environ.setdefault("MASTER_ADDR", self.config.master_addr)
            os.environ.setdefault("MASTER_PORT", str(self.config.master_port))

            backend = self.config.backend
            if backend == "nccl" and not torch.cuda.is_available():
                logger.warning("NCCL requested but CUDA not available. Falling back to Gloo.")
                backend = "gloo"

            init_method = self.config.init_method or "env://"

            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=self.world_size,
                rank=self.rank,
                timeout=torch.distributed.distributed_c10d._DEFAULT_PG_TIMEOUT
                if hasattr(torch.distributed, "distributed_c10d")
                else None,
            )

            self.is_distributed = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

            # Set CUDA device for this rank
            if torch.cuda.is_available() and self.device_count > 0:
                torch.cuda.set_device(self.local_rank % self.device_count)

            logger.info(
                f"Distributed backend '{backend}' initialized. "
                f"Rank {self.rank}/{self.world_size}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize distributed backend: {e}")
            self.is_distributed = False

    def _create_parallel_groups(self) -> None:
        """Create process groups for Tensor Parallelism and Pipeline Parallelism.

        With TP_SIZE=T and PP_SIZE=P, we need T*P = WORLD_SIZE.

        Layout example with 8 GPUs, TP=4, PP=2:
          PP Stage 0: [GPU 0, 1, 2, 3]  ← TP group
          PP Stage 1: [GPU 4, 5, 6, 7]  ← TP group
          PP group for TP-rank 0: [GPU 0, 4]
          PP group for TP-rank 1: [GPU 1, 5]
          ...
        """
        tp_size = min(self.config.tp_size, self.world_size)
        pp_size = min(self.config.pp_size, self.world_size // tp_size)

        # Validate configuration
        if tp_size * pp_size > self.world_size:
            logger.warning(
                f"TP_SIZE({tp_size}) * PP_SIZE({pp_size}) = {tp_size * pp_size} > "
                f"WORLD_SIZE({self.world_size}). Falling back to TP-only."
            )
            pp_size = 1

        self.tp_size = tp_size
        self.pp_size = pp_size

        # Determine this rank's TP and PP positions
        self.pp_rank = self.rank // tp_size
        self.tp_rank = self.rank % tp_size

        # Create TP groups: ranks within the same PP stage
        for pp_stage in range(pp_size):
            tp_ranks = list(range(pp_stage * tp_size, (pp_stage + 1) * tp_size))
            group = dist.new_group(tp_ranks)
            if self.rank in tp_ranks:
                self.tp_group = group
                logger.info(f"Rank {self.rank}: Joined TP group {tp_ranks} (TP rank={self.tp_rank})")

        # Create PP groups: same TP-rank across PP stages
        for tp_idx in range(tp_size):
            pp_ranks = [pp_stage * tp_size + tp_idx for pp_stage in range(pp_size)]
            group = dist.new_group(pp_ranks)
            if self.rank in pp_ranks:
                self.pp_group = group
                logger.info(f"Rank {self.rank}: Joined PP group {pp_ranks} (PP rank={self.pp_rank})")

    def _register_node(self) -> None:
        """Register this process as a node in the topology."""
        devices_info = []
        if torch.cuda.is_available():
            for i in range(self.device_count):
                props = torch.cuda.get_device_properties(i)
                devices_info.append({
                    "id": i,
                    "name": props.name,
                    "vram_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })

        self._nodes[self.rank] = NodeInfo(
            hostname=socket.gethostname(),
            rank=self.rank,
            local_rank=self.local_rank,
            world_size=self.world_size,
            device_count=self.device_count,
            devices=devices_info,
            is_alive=True,
            last_heartbeat=time.time(),
        )

    # ─────────────────────────────────────────────
    # Communication Primitives
    # ─────────────────────────────────────────────

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Perform AllReduce across the specified process group.

        Sums (or applies op to) tensor values across all ranks in the group
        and distributes the result back to all ranks. Used in TP for combining
        partial matrix multiplication results.

        Args:
            tensor: Input tensor to reduce. Modified in-place.
            op: Reduction operation (SUM, AVG, MAX, MIN).
            group: Process group. Defaults to TP group.

        Returns:
            The reduced tensor (same object, modified in-place).
        """
        if not self.is_distributed:
            return tensor

        target_group = group or self.tp_group
        if target_group is None:
            return tensor

        dist.all_reduce(tensor, op=op, group=target_group)
        return tensor

    def all_gather(
        self,
        tensor: torch.Tensor,
        group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Perform AllGather across the specified process group.

        Gathers tensors from all ranks and concatenates them along dim 0.
        Used in TP for reconstructing the full output from column-parallel layers.

        Args:
            tensor: Local tensor to gather.
            group: Process group. Defaults to TP group.

        Returns:
            Concatenated tensor from all ranks.
        """
        if not self.is_distributed:
            return tensor

        target_group = group or self.tp_group
        if target_group is None:
            return tensor

        group_size = dist.get_world_size(target_group)
        gathered = [torch.empty_like(tensor) for _ in range(group_size)]
        dist.all_gather(gathered, tensor, group=target_group)
        return torch.cat(gathered, dim=-1)

    def send(
        self,
        tensor: torch.Tensor,
        dst: int,
        group: dist.ProcessGroup | None = None,
        tag: int = 0,
    ) -> None:
        """Send a tensor to a specific rank (point-to-point).

        Used in Pipeline Parallelism for sending activations between stages.

        Args:
            tensor: Tensor to send.
            dst: Destination rank.
            group: Process group. Defaults to PP group.
            tag: Message tag for matching send/recv pairs.
        """
        if not self.is_distributed:
            return

        target_group = group or self.pp_group
        dist.send(tensor, dst=dst, group=target_group, tag=tag)

    def recv(
        self,
        tensor: torch.Tensor,
        src: int,
        group: dist.ProcessGroup | None = None,
        tag: int = 0,
    ) -> torch.Tensor:
        """Receive a tensor from a specific rank (point-to-point).

        Used in Pipeline Parallelism for receiving activations between stages.

        Args:
            tensor: Pre-allocated buffer to receive into.
            src: Source rank.
            group: Process group. Defaults to PP group.
            tag: Message tag for matching send/recv pairs.

        Returns:
            The received tensor.
        """
        if not self.is_distributed:
            return tensor

        target_group = group or self.pp_group
        dist.recv(tensor, src=src, group=target_group, tag=tag)
        return tensor

    def isend(
        self,
        tensor: torch.Tensor,
        dst: int,
        group: dist.ProcessGroup | None = None,
        tag: int = 0,
    ) -> dist.Work:
        """Non-blocking send for overlapping communication with computation.

        Args:
            tensor: Tensor to send.
            dst: Destination rank.
            group: Process group.
            tag: Message tag.

        Returns:
            A Work handle that can be waited on.
        """
        target_group = group or self.pp_group
        return dist.isend(tensor, dst=dst, group=target_group, tag=tag)

    def irecv(
        self,
        tensor: torch.Tensor,
        src: int,
        group: dist.ProcessGroup | None = None,
        tag: int = 0,
    ) -> dist.Work:
        """Non-blocking receive for overlapping communication with computation.

        Args:
            tensor: Pre-allocated buffer.
            src: Source rank.
            group: Process group.
            tag: Message tag.

        Returns:
            A Work handle that can be waited on.
        """
        target_group = group or self.pp_group
        return dist.irecv(tensor, src=src, group=target_group, tag=tag)

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
        group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Broadcast a tensor from src rank to all ranks in the group.

        Args:
            tensor: Tensor to broadcast. Modified in-place on non-src ranks.
            src: Source rank within the group.
            group: Process group. Defaults to world group.

        Returns:
            The broadcast tensor.
        """
        if not self.is_distributed:
            return tensor

        dist.broadcast(tensor, src=src, group=group)
        return tensor

    def barrier(self, group: dist.ProcessGroup | None = None) -> None:
        """Synchronize all ranks in the group.

        Blocks until all ranks have reached this point. Used before
        CUDA graph capture to ensure all ranks capture simultaneously.

        Args:
            group: Process group. Defaults to world group.
        """
        if not self.is_distributed:
            return

        dist.barrier(group=group)

    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: list[torch.Tensor],
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Reduce-scatter: reduce and scatter the result to all ranks.

        Each rank receives a different portion of the reduced result.
        More bandwidth-efficient than AllReduce when each rank only needs
        a subset of the output.

        Args:
            output: Output tensor buffer for this rank's portion.
            input_list: List of tensors, one per rank, to reduce.
            op: Reduction operation.
            group: Process group.

        Returns:
            The output tensor.
        """
        if not self.is_distributed:
            return output

        target_group = group or self.tp_group
        dist.reduce_scatter(output, input_list, op=op, group=target_group)
        return output

    # ─────────────────────────────────────────────
    # Device & Topology Queries
    # ─────────────────────────────────────────────

    def get_device(self, rank_id: int | None = None) -> torch.device:
        """Return target PyTorch CUDA device for a specific rank.

        Args:
            rank_id: Global rank. Defaults to this process's rank.

        Returns:
            torch.device for the specified rank.
        """
        if not torch.cuda.is_available():
            return torch.device("cpu")
        target_rank = rank_id if rank_id is not None else self.rank
        device_id = target_rank % max(1, self.device_count)
        return torch.device(f"cuda:{device_id}")

    def get_tp_src_rank(self) -> int:
        """Get the first rank in this TP group (the 'source' for broadcasts)."""
        return self.pp_rank * self.tp_size

    def get_pp_prev_rank(self) -> int | None:
        """Get the rank of the previous pipeline stage, or None if first stage."""
        if self.pp_rank == 0:
            return None
        return (self.pp_rank - 1) * self.tp_size + self.tp_rank

    def get_pp_next_rank(self) -> int | None:
        """Get the rank of the next pipeline stage, or None if last stage."""
        if self.pp_rank >= self.pp_size - 1:
            return None
        return (self.pp_rank + 1) * self.tp_size + self.tp_rank

    def is_first_pp_stage(self) -> bool:
        """Check if this rank is in the first pipeline stage."""
        return self.pp_rank == 0

    def is_last_pp_stage(self) -> bool:
        """Check if this rank is in the last pipeline stage."""
        return self.pp_rank == self.pp_size - 1

    def get_p2p_capability(self) -> dict[str, Any]:
        """Detect P2P (NVLink/PCIe) capability between GPU pairs.

        Returns:
            Dictionary mapping device pairs to their P2P access capability.
        """
        p2p_matrix: dict[str, bool] = {}
        if torch.cuda.is_available() and self.device_count > 1:
            for i in range(self.device_count):
                for j in range(self.device_count):
                    if i != j:
                        can_access = torch.cuda.can_device_access_peer(i, j)
                        p2p_matrix[f"{i}->{j}"] = can_access

        return {
            "device_count": self.device_count,
            "p2p_matrix": p2p_matrix,
            "nvlink_available": any(p2p_matrix.values()) if p2p_matrix else False,
        }

    def get_topology_info(self) -> dict[str, Any]:
        """Return comprehensive diagnostic dictionary of the distributed topology."""
        devices_info = []
        if torch.cuda.is_available():
            for i in range(self.device_count):
                props = torch.cuda.get_device_properties(i)
                devices_info.append({
                    "id": i,
                    "name": props.name,
                    "vram_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })

        return {
            "device_count": self.device_count,
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "is_distributed": self.is_distributed,
            "parallelism": {
                "mode": self._get_parallelism_mode().name,
                "tp_size": self.tp_size,
                "tp_rank": self.tp_rank,
                "pp_size": self.pp_size,
                "pp_rank": self.pp_rank,
                "has_tp_group": self.tp_group is not None,
                "has_pp_group": self.pp_group is not None,
            },
            "devices": devices_info,
            "p2p": self.get_p2p_capability(),
            "nodes": {k: v.to_dict() for k, v in self._nodes.items()},
            "hostname": socket.gethostname(),
        }

    def _get_parallelism_mode(self) -> ParallelismMode:
        """Determine the active parallelism mode."""
        if self.tp_size > 1 and self.pp_size > 1:
            return ParallelismMode.HYBRID
        elif self.tp_size > 1:
            return ParallelismMode.TENSOR_PARALLEL
        elif self.pp_size > 1:
            return ParallelismMode.PIPELINE_PARALLEL
        return ParallelismMode.NONE

    # ─────────────────────────────────────────────
    # Multi-Node Health Monitoring
    # ─────────────────────────────────────────────

    def start_heartbeat(self) -> None:
        """Start background heartbeat thread for multi-node health monitoring."""
        if self._running or not self.is_distributed:
            return

        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="pravaha-heartbeat",
        )
        self._heartbeat_thread.start()
        logger.info("Multi-node heartbeat monitoring started.")

    def stop_heartbeat(self) -> None:
        """Stop the heartbeat monitoring thread."""
        self._running = False
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=3.0)

    def _heartbeat_loop(self) -> None:
        """Background loop that exchanges heartbeats between nodes."""
        while self._running:
            try:
                # Update local heartbeat
                if self.rank in self._nodes:
                    self._nodes[self.rank].last_heartbeat = time.time()

                # In a real multi-node setup, we'd use a distributed store
                # or AllGather to exchange heartbeats. For now, we use a
                # tensor-based approach.
                if self.is_distributed and dist.is_initialized():
                    # Create a heartbeat tensor (1 = alive)
                    heartbeat = torch.tensor([1.0], device=self.get_device())
                    gathered = [torch.zeros(1, device=self.get_device()) for _ in range(self.world_size)]

                    try:
                        dist.all_gather(gathered, heartbeat)
                        for i, hb in enumerate(gathered):
                            if i in self._nodes:
                                self._nodes[i].is_alive = hb.item() > 0
                                if hb.item() > 0:
                                    self._nodes[i].last_heartbeat = time.time()
                    except Exception as e:
                        logger.debug(f"Heartbeat exchange failed: {e}")

            except Exception as e:
                logger.warning(f"Heartbeat loop error: {e}")

            time.sleep(self.config.heartbeat_interval)

    def get_live_nodes(self) -> list[int]:
        """Get list of ranks that are currently alive."""
        cutoff = time.time() - (self.config.heartbeat_interval * 3)
        return [
            rank for rank, node in self._nodes.items()
            if node.is_alive and node.last_heartbeat > cutoff
        ]

    # ─────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────

    def shutdown(self) -> None:
        """Gracefully shut down distributed resources."""
        self.stop_heartbeat()
        if self.is_distributed and dist.is_initialized():
            try:
                dist.destroy_process_group()
                logger.info("Distributed process group destroyed.")
            except Exception as e:
                logger.warning(f"Error destroying process group: {e}")
        self.is_distributed = False


# ─────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────

_topology_manager: DistributedTopologyManager | None = None


def get_topology_manager(config: ParallelConfig | None = None) -> DistributedTopologyManager:
    """Singleton getter for DistributedTopologyManager.

    Args:
        config: Optional parallel config. Only used on first call.

    Returns:
        The global DistributedTopologyManager instance.
    """
    global _topology_manager
    if _topology_manager is None:
        _topology_manager = DistributedTopologyManager(config)
    return _topology_manager


def reset_topology_manager() -> None:
    """Reset the singleton (for testing)."""
    global _topology_manager
    if _topology_manager is not None:
        _topology_manager.shutdown()
    _topology_manager = None
