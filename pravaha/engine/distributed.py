"""Pravāha Multi-GPU & Distributed Topology Manager."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)


class DistributedTopologyManager:
    """Manages multi-GPU placement, Tensor Parallelism (TP) rank assignment, and device topologies.

    Attributes:
        world_size: Total number of GPU devices / ranks.
        rank: Local process rank ID.
        is_distributed: True if PyTorch distributed backend is initialized.
    """

    def __init__(self) -> None:
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", max(1, self.device_count)))
        self.is_distributed = torch.distributed.is_initialized() if hasattr(torch, "distributed") else False

        logger.info(
            f"DistributedTopologyManager: Detected {self.device_count} GPUs. "
            f"World Size={self.world_size}, Rank={self.rank}, Distributed={self.is_distributed}"
        )

    def get_device(self, rank_id: int | None = None) -> torch.device:
        """Return target PyTorch CUDA device for a specific rank."""
        if not torch.cuda.is_available():
            return torch.device("cpu")
        target_rank = rank_id if rank_id is not None else self.rank
        device_id = target_rank % max(1, self.device_count)
        return torch.device(f"cuda:{device_id}")

    def get_topology_info(self) -> dict[str, Any]:
        """Return diagnostic dictionary of available GPU devices and P2P capability."""
        devices_info = []
        if torch.cuda.is_available():
            for i in range(self.device_count):
                devices_info.append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                })

        return {
            "device_count": self.device_count,
            "world_size": self.world_size,
            "rank": self.rank,
            "is_distributed": self.is_distributed,
            "devices": devices_info,
        }


_topology_manager: DistributedTopologyManager | None = None


def get_topology_manager() -> DistributedTopologyManager:
    """Singleton getter for DistributedTopologyManager."""
    global _topology_manager
    if _topology_manager is None:
        _topology_manager = DistributedTopologyManager()
    return _topology_manager
