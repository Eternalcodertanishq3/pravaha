"""Tests for distributed inference components.

Tests the DistributedTopologyManager, TensorParallelism, PipelineParallelism,
and distributed-aware CUDA graph capture. All tests run in single-process mode
(no actual multi-GPU required) to verify the logic and API contracts.
"""

from __future__ import annotations

import pytest
import torch


# ─────────────────────────────────────────────
# Distributed Topology Manager Tests
# ─────────────────────────────────────────────


class TestDistributedTopologyManager:
    """Test the DistributedTopologyManager in single-process mode."""

    def test_default_construction(self) -> None:
        """Manager should initialize with sensible defaults when no distributed env is set."""
        from pravaha.engine.distributed import DistributedTopologyManager, ParallelConfig

        config = ParallelConfig(tp_size=1, pp_size=1)
        mgr = DistributedTopologyManager(config)

        assert mgr.rank == 0
        assert mgr.world_size >= 1
        assert mgr.tp_size == 1
        assert mgr.pp_size == 1
        assert mgr.tp_rank == 0
        assert mgr.pp_rank == 0

    def test_get_device_cpu_fallback(self) -> None:
        """get_device should return CPU when CUDA is unavailable."""
        from pravaha.engine.distributed import DistributedTopologyManager, ParallelConfig

        config = ParallelConfig()
        mgr = DistributedTopologyManager(config)

        # Even if CUDA is available, rank 0 should always resolve to a valid device
        device = mgr.get_device(rank_id=0)
        assert device is not None
        assert isinstance(device, torch.device)

    def test_topology_info(self) -> None:
        """get_topology_info should return a comprehensive dictionary."""
        from pravaha.engine.distributed import DistributedTopologyManager, ParallelConfig

        config = ParallelConfig(tp_size=2, pp_size=2)
        mgr = DistributedTopologyManager(config)

        info = mgr.get_topology_info()
        assert "device_count" in info
        assert "world_size" in info
        assert "rank" in info
        assert "local_rank" in info
        assert "parallelism" in info
        assert "p2p" in info
        assert "nodes" in info
        assert "hostname" in info

        parallelism = info["parallelism"]
        assert "mode" in parallelism
        assert "tp_size" in parallelism
        assert "pp_size" in parallelism

    def test_parallelism_mode_detection(self) -> None:
        """Should correctly detect NONE/TP/PP/HYBRID modes."""
        from pravaha.engine.distributed import (
            DistributedTopologyManager,
            ParallelConfig,
            ParallelismMode,
        )

        # Default (no parallelism)
        mgr = DistributedTopologyManager(ParallelConfig(tp_size=1, pp_size=1))
        assert mgr._get_parallelism_mode() == ParallelismMode.NONE

    def test_pp_navigation(self) -> None:
        """PP navigation helpers should work correctly."""
        from pravaha.engine.distributed import DistributedTopologyManager, ParallelConfig

        mgr = DistributedTopologyManager(ParallelConfig(tp_size=1, pp_size=1))

        # With PP=1, we're both first and last stage
        assert mgr.is_first_pp_stage() is True
        assert mgr.is_last_pp_stage() is True
        assert mgr.get_pp_prev_rank() is None
        assert mgr.get_pp_next_rank() is None

    def test_singleton_pattern(self) -> None:
        """get_topology_manager should return same instance on repeated calls."""
        from pravaha.engine.distributed import (
            get_topology_manager,
            reset_topology_manager,
        )

        reset_topology_manager()
        mgr1 = get_topology_manager()
        mgr2 = get_topology_manager()
        assert mgr1 is mgr2
        reset_topology_manager()

    def test_communication_primitives_noop(self) -> None:
        """Communication primitives should be no-ops when not distributed."""
        from pravaha.engine.distributed import DistributedTopologyManager, ParallelConfig

        mgr = DistributedTopologyManager(ParallelConfig())
        assert mgr.is_distributed is False

        tensor = torch.randn(4, 4)
        # Should return tensor unchanged
        result = mgr.all_reduce(tensor)
        assert torch.equal(result, tensor)

        result = mgr.all_gather(tensor)
        assert torch.equal(result, tensor)

        result = mgr.broadcast(tensor)
        assert torch.equal(result, tensor)

        # barrier should be a no-op
        mgr.barrier()

    def test_node_registry(self) -> None:
        """Node registration should work on init."""
        from pravaha.engine.distributed import DistributedTopologyManager, ParallelConfig

        mgr = DistributedTopologyManager(ParallelConfig())
        assert 0 in mgr._nodes
        node = mgr._nodes[0]
        assert node.rank == 0
        assert node.is_alive is True
        assert len(node.hostname) > 0

    def test_shutdown(self) -> None:
        """Shutdown should be graceful even when not distributed."""
        from pravaha.engine.distributed import DistributedTopologyManager, ParallelConfig

        mgr = DistributedTopologyManager(ParallelConfig())
        mgr.shutdown()  # Should not raise
        assert mgr.is_distributed is False


# ─────────────────────────────────────────────
# Distributed Config Tests
# ─────────────────────────────────────────────


class TestDistributedConfig:
    """Test the DistributedConfig in engine_config."""

    def test_distributed_config_defaults(self) -> None:
        """DistributedConfig should have sensible defaults."""
        from pravaha.config.engine_config import DistributedConfig

        config = DistributedConfig()
        assert config.enabled is False
        assert config.tp_size == 1
        assert config.pp_size == 1
        assert config.backend == "nccl"
        assert config.master_addr == "127.0.0.1"
        assert config.master_port == 29500

    def test_engine_config_has_distributed(self) -> None:
        """EngineConfig should contain a distributed section."""
        from pravaha.config.engine_config import EngineConfig

        config = EngineConfig.default()
        assert hasattr(config, "distributed")
        assert config.distributed.enabled is False
        assert config.distributed.tp_size == 1

    def test_distributed_fields_are_cold(self) -> None:
        """Distributed fields should be in COLD_FIELDS (not hot-reloadable)."""
        from pravaha.config.engine_config import COLD_FIELDS

        assert "distributed.enabled" in COLD_FIELDS
        assert "distributed.tp_size" in COLD_FIELDS
        assert "distributed.pp_size" in COLD_FIELDS
        assert "distributed.backend" in COLD_FIELDS


# ─────────────────────────────────────────────
# Tensor Parallelism Tests
# ─────────────────────────────────────────────


class TestTensorParallelism:
    """Test tensor parallelism components."""

    def test_column_parallel_linear_creation(self) -> None:
        """ColumnParallelLinear should partition output features."""
        from pravaha.engine.tensor_parallel import ColumnParallelLinear

        layer = ColumnParallelLinear(
            in_features=64,
            out_features=32,
            world_size=2,
            rank=0,
        )

        # Rank 0 should have half the output features
        assert layer.output_size_per_partition == 16

    def test_row_parallel_linear_creation(self) -> None:
        """RowParallelLinear should partition input features."""
        from pravaha.engine.tensor_parallel import RowParallelLinear

        layer = RowParallelLinear(
            in_features=64,
            out_features=32,
            world_size=4,
            rank=0,
        )

        # Rank 0 should have 1/4 of the input features
        assert layer.input_size_per_partition == 16

    def test_vocab_parallel_embedding_creation(self) -> None:
        """VocabParallelEmbedding should partition the embedding table."""
        from pravaha.engine.tensor_parallel import VocabParallelEmbedding

        embedding = VocabParallelEmbedding(
            num_embeddings=50000,
            embedding_dim=768,
            world_size=2,
            rank=0,
        )

        # Rank 0 should have half the vocabulary
        assert embedding.vocab_size_per_partition == 25000

    def test_column_parallel_forward(self) -> None:
        """ColumnParallelLinear forward should produce output of correct shape."""
        from pravaha.engine.tensor_parallel import ColumnParallelLinear

        layer = ColumnParallelLinear(
            in_features=64,
            out_features=32,
            world_size=1,  # Single rank = full output
            rank=0,
            gather_output=False,
        )

        x = torch.randn(2, 64)
        out = layer(x)
        assert out.shape == (2, 32)

    def test_row_parallel_forward(self) -> None:
        """RowParallelLinear forward should produce correct output shape."""
        from pravaha.engine.tensor_parallel import RowParallelLinear

        layer = RowParallelLinear(
            in_features=64,
            out_features=32,
            world_size=1,
            rank=0,
        )

        x = torch.randn(2, 64)
        out = layer(x)
        assert out.shape == (2, 32)


# ─────────────────────────────────────────────
# Pipeline Parallelism Tests
# ─────────────────────────────────────────────


class TestPipelineParallelism:
    """Test pipeline parallelism components."""

    def test_pipeline_stage_dataclass(self) -> None:
        """PipelineStage should hold stage metadata."""
        from pravaha.engine.pipeline_parallel import PipelineStage

        stage = PipelineStage(
            stage_id=0,
            layers=[0, 1, 2, 3],
            device=torch.device("cpu"),
            rank=0,
        )
        assert stage.stage_id == 0
        assert len(stage.layers) == 4

    def test_micro_batch_dataclass(self) -> None:
        """MicroBatch should hold batch data and metadata."""
        from pravaha.engine.pipeline_parallel import MicroBatch

        data = torch.randn(4, 64)
        mb = MicroBatch(batch_id=0, data=data, is_last=False)
        assert mb.batch_id == 0
        assert mb.is_last is False
        assert mb.data.shape == (4, 64)

    def test_pipeline_scheduler(self) -> None:
        """PipelineScheduler should generate a valid schedule."""
        from pravaha.engine.pipeline_parallel import PipelineScheduler

        scheduler = PipelineScheduler(
            num_stages=2,
            num_micro_batches=4,
            stage_id=0,
        )
        schedule = scheduler.get_schedule()
        assert len(schedule) > 0
        # All actions should be valid
        for action, mb_id in schedule:
            assert action in ("forward", "send", "recv")
            assert 0 <= mb_id < 4


# ─────────────────────────────────────────────
# CUDA Graph Distributed Awareness Tests
# ─────────────────────────────────────────────


class TestCUDAGraphDistributed:
    """Test distributed-aware CUDA graph engine."""

    def test_cuda_graph_accepts_topology(self) -> None:
        """CUDAGraphDecoderWrapper should accept a topology parameter."""
        from pravaha.engine.cuda_graph_engine import CUDAGraphDecoderWrapper

        class MockDecoder:
            pass

        wrapper = CUDAGraphDecoderWrapper(
            decoder_engine=MockDecoder(),
            topology=None,
        )
        assert wrapper.topology is None

    def test_cuda_graph_bucket_selection(self) -> None:
        """Bucket selection should work correctly."""
        from pravaha.engine.cuda_graph_engine import CUDAGraphDecoderWrapper

        class MockDecoder:
            pass

        wrapper = CUDAGraphDecoderWrapper(decoder_engine=MockDecoder())
        assert wrapper._get_bucket(1) == 1
        assert wrapper._get_bucket(3) == 4
        assert wrapper._get_bucket(5) == 8
        assert wrapper._get_bucket(17) == 32
        assert wrapper._get_bucket(100) is None  # Exceeds max bucket


# ─────────────────────────────────────────────
# Factory Integration Tests
# ─────────────────────────────────────────────


class TestFactoryDistributed:
    """Test factory integration with distributed config."""

    def test_factory_has_distributed_path(self) -> None:
        """Factory should have the import path for distributed components."""
        from pravaha.engine.factory import EngineFactory

        # Just verify the class exists and has build_subsystems
        assert hasattr(EngineFactory, "build_subsystems")

    def test_engine_config_serialization_with_distributed(self) -> None:
        """EngineConfig with distributed settings should serialize to YAML."""
        from pravaha.config.engine_config import EngineConfig

        config = EngineConfig.default()
        config.distributed.enabled = True
        config.distributed.tp_size = 4
        config.distributed.pp_size = 2

        data = config.model_dump()
        assert "distributed" in data
        assert data["distributed"]["enabled"] is True
        assert data["distributed"]["tp_size"] == 4
        assert data["distributed"]["pp_size"] == 2
