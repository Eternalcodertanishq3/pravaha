"""Tests for configuration system."""

import tempfile
from pathlib import Path

import pytest

from pravaha.config import (
    CacheConfig,
    EngineConfig,
    ModelConfig,
    SamplingConfig,
    SchedulerConfig,
    ServerConfig,
)


class TestModelConfig:
    def test_default_values(self):
        config = ModelConfig()
        assert config.model_path == "gpt2"
        assert config.dtype == "float16"
        assert config.device == "cuda"

    def test_torch_dtype(self):
        import torch

        config = ModelConfig(dtype="float16")
        assert config.torch_dtype == torch.float16

        config = ModelConfig(dtype="float32")
        assert config.torch_dtype == torch.float32


class TestEngineConfig:
    def test_default_config(self):
        config = EngineConfig.default()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.scheduler, SchedulerConfig)
        assert isinstance(config.sampling, SamplingConfig)
        assert isinstance(config.server, ServerConfig)

    def test_yaml_roundtrip(self):
        config = EngineConfig.default()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"
            config.to_yaml(path)
            loaded = EngineConfig.from_yaml(path)
            assert loaded.model.model_path == config.model.model_path
            assert loaded.cache.block_size == config.cache.block_size
            assert loaded.sampling.temperature == config.sampling.temperature

    def test_from_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            EngineConfig.from_yaml("/nonexistent/config.yaml")

    def test_load_project_config(self):
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        if config_path.exists():
            config = EngineConfig.from_yaml(config_path)
            assert config.model.model_path == "gpt2"
            assert config.cache.block_size == 16
