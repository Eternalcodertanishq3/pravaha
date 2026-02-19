"""Tests for model configuration parsing and architecture detection."""

import pytest

from pravaha.models.model_config import (
    ModelArchConfig,
    _detect_architecture,
    parse_model_config,
)


class TestDetectArchitecture:
    """Test architecture detection from HuggingFace config dicts."""

    def test_detect_gpt2_by_model_type(self):
        config = {"model_type": "gpt2"}
        assert _detect_architecture(config) == "gpt2"

    def test_detect_llama_by_model_type(self):
        config = {"model_type": "llama"}
        assert _detect_architecture(config) == "llama"

    def test_detect_mistral_by_model_type(self):
        config = {"model_type": "mistral"}
        assert _detect_architecture(config) == "mistral"

    def test_detect_llama_from_architectures_list(self):
        config = {"model_type": "", "architectures": ["LlamaForCausalLM"]}
        assert _detect_architecture(config) == "llama"

    def test_unsupported_architecture_raises(self):
        config = {"model_type": "unknown_arch", "architectures": []}
        with pytest.raises(ValueError, match="Unsupported model architecture"):
            _detect_architecture(config)


class TestModelArchConfig:
    """Test ModelArchConfig properties and memory estimation."""

    @pytest.fixture
    def gpt2_config(self) -> ModelArchConfig:
        return ModelArchConfig(
            arch_name="gpt2",
            num_layers=12,
            hidden_size=768,
            num_heads=12,
            num_kv_heads=12,
            intermediate_size=3072,
            vocab_size=50257,
            max_position_embeddings=1024,
        )

    def test_head_dim_computed(self, gpt2_config: ModelArchConfig):
        assert gpt2_config.head_dim == 64  # 768 / 12

    def test_kv_cache_size_per_token(self, gpt2_config: ModelArchConfig):
        # 2 (K+V) × 12 heads × 64 head_dim × 2 bytes = 3072 bytes
        assert gpt2_config.kv_cache_size_per_token == 3072

    def test_total_kv_cache_per_token(self, gpt2_config: ModelArchConfig):
        # 3072 × 12 layers = 36864 bytes
        assert gpt2_config.total_kv_cache_per_token == 36864

    def test_memory_estimation_positive(self, gpt2_config: ModelArchConfig):
        mem = gpt2_config.estimate_model_memory_bytes(dtype_bytes=2)
        assert mem > 0
        # GPT-2 small is ~124M params → ~248MB in fp16
        assert 200_000_000 < mem < 400_000_000


class TestParseModelConfig:
    """Test parsing real HuggingFace model configs.

    These tests require network access to download model configs.
    Mark as slow/integration tests if needed.
    """

    @pytest.mark.slow
    def test_parse_gpt2(self):
        config = parse_model_config("gpt2")
        assert config.arch_name == "gpt2"
        assert config.num_layers == 12
        assert config.hidden_size == 768
        assert config.num_heads == 12
        assert config.vocab_size == 50257

    @pytest.mark.slow
    def test_parse_gpt2_kv_cache_size(self):
        config = parse_model_config("gpt2")
        # KV-cache per token should be reasonable
        assert config.total_kv_cache_per_token > 0
        assert config.total_kv_cache_per_token < 1_000_000  # < 1MB per token
