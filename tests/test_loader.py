"""Tests for the model loader."""

import pytest
import torch

from pravaha.models.loader import ModelLoader


@pytest.fixture(scope="module")
def loader() -> ModelLoader:
    return ModelLoader()


class TestModelLoader:
    """Test model loading functionality."""

    @pytest.mark.slow
    def test_load_gpt2_fp32(self, loader: ModelLoader):
        model, arch = loader.load_model("gpt2", dtype=torch.float32, device="cpu")
        assert model is not None
        assert arch.arch_name == "gpt2"
        assert arch.num_layers == 12
        # Model should be in eval mode
        assert not model.training

    @pytest.mark.slow
    def test_load_gpt2_fp16_cpu(self, loader: ModelLoader):
        model, arch = loader.load_model("gpt2", dtype=torch.float16, device="cpu")
        assert model is not None
        # Check params are fp16
        for param in model.parameters():
            if param.is_floating_point():
                assert param.dtype == torch.float16
                break

    @pytest.mark.slow
    def test_no_gradients(self, loader: ModelLoader):
        model, _ = loader.load_model("gpt2", dtype=torch.float32, device="cpu")
        for param in model.parameters():
            assert not param.requires_grad


class TestMemoryEstimation:
    """Test memory estimation utilities."""

    def test_estimate_memory_gpt2(self):
        loader = ModelLoader()
        estimates = loader.estimate_memory("gpt2", dtype=torch.float16)
        assert estimates["model_weights_gb"] > 0
        assert estimates["kv_cache_gb"] > 0
        assert estimates["total_gb"] > 0
        assert estimates["architecture"] == "gpt2"
        # GPT-2 small is ~124M params
        assert 0.05 < estimates["num_params_billions"] < 0.5

    def test_dtype_to_bytes(self):
        assert ModelLoader._dtype_to_bytes(torch.float32) == 4
        assert ModelLoader._dtype_to_bytes(torch.float16) == 2
        assert ModelLoader._dtype_to_bytes(torch.bfloat16) == 2
        assert ModelLoader._dtype_to_bytes(torch.int8) == 1
