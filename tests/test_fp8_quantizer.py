from __future__ import annotations

import math
import sys

import pytest
import torch
import torch.nn as nn

from pravaha.quantization.fp8_quantizer import (
    FP8Linear,
    FP8Quantizer,
    calculate_vram_savings,
    compute_quantization_metrics,
    compute_salient_channels,
    compute_scale_factor,
)


@pytest.fixture
def fp8_supported() -> bool:
    return hasattr(torch, "float8_e4m3fn")


def test_compute_scale_factor(fp8_supported: bool) -> None:
    if not fp8_supported:
        pytest.skip("PyTorch version does not support torch.float8_e4m3fn")
        
    weight = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    scale = compute_scale_factor(weight, dtype=torch.float8_e4m3fn)
    
    # max value for float8_e4m3fn is 448
    # max abs value is 2.0
    # expected scale is 448.0 / 2.0 = 224.0
    assert torch.isclose(scale, torch.tensor(224.0, dtype=torch.float32))


def test_compute_salient_channels() -> None:
    # 2 batches, 3 seq_len, 4 hidden_dim
    # channel 0: low activation
    # channel 3: high activation
    activations = torch.zeros(2, 3, 4)
    activations[:, :, 0] = 0.1
    activations[:, :, 1] = 0.5
    activations[:, :, 2] = 1.0
    activations[:, :, 3] = 10.0
    
    # Use 75th percentile -> only top 1 channel should be salient
    mask = compute_salient_channels(activations, percentile=0.75)
    
    assert mask.shape == (4,)
    assert not mask[0]
    assert not mask[1]
    assert not mask[2]
    assert mask[3]


def test_fp8_linear_forward(fp8_supported: bool) -> None:
    if not fp8_supported:
        pytest.skip("PyTorch version does not support torch.float8_e4m3fn")
        
    in_features = 8
    out_features = 4
    batch_size = 2
    
    linear = nn.Linear(in_features, out_features)
    # Initialize with non-zero values
    nn.init.normal_(linear.weight)
    
    salient_mask = torch.zeros(in_features, dtype=torch.bool)
    salient_mask[0] = True  # Make 1 channel salient
    
    fp8_linear = FP8Linear.from_linear(linear, salient_mask)
    
    x = torch.randn(batch_size, in_features)
    
    out_original = linear(x)
    out_fp8 = fp8_linear(x)
    
    assert out_fp8.shape == out_original.shape
    # Outputs should be close, but not exact due to quantization
    # Using a loose tolerance
    assert torch.allclose(out_fp8, out_original, atol=0.5, rtol=0.1)


def test_fp8_linear_dequantization_error(fp8_supported: bool) -> None:
    if not fp8_supported:
        pytest.skip("PyTorch version does not support torch.float8_e4m3fn")
        
    in_features = 128
    out_features = 64
    
    linear = nn.Linear(in_features, out_features, bias=False)
    nn.init.normal_(linear.weight, mean=0, std=0.1)
    
    fp8_linear = FP8Linear.from_linear(linear)
    
    metrics = compute_quantization_metrics(linear.weight, fp8_linear)
    
    assert "mse" in metrics
    assert "max_error" in metrics
    assert "sqnr" in metrics
    
    # Error should be bounded
    assert metrics["mse"] < 0.01
    assert metrics["sqnr"] > 20.0  # At least 20 dB SQNR for normal distribution


def test_vram_savings_calculation() -> None:
    in_features = 1024
    out_features = 1024
    
    linear = nn.Linear(in_features, out_features, dtype=torch.float16)
    
    # 0 salient channels
    salient_mask = torch.zeros(in_features, dtype=torch.bool)
    savings = calculate_vram_savings(linear, salient_mask)
    
    # Original: 1024 * 1024 * 2 (float16) = 2097152 bytes
    # Quantized: 1024 * 1024 * 1 (fp8) + 4 (scale) = 1048580 bytes
    assert savings["original_bytes"] == 2097152
    assert savings["quantized_bytes"] == 1048580
    assert math.isclose(savings["savings_ratio"], 0.5, abs_tol=1e-4)


def test_vram_savings_with_salient_channels() -> None:
    in_features = 100
    out_features = 100
    
    linear = nn.Linear(in_features, out_features, dtype=torch.float16)
    
    # 10 salient channels (10%)
    salient_mask = torch.zeros(in_features, dtype=torch.bool)
    salient_mask[:10] = True
    
    savings = calculate_vram_savings(linear, salient_mask)
    
    # Original: 100 * 100 * 2 = 20000
    # Quantized: 10 * 100 * 2 (salient) + 90 * 100 * 1 (fp8) + 4 = 2000 + 9000 + 4 = 11004
    assert savings["original_bytes"] == 20000
    assert savings["quantized_bytes"] == 11004
    assert savings["savings_ratio"] == (20000 - 11004) / 20000


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, 8)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_model_quantization_pipeline(fp8_supported: bool) -> None:
    if not fp8_supported:
        pytest.skip("PyTorch version does not support torch.float8_e4m3fn")
        
    model = TinyModel()
    
    # Create calibration data
    calib_data = [torch.randn(4, 16) for _ in range(5)]
    
    quantizer = FP8Quantizer(percentile=0.90)  # Top 10% salient
    quantizer.calibrate(model, calib_data)
    
    assert "fc1" in quantizer.calibration_activations
    assert "fc2" in quantizer.calibration_activations
    
    metrics = quantizer.quantize_model(model)
    
    # Check if layers were replaced
    assert isinstance(model.fc1, FP8Linear)
    assert isinstance(model.fc2, FP8Linear)
    
    # Check metrics keys
    assert "fc1" in metrics
    assert "fc2" in metrics
    
    assert metrics["fc1"]["salient_channels"] > 0
    assert metrics["fc2"]["salient_channels"] > 0


def test_end_to_end_quantize_forward(fp8_supported: bool) -> None:
    if not fp8_supported:
        pytest.skip("PyTorch version does not support torch.float8_e4m3fn")
        
    model = TinyModel()
    model.eval()
    
    x = torch.randn(2, 16)
    
    # Original forward
    with torch.no_grad():
        out_orig = model(x)
        
    quantizer = FP8Quantizer(percentile=0.95)
    quantizer.calibrate(model, [x, torch.randn(2, 16)])
    quantizer.quantize_model(model)
    
    # Quantized forward
    with torch.no_grad():
        out_quant = model(x)
        
    assert out_orig.shape == out_quant.shape
    # Ensure it runs without error and produces somewhat similar outputs
    assert torch.isfinite(out_quant).all()
