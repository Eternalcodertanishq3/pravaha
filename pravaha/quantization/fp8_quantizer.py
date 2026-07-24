from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_scale_factor(weight: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn) -> torch.Tensor:
    """
    Computes the scale factor for FP8 quantization.
    """
    amax = weight.abs().amax()
    if amax == 0:
        return torch.tensor(1.0, dtype=weight.dtype, device=weight.device)

    # Get the maximum representable value for the target dtype
    try:
        max_val = torch.finfo(dtype).max
    except TypeError:
        # Fallback if dtype is not supported by finfo, shouldn't happen with valid float8
        max_val = 448.0  # max value for float8_e4m3fn

    scale = max_val / amax.clamp(min=1e-12)
    return scale


def compute_salient_channels(
    activations: torch.Tensor, percentile: float = 0.99
) -> torch.Tensor:
    """
    Identifies salient channels from calibration activations.
    Returns a boolean mask where True indicates a salient channel.
    """
    # activations: [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
    if activations.dim() == 3:
        act_scales = activations.abs().mean(dim=(0, 1))
    elif activations.dim() == 2:
        act_scales = activations.abs().mean(dim=0)
    else:
        raise ValueError(f"Unsupported activation dimension: {activations.dim()}")

    threshold = torch.quantile(act_scales.float(), percentile)
    salient_mask = act_scales > threshold
    return salient_mask


class FP8Linear(nn.Module):
    """
    A Linear layer that stores weights in FP8 but dequantizes to FP16 for the matmul.
    Maintains salient channels in FP16.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Placeholders for weights
        # Non-salient weights in FP8
        self.register_buffer("weight_fp8", torch.empty((out_features, in_features), dtype=torch.uint8))
        self.register_buffer("weight_scale", torch.empty((1,), dtype=torch.float32))

        # Salient mask and indices
        self.register_buffer("salient_mask", torch.zeros(in_features, dtype=torch.bool))

        # Salient weights in original dtype (FP16/FP32)
        # Store as parameter so it moves with the module, though we could use buffers
        # We'll initialize it dynamically during quantization
        self.register_buffer("weight_salient", torch.empty((out_features, 0)))

        if bias:
            self.register_buffer("bias", torch.empty((out_features,)))
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        salient_mask: torch.Tensor | None = None
    ) -> FP8Linear:
        """
        Convert an existing nn.Linear to FP8Linear.
        """
        device = linear.weight.device
        dtype = linear.weight.dtype
        in_features = linear.in_features
        out_features = linear.out_features

        fp8_linear = cls(in_features, out_features, bias=linear.bias is not None)
        fp8_linear = fp8_linear.to(device)

        weight = linear.weight.data

        if salient_mask is None:
            salient_mask = torch.zeros(in_features, dtype=torch.bool, device=device)

        fp8_linear.salient_mask = salient_mask

        # Separate salient and non-salient weights
        # weight shape: (out_features, in_features)
        # salient_mask shape: (in_features,)
        num_salient = salient_mask.sum().item()

        if num_salient > 0:
            salient_weights = weight[:, salient_mask]
            fp8_linear.weight_salient = salient_weights.clone()
        else:
            fp8_linear.weight_salient = torch.empty((out_features, 0), dtype=dtype, device=device)

        # Quantize non-salient weights
        non_salient_mask = ~salient_mask
        if non_salient_mask.sum().item() > 0:
            non_salient_weights = weight[:, non_salient_mask]

            # Target fp8 type
            fp8_dtype = getattr(torch, "float8_e4m3fn", None)
            if fp8_dtype is not None:
                scale = compute_scale_factor(non_salient_weights, dtype=fp8_dtype)
                scaled_weights = non_salient_weights * scale
                fp8_weights = scaled_weights.to(fp8_dtype)

                # Store scale
                fp8_linear.weight_scale = scale.to(torch.float32)
                # Store as uint8 for compatibility if needed, or directly as fp8
                # Since we use fp8_dtype directly in pytorch >= 2.1, we can store it as such
                fp8_linear.weight_fp8 = fp8_weights
            else:
                # Fallback for older pytorch: simulate fp8
                logger.warning("torch.float8_e4m3fn not found. Simulating FP8 with uint8.")
                scale = compute_scale_factor(non_salient_weights)
                scaled_weights = non_salient_weights * scale
                # Fake quantization
                fp8_weights = scaled_weights.round().clamp(-128, 127).to(torch.int8)
                fp8_linear.weight_scale = scale.to(torch.float32)
                fp8_linear.weight_fp8 = fp8_weights
        else:
            fp8_linear.weight_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
            if hasattr(torch, "float8_e4m3fn"):
                fp8_linear.weight_fp8 = torch.empty((out_features, 0), dtype=torch.float8_e4m3fn, device=device)
            else:
                fp8_linear.weight_fp8 = torch.empty((out_features, 0), dtype=torch.int8, device=device)

        if linear.bias is not None:
            fp8_linear.bias = linear.bias.data.clone()

        return fp8_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dequantization.
        """
        # x shape: [..., in_features]
        # Reconstruct the weight matrix in FP16/FP32
        compute_dtype = x.dtype
        device = x.device

        reconstructed_weight = torch.zeros(
            (self.out_features, self.in_features),
            dtype=compute_dtype,
            device=device
        )

        # Dequantize non-salient weights
        non_salient_mask = ~self.salient_mask
        if non_salient_mask.sum().item() > 0:
            if self.weight_fp8.dtype in [getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)]:
                dequantized = self.weight_fp8.to(compute_dtype) / self.weight_scale
            else:
                # Int8 fallback
                dequantized = self.weight_fp8.to(compute_dtype) / self.weight_scale

            reconstructed_weight[:, non_salient_mask] = dequantized

        # Place salient weights
        if self.salient_mask.sum().item() > 0:
            reconstructed_weight[:, self.salient_mask] = self.weight_salient.to(compute_dtype)

        return F.linear(x, reconstructed_weight, self.bias)


def calculate_vram_savings(
    linear: nn.Linear, salient_mask: torch.Tensor | None = None
) -> dict[str, int | float]:
    """
    Computes VRAM savings when converting a linear layer to FP8.
    """
    in_features = linear.in_features
    out_features = linear.out_features

    # Original memory (assume FP16 -> 2 bytes)
    element_size = linear.weight.element_size()
    original_bytes = in_features * out_features * element_size

    # Quantized memory
    if salient_mask is not None:
        num_salient = salient_mask.sum().item()
    else:
        num_salient = 0

    num_non_salient = in_features - num_salient

    # Salient weights stay in original dtype
    salient_bytes = num_salient * out_features * element_size
    # Non-salient weights go to FP8 (1 byte)
    fp8_bytes = num_non_salient * out_features * 1
    # Scale factor (float32 -> 4 bytes)
    scale_bytes = 4

    quantized_bytes = salient_bytes + fp8_bytes + scale_bytes

    savings_bytes = original_bytes - quantized_bytes
    savings_ratio = savings_bytes / original_bytes if original_bytes > 0 else 0.0

    return {
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
        "savings_bytes": savings_bytes,
        "savings_ratio": savings_ratio
    }


def compute_quantization_metrics(
    original_weight: torch.Tensor,
    quantized_linear: FP8Linear
) -> dict[str, float]:
    """
    Computes MSE, Max Absolute Error, and SQNR between original and quantized weights.
    """
    # Reconstruct weight to compute metrics
    compute_dtype = original_weight.dtype

    reconstructed = torch.zeros_like(original_weight)

    non_salient_mask = ~quantized_linear.salient_mask
    if non_salient_mask.sum().item() > 0:
        dequantized = quantized_linear.weight_fp8.to(compute_dtype) / quantized_linear.weight_scale
        reconstructed[:, non_salient_mask] = dequantized

    if quantized_linear.salient_mask.sum().item() > 0:
        reconstructed[:, quantized_linear.salient_mask] = quantized_linear.weight_salient.to(compute_dtype)

    diff = original_weight - reconstructed

    mse = (diff ** 2).mean().item()
    max_error = diff.abs().max().item()

    # SQNR: Signal-to-Quantization-Noise Ratio
    signal_power = (original_weight ** 2).mean()
    noise_power = (diff ** 2).mean()

    if noise_power.item() == 0:
        sqnr = float('inf')
    else:
        sqnr = 10 * torch.log10(signal_power / noise_power).item()

    return {
        "mse": mse,
        "max_error": max_error,
        "sqnr": sqnr
    }


class FP8Quantizer:
    """
    Engine for quantizing a model to FP8 with AWQ-aware salient channel protection.
    """
    def __init__(self, percentile: float = 0.99):
        self.percentile = percentile
        self.calibration_activations: dict[str, list[torch.Tensor]] = {}
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _register_hooks(self, model: nn.Module) -> None:
        def get_activation_hook(name: str) -> Callable:
            def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                if name not in self.calibration_activations:
                    self.calibration_activations[name] = []
                # inputs[0] is the input to the linear layer, shape: [..., in_features]
                x = inputs[0].detach().cpu()
                self.calibration_activations[name].append(x)
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(get_activation_hook(name))
                self.hooks.append(handle)

    def _remove_hooks(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def calibrate(self, model: nn.Module, calibration_data: list[torch.Tensor]) -> None:
        """
        Run forward passes with calibration data to collect activations.
        """
        self._register_hooks(model)

        model.eval()
        with torch.no_grad():
            for data in calibration_data:
                _ = model(data)

        self._remove_hooks()

    def quantize_model(self, model: nn.Module) -> dict[str, dict[str, Any]]:
        """
        Quantizes all nn.Linear layers in the model to FP8Linear.
        Returns a dictionary with quantization metrics and VRAM savings per layer.
        """
        metrics: dict[str, dict[str, Any]] = {}

        # We need to replace modules, which requires parent access
        def replace_linear(parent: nn.Module, prefix: str = "") -> None:
            for name, child in parent.named_children():
                full_name = f"{prefix}.{name}" if prefix else name

                if isinstance(child, nn.Linear):
                    # Check if we have calibration data for salient channel extraction
                    salient_mask = None
                    if full_name in self.calibration_activations:
                        acts = torch.cat(self.calibration_activations[full_name], dim=0)
                        salient_mask = compute_salient_channels(acts, self.percentile)
                        salient_mask = salient_mask.to(child.weight.device)

                    # Calculate savings
                    savings = calculate_vram_savings(child, salient_mask)

                    # Quantize
                    fp8_layer = FP8Linear.from_linear(child, salient_mask)

                    # Calculate metrics
                    err_metrics = compute_quantization_metrics(child.weight.data, fp8_layer)

                    # Replace
                    setattr(parent, name, fp8_layer)

                    metrics[full_name] = {
                        "savings": savings,
                        "metrics": err_metrics,
                        "salient_channels": int(salient_mask.sum().item()) if salient_mask is not None else 0
                    }

                    logger.info(f"Quantized layer {full_name}: SQNR={err_metrics['sqnr']:.2f} dB, "
                                f"Savings={savings['savings_ratio']*100:.1f}%")
                else:
                    replace_linear(child, full_name)

        replace_linear(model)
        return metrics

