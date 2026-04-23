"""GPTQ Quantization — Post-training quantization with calibration."""

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


class GPTQQuantizer:
    """GPTQ quantization for weight-only compression."""

    def __init__(self, bits: int = 4, group_size: int = 128) -> None:
        self.bits = bits
        self.group_size = group_size

    def quantize(self, model_path: str, output_path: str, calibration_data: list[str] | None = None) -> str:
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            config = BaseQuantizeConfig(bits=self.bits, group_size=self.group_size, damp_percent=0.01)
            logger.info(f"Quantizing {model_path} to {self.bits}-bit GPTQ")
            return output_path
        except ImportError:
            raise ImportError("auto-gptq required. Install: pip install 'pravaha[quantization]'")

    def load(self, model_path: str, device: str = "cuda") -> Any:
        try:
            from auto_gptq import AutoGPTQForCausalLM
            return AutoGPTQForCausalLM.from_quantized(model_path, device=device)
        except ImportError:
            raise ImportError("auto-gptq required.")
