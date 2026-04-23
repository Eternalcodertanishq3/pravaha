"""AWQ Quantization — Activation-Aware Weight Quantization."""

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


class AWQQuantizer:
    """AWQ quantization using activation-aware weight scaling."""

    def __init__(self, bits: int = 4) -> None:
        self.bits = bits

    def load(self, model_path: str, device: str = "cuda") -> Any:
        try:
            from awq import AutoAWQForCausalLM
            return AutoAWQForCausalLM.from_quantized(model_path, device_map=device)
        except ImportError:
            raise ImportError("autoawq required. Install: pip install 'pravaha[quantization]'")
