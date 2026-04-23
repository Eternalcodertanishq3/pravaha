"""BitsAndBytes Quantization — INT4/INT8 weight compression.

Integrates with the bitsandbytes library for on-the-fly quantization
during model loading. Supports 4-bit NF4 and 8-bit LLM.int8().
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BitsAndBytesQuantizer:
    """Quantize models using bitsandbytes."""

    def __init__(self, bits: int = 4) -> None:
        self.bits = bits

    def get_config(self) -> Any:
        """Get a BitsAndBytesConfig for transformers loading."""
        try:
            from transformers import BitsAndBytesConfig
            import torch
            if self.bits == 4:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            elif self.bits == 8:
                return BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError(f"Unsupported bits: {self.bits}")
        except ImportError:
            raise ImportError("bitsandbytes required. Install: pip install 'pravaha[quantization]'")
