"""GGUF Model Loader — Load quantized GGUF models via llama-cpp-python.

Feature: Enables loading GGUF-format models (used by llama.cpp ecosystem)
directly into Pravaha for CPU and GPU inference.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GGUFLoader:
    """Load and run GGUF-format models.

    Uses llama-cpp-python as the backend for GGUF inference.
    """

    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 2048) -> None:
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self._model = None

    def load(self) -> Any:
        """Load the GGUF model."""
        try:
            from llama_cpp import Llama

            self._model = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=False,
            )
            logger.info(f"GGUF model loaded: {self.model_path}")
            return self._model
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF support. "
                "Install with: pip install 'pravaha[gguf]'"
            )

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> str:
        """Generate text using the GGUF model."""
        if self._model is None:
            self.load()
        assert self._model is not None
        output = self._model(prompt, max_tokens=max_tokens, **kwargs)
        return output["choices"][0]["text"]

    def generate_stream(self, prompt: str, max_tokens: int = 256, **kwargs: Any):
        """Stream tokens from the GGUF model."""
        if self._model is None:
            self.load()
        assert self._model is not None
        for chunk in self._model(prompt, max_tokens=max_tokens, stream=True, **kwargs):
            yield chunk["choices"][0]["text"]
