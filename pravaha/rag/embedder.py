"""RAG Embedder — Generate vector embeddings for text chunks."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Embedding model loaded: {self.model_name}")
            except ImportError:
                raise ImportError("sentence-transformers required for RAG")

    def embed(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return self._model.get_sentence_embedding_dimension()
