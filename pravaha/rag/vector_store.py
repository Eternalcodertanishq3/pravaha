"""RAG Vector Store — FAISS-based vector storage and retrieval."""

from __future__ import annotations
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from pravaha.rag.ingester import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-backed vector store for RAG retrieval."""

    def __init__(self, dimension: int = 384, persist_path: Optional[str] = None) -> None:
        self.dimension = dimension
        self.persist_path = persist_path
        self._index = None
        self._chunks: list[DocumentChunk] = []
        self._init_index()

    def _init_index(self) -> None:
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)
            if self.persist_path:
                p = Path(self.persist_path)
                if (p / "index.faiss").exists():
                    self._index = faiss.read_index(str(p / "index.faiss"))
                    logger.info(f"Loaded FAISS index from {self.persist_path}")
        except ImportError:
            logger.warning("faiss-cpu not installed, vector store disabled")

    def add(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        if self._index is None:
            return
        import faiss
        self._index.add(embeddings.astype(np.float32))
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        import faiss
        q = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(q, min(top_k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._chunks):
                results.append((self._chunks[idx], float(score)))
        return results

    def save(self) -> None:
        if self._index is None or not self.persist_path:
            return
        import faiss
        p = Path(self.persist_path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(p / "index.faiss"))

    @property
    def count(self) -> int:
        return self._index.ntotal if self._index else 0
