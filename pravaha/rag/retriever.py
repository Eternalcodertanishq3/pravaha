"""RAG Retriever — Query the vector store and format context."""

from __future__ import annotations

import logging

from pravaha.rag.embedder import Embedder
from pravaha.rag.ingester import DocumentChunk
from pravaha.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieve relevant context from the vector store."""

    def __init__(
        self, embedder: Embedder, store: VectorStore, top_k: int = 5, threshold: float = 0.3
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(self, query: str) -> list[tuple[DocumentChunk, float]]:
        embedding = self.embedder.embed_single(query)
        results = self.store.search(embedding, self.top_k)
        return [(chunk, score) for chunk, score in results if score >= self.threshold]

    def format_context(self, results: list[tuple[DocumentChunk, float]]) -> str:
        if not results:
            return ""
        parts = []
        for chunk, score in results:
            parts.append(f"[Source: {chunk.source}, Score: {score:.2f}]\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    def query_and_format(self, query: str) -> str:
        results = self.retrieve(query)
        return self.format_context(results)
