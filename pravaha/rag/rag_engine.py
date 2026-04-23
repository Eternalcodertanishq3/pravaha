"""RAG Engine — Unified RAG pipeline combining ingestion, embedding, retrieval."""

from __future__ import annotations

import logging

from pravaha.config.rag_config import RAGConfig
from pravaha.rag.embedder import Embedder
from pravaha.rag.ingester import Ingester
from pravaha.rag.retriever import Retriever
from pravaha.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGEngine:
    """Unified RAG pipeline."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        self.ingester = Ingester(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
        )
        self.embedder = Embedder(
            model_name=self.config.embedding.model_name, device=self.config.embedding.device
        )
        self.store = VectorStore(persist_path=self.config.vector_store.persist_path)
        self.retriever = Retriever(
            embedder=self.embedder,
            store=self.store,
            top_k=self.config.retrieval.top_k,
            threshold=self.config.retrieval.score_threshold,
        )

    def ingest(self, path: str) -> int:
        chunks = self.ingester.ingest_file(path)
        if chunks:
            embeddings = self.embedder.embed([c.text for c in chunks])
            self.store.add(chunks, embeddings)
            self.store.save()
        logger.info(f"Ingested {len(chunks)} chunks from {path}")
        return len(chunks)

    def query(self, prompt: str) -> str:
        return self.retriever.query_and_format(prompt)

    def augment_prompt(self, prompt: str) -> str:
        context = self.query(prompt)
        if not context:
            return prompt
        template = self.config.retrieval.context_template
        return template.format(context=context) + prompt

    def get_stats(self) -> dict:
        return {"documents": self.store.count, "enabled": self.config.enabled}
