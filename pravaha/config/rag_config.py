"""RAG configuration — chunking, embedding, retrieval parameters.

Controls the built-in RAG pipeline: document ingestion, embedding model,
chunking strategy, vector store backend, and retrieval settings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChunkingConfig(BaseModel):
    """Document chunking configuration.

    Attributes:
        strategy: Chunking strategy — 'fixed' splits by token count,
            'sentence' splits at sentence boundaries, 'paragraph' at double newlines.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Number of overlapping tokens between adjacent chunks.
        min_chunk_size: Minimum chunk size to keep (smaller chunks are merged).
    """

    strategy: Literal["fixed", "sentence", "paragraph"] = "sentence"
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50


class EmbeddingConfig(BaseModel):
    """Embedding model configuration.

    Attributes:
        model_name: HuggingFace model identifier for the embedding model.
        device: Device for embedding computation.
        batch_size: Batch size for embedding generation.
        normalize: Whether to L2-normalize embeddings.
    """

    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True


class VectorStoreConfig(BaseModel):
    """Vector store configuration.

    Attributes:
        backend: Vector store backend. Currently only FAISS is supported.
        index_type: FAISS index type — 'flat' for exact search, 'ivf' for approximate.
        nprobe: Number of probes for IVF index search (higher = more accurate but slower).
        persist_path: Path to persist the vector store index.
    """

    backend: Literal["faiss"] = "faiss"
    index_type: Literal["flat", "ivf"] = "flat"
    nprobe: int = 10
    persist_path: str = "data/rag/index"


class RetrievalConfig(BaseModel):
    """Retrieval settings.

    Attributes:
        top_k: Number of top results to retrieve.
        score_threshold: Minimum similarity score to include a result.
        rerank: Whether to rerank results using a cross-encoder.
        context_template: Template for injecting retrieved context into the prompt.
    """

    top_k: int = 5
    score_threshold: float = 0.3
    rerank: bool = False
    context_template: str = "[Context from documents:]\n{context}\n[End context]\n\n"


class RAGConfig(BaseModel):
    """Full RAG pipeline configuration.

    Loaded from configs/rag_default.yaml. Controls every aspect of the
    built-in RAG system from ingestion to retrieval.
    """

    enabled: bool = False
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    supported_formats: list[str] = Field(
        default_factory=lambda: ["pdf", "txt", "md", "html", "url"]
    )
    max_documents: int = 10000
    store_path: str = "data/rag"

    @classmethod
    def from_yaml(cls, path: str | Path) -> RAGConfig:
        """Load RAG configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"RAG config not found: {path}, using defaults.")
            return cls()

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls.model_validate(raw or {})
