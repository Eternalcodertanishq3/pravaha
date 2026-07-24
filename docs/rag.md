# Pravāha v3.3 Retrieval-Augmented Generation (RAG) Architecture

This guide provides deep technical documentation for the **Retrieval-Augmented Generation (RAG)** pipeline in **Pravāha v3.3**. Pravāha's RAG subsystem combines multi-format document ingestion, semantic chunking, vector embedding generation, multi-backend vector databases (FAISS, Qdrant, ChromaDB, PGvector), hybrid dense/sparse retrieval with Reciprocal Rank Fusion (RRF), Cross-Encoder re-ranking, and integration with the 52-agent autonomous ReAct swarm.

---

## 1. RAG Subsystem Architecture Overview

Pravāha v3.3 implements a decoupled, end-to-end RAG architecture consisting of two primary operational pipelines: the **Ingestion & Indexing Pipeline** and the **Retrieval & Inference Pipeline**.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Operational Pipeline 1: Document Ingestion & Indexing                   │
│                                                                         │
│  ┌────────────┐    ┌─────────────────┐    ┌──────────────────┐          │
│  │ Document   │ ─► │ Text Normalizer │ ─► │ Semantic Chunker │          │
│  │ Source     │    │ & Metadata Ext. │    │ (Tokens/Sentences)          │
│  └────────────┘    └─────────────────┘    └────────┬─────────┘          │
│                                                    │                    │
│                                                    ▼                    │
│  ┌────────────┐    ┌─────────────────┐    ┌──────────────────┐          │
│  │ Vector DB  │ ◄─ │ SHA-256 Dedupe  │ ◄─ │ Embedding Engine │          │
│  │ Storage    │    │ & Ledger Log    │    │ (MiniLM/OpenAI)  │          │
│  └────────────┘    └─────────────────┘    └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Operational Pipeline 2: Hybrid Retrieval & Swarm Synthesizer            │
│                                                                         │
│  ┌────────────┐    ┌─────────────────┐    ┌──────────────────┐          │
│  │ Client     │ ─► │ HyDE Query      │ ─► │ Hybrid Search    │          │
│  │ Query      │    │ Expander        │    │ (Dense + BM25)   │          │
│  └────────────┘    └─────────────────┘    └────────┬─────────┘          │
│                                                    │                    │
│                                                    ▼                    │
│  ┌────────────┐    ┌─────────────────┐    ┌──────────────────┐          │
│  │ AsyncEngine│ ◄─ │ ReAct Swarm     │ ◄─ │ RRF Fusion       │          │
│  │ Generation │    │ Context Verification │ Cross-Encoder    │          │
│  └────────────┘    └─────────────────┘    └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Configuration Deep Dive (`configs/rag_default.yaml`)

Below is the complete configuration definition for Pravāha's RAG pipeline:

```yaml
# ==============================================================================
# Pravāha v3.3 Enterprise RAG Configuration
# ==============================================================================

rag:
  enabled: true
  
  # Embedding Model Configuration
  embedding:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    device: "cuda"               # Options: cuda, cpu, auto
    batch_size: 64
    normalize_embeddings: true
    dimension: 384

  # Text Splitting & Chunking Parameters
  chunking:
    strategy: "semantic"         # Options: fixed_size, sentence, semantic, markdown
    chunk_size: 512              # Target token count per chunk
    chunk_overlap: 64            # Overlapping tokens between adjacent chunks
    respect_document_structure: true

  # Vector Store Backend Selection
  vector_store:
    backend: "faiss"             # Options: faiss, qdrant, chroma, pgvector
    index_type: "HNSW"           # FAISS Index: Flat, IVF, HNSW
    distance_metric: "cosine"    # Options: cosine, L2, inner_product
    persist_directory: "./data/rag/indices"
    
    # Qdrant Specific Settings (Used if backend: qdrant)
    qdrant:
      host: "localhost"
      port: 6333
      collection_name: "pravaha_knowledge_base"

    # PGvector Specific Settings (Used if backend: pgvector)
    pgvector:
      connection_string: "postgresql://pravaha:secret@localhost:5432/pravaha_db"
      table_name: "document_embeddings"

  # Hybrid Retrieval & Re-ranking Settings
  retrieval:
    top_k: 10                    # Initial candidates retrieved from vector store
    similarity_threshold: 0.65   # Minimum cosine similarity cutoff
    enable_hybrid_search: true   # Combine Dense Embeddings + BM25 Sparse Search
    rrf_k: 60                    # Reciprocal Rank Fusion constant
    
    # Cross-Encoder Re-Ranking Stage
    reranking:
      enabled: true
      model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
      top_n: 4                   # Final candidate count sent to LLM prompt context

  # Security & Governance Integration
  security:
    enforce_rbac: true
    audit_ingestion: true
```

---

## 3. Document Ingestion Engine (`pravaha.rag.ingestion`)

Pravāha supports multi-format parsing with metadata preservation (file name, page numbers, author, creation timestamp, and SHA-256 content hashes).

### Supported File Formats

- **PDF Documents** (`.pdf`): Multi-page text extraction via `pypdf` with section heading preservation.
- **Markdown & Technical Specs** (`.md`): Structural parsing splitting on header boundaries (`#`, `##`, `###`).
- **Plain Text** (`.txt`): Arbitrary unstructured text processing.
- **Web Pages & HTML** (`.html`): DOM extraction stripping scripts, navigation headers, and boilerplate styling via `BeautifulSoup`.

### Python Ingestion Pipeline Implementation

```python
"""Document Ingestion & Indexing Engine for Pravāha v3.3."""

import os
import hashlib
from typing import List, Dict, Any
from pathlib import Path

from pravaha.rag.chunker import SemanticChunker
from pravaha.rag.embedder import VectorEmbedder
from pravaha.rag.vector_stores.base import BaseVectorStore


class DocumentIngestionEngine:
    """Manages document parsing, deduplication, chunking, and embedding generation."""

    def __init__(
        self,
        embedder: VectorEmbedder,
        vector_store: BaseVectorStore,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.indexed_hashes: set[str] = set()

    def _compute_sha256(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def ingest_file(self, file_path: str, collection: str = "default") -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        print(f"[*] Parsing document: {path.name}")
        raw_text = path.read_text(encoding="utf-8")
        file_hash = self._compute_sha256(raw_text)

        # SHA-256 Deduplication check
        if file_hash in self.indexed_hashes:
            print(f"[!] Skipping duplicate document (SHA-256: {file_hash[:12]})")
            return {"status": "skipped", "reason": "duplicate", "sha256": file_hash}

        # Generate semantic chunks
        chunks = self.chunker.split_text(raw_text)
        print(f"[*] Generated {len(chunks)} semantic chunks from {path.name}")

        # Compute embeddings in batched tensor pass
        chunk_texts = [c.text for c in chunks]
        embeddings = await self.embedder.embed_documents(chunk_texts)

        # Prepare payload records
        records = []
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            records.append({
                "id": f"{file_hash[:16]}_{idx}",
                "vector": emb,
                "text": chunk.text,
                "metadata": {
                    "source": str(path.resolve()),
                    "filename": path.name,
                    "chunk_index": idx,
                    "sha256": file_hash,
                    "collection": collection,
                },
            })

        # Insert records into Vector Database
        await self.vector_store.upsert(records, collection=collection)
        self.indexed_hashes.add(file_hash)

        return {
            "status": "success",
            "chunks_indexed": len(chunks),
            "file_name": path.name,
            "sha256": file_hash,
        }
```

---

## 4. Vector Store Backend Abstractions

Pravāha provides a unified abstraction layer (`BaseVectorStore`) supporting multiple enterprise vector databases.

### 1. FAISS Backend (`pravaha.rag.vector_stores.faiss_store`)

FAISS (Facebook AI Similarity Search) is Pravāha's default in-memory and disk-persisted vector engine, featuring HNSW (Hierarchical Navigable Small World) indexing for ultra-low latency searches.

```python
"""FAISS Vector Store Implementation with HNSW Indexing."""

import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any
from pravaha.rag.vector_stores.base import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):
    """FAISS HNSW Vector Store for fast local similarity search."""

    def __init__(self, dimension: int = 384, persist_dir: str = "./data/rag/faiss"):
        self.dimension = dimension
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct HNSW Index (M=32 connections per node)
        self.index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        self.metadata_db: List[Dict[str, Any]] = []

    async def upsert(self, records: List[Dict[str, Any]], collection: str = "default") -> None:
        vectors = np.array([r["vector"] for r in records], dtype=np.float32)
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)
        
        self.index.add(vectors)
        for r in records:
            self.metadata_db.append({
                "id": r["id"],
                "text": r["text"],
                "metadata": r["metadata"]
            })
            
    async def search(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        q_vec = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(q_vec)
        
        distances, indices = self.index.search(q_vec, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata_db):
                item = self.metadata_db[idx].copy()
                item["score"] = float(dist)
                results.append(item)
        return results
```

---

### 2. Qdrant & PGvector Integration Patterns

Pravāha also provides native connectors for distributed vector databases:

- **Qdrant Connector** (`pravaha.rag.vector_stores.qdrant_store`): Utilizes Qdrant's REST/gRPC client to perform collection filtering, payload index queries, and multi-tenant partition isolation.
- **PGvector Connector** (`pravaha.rag.vector_stores.pgvector_store`): Executes vectorized SQL queries over PostgreSQL using the `pgvector` extension:

```sql
-- Pravāha PGvector Similarity Query Pattern
SELECT id, chunk_text, metadata, 1 - (embedding <=> $1) AS similarity_score
FROM document_embeddings
WHERE metadata->>'collection' = 'enterprise_docs'
  AND 1 - (embedding <=> $1) > 0.65
ORDER BY embedding <=> $1 ASC
LIMIT 10;
```

---

## 5. Advanced Hybrid Search & Re-Ranking

Dense vector embeddings excel at capturing broad semantic context, while sparse keyword search (BM25) excels at exact matches (such as product IDs, error codes, and function names). Pravāha combines both modalities using **Reciprocal Rank Fusion (RRF)**.

### Hybrid Retrieval & RRF Fusion Algorithm

$$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where $M$ is the set of retrieval models (Dense Vector + BM25 Sparse), $r_m(d)$ is the rank position of document $d$ in model $m$, and $k$ is a smoothing constant (default: 60).

```python
"""Hybrid Search Engine with Reciprocal Rank Fusion (RRF) & Re-Ranking."""

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder


class HybridRAGRetriever:
    """Combines Dense Vector search with BM25 Sparse Search and Cross-Encoder Re-Ranking."""

    def __init__(self, vector_store: Any, bm25_searcher: Any, reranker_model: str):
        self.vector_store = vector_store
        self.bm25 = bm25_searcher
        self.reranker = CrossEncoder(reranker_model)

    async def hybrid_retrieve(self, query: str, top_k: int = 10, rrf_k: int = 60) -> List[Dict[str, Any]]:
        # 1. Execute Dense Vector Retrieval
        dense_results = await self.vector_store.search_by_text(query, top_k=top_k * 2)
        
        # 2. Execute BM25 Sparse Keyword Search
        sparse_results = self.bm25.search(query, top_k=top_k * 2)

        # 3. Apply Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            doc_map[doc_id] = doc

        for rank, doc in enumerate(sparse_results):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            doc_map[doc_id] = doc

        # Sort documents by RRF score
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        fused_candidates = [doc_map[did] for did in sorted_doc_ids]

        # 4. Perform Cross-Encoder Re-Ranking Pass
        pairs = [[query, doc["text"]] for doc in fused_candidates]
        cross_scores = self.reranker.predict(pairs)

        for doc, score in zip(fused_candidates, cross_scores):
            doc["rerank_score"] = float(score)

        # Final sort by Cross-Encoder score
        fused_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return fused_candidates[:4]  # Return top 4 re-ranked chunks
```

---

## 6. Agentic RAG Swarm Tools (`rag_search`)

Pravāha exposes the RAG pipeline to its 52-agent swarm as a sandboxed tool registered in `ToolRegistry`. ReAct agents can dynamically invoke retrieval during multi-step reasoning.

```python
# ReAct Agent Invocation Example inside Swarm Orchestrator
"""
Thought: The user asks about Pravāha's continuous scheduler configuration parameters.
Action : rag_search({"query": "continuous scheduler configuration parameters max_num_seqs"})
Observation: Retrieved 2 relevant chunks:
  [Source: configs/default.yaml] max_num_seqs specifies maximum active sequence concurrency limit (default: 256).
  [Source: docs/deployment.md] PRAVAHA_MAX_NUM_SEQS environment variable overrides scheduler concurrency.
Thought: I now have precise documentation context to answer the user query.
Final Answer: Pravāha's continuous scheduler concurrency is governed by...
"""
```

---

## 7. REST API & CLI Interfaces

### REST API Endpoints

#### 1. Ingest Document (`POST /v1/rag/ingest`)

```bash
curl -X POST http://localhost:8000/v1/rag/ingest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${PRAVAHA_API_KEY}" \
  -H "X-User-Role: operator" \
  -d '{
    "source": "./docs/architecture.md",
    "collection": "system_docs"
  }'
```

#### 2. Query Knowledge Base (`POST /v1/rag/query`)

```bash
curl -X POST http://localhost:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${PRAVAHA_API_KEY}" \
  -d '{
    "query": "How does PagedAttention eliminate memory fragmentation?",
    "top_k": 5,
    "hybrid_search": true
  }'
```

### CLI Command Suite

```bash
# Ingest single file or entire directory into RAG index
pravaha rag ingest ./docs/ --collection architecture_docs

# Query knowledge base from CLI
pravaha rag query "What is the role of BlockAllocator in Rust core?" --top-k 3

# List all indexed collections and document metadata
pravaha rag list

# Purge a specific collection or document from index
pravaha rag purge --collection architecture_docs
```

---

## 8. Security & Cryptographic Audit Integration

Every document ingested and query executed through the RAG pipeline is subjected to access control checks and audit logging:

1. **Role-Based Access Control (`RBACManager`)**: Ingestion endpoints require minimum `OPERATOR` role permissions.
2. **Cryptographic SHA-256 Audit Ledger (`SHA256AuditTrail`)**: Whenever a document is ingested, its SHA-256 fingerprint, chunk count, and executing user identity are written to an append-only log file.
3. **Data Sanitization**: Documents are scanned by `PIIFilter` before chunking to ensure sensitive enterprise secrets are not permanently indexed into vector storage.
