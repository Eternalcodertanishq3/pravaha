# RAG Pipeline Guide

## Overview

Pravāha includes a built-in Retrieval-Augmented Generation (RAG) pipeline for grounding LLM responses in your own documents.

## Quick Start

```bash
# Enable RAG and serve
pravaha serve gpt2 --rag

# Ingest documents
pravaha rag ingest ./docs/
pravaha rag ingest https://example.com/page

# Query
pravaha rag query "How does caching work?"
```

## Architecture

```
Document → Chunker → Embedder → Vector Store (FAISS)
                                      ↓
Query → Embedder → Similarity Search → Top-K chunks → LLM Context
```

## Configuration

```yaml
rag:
  enabled: true
  embedding_model: all-MiniLM-L6-v2  # sentence-transformers model
  chunk_size: 512                      # Tokens per chunk
  chunk_overlap: 64                    # Overlap between chunks
  top_k: 5                            # Results to retrieve
  similarity_threshold: 0.7           # Minimum similarity score
  vector_store:
    type: faiss                        # Vector store backend
    save_path: ./data/rag/index.faiss  # Persistence path
```

## Supported Formats

- PDF (`.pdf`)
- Plain text (`.txt`)
- Markdown (`.md`)
- HTML (`.html`)
- URLs (web pages)

## API Endpoints

### Ingest

```bash
POST /v1/rag/ingest
{"source": "./my_document.pdf"}
```

### Query

```bash
GET /v1/rag/query?query=How+does+X+work&top_k=5
```

### List Sources

```bash
GET /v1/rag/sources
```

## CLI Commands

```bash
pravaha rag ingest <file_or_url>
pravaha rag query "your question" --top-k 10
pravaha rag list
pravaha rag remove <doc-id>
```
