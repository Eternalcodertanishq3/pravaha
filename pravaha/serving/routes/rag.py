"""RAG API — /v1/rag endpoints for document management and retrieval."""

from __future__ import annotations

from fastapi import APIRouter, File, Request, UploadFile
from pydantic import BaseModel

router = APIRouter(tags=["RAG"])


class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/rag/query")
async def rag_query(request: RAGQueryRequest, raw_request: Request):
    from pravaha.rag.rag_engine import RAGEngine

    rag = RAGEngine()
    context = rag.query(request.query)
    return {"context": context, "query": request.query}


@router.post("/rag/ingest")
async def rag_ingest(file: UploadFile = File(...)):
    import os
    import tempfile

    from pravaha.rag.rag_engine import RAGEngine

    rag = RAGEngine()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        chunks = rag.ingest(tmp_path)
        return {"status": "ok", "chunks_ingested": chunks, "filename": file.filename}
    finally:
        os.unlink(tmp_path)


@router.get("/rag/stats")
async def rag_stats():
    from pravaha.rag.rag_engine import RAGEngine

    rag = RAGEngine()
    return rag.get_stats()
