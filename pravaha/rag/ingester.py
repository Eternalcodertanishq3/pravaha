"""RAG Ingester — Document loading and chunking pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of text from an ingested document."""

    text: str
    source: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


class Ingester:
    """Ingest documents from various formats and split into chunks."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_file(self, path: str) -> list[DocumentChunk]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        ext = p.suffix.lower()
        if ext == ".pdf":
            text = self._load_pdf(p)
        elif ext in (".txt", ".md"):
            text = p.read_text(encoding="utf-8")
        elif ext in (".html", ".htm"):
            text = self._load_html(p)
        else:
            text = p.read_text(encoding="utf-8")
        return self._chunk_text(text, source=str(p))

    def ingest_text(self, text: str, source: str = "direct") -> list[DocumentChunk]:
        return self._chunk_text(text, source=source)

    def _chunk_text(self, text: str, source: str) -> list[DocumentChunk]:
        words = text.split()
        chunks = []
        i = 0
        idx = 0
        while i < len(words):
            end = min(i + self.chunk_size, len(words))
            chunk_text = " ".join(words[i:end])
            chunks.append(DocumentChunk(text=chunk_text, source=source, chunk_index=idx))
            i += self.chunk_size - self.chunk_overlap
            idx += 1
        return chunks

    def _load_pdf(self, path: Path) -> str:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise ImportError("pypdf required for PDF support")

    def _load_html(self, path: Path) -> str:
        import re

        html = path.read_text(encoding="utf-8")
        return re.sub(r"<[^>]+>", " ", html)
