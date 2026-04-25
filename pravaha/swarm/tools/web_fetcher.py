"""Web Fetcher — Fetch content from URLs and extract text."""

from __future__ import annotations

import logging
from html.parser import HTMLParser

logger = logging.getLogger(__name__)


class _TextExtractor(HTMLParser):
    """Simple HTML-to-text extractor."""

    def __init__(self):
        super().__init__()
        self.text_parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            stripped = data.strip()
            if stripped:
                self.text_parts.append(stripped)


class WebFetcher:
    """Fetch content from a URL and extract readable text."""

    name = "fetch_url"
    description = "Fetch content from a URL and extract text"
    arg_schema = '{"url": "string", "timeout_s": 10}'

    MAX_OUTPUT_BYTES = 4096

    def execute(self, url: str, timeout_s: int = 10) -> dict:
        """Fetch URL and return extracted text."""
        timeout_s = min(timeout_s, 15)
        try:
            import httpx

            resp = httpx.get(
                url,
                timeout=timeout_s,
                follow_redirects=True,
                headers={"User-Agent": "Pravaha/3.1 (Research Agent)"},
            )
            resp.raise_for_status()

            parser = _TextExtractor()
            parser.feed(resp.text)
            text = " ".join(parser.text_parts)[: self.MAX_OUTPUT_BYTES]

            return {
                "url": url,
                "status": resp.status_code,
                "text": text,
                "content_type": resp.headers.get("content-type", ""),
                "success": True,
            }
        except Exception as e:
            return {"error": str(e), "url": url, "success": False}
