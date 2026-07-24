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

    def _validate_url(self, url: str) -> None:
        import ipaddress
        import socket
        import urllib.parse

        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

        if not parsed.hostname:
            raise ValueError("Missing hostname in URL")

        try:
            addr_info = socket.getaddrinfo(parsed.hostname, None)
        except socket.gaierror:
            raise ValueError(f"Could not resolve hostname: {parsed.hostname}")

        for addr in addr_info:
            ip_str = addr[4][0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                raise ValueError(f"URL resolves to private/reserved IP: {ip_str}")
            if ip in ipaddress.ip_network("10.0.0.0/8") or \
               ip in ipaddress.ip_network("127.0.0.0/8") or \
               ip in ipaddress.ip_network("172.16.0.0/12") or \
               ip in ipaddress.ip_network("192.168.0.0/16") or \
               ip in ipaddress.ip_network("169.254.0.0/16") or \
               ip in ipaddress.ip_network("0.0.0.0/8") or \
               ip == ipaddress.ip_address("::1") or \
               ip in ipaddress.ip_network("fc00::/7") or \
               ip in ipaddress.ip_network("fe80::/10"):
                raise ValueError(f"URL resolves to blocked IP range: {ip_str}")

    def execute(self, url: str, timeout_s: int = 10) -> dict:
        """Fetch URL and return extracted text."""
        try:
            self._validate_url(url)
        except ValueError as e:
            return {"error": str(e), "url": url, "success": False}

        timeout_s = min(timeout_s, 15)
        try:
            import httpx

            with httpx.Client(
                timeout=timeout_s,
                follow_redirects=True,
                max_redirects=3,
                headers={"User-Agent": "Pravaha/3.1 (Research Agent)"},
            ) as client:
                resp = client.get(url)
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
