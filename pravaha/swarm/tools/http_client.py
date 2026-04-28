"""HTTP Client Tool — Full HTTP client for agent use.

Supports GET, POST, PUT, DELETE, PATCH with JSON bodies,
custom headers, and response parsing.
"""

from __future__ import annotations

import json
from typing import Any


class HttpClient:
    """Full HTTP client: GET/POST/PUT/DELETE/PATCH."""

    name = "http_request"
    description = "Make HTTP requests (GET/POST/PUT/DELETE/PATCH)"
    arg_schema = '{"method": "GET", "url": "...", "body": {}, "headers": {}, "timeout_s": 15}'

    def execute(
        self,
        method: str = "GET",
        url: str = "",
        body: dict[str, Any] | str | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: int = 15,
    ) -> dict[str, Any]:
        """Execute an HTTP request."""
        if not url:
            return {"error": "URL is required", "success": False}

        method = method.upper()
        if method not in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
            return {"error": f"Unsupported method: {method}", "success": False}

        try:
            import httpx
        except ImportError:
            return {"error": "httpx not installed", "success": False}

        req_headers = {"User-Agent": "Pravaha-Agent/3.3"}
        if headers:
            req_headers.update(headers)

        try:
            with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
                if method in ("POST", "PUT", "PATCH") and body:
                    if isinstance(body, dict):
                        req_headers.setdefault("Content-Type", "application/json")
                        response = client.request(
                            method, url, json=body, headers=req_headers
                        )
                    else:
                        response = client.request(
                            method, url, content=str(body), headers=req_headers
                        )
                else:
                    response = client.request(method, url, headers=req_headers)

            # Parse response body
            resp_body: Any
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    resp_body = response.json()
                except Exception:
                    resp_body = response.text[:4096]
            else:
                resp_body = response.text[:4096]

            return {
                "status_code": response.status_code,
                "body": resp_body,
                "headers": dict(response.headers),
                "url": str(response.url),
                "success": 200 <= response.status_code < 400,
            }

        except httpx.TimeoutException:
            return {"error": f"Request timed out after {timeout_s}s", "success": False}
        except httpx.ConnectError as e:
            return {"error": f"Connection failed: {e}", "success": False}
        except Exception as e:
            return {"error": f"HTTP error: {e}", "success": False}
