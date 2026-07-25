"""Serving middleware — RequestID, Timing, Error handling, Auth, RateLimit."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from pravaha.observability.structured_logger import request_id_ctx

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique X-Request-ID header into every request/response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        token = request_id_ctx.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_ctx.reset(token)


class TimingMiddleware(BaseHTTPMiddleware):
    """Add X-Process-Time header measuring request duration."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.time()
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        response.headers["X-Process-Time"] = f"{duration:.1f}ms"
        return response


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Catch unhandled exceptions and return structured JSON errors."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "type": type(e).__name__}},
            )

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter by client IP."""

    def __init__(self, app: Any, max_requests: int = 100, window_seconds: int = 60) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self._counts: dict[str, list[float]] = {}
        self._last_cleanup = time.time()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        now = time.time()

        if now - self._last_cleanup > self.window:
            self._counts = {
                ip: timestamps
                for ip, timestamps in self._counts.items()
                if timestamps and now - timestamps[-1] < self.window
            }
            self._last_cleanup = now

        client_ip = request.client.host if request.client else "unknown"
        if client_ip not in self._counts:
            self._counts[client_ip] = []
        self._counts[client_ip] = [t for t in self._counts[client_ip] if now - t < self.window]
        if len(self._counts[client_ip]) >= self.max_requests:
            return JSONResponse(
                status_code=429, content={"error": {"message": "Rate limit exceeded"}}
            )
        self._counts[client_ip].append(now)
        return await call_next(request)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """API Key authentication middleware."""

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        import os
        self.api_key = os.environ.get("PRAVAHA_API_KEY")
        if not self.api_key:
            logger.warning("PRAVAHA_API_KEY is not set. API authentication is DISABLED.")
        self.excluded_paths = {"/health", "/health/ready", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        if not self.api_key:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != self.api_key:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Missing or invalid API key", "type": "authentication_error"}},
            )

        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response
