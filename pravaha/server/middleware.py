"""Middleware stack for the Pravāha API Server.

Provides:
- RequestIDMiddleware: Injects unique X-Request-ID header for traceability.
- TimingMiddleware: Measures and logs request duration.
- ErrorHandlerMiddleware: Catches unhandled exceptions, returns structured JSON.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique X-Request-ID header into every response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        # Store on request state for downstream access
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Measure and log request duration, attach X-Response-Time header."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Response-Time"] = f"{elapsed_ms:.2f}ms"

        # Log only non-health endpoints to avoid noise
        if not request.url.path.startswith("/health"):
            logger.info(
                f"{request.method} {request.url.path} → {response.status_code} "
                f"({elapsed_ms:.1f}ms)"
            )
        return response


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Catch unhandled exceptions and return structured JSON errors."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"Unhandled error on {request.url.path}: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "object": "error",
                    "message": str(e),
                    "type": "internal_server_error",
                    "code": "500",
                },
            )
