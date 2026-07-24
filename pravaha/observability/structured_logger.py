"""Structured JSON Logging — Production JSON log formatter with correlation ID context.

Formats python log records as structured JSON for log aggregators (Datadog, ELK, CloudWatch).
Automatically attaches Request ID from contextvars and installs secrets/PII filters.
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
from typing import Any

from pravaha.observability.log_filter import install_secrets_filter
from pravaha.observability.pii_filter import install_pii_filter

# Context variable for cross-thread / async correlation ID tracking
request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """Custom logging formatter that outputs log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        # Attach Correlation ID if set
        req_id = request_id_ctx.get()
        if req_id:
            log_data["request_id"] = req_id

        # Attach exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Attach extra fields passed to logger
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_data.update(record.extra)

        return json.dumps(log_data)


def setup_structured_logging(level: int = logging.INFO, json_format: bool = True) -> None:
    """Configure structured logging on the root logger.

    Args:
        level: Logging level (default: logging.INFO).
        json_format: If True, uses JSONFormatter; otherwise standard text format.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    if json_format:
        stream_handler.setFormatter(JSONFormatter())
    else:
        stream_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s in %(name)s: %(message)s")
        )

    root_logger.addHandler(stream_handler)

    # Automatically install redaction filters
    install_secrets_filter()
    install_pii_filter()

    logging.getLogger(__name__).info("Structured JSON logging initialized.")
