"""Distributed Tracing — OpenTelemetry integration.

Traces inference requests across engine components for debugging
and performance analysis.
"""

from __future__ import annotations
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Tracer:
    """OpenTelemetry tracer for Pravaha engine."""

    def __init__(self, service_name: str = "pravaha") -> None:
        self.service_name = service_name
        self._tracer = None
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            provider = TracerProvider()
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(service_name)
            self._available = True
        except ImportError:
            self._available = False

    def start_span(self, name: str, attributes: Optional[dict] = None) -> Any:
        if not self._available or self._tracer is None:
            return _NoOpSpan()
        span = self._tracer.start_span(name, attributes=attributes or {})
        return span

    def trace(self, name: str):
        """Decorator for tracing a function."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self._available:
                    return func(*args, **kwargs)
                with self._tracer.start_as_current_span(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


class _NoOpSpan:
    def end(self): pass
    def set_attribute(self, key, value): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
