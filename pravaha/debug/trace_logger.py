"""Trace Logger — Structured logging for inference pipeline debugging."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class TraceLogger:
    """Log structured traces of the inference pipeline."""

    def __init__(self, output_path: str = "data/debug/traces") -> None:
        self.output_path = Path(output_path)
        self._traces: list[dict] = []

    def log(self, component: str, action: str, data: dict | None = None) -> None:
        entry = {
            "timestamp": time.time(),
            "component": component,
            "action": action,
            "data": data or {},
        }
        self._traces.append(entry)

    def flush(self, request_id: str) -> None:
        if not self._traces:
            return
        self.output_path.mkdir(parents=True, exist_ok=True)
        path = self.output_path / f"{request_id}.jsonl"
        with open(path, "a") as f:
            for trace in self._traces:
                f.write(json.dumps(trace) + "\n")
        self._traces.clear()

    def get_traces(self) -> list[dict]:
        return self._traces.copy()

    def get_trace(self, request_id: str) -> list[dict]:
        """Load trace from disk for a given request ID (route-compatible)."""
        path = self.output_path / f"{request_id}.jsonl"
        if not path.exists():
            return []
        traces: list[dict] = []
        with open(path) as f:
            for line in f:
                try:
                    traces.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
        return traces
