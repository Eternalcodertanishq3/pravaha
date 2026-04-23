"""Shared Memory — Thread-safe shared state for swarm agents."""

from __future__ import annotations

import threading
from typing import Any


class SharedMemory:
    """Thread-safe shared memory for inter-agent communication."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def update(self, data: dict[str, Any]) -> None:
        with self._lock:
            self._data.update(data)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._data.copy()
