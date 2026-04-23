"""Request Replayer — Replay saved requests for debugging."""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


@dataclass
class ReplayRecord:
    request_id: str = ""
    prompt: str = ""
    params: dict = field(default_factory=dict)
    generated_tokens: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


class Replayer:
    """Record and replay inference requests for debugging."""

    def __init__(self, store_path: str = "data/debug/replays") -> None:
        self.store_path = Path(store_path)
        self._records: dict[str, ReplayRecord] = {}
        self._recording = False

    def start_recording(self) -> None:
        self._recording = True

    def stop_recording(self) -> None:
        self._recording = False

    def record(self, request_id: str, prompt: str, params: dict) -> None:
        if not self._recording:
            return
        self._records[request_id] = ReplayRecord(request_id=request_id, prompt=prompt, params=params)

    def record_token(self, request_id: str, token: str) -> None:
        if request_id in self._records:
            self._records[request_id].generated_tokens.append(token)

    def save(self, request_id: str) -> None:
        record = self._records.get(request_id)
        if not record:
            return
        self.store_path.mkdir(parents=True, exist_ok=True)
        path = self.store_path / f"{request_id}.json"
        with open(path, "w") as f:
            json.dump({"request_id": record.request_id, "prompt": record.prompt, "params": record.params, "generated_tokens": record.generated_tokens}, f, indent=2)

    def load(self, request_id: str) -> ReplayRecord:
        path = self.store_path / f"{request_id}.json"
        with open(path) as f:
            data = json.load(f)
        return ReplayRecord(**data)

    async def replay(self, record: ReplayRecord) -> AsyncGenerator[str, None]:
        for token in record.generated_tokens:
            yield token
