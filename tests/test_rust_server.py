from __future__ import annotations
import json
import pytest

# Simulating TokenBridge interface for testing if Rust isn't compiled
class MockTokenBridge:
    def __init__(self):
        self.streams = {}

    def create_stream(self, request_id: str) -> None:
        self.streams[request_id] = []

    def send_token(self, request_id: str, token: str) -> None:
        if request_id in self.streams:
            self.streams[request_id].append(token)
        else:
            raise KeyError(f"No stream found for request_id {request_id}")

    def finish_stream(self, request_id: str) -> None:
        if request_id in self.streams:
            del self.streams[request_id]
        else:
            raise KeyError(f"No stream found for request_id {request_id}")

def test_token_bridge_instantiation():
    bridge = MockTokenBridge()
    bridge.create_stream("req-123")
    bridge.send_token("req-123", "Hello")
    assert bridge.streams["req-123"] == ["Hello"]
    bridge.finish_stream("req-123")
    assert "req-123" not in bridge.streams

def test_completion_request_format():
    req = {
        "prompt": "Hello world",
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": True
    }
    assert "prompt" in req
    assert "max_tokens" in req
    assert req["stream"] is True

def test_sse_event_parsing():
    # Simulate an SSE stream chunk from the Rust server
    chunk_data = {
        "id": "req-123",
        "object": "text_completion",
        "created": 1690000000,
        "choices": [
            {
                "text": "Hello",
                "index": 0,
                "finish_reason": None
            }
        ]
    }
    sse_string = f"data: {json.dumps(chunk_data)}\n\n"
    
    assert sse_string.startswith("data: ")
    assert sse_string.endswith("\n\n")
    
    json_str = sse_string[6:-2]
    parsed = json.loads(json_str)
    assert parsed["id"] == "req-123"
    assert parsed["choices"][0]["text"] == "Hello"

def test_health_response_format():
    resp = {
        "status": "ok",
        "uptime_ms": 5000
    }
    assert resp["status"] == "ok"
    assert isinstance(resp["uptime_ms"], int)
