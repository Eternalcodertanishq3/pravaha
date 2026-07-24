"""Tests for Phase 2 Memory & Stability features."""

import asyncio
import pytest
from pravaha.scheduler.continuous_scheduler import ContinuousScheduler
from pravaha.scheduler.request import InferenceRequest
from pravaha.decoder.sampling import SamplingParams
from pravaha.memory.session_cache import SessionKVCache


def test_scheduler_bounded_queues():
    """Verify that ContinuousScheduler enforces waiting queue bounds."""
    scheduler = ContinuousScheduler(
        num_blocks=100,
        block_size=16,
        max_batch_size=32,
        max_seq_len=1024,
        max_waiting_requests=5,
    )

    # Fill up queue
    for i in range(5):
        req = InferenceRequest(
            request_id=f"req-{i}",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
        )
        assert scheduler.add_request(req) is True

    # 6th request should fail
    overflow_req = InferenceRequest(
        request_id="req-overflow",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
    )
    assert scheduler.add_request(overflow_req) is False
    assert len(scheduler.waiting) == 5


def test_scheduler_is_overloaded():
    """Verify backpressure check in ContinuousScheduler."""
    scheduler = ContinuousScheduler(
        num_blocks=10,
        block_size=16,
        max_batch_size=32,
        max_seq_len=1024,
        max_waiting_requests=10,
    )

    assert scheduler.is_overloaded(threshold_pct=0.8) is False

    # Add 8 requests (80% capacity)
    for i in range(8):
        req = InferenceRequest(
            request_id=f"req-{i}",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(req)

    assert scheduler.is_overloaded(threshold_pct=0.8) is True


def test_session_cache_context_bounding():
    """Verify that SessionKVCache truncates context exceeding max_context_len."""
    cache = SessionKVCache(
        max_sessions=10,
        ttl_seconds=3600,
        max_context_len=100,
    )

    # Save a session with context length 200 (exceeds max_context_len of 100)
    blocks = list(range(20))  # 20 blocks
    cache.save(session_id="session-long", block_table=blocks, context_len=200)

    loaded = cache.load("session-long")
    assert loaded is not None
    loaded_blocks, loaded_context_len = loaded

    assert loaded_context_len == 100
    assert len(loaded_blocks) <= 8  # truncated blocks
