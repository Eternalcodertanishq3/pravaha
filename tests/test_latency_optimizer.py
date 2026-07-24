"""Unit tests for pravaha.engine.latency_optimizer."""

from pravaha.engine.latency_optimizer import (
    AdaptiveAcceptanceTracker,
    DynamicCUDAGraphManager,
    NGramLookaheadDecoder,
)


def test_ngram_lookahead_decoder():
    decoder = NGramLookaheadDecoder(n_gram_size=2, max_candidates=2)
    prompt = [101, 200, 300, 101, 200, 400]
    decoder.build_ngram_table(prompt)

    # Context [101, 200] maps to candidates [300, 400]
    candidates = decoder.propose_candidates([101, 200])
    assert candidates == [300, 400]


def test_adaptive_acceptance_tracker():
    tracker = AdaptiveAcceptanceTracker(min_acceptance_rate=0.50, window_size=10)
    assert tracker.is_speculation_enabled() is True

    # Record 5 failures
    for _ in range(5):
        tracker.record_attempt(accepted_tokens=0, proposed_tokens=4)

    assert tracker.is_speculation_enabled() is False


def test_dynamic_cuda_graph_manager():
    manager = DynamicCUDAGraphManager(enabled=True)
    bucket = manager.select_bucket(3)
    assert bucket == 4

    warmed = manager.warmup_bucket(4)
    assert warmed is True
    assert manager.graphed_buckets[4] is True
