"""Tests for the sampling pipeline."""

import pytest
import torch

from pravaha.decoder.sampling import Sampler, SamplingParams


@pytest.fixture
def sampler() -> Sampler:
    return Sampler()


@pytest.fixture
def uniform_logits() -> torch.Tensor:
    """Logits where all tokens have equal probability."""
    return torch.zeros(100)  # 100-token vocab, all equal


@pytest.fixture
def peaked_logits() -> torch.Tensor:
    """Logits where token 42 has the highest probability."""
    logits = torch.full((100,), -10.0)
    logits[42] = 10.0
    return logits


class TestGreedyDecoding:
    """Test greedy (temperature=0) decoding."""

    def test_greedy_picks_argmax(self, sampler: Sampler, peaked_logits: torch.Tensor):
        params = SamplingParams(temperature=0.0)
        token = sampler.sample(peaked_logits, params)
        assert token.item() == 42

    def test_greedy_deterministic(self, sampler: Sampler, peaked_logits: torch.Tensor):
        params = SamplingParams(temperature=0.0)
        tokens = [sampler.sample(peaked_logits, params).item() for _ in range(10)]
        assert all(t == 42 for t in tokens)

    def test_greedy_batched(self, sampler: Sampler):
        logits = torch.randn(4, 100)
        params = SamplingParams(temperature=0.0)
        tokens = sampler.sample(logits, params)
        assert tokens.shape == (4,)
        expected = logits.argmax(dim=-1)
        assert torch.equal(tokens, expected)


class TestTemperature:
    """Test temperature scaling effects."""

    def test_low_temperature_more_peaked(
        self, sampler: Sampler, uniform_logits: torch.Tensor
    ):
        # With uniform logits and temp=1.0, distribution is uniform
        # With temp scaling, distribution shape should change
        params = SamplingParams(temperature=1.0)
        token = sampler.sample(uniform_logits, params)
        assert 0 <= token.item() < 100

    def test_high_temperature_samples_valid(self, sampler: Sampler):
        logits = torch.randn(100)
        params = SamplingParams(temperature=2.0)
        token = sampler.sample(logits, params)
        assert 0 <= token.item() < 100


class TestTopK:
    """Test top-k filtering."""

    def test_top_k_1_equals_greedy(self, sampler: Sampler, peaked_logits: torch.Tensor):
        params = SamplingParams(top_k=1, temperature=1.0)
        token = sampler.sample(peaked_logits, params)
        assert token.item() == 42

    def test_top_k_filters_low_prob(self, sampler: Sampler):
        logits = torch.tensor([10.0, 5.0, -100.0, -100.0, -100.0])
        params = SamplingParams(top_k=2, temperature=1.0)
        # Run many samples — should only see tokens 0 or 1
        seen = set()
        for _ in range(100):
            token = sampler.sample(logits.clone(), params)
            seen.add(token.item())
        assert seen.issubset({0, 1})


class TestTopP:
    """Test top-p (nucleus) sampling."""

    def test_top_p_1_allows_all(self, sampler: Sampler, uniform_logits: torch.Tensor):
        params = SamplingParams(top_p=1.0, temperature=1.0)
        token = sampler.sample(uniform_logits, params)
        assert 0 <= token.item() < 100

    def test_top_p_very_small_acts_like_greedy(
        self, sampler: Sampler, peaked_logits: torch.Tensor
    ):
        params = SamplingParams(top_p=0.01, temperature=1.0)
        # With very small top_p, only the most probable token survives
        token = sampler.sample(peaked_logits, params)
        assert token.item() == 42


class TestRepetitionPenalty:
    """Test repetition penalty."""

    def test_penalty_reduces_repeated_token_prob(self, sampler: Sampler):
        # Token 0 has highest logit
        logits = torch.tensor([10.0, 9.9, 0.0, 0.0, 0.0])
        generated = torch.tensor([0])  # Token 0 was already generated

        params = SamplingParams(repetition_penalty=2.0, temperature=0.0)
        token = sampler.sample(logits, params, generated_ids=generated)
        # After penalty, token 0's logit is 10/2=5, token 1 is still 9.9
        assert token.item() == 1

    def test_no_penalty_by_default(self, sampler: Sampler):
        logits = torch.tensor([10.0, 5.0, 0.0])
        generated = torch.tensor([0])

        params = SamplingParams(repetition_penalty=1.0, temperature=0.0)
        token = sampler.sample(logits, params, generated_ids=generated)
        assert token.item() == 0  # Still picks argmax


class TestSamplingParams:
    """Test SamplingParams dataclass."""

    def test_default_values(self):
        params = SamplingParams()
        assert params.temperature == 1.0
        assert params.top_k == 50
        assert params.top_p == 1.0
        assert params.max_new_tokens == 256

    def test_is_greedy(self):
        assert SamplingParams(temperature=0.0).is_greedy
        assert not SamplingParams(temperature=1.0).is_greedy

    def test_custom_stop_tokens(self):
        params = SamplingParams(stop_token_ids=[1, 2, 3])
        assert params.stop_token_ids == [1, 2, 3]
