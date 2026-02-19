"""Tests for the decoder engine."""

import pytest
import torch

from pravaha.decoder.decoder import DecoderEngine
from pravaha.decoder.sampling import Sampler, SamplingParams
from pravaha.models.loader import ModelLoader
from pravaha.tokenizer.tokenizer import PravahaTokenizer


@pytest.fixture(scope="module")
def decoder_engine() -> DecoderEngine:
    """Load GPT-2 and create decoder engine (once per test module)."""
    loader = ModelLoader()
    model, _ = loader.load_model("gpt2", dtype=torch.float32, device="cpu")
    tokenizer = PravahaTokenizer("gpt2")
    sampler = Sampler()
    return DecoderEngine(model=model, tokenizer=tokenizer, sampler=sampler, device="cpu")


class TestDecoderEngine:
    """Test the autoregressive decoder."""

    @pytest.mark.slow
    def test_generate_produces_tokens(self, decoder_engine: DecoderEngine):
        tokens = list(decoder_engine.generate(
            "Hello",
            SamplingParams(max_new_tokens=10, temperature=0.0),
        ))
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    @pytest.mark.slow
    def test_generate_respects_max_tokens(self, decoder_engine: DecoderEngine):
        tokens = list(decoder_engine.generate(
            "Once upon a time",
            SamplingParams(max_new_tokens=5, temperature=0.0),
        ))
        # Should generate at most 5 tokens (may be fewer if EOS hit)
        assert len(tokens) <= 5

    @pytest.mark.slow
    def test_generate_text_returns_string(self, decoder_engine: DecoderEngine):
        text = decoder_engine.generate_text(
            "The capital of France is",
            SamplingParams(max_new_tokens=10, temperature=0.0),
        )
        assert isinstance(text, str)
        assert len(text) > 0

    @pytest.mark.slow
    def test_greedy_is_deterministic(self, decoder_engine: DecoderEngine):
        params = SamplingParams(max_new_tokens=10, temperature=0.0)
        text1 = decoder_engine.generate_text("Hello world", params)
        text2 = decoder_engine.generate_text("Hello world", params)
        assert text1 == text2

    @pytest.mark.slow
    def test_streaming_yields_incrementally(self, decoder_engine: DecoderEngine):
        params = SamplingParams(max_new_tokens=5, temperature=0.0)
        gen = decoder_engine.generate("Test", params)
        # Should be able to get first token without consuming all
        first_token = next(gen)
        assert isinstance(first_token, str)
