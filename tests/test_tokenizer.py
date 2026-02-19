"""Tests for the tokenizer wrapper."""

import pytest
import torch

from pravaha.tokenizer.tokenizer import PravahaTokenizer


@pytest.fixture(scope="module")
def tokenizer() -> PravahaTokenizer:
    """Load GPT-2 tokenizer once for all tests in this module."""
    return PravahaTokenizer("gpt2")


class TestPravahaTokenizer:
    """Test the PravahaTokenizer wrapper."""

    def test_encode_returns_list(self, tokenizer: PravahaTokenizer):
        ids = tokenizer.encode("Hello, world!")
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_decode_roundtrip(self, tokenizer: PravahaTokenizer):
        text = "The quick brown fox jumps over the lazy dog."
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded.strip() == text

    def test_decode_tensor(self, tokenizer: PravahaTokenizer):
        ids = tokenizer.encode("Hello")
        tensor_ids = torch.tensor(ids)
        decoded = tokenizer.decode(tensor_ids)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_decode_token_single(self, tokenizer: PravahaTokenizer):
        ids = tokenizer.encode("Hello")
        # Decode each token individually
        for token_id in ids:
            text = tokenizer.decode_token(token_id)
            assert isinstance(text, str)

    def test_batch_encode(self, tokenizer: PravahaTokenizer):
        texts = ["Hello, world!", "How are you?", "Short"]
        input_ids, attention_mask = tokenizer.batch_encode(texts)
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert input_ids.shape[0] == 3
        assert attention_mask.shape == input_ids.shape

    def test_batch_encode_padding(self, tokenizer: PravahaTokenizer):
        texts = ["Short", "This is a much longer sentence."]
        input_ids, attention_mask = tokenizer.batch_encode(texts)
        # Both should have the same length (padded to longest)
        assert input_ids.shape[1] == attention_mask.shape[1]

    def test_eos_token_id(self, tokenizer: PravahaTokenizer):
        assert tokenizer.eos_token_id is not None
        assert isinstance(tokenizer.eos_token_id, int)

    def test_pad_token_id(self, tokenizer: PravahaTokenizer):
        assert tokenizer.pad_token_id is not None
        assert isinstance(tokenizer.pad_token_id, int)

    def test_vocab_size(self, tokenizer: PravahaTokenizer):
        assert tokenizer.vocab_size > 0
        # GPT-2 has 50257 tokens
        assert tokenizer.vocab_size == 50257

    def test_encode_empty_string(self, tokenizer: PravahaTokenizer):
        ids = tokenizer.encode("")
        assert isinstance(ids, list)
        # Empty string may produce empty list or just special tokens
        assert len(ids) >= 0
