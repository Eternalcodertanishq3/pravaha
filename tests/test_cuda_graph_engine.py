from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch

from pravaha.engine.cuda_graph_engine import CUDAGraphDecoderWrapper


class MockModel:
    def __call__(self, token_ids: torch.Tensor, block_tables: torch.Tensor, context_lens: torch.Tensor) -> torch.Tensor:
        # Return dummy output with same batch size
        return torch.ones_like(token_ids, dtype=torch.float32)


class MockDecoderEngine:
    def __init__(self) -> None:
        self.model = MockModel()

    def step_decode(
        self,
        token_ids: torch.Tensor,
        request_ids: list[str],
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(token_ids, block_tables, context_lens)


class TestCUDAGraphDecoderWrapper(unittest.TestCase):
    def setUp(self) -> None:
        self.decoder = MockDecoderEngine()
        self.wrapper = CUDAGraphDecoderWrapper(
            decoder_engine=self.decoder,
            buckets=[1, 4, 16],
            warmup_steps=3,
            device=torch.device("cpu"),
        )
        self.token_ids = torch.zeros((3, 10), dtype=torch.long)
        self.block_tables = torch.zeros((3, 5), dtype=torch.long)
        self.context_lens = torch.zeros((3,), dtype=torch.long)
        self.request_ids = ["req1", "req2", "req3"]

    def test_bucket_selection(self) -> None:
        self.assertEqual(self.wrapper._get_bucket(1), 1)
        self.assertEqual(self.wrapper._get_bucket(3), 4)
        self.assertEqual(self.wrapper._get_bucket(4), 4)
        self.assertEqual(self.wrapper._get_bucket(5), 16)
        self.assertEqual(self.wrapper._get_bucket(16), 16)
        self.assertIsNone(self.wrapper._get_bucket(17))

    def test_static_buffer_allocation(self) -> None:
        inputs = {
            "token_ids": self.token_ids,
            "block_tables": self.block_tables,
            "context_lens": self.context_lens,
        }
        self.wrapper._allocate_static_buffers(4, inputs)
        static_inputs = self.wrapper._static_inputs[4]

        self.assertEqual(static_inputs["token_ids"].shape, (4, 10))
        self.assertEqual(static_inputs["block_tables"].shape, (4, 5))
        self.assertEqual(static_inputs["context_lens"].shape, (4,))

    @patch("torch.cuda.is_available", return_value=True)
    def test_warmup_counter_increments(self, mock_is_available: MagicMock) -> None:
        # Send batch size 3 -> bucket 4
        # Need 3 warmup steps
        for _ in range(3):
            out = self.wrapper.step_decode_graphed(
                self.token_ids, self.request_ids, self.block_tables, self.context_lens
            )
            self.assertEqual(out.shape, (3, 10))

        self.assertEqual(self.wrapper._warmup_counters[4], 3)
        self.assertNotIn(4, self.wrapper._graphs)

    @patch("torch.cuda.memory_allocated", side_effect=[1000, 5000])
    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.graph")
    @patch("torch.cuda.CUDAGraph")
    @patch("torch.cuda.is_available", return_value=True)
    def test_graph_capture_and_vram_accounting(
        self,
        mock_is_available: MagicMock,
        mock_cuda_graph: MagicMock,
        mock_graph_ctx: MagicMock,
        mock_sync: MagicMock,
        mock_mem: MagicMock,
    ) -> None:
        # Run 3 warmups
        for _ in range(3):
            self.wrapper.step_decode_graphed(
                self.token_ids, self.request_ids, self.block_tables, self.context_lens
            )
        
        # 4th run should trigger capture and replay
        mock_graph_instance = MagicMock()
        mock_cuda_graph.return_value = mock_graph_instance

        # Simulate context manager
        mock_graph_ctx.return_value.__enter__ = MagicMock()
        mock_graph_ctx.return_value.__exit__ = MagicMock()

        # To avoid error during context manager body mock, we set output explicitly
        self.wrapper._static_outputs[4] = torch.ones((4, 10), dtype=torch.float32)

        out = self.wrapper.step_decode_graphed(
            self.token_ids, self.request_ids, self.block_tables, self.context_lens
        )
        
        self.assertIn(4, self.wrapper._graphs)
        self.assertEqual(self.wrapper._memory_usage_bytes[4], 4000)
        mock_graph_instance.replay.assert_called_once()
        self.assertEqual(out.shape, (3, 10))

    @patch("torch.cuda.is_available", return_value=False)
    def test_fallback_cuda_unavailable(self, mock_is_available: MagicMock) -> None:
        # Should always use eager path
        for _ in range(5):
            out = self.wrapper.step_decode_graphed(
                self.token_ids, self.request_ids, self.block_tables, self.context_lens
            )
            self.assertEqual(out.shape, (3, 10))
        
        self.assertEqual(self.wrapper._warmup_counters[4], 0)
        self.assertNotIn(4, self.wrapper._graphs)

    @patch("torch.cuda.is_available", return_value=True)
    def test_fallback_large_batch(self, mock_is_available: MagicMock) -> None:
        large_token_ids = torch.zeros((17, 10), dtype=torch.long)
        large_block_tables = torch.zeros((17, 5), dtype=torch.long)
        large_context_lens = torch.zeros((17,), dtype=torch.long)
        large_req_ids = [f"req{i}" for i in range(17)]

        out = self.wrapper.step_decode_graphed(
            large_token_ids, large_req_ids, large_block_tables, large_context_lens
        )
        self.assertEqual(out.shape, (17, 10))
        # Warmup should not be incremented because bucket is None
        self.assertTrue(all(v == 0 for v in self.wrapper._warmup_counters.values()))
