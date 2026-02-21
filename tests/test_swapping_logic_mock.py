import logging
import torch
import unittest
from unittest.mock import MagicMock

from pravaha.scheduler.scheduler import ContinuousScheduler
from pravaha.scheduler.request import InferenceRequest, SamplingParams
from pravaha.kv_cache.paged_cache import PagedKVCache
from pravaha.models.model_config import ModelArchConfig

# Minimal logger setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestSwappingLogic(unittest.TestCase):
    def setUp(self):
        self.num_blocks = 4
        self.block_size = 4
        self.max_batch_size = 2
        
        # 1. Setup Scheduler
        self.scheduler = ContinuousScheduler(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            max_batch_size=self.max_batch_size,
            max_seq_len=128
        )
        
        # 2. Setup Cache
        # Mocking an arch config
        self.arch_config = MagicMock(spec=ModelArchConfig)
        self.arch_config.num_layers = 2
        self.arch_config.num_heads = 4
        self.arch_config.num_kv_heads = 4
        self.arch_config.head_dim = 16
        
        self.cache = PagedKVCache(
            num_layers=2,
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            num_kv_heads=4,
            head_dim=16,
            device="cpu" # Use CPU to avoid CUDA dependency in logic test
        )

    def test_preemption_and_resume(self):
        """Verify that scheduler triggers preemption and resume correctly."""
        
        # Request 1: 8 tokens (needs 2 blocks). Use distinct tokens per block.
        # Block 0: [1, 1, 1, 1], Block 1: [2, 2, 2, 2]
        tokens1 = [1, 1, 1, 1, 2, 2, 2, 2]
        req1 = InferenceRequest("req1", "prompt1", tokens1, SamplingParams())
        self.scheduler.add_request(req1)
        
        # Step 1: Schedule req1
        scheduled = self.scheduler.step()
        self.assertEqual(len(scheduled["prefill"]), 1)
        self.assertEqual(len(req1.block_table), 2)
        # 4 total - 2 used = 2 free
        self.assertEqual(self.scheduler.allocator.num_free_blocks(), 2)
        
        # Request 2: 8 tokens (needs 2 blocks). Use distinct tokens.
        # Block 2: [3, 3, 3, 3], Block 3: [4, 4, 4, 4]
        tokens2 = [3, 3, 3, 3, 4, 4, 4, 4]
        req2 = InferenceRequest("req2", "prompt2", tokens2, SamplingParams())
        self.scheduler.add_request(req2)
        
        # Step 2: Schedule req2 (should work because 2 blocks remain)
        scheduled = self.scheduler.step()
        self.assertEqual(len(scheduled["prefill"]), 1)
        self.assertEqual(len(req2.block_table), 2)
        self.assertEqual(self.scheduler.allocator.num_free_blocks(), 0)
        
        # Both are now running. Now try to decode to force OOM
        # Request 3: Waiting...
        req3 = InferenceRequest("req3", "prompt3", [3]*4, SamplingParams())
        self.scheduler.add_request(req3)
        
        # Step 3: Decode phase. 
        # Current state: Total tokens = 8 for req1 and req2. Block size = 4.
        # Both req1 and req2 want a new block for the NEXT token because 8 % 4 == 0.
        # Since free blocks = 0, they should both be preempted.
        
        scheduled = self.scheduler.step()
        
        # Both should be preempted because they both needed a new block and none were available.
        self.assertEqual(len(scheduled["swap_out"]), 2)
        self.assertEqual(len(self.scheduler.running), 0)
        self.assertEqual(len(self.scheduler.swapped), 2)
        
        # Verify allocator state
        self.assertEqual(self.scheduler.allocator.num_free_blocks(), 0) # Blocks are still "allocated" but marked as CPU
        
        # Finish req1 blocks to free space
        for bid in req1.block_table:
            self.scheduler.allocator.free(bid)
        req1.block_table = []
        self.scheduler.swapped.remove(req1)
        
        # Now req2 should be able to resume!
        # Reset req2 to fit back (well, resume logic just checks free blocks)
        scheduled = self.scheduler.step()
        self.assertEqual(len(scheduled["swap_in"]), 1)
        self.assertEqual(scheduled["swap_in"][0].request_id, "req2")

if __name__ == "__main__":
    unittest.main()
