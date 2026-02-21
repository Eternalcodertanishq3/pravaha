"""Continuous Batching Scheduler for Pravaha.

Manages the lifecycle of InferenceRequests (WAITING -> RUNNING -> FINISHED).
Implements Paged Attention and Prefix Sharing (Phase 4).
"""

from __future__ import annotations

import collections
import hashlib
import logging
from typing import Optional, List, Tuple, Dict

from pravaha.scheduler.request import InferenceRequest, SequenceStatus, FinishReason
from pravaha_core import BlockAllocator

logger = logging.getLogger(__name__)


class ContinuousScheduler:
    """Manages concurrent sequence execution using Paged KV-Cache blocks.

    Phase 4: Paged attention scheduling. Integrates with the Rust-based
    BlockAllocator to manage physical memory pages and logical-to-physical
    block tables per request.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        max_batch_size: int,
        max_seq_len: int
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # Queues
        self.waiting: collections.deque[InferenceRequest] = collections.deque()
        self.running: list[InferenceRequest] = []
        self.swapped: collections.deque[InferenceRequest] = collections.deque()
        self.finished: list[InferenceRequest] = []

        # High-performance memory management (Rust extension)
        self.allocator = BlockAllocator(num_blocks)

        # Mapping: content_hash -> block_id (for Prefix Sharing)
        self.hash_to_block: dict[str, int] = {}

    def add_request(self, request: InferenceRequest) -> None:
        """Add a new inference request to the waiting queue."""
        self.waiting.append(request)
        logger.debug(f"Scheduler: Added {request.request_id} to waiting queue.")

    def has_unfinished_requests(self) -> bool:
        """True if any requests are waiting or running."""
        return len(self.waiting) > 0 or len(self.running) > 0 or len(self.swapped) > 0

    def step(self) -> Dict[str, List[InferenceRequest]]:
        """Perform one scheduling iteration.
        
        Returns a dictionary with:
            - "prefill": Requests starting their first pass.
            - "decode": Requests generating the next token.
            - "swap_out": Requests moved to CPU to free memory.
            - "swap_in": Requests resumed from CPU.
        """
        # 1. Clean up finished
        self._free_finished_resources()

        # 2. Results container
        scheduled = {
            "prefill": [],
            "decode": [],
            "swap_out": [],
            "swap_in": []
        }

        # 3. Resume swapped requests if possible
        while self.swapped and len(self.running) < self.max_batch_size:
            req = self.swapped[0]
            if self.allocator.num_free_blocks() >= len(req.block_table):
                self.swapped.popleft()
                self.running.append(req)
                for bid in req.block_table:
                    self.allocator.swap_in(bid)
                scheduled["swap_in"].append(req)
                logger.info(f"Scheduler: Resumed request {req.request_id} from swap.")
            else:
                break

        # 4. Schedule new requests (Prefill)
        # We only prefill if no swapped requests are waiting (or can be mixed)
        # To keep it simple, we prioritize Resuming over starting New.
        while self.waiting and len(self.running) < self.max_batch_size:
            request = self.waiting[0]
            
            if request.num_prompt_tokens > self.max_seq_len:
                self.waiting.popleft()
                request.mark_finished(FinishReason.ABORTED)
                self.finished.append(request)
                continue
            
            # Prefix Sharing + Allocation
            num_prompt_tokens = request.num_prompt_tokens
            full_blocks = num_prompt_tokens // self.block_size
            has_partial = (num_prompt_tokens % self.block_size) != 0
            
            assigned_blocks = []
            prompt_tokens = request.prompt_token_ids
            possible = True
            
            for i in range(full_blocks):
                block_content = tuple(prompt_tokens[i * self.block_size : (i + 1) * self.block_size])
                content_hash = hashlib.sha256(str(block_content).encode()).hexdigest()
                
                shared_id = self.hash_to_block.get(content_hash)
                if shared_id is not None:
                    try:
                        if self.allocator.get_ref_count(shared_id) > 0:
                            self.allocator.increment_ref(shared_id)
                            assigned_blocks.append(shared_id)
                            continue
                    except Exception:
                        self.hash_to_block.pop(content_hash, None)
                
                if self.allocator.num_free_blocks() > 0:
                    new_id = self.allocator.allocate(1)[0]
                    assigned_blocks.append(new_id)
                    self.hash_to_block[content_hash] = new_id
                else:
                    possible = False
                    break
            
            if possible and has_partial:
                if self.allocator.num_free_blocks() > 0:
                    assigned_blocks.append(self.allocator.allocate(1)[0])
                else:
                    possible = False
            
            if possible:
                request.block_table.extend(assigned_blocks)
                self.waiting.popleft()
                request.mark_running()
                self.running.append(request)
                scheduled["prefill"].append(request)
            else:
                # Rollback and stop prefilling
                for bid in assigned_blocks:
                    self.allocator.free(bid)
                break

        # 5. Handle Running requests (Decode Phase)
        # If we have prefill, we return immediately (Disjoint Phase strategy)
        if scheduled["prefill"]:
            return scheduled

        # Decode tokens for all running sequences
        to_preempt = []
        for request in self.running:
            # How many blocks do we need to store the NEXT token?
            # total_tokens is the number of tokens ALREADY in the cache.
            # We are about to compute one more.
            needed_blocks = (request.total_tokens + 1 + self.block_size - 1) // self.block_size
            
            if len(request.block_table) < needed_blocks:
                num_to_alloc = needed_blocks - len(request.block_table)
                if self.allocator.num_free_blocks() >= num_to_alloc:
                    new_blocks = self.allocator.allocate(num_to_alloc)
                    request.block_table.extend(new_blocks)
                else:
                    # OOM during decode -> Mark for preemption
                    to_preempt.append(request)
                    continue

            for bid in request.block_table:
                self.allocator.touch(bid)
            scheduled["decode"].append(request)
            
        # Perform preemption after the loop to avoid modification-while-iteration
        for request in to_preempt:
            self.preempt_request(request)
            scheduled["swap_out"].append(request)
            
        return scheduled

    def preempt_request(self, request: Optional[InferenceRequest] = None) -> bool:
        """Move a request to the swapped queue."""
        if request is None:
            if not self.running: return False
            request = self.running.pop(0) # FIFO preemption
        else:
            if request in self.running:
                self.running.remove(request)
            else:
                return False

        for bid in request.block_table:
            self.allocator.swap_out(bid)
            
        self.swapped.append(request)
        logger.warning(f"Scheduler: Preempted {request.request_id} to CPU.")
        return True

    def abort_request(self, request_id: str) -> bool:
        """Abort a request by ID."""
        for q in [self.waiting, self.running, self.swapped]:
            for i, req in enumerate(q):
                if req.request_id == request_id:
                    req.mark_finished(FinishReason.ABORTED)
                    if q == self.waiting:
                        self.finished.append(req)
                        q.remove(req)
                    return True
        return False

    def _free_finished_resources(self) -> None:
        """Remove finished requests and release their KV-cache blocks."""
        still_running = []
        for request in self.running:
            if request.is_finished:
                for block_id in request.block_table:
                    self.allocator.free(block_id)
                self.finished.append(request)
            else:
                still_running.append(request)
        self.running = still_running

    def get_queue_stats(self) -> dict:
        """Get current queue sizes for metrics."""
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "swapped": len(self.swapped),
            "finished": len(self.finished),
            "free_blocks": self.allocator.num_free_blocks(),
            "total_blocks": self.num_blocks
        }
