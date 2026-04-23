"""Continuous Scheduler — Paged attention scheduling (Phase 4).

Manages the lifecycle of InferenceRequests (WAITING → RUNNING → FINISHED).
Uses the BlockManager for prefix sharing and Rust-based allocation.
"""

from __future__ import annotations

import collections
import logging

from pravaha.memory.block_manager import BlockManager
from pravaha.scheduler.request import FinishReason, InferenceRequest

logger = logging.getLogger(__name__)


class ContinuousScheduler:
    """Continuous batching scheduler with paged attention.

    Implements disjoint phase strategy: each step either prefills
    new requests OR decodes running requests, never both.
    """

    def __init__(
        self, num_blocks: int, block_size: int, max_batch_size: int, max_seq_len: int
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.waiting: collections.deque[InferenceRequest] = collections.deque()
        self.running: list[InferenceRequest] = []
        self.swapped: collections.deque[InferenceRequest] = collections.deque()
        self.finished: list[InferenceRequest] = []
        self.block_manager = BlockManager(num_blocks, block_size)
        self.hash_to_block: dict[str, int] = {}

    def add_request(self, request: InferenceRequest) -> None:
        self.waiting.append(request)

    def has_unfinished_requests(self) -> bool:
        return len(self.waiting) > 0 or len(self.running) > 0 or len(self.swapped) > 0

    def step(self) -> dict[str, list[InferenceRequest]]:
        self._free_finished_resources()
        scheduled: dict[str, list[InferenceRequest]] = {
            "prefill": [],
            "decode": [],
            "swap_out": [],
            "swap_in": [],
        }

        # Resume swapped
        while self.swapped and len(self.running) < self.max_batch_size:
            req = self.swapped[0]
            if self.block_manager.num_free_blocks() >= len(req.block_table):
                self.swapped.popleft()
                self.running.append(req)
                for bid in req.block_table:
                    self.block_manager.swap_in(bid)
                scheduled["swap_in"].append(req)
            else:
                break

        # Prefill new requests
        while self.waiting and len(self.running) < self.max_batch_size:
            request = self.waiting[0]
            if request.num_prompt_tokens > self.max_seq_len:
                self.waiting.popleft()
                request.mark_finished(FinishReason.ABORTED)
                self.finished.append(request)
                continue

            full_blocks = request.num_prompt_tokens // self.block_size
            has_partial = (request.num_prompt_tokens % self.block_size) != 0
            assigned: list[int] = []
            possible = True

            for i in range(full_blocks):
                content = request.prompt_token_ids[i * self.block_size : (i + 1) * self.block_size]
                h = self.block_manager.compute_content_hash(content)
                shared = self.hash_to_block.get(h)
                if shared is not None:
                    try:
                        if self.block_manager.get_ref_count(shared) > 0:
                            self.block_manager.increment_ref(shared)
                            assigned.append(shared)
                            continue
                    except Exception:
                        self.hash_to_block.pop(h, None)
                if self.block_manager.num_free_blocks() > 0:
                    new_id = self.block_manager.allocate(1)[0]
                    assigned.append(new_id)
                    self.hash_to_block[h] = new_id
                else:
                    possible = False
                    break

            if possible and has_partial:
                if self.block_manager.num_free_blocks() > 0:
                    assigned.append(self.block_manager.allocate(1)[0])
                else:
                    possible = False

            if possible:
                request.block_table.extend(assigned)
                self.waiting.popleft()
                request.mark_running()
                self.running.append(request)
                scheduled["prefill"].append(request)
            else:
                for bid in assigned:
                    self.block_manager.free(bid)
                break

        if scheduled["prefill"]:
            return scheduled

        # Decode running requests
        to_preempt: list[InferenceRequest] = []
        for request in self.running:
            needed = (request.total_tokens + 1 + self.block_size - 1) // self.block_size
            if len(request.block_table) < needed:
                n = needed - len(request.block_table)
                if self.block_manager.num_free_blocks() >= n:
                    request.block_table.extend(self.block_manager.allocate(n))
                else:
                    to_preempt.append(request)
                    continue
            for bid in request.block_table:
                self.block_manager.touch(bid)
            scheduled["decode"].append(request)

        for request in to_preempt:
            self.preempt_request(request)
            scheduled["swap_out"].append(request)

        return scheduled

    def preempt_request(self, request: InferenceRequest | None = None) -> bool:
        if request is None:
            if not self.running:
                return False
            request = self.running.pop(0)
        else:
            if request in self.running:
                self.running.remove(request)
            else:
                return False
        for bid in request.block_table:
            self.block_manager.swap_out(bid)
        self.swapped.append(request)
        return True

    def abort_request(self, request_id: str) -> bool:
        for q in [self.waiting, self.running, self.swapped]:
            for req in q:
                if req.request_id == request_id:
                    req.mark_finished(FinishReason.ABORTED)
                    if q is self.waiting:
                        self.finished.append(req)
                        q.remove(req)
                    return True
        return False

    def _free_finished_resources(self) -> None:
        still_running = []
        for r in self.running:
            if r.is_finished:
                for bid in r.block_table:
                    self.block_manager.free(bid)
                self.finished.append(r)
            else:
                still_running.append(r)
        self.running = still_running

    def get_queue_stats(self) -> dict:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "swapped": len(self.swapped),
            "finished": len(self.finished),
            "free_blocks": self.block_manager.num_free_blocks(),
            "total_blocks": self.num_blocks,
            "requests": [
                {
                    "id": r.request_id,
                    "status": "running",
                    "tokens": len(r.generated_token_ids),
                    "progress": round(
                        len(r.generated_token_ids) / max(1, r.sampling_params.max_new_tokens) * 100,
                        1,
                    ),
                }
                for r in self.running
            ],
        }

    def get_usage_pct(self) -> float:
        used = self.num_blocks - self.block_manager.num_free_blocks()
        return used / self.num_blocks if self.num_blocks > 0 else 0.0
