"""Continuous Batching Scheduler for Pravaha.

Manages the lifecycle of InferenceRequests (WAITING -> RUNNING -> FINISHED).
Implements a "Slot-based" approach for HF model compatibility (Phase 3).
"""

from __future__ import annotations

import collections
import logging
from typing import Optional

from pravaha.scheduler.request import InferenceRequest, SequenceStatus, FinishReason

logger = logging.getLogger(__name__)


class ContinuousScheduler:
    """Manages concurrent sequence execution using fixed cache slots.

    Phase 3: Slot-based scheduling. We pre-allocate `max_batch_size` slots
    in the NaiveKVCache. The scheduler assigns waiting requests to free slots.
    """

    def __init__(self, max_batch_size: int, max_seq_len: int):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # Queues
        self.waiting: collections.deque[InferenceRequest] = collections.deque()
        self.running: list[InferenceRequest] = []
        self.finished: list[InferenceRequest] = []

        # Slot Management: Available slot indices (0 to max_batch_size - 1)
        self.free_slots: set[int] = set(range(max_batch_size))
        
        # Mapping: request_id -> active slot index
        self.request_to_slot: dict[str, int] = {}

    def add_request(self, request: InferenceRequest) -> None:
        """Add a new inference request to the waiting queue."""
        self.waiting.append(request)
        logger.debug(f"Scheduler: Added {request.request_id} to waiting queue.")

    def has_unfinished_requests(self) -> bool:
        """True if any requests are waiting or running."""
        return len(self.waiting) > 0 or len(self.running) > 0

    def get_slot_for_request(self, request_id: str) -> Optional[int]:
        """Get the KV-cache slot index assigned to a running request."""
        return self.request_to_slot.get(request_id)

    def step(self) -> tuple[list[InferenceRequest], list[InferenceRequest]]:
        """Perform one scheduling iteration.
        
        Disjoint Execution Strategy (Phase 3):
        If there are waiting requests AND free slots -> Schedule Prefill batch.
        Else If there are running requests -> Schedule Decode batch.
        
        Returns:
            Tuple of (requests_to_prefill, requests_to_decode)
        """
        # 1. Clean up finished requests from the running pool and free their slots
        self._free_finished_slots()

        # 2. Can we schedule new requests for prefill?
        # We must have waiting requests AND free slots.
        requests_to_prefill = []
        
        while self.waiting and self.free_slots:
            request = self.waiting[0]
            
            # Check maximum sequence length constraints
            if request.num_prompt_tokens > self.max_seq_len:
                logger.warning(f"Request {request.request_id} prompt too long ({request.num_prompt_tokens} > {self.max_seq_len}). Rejecting.")
                self.waiting.popleft()
                request.mark_finished(FinishReason.ABORTED)
                self.finished.append(request)
                continue
                
            # Assign a slot
            slot_idx = self.free_slots.pop()
            self.request_to_slot[request.request_id] = slot_idx
            
            # Move to executing
            popped_request = self.waiting.popleft()
            popped_request.mark_running()
            self.running.append(popped_request)
            requests_to_prefill.append(popped_request)
            
            logger.debug(f"Scheduler: Assigned slot {slot_idx} to {request.request_id} (Prefill).")

        # Disjoint phases: If we scheduled new requests, we MUST do a prefill step first
        # so their KV-cache states are initialized before joining the normal decode loop.
        if requests_to_prefill:
            return requests_to_prefill, []
            
        # 3. Otherwise, if we have running requests, we schedule a decode step.
        requests_to_decode = []
        if self.running:
            requests_to_decode = list(self.running)
            
        return [], requests_to_decode

    def abort_request(self, request_id: str) -> bool:
        """Abort a request by ID. Returns True if aborted, False if not found."""
        # Check waiting
        for i, req in enumerate(self.waiting):
            if req.request_id == request_id:
                req.mark_finished(FinishReason.ABORTED)
                self.finished.append(req)
                del self.waiting[i]
                return True
                
        # Check running
        for req in self.running:
            if req.request_id == request_id:
                req.mark_finished(FinishReason.ABORTED)
                # Next step() will clean it up and free the slot
                return True
                
        return False

    def _free_finished_slots(self) -> None:
        """Remove finished requests from running pool and release slots."""
        still_running = []
        
        for request in self.running:
            if request.is_finished:
                # Release slot
                slot_idx = self.request_to_slot.pop(request.request_id, None)
                if slot_idx is not None:
                    self.free_slots.add(slot_idx)
                    logger.debug(f"Scheduler: Freed slot {slot_idx} from finished {request.request_id}.")
                self.finished.append(request)
            else:
                still_running.append(request)
                
        self.running = still_running

    def get_queue_stats(self) -> dict:
        """Get current queue sizes for metrics."""
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "finished": len(self.finished),
            "free_slots": len(self.free_slots),
            "max_slots": self.max_batch_size
        }
