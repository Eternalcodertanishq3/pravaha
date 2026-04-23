"""Inference Checkpointing — Engine-level checkpoint management.

Feature D: Wraps the decoder-level checkpoint with engine lifecycle management.
Handles KV cache block preservation during pause and restoration during resume.
"""

from __future__ import annotations

import logging

from pravaha.decoder.checkpoint_decoder import CheckpointManager, InferenceCheckpoint

logger = logging.getLogger(__name__)

# Re-export for convenience
__all__ = ["InferenceCheckpoint", "CheckpointManager"]
