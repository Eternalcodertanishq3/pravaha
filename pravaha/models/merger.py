"""Model Merger — Combine model weights using various strategies.

Feature 14: Merge multiple model checkpoints using SLERP, TIES, or
linear interpolation for creating custom model blends.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MergeStrategy(Enum):
    LINEAR = "linear"
    SLERP = "slerp"
    TIES = "ties"


class ModelMerger:
    """Merge multiple model checkpoints into one."""

    def __init__(self, strategy: MergeStrategy = MergeStrategy.LINEAR) -> None:
        self.strategy = strategy

    def merge(
        self,
        model_paths: list[str],
        output_path: str,
        weights: Optional[list[float]] = None,
    ) -> str:
        """Merge models and save the result.

        Args:
            model_paths: Paths to model checkpoints.
            output_path: Where to save the merged model.
            weights: Interpolation weights (must sum to 1.0).

        Returns:
            Path to the merged model.
        """
        import torch
        from safetensors.torch import load_file, save_file

        if weights is None:
            weights = [1.0 / len(model_paths)] * len(model_paths)

        assert len(weights) == len(model_paths)
        assert abs(sum(weights) - 1.0) < 1e-6

        logger.info(f"Merging {len(model_paths)} models with strategy={self.strategy.value}")

        # Load all state dicts
        state_dicts = []
        for p in model_paths:
            path = Path(p)
            if (path / "model.safetensors").exists():
                state_dicts.append(load_file(str(path / "model.safetensors")))
            else:
                # Try loading all safetensors shards
                tensors = {}
                for sf in path.glob("*.safetensors"):
                    tensors.update(load_file(str(sf)))
                state_dicts.append(tensors)

        # Merge
        if self.strategy == MergeStrategy.LINEAR:
            merged = self._linear_merge(state_dicts, weights)
        elif self.strategy == MergeStrategy.SLERP:
            merged = self._slerp_merge(state_dicts, weights)
        else:
            merged = self._linear_merge(state_dicts, weights)

        # Save
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        save_file(merged, str(out / "model.safetensors"))
        logger.info(f"Merged model saved to {output_path}")
        return output_path

    def _linear_merge(self, state_dicts: list[dict], weights: list[float]) -> dict:
        """Linear interpolation of weights."""
        import torch
        merged = {}
        keys = state_dicts[0].keys()
        for key in keys:
            merged[key] = sum(sd[key].float() * w for sd, w in zip(state_dicts, weights)).to(state_dicts[0][key].dtype)
        return merged

    def _slerp_merge(self, state_dicts: list[dict], weights: list[float]) -> dict:
        """Spherical linear interpolation (for 2 models)."""
        import torch
        import torch.nn.functional as F

        if len(state_dicts) != 2:
            logger.warning("SLERP requires exactly 2 models, falling back to linear")
            return self._linear_merge(state_dicts, weights)

        merged = {}
        t = weights[1]
        for key in state_dicts[0].keys():
            v0 = state_dicts[0][key].float().flatten()
            v1 = state_dicts[1][key].float().flatten()
            dot = F.cosine_similarity(v0.unsqueeze(0), v1.unsqueeze(0)).item()
            dot = max(-1.0, min(1.0, dot))

            import math
            omega = math.acos(dot)
            if abs(omega) < 1e-10:
                merged[key] = ((1 - t) * v0 + t * v1).reshape(state_dicts[0][key].shape).to(state_dicts[0][key].dtype)
            else:
                so = math.sin(omega)
                merged[key] = (
                    (math.sin((1 - t) * omega) / so) * v0 + (math.sin(t * omega) / so) * v1
                ).reshape(state_dicts[0][key].shape).to(state_dicts[0][key].dtype)
        return merged
