"""Model Registry — Discover, list, and manage available models.

Scans local paths and HuggingFace Hub for compatible models.
Tracks loaded models and manages LoRA adapters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """A registered model entry."""

    name: str
    path: str
    format: str = "safetensors"  # safetensors, gguf, pytorch
    size_gb: float = 0.0
    quantization: str | None = None
    loaded: bool = False
    lora_adapters: list[str] = field(default_factory=list)


class ModelRegistry:
    """Registry for discovering and managing available models.

    Scans configured directories for model files and provides a
    unified API for listing and loading models.
    """

    def __init__(self, model_dirs: list[str] | None = None) -> None:
        self._models: dict[str, ModelEntry] = {}
        self._model_dirs = model_dirs or []

    def scan(self) -> list[ModelEntry]:
        """Scan configured directories for models."""
        found: list[ModelEntry] = []
        for dir_path in self._model_dirs:
            p = Path(dir_path)
            if not p.exists():
                continue
            for model_dir in p.iterdir():
                if model_dir.is_dir():
                    config_file = model_dir / "config.json"
                    if config_file.exists():
                        entry = ModelEntry(name=model_dir.name, path=str(model_dir))
                        self._models[entry.name] = entry
                        found.append(entry)
                # Check for GGUF files
                elif model_dir.suffix == ".gguf":
                    entry = ModelEntry(name=model_dir.stem, path=str(model_dir), format="gguf")
                    self._models[entry.name] = entry
                    found.append(entry)
        return found

    def register(self, name: str, path: str, **kwargs: object) -> ModelEntry:
        """Manually register a model."""
        entry = ModelEntry(name=name, path=path, **kwargs)  # type: ignore[arg-type]
        self._models[name] = entry
        return entry

    def get(self, name: str) -> ModelEntry | None:
        return self._models.get(name)

    def list_models(self) -> list[ModelEntry]:
        return list(self._models.values())

    def list_loaded(self) -> list[ModelEntry]:
        return [m for m in self._models.values() if m.loaded]

    def mark_loaded(self, name: str) -> None:
        if name in self._models:
            self._models[name].loaded = True
