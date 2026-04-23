"""Plugin Base — Abstract base class for Pravaha plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePlugin(ABC):
    """Base class for all Pravaha plugins.

    Plugins can extend the engine with custom agents, tools, routers,
    and preprocessing/postprocessing hooks.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique plugin name."""
        ...

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return ""

    def on_load(self, engine: Any) -> None:
        """Called when the plugin is loaded into the engine."""
        pass

    def on_unload(self) -> None:
        """Called when the plugin is unloaded."""
        pass

    def pre_generate(self, prompt: str, params: Any) -> tuple[str, Any]:
        """Hook: modify prompt/params before generation."""
        return prompt, params

    def post_generate(self, prompt: str, response: str) -> str:
        """Hook: modify response after generation."""
        return response

    def get_config_schema(self) -> dict:
        """Return JSON schema for plugin configuration."""
        return {}
