"""Plugin Registry — Discover, register, and manage plugins."""

from __future__ import annotations
import logging
from typing import Any, Optional
from pravaha.plugins.base_plugin import BasePlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Central registry for all loaded plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, BasePlugin] = {}

    def register(self, plugin: BasePlugin) -> None:
        self._plugins[plugin.name] = plugin
        logger.info(f"Plugin registered: {plugin.name} v{plugin.version}")

    def unregister(self, name: str) -> None:
        plugin = self._plugins.pop(name, None)
        if plugin:
            plugin.on_unload()

    def get(self, name: str) -> Optional[BasePlugin]:
        return self._plugins.get(name)

    def list_plugins(self) -> list[dict]:
        return [{"name": p.name, "version": p.version, "description": p.description} for p in self._plugins.values()]

    def apply_pre_hooks(self, prompt: str, params: Any) -> tuple[str, Any]:
        for plugin in self._plugins.values():
            prompt, params = plugin.pre_generate(prompt, params)
        return prompt, params

    def apply_post_hooks(self, prompt: str, response: str) -> str:
        for plugin in self._plugins.values():
            response = plugin.post_generate(prompt, response)
        return response
