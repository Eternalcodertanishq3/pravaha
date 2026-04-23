"""Plugin Loader — Discover and load plugins from directories."""

from __future__ import annotations
import importlib
import logging
from pathlib import Path
from typing import Optional
from pravaha.plugins.base_plugin import BasePlugin
from pravaha.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


class PluginLoader:
    """Load plugins from Python modules and directories."""

    def __init__(self, registry: Optional[PluginRegistry] = None) -> None:
        self.registry = registry or PluginRegistry()

    def load_module(self, module_path: str) -> Optional[BasePlugin]:
        try:
            module = importlib.import_module(module_path)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, BasePlugin) and attr is not BasePlugin:
                    plugin = attr()
                    self.registry.register(plugin)
                    return plugin
        except Exception as e:
            logger.error(f"Failed to load plugin {module_path}: {e}")
        return None

    def load_directory(self, path: str) -> list[BasePlugin]:
        loaded = []
        p = Path(path)
        if not p.exists():
            return loaded
        for py_file in p.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            module_name = py_file.stem
            plugin = self.load_module(f"plugins.{module_name}")
            if plugin:
                loaded.append(plugin)
        return loaded
