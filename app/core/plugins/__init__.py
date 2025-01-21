"""Core plugin system interfaces and implementation."""

from app.core.plugins.interface import PluginBase, PluginConfig
from app.core.plugins.manager import PluginManager
from app.core.plugins.protocol import PluginProtocol

__all__ = ["PluginBase", "PluginConfig", "PluginManager", "PluginProtocol"]
