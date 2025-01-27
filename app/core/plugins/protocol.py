"""Core plugin system interfaces and protocols."""

from abc import ABC, abstractmethod
from typing import Any, Protocol

from app.core.events import EventBus


class PluginProtocol(Protocol):
    """Protocol defining the plugin interface."""

    @property
    def name(self) -> str:
        """The name of the plugin."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Whether the plugin has been initialized."""
        ...

    async def initialize(self) -> None:
        """Initialize the plugin."""
        ...

    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        ...


class PluginBase(ABC):
    """Abstract base class for plugins implementing core functionality."""

    def __init__(self, config: Any, event_bus: EventBus | None = None) -> None:
        """Initialize the plugin.

        Args:
            config: Plugin configuration
            event_bus: Optional event bus instance
        """
        self.config = config
        self.event_bus = event_bus
        self.logger = None  # Will be set by plugin manager

    async def initialize(self) -> None:
        """Initialize the plugin. Called by plugin manager after instantiation."""
        await self._initialize()

    async def shutdown(self) -> None:
        """Shutdown the plugin. Called by plugin manager before destruction."""
        await self._shutdown()

    @abstractmethod
    async def _initialize(self) -> None:
        """Initialize plugin implementation. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def _shutdown(self) -> None:
        """Shutdown plugin implementation. Must be implemented by subclasses."""
        pass
