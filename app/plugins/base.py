from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from ..models.event_bus import EventBus, EventData
from ..utils import get_logger

# Initialize logging when the module is imported
logger = get_logger(__name__)


class PluginConfig(BaseModel):
    """Base configuration model for plugins"""

    name: str
    version: str
    enabled: bool = True
    dependencies: list[str] = []
    config: dict[str, Any] | None = None


class PluginBase(ABC):
    """Base class for all plugins"""

    def __init__(self, config: PluginConfig, event_bus: EventBus | None = None) -> None:
        self.config = config
        self.logger = get_logger(f"plugin.{config.name}")
        self._initialized = False
        self.event_bus = event_bus

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def version(self) -> str:
        return self.config.version

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the plugin"""
        try:
            self.logger.info(
                "Initializing plugin",
                extra={
                    "plugin_name": self.name,
                    "plugin_version": self.version,
                    "plugin_config": self.config.dict(),
                },
            )
            await self._initialize()
            self._initialized = True
            self.logger.info(
                "Plugin initialized successfully", extra={"plugin_name": self.name}
            )
        except Exception as e:
            self.logger.error(
                "Failed to initialize plugin",
                extra={
                    "plugin_name": self.name,
                    "error": str(e),
                    "plugin_config": self.config.dict(),
                },
            )
            raise

    async def shutdown(self) -> None:
        """Shutdown the plugin"""
        try:
            self.logger.info("Shutting down plugin", extra={"plugin_name": self.name})
            await self._shutdown()
            self._initialized = False
            self.logger.info(
                "Plugin shutdown successfully", extra={"plugin_name": self.name}
            )
        except Exception as e:
            self.logger.error(
                "Failed to shutdown plugin",
                extra={"plugin_name": self.name, "error": str(e)},
            )
            raise

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        if self.config.config is None:
            return default
        value = self.config.config.get(key, default)
        self.logger.debug(
            "Retrieved plugin config",
            extra={"plugin_name": self.name, "config_key": key, "config_value": value},
        )
        return value

    async def subscribe(
        self, event_type: str, callback: Callable[[EventData], Any]
    ) -> None:
        """Subscribe to events safely."""
        if self.event_bus is not None:
            await self.event_bus.subscribe(event_type, callback)

    async def unsubscribe(
        self, event_type: str, callback: Callable[[EventData], Any]
    ) -> None:
        """Unsubscribe from events safely."""
        if self.event_bus is not None:
            await self.event_bus.unsubscribe(event_type, callback)

    async def publish(self, event: EventData) -> None:
        """Publish an event safely."""
        if self.event_bus is not None:
            await self.event_bus.publish(event)

    async def emit(self, event: EventData) -> None:
        """Emit an event safely."""
        if self.event_bus is not None:
            await self.event_bus.emit(event)

    @abstractmethod
    async def _initialize(self) -> None:
        """Custom initialization logic to be implemented by plugins"""
        pass

    @abstractmethod
    async def _shutdown(self) -> None:
        """Custom shutdown logic to be implemented by plugins"""
        pass
