import logging
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from app.core.events import ConcreteEventBus as EventBus
from app.core.events.types import EventHandler
from app.core.plugins.protocol import PluginProtocol
from app.models.recording.events import (
    RecordingEndRequest,
    RecordingEvent,
    RecordingStartRequest,
)
from app.utils.logging_config import get_logger

# Define a type for all possible event types
EventType = (
    dict[str, Any] | RecordingEvent | RecordingStartRequest | RecordingEndRequest
)

# Initialize logging when the module is imported
logger = get_logger(__name__)


class PluginConfig(BaseModel):
    """Plugin configuration"""

    name: str
    version: str
    enabled: bool = True
    dependencies: list[str] = []
    config: dict[str, Any] | None = None


class PluginBase(ABC, PluginProtocol):
    """Base implementation for all plugins"""

    def __init__(self, config: PluginConfig, event_bus: EventBus | None = None) -> None:
        self.config = config
        self.event_bus = event_bus
        self.version = config.version
        self.logger = logging.getLogger(f"plugin.{config.name}")
        self._initialized = False
        self._req_id = str(uuid4())  # Add request ID for tracing

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the plugin.

        This method should be called only once. Subsequent calls will be ignored.
        """
        if self._initialized:
            self.logger.debug(
                f"Plugin {self.name} already initialized",
                extra={
                    "req_id": self._req_id,
                    "plugin": self.name,
                    "version": self.version,
                },
            )
            return

        try:
            self.logger.debug(
                f"Initializing plugin {self.name}",
                extra={
                    "req_id": self._req_id,
                    "plugin": self.name,
                    "version": self.version,
                },
            )
            await self._initialize()
            self._initialized = True
            self.logger.info(
                f"Plugin {self.name} initialized successfully",
                extra={
                    "req_id": self._req_id,
                    "plugin": self.name,
                    "version": self.version,
                },
            )
        except Exception as e:
            self.logger.error(
                f"Failed to initialize plugin {self.name}",
                extra={
                    "req_id": self._req_id,
                    "plugin": self.name,
                    "version": self.version,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise

    async def shutdown(self) -> None:
        """Shutdown the plugin"""
        try:
            self.logger.info(
                "Shutting down plugin",
                extra={"req_id": self._req_id, "plugin_name": self.name},
            )
            await self._shutdown()
            self._initialized = False
            self.logger.info(
                "Plugin shutdown complete",
                extra={"req_id": self._req_id, "plugin_name": self.name},
            )
        except Exception as e:
            self.logger.error(
                "Error during plugin shutdown",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "error": str(e),
                },
            )
            raise

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        if self.config.config is None:
            return default
        value = self.config.config.get(key, default)
        self.logger.debug(
            "Retrieved plugin config",
            extra={
                "req_id": self._req_id,
                "plugin_name": self.name,
                "config_key": key,
                "config_value": value,
            },
        )
        return value

    async def subscribe(self, event_type: str, callback: EventHandler) -> None:
        """Subscribe to events safely."""
        if self.event_bus is None:
            self.logger.warning(
                "Cannot subscribe to events: no event bus available",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "event_type": event_type,
                },
            )
            return
        await self.event_bus.subscribe(event_type, callback)

    async def unsubscribe(self, event_type: str, callback: EventHandler) -> None:
        """Unsubscribe from events safely."""
        if self.event_bus is not None:
            await self.event_bus.unsubscribe(event_type, callback)

    async def publish(self, event: EventType) -> None:
        """Publish an event safely."""
        if self.event_bus is not None:
            await self.event_bus.publish(event)

    async def emit_event(
        self, name: str, data: dict[str, Any], correlation_id: str | None = None
    ) -> None:
        """Helper method to emit events with proper context"""
        if self.event_bus is None:
            self.logger.warning(
                f"Cannot emit event {name}: no event bus available",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "event_name": name,
                },
            )
            return

        # Convert the event to a dict type that EventBus accepts
        event_data = {
            "name": name,
            "data": data,
            "correlation_id": correlation_id or "unknown",
            "source_plugin": self.name,
        }
        await self.event_bus.publish(event_data)

    @abstractmethod
    async def _initialize(self) -> None:
        """Plugin-specific initialization"""
        pass

    @abstractmethod
    async def _shutdown(self) -> None:
        """Plugin-specific shutdown"""
        pass
