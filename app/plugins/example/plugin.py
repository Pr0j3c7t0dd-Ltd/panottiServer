"""Example plugin implementation."""

import asyncio
from datetime import datetime
from collections.abc import Callable, Coroutine
from typing import Any
import uuid

from app.models.recording.events import (
    RecordingEndRequest,
    RecordingEvent,
    RecordingStartRequest,
)
from app.plugins.base import EventType, PluginBase, PluginConfig
from app.plugins.events.bus import EventBus
from app.plugins.events.models import EventContext
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Type alias for event handler
EventHandler = Callable[[EventType], Coroutine[Any, Any, None]]


class ExamplePlugin(PluginBase):
    """Example plugin implementation."""

    def __init__(self, config: PluginConfig, event_bus: EventBus | None = None) -> None:
        """Initialize plugin.
        
        Args:
            config: Plugin configuration
            event_bus: Optional event bus instance
        """
        super().__init__(config, event_bus)
        self._event_delay = config.config.get("event_delay", 0) if config.config else 0
        self._subscribed_events = set()
        logger.info(
            "Example plugin initialized",
            extra={
                "plugin": self.name,
                "event_delay": self._event_delay
            }
        )

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "example"

    async def _initialize(self) -> None:
        """Initialize plugin by subscribing to events."""
        if self._initialized:
            logger.debug(
                "Plugin already initialized",
                extra={"plugin": self.name}
            )
            return

        try:
            # Subscribe to events
            events_to_subscribe = [
                ("recording.started", self._handle_recording_started),
                ("recording.ended", self._handle_recording_ended)
            ]

            for event_name, handler in events_to_subscribe:
                if event_name not in self._subscribed_events:
                    await self.event_bus.subscribe(event_name, handler)
                    self._subscribed_events.add(event_name)
                    logger.debug(
                        f"Subscribed to {event_name}",
                        extra={
                            "plugin": self.name,
                            "event": event_name,
                            "handler": handler.__name__
                        }
                    )

            self._initialized = True
            logger.info(
                "Plugin initialization complete",
                extra={
                    "plugin": self.name,
                    "subscribed_events": list(self._subscribed_events)
                }
            )

        except Exception as e:
            logger.error(
                "Failed to initialize plugin",
                extra={
                    "plugin": self.name,
                    "error": str(e)
                },
                exc_info=True
            )
            raise

    async def _handle_recording_started(self, event: EventType) -> None:
        """Handle recording started event."""
        try:
            logger.debug(
                "Received recording started event",
                extra={
                    "plugin": self.name,
                    "event_type": type(event).__name__,
                    "event_data": str(event)
                }
            )

            # Extract recording ID
            if isinstance(event, dict):
                recording_id = event.get("recording_id")
            else:
                recording_id = event.recording_id if hasattr(event, "recording_id") else None

            if not recording_id:
                logger.error(
                    "No recording_id found in event data",
                    extra={
                        "plugin": self.name,
                        "event_data": str(event)
                    }
                )
                return

            # Check if this event originated from us to prevent loops
            if isinstance(event, RecordingEvent) and event.context and event.context.source_plugin == self.name:
                logger.debug(
                    "Skipping event that originated from us",
                    extra={
                        "plugin": self.name,
                        "recording_id": recording_id
                    }
                )
                return

            # Simulate processing delay if configured
            if self._event_delay > 0:
                await asyncio.sleep(self._event_delay)

            # Create and emit response event
            response_event = RecordingEvent(
                recording_timestamp=datetime.utcnow().isoformat(),
                recording_id=recording_id,
                event="example.processed",
                data={
                    "recording_id": recording_id,
                    "status": "processed",
                    "timestamp": datetime.utcnow().isoformat()
                },
                context=EventContext(
                    correlation_id=str(uuid.uuid4()),
                    source_plugin=self.name
                )
            )

            await self.event_bus.publish(response_event)

            logger.info(
                "Processed recording started event",
                extra={
                    "plugin": self.name,
                    "recording_id": recording_id
                }
            )

        except Exception as e:
            logger.error(
                "Failed to handle recording started event",
                extra={
                    "plugin": self.name,
                    "error": str(e),
                    "event_data": str(event)
                },
                exc_info=True
            )
            raise

    async def _handle_recording_ended(self, event: EventType) -> None:
        """Handle recording ended event."""
        try:
            logger.debug(
                "Received recording ended event",
                extra={
                    "plugin": self.name,
                    "event_type": type(event).__name__,
                    "event_data": str(event)
                }
            )

            # Extract recording ID
            if isinstance(event, dict):
                recording_id = event.get("recording_id")
            else:
                recording_id = event.recording_id if hasattr(event, "recording_id") else None

            if not recording_id:
                logger.error(
                    "No recording_id found in event data",
                    extra={
                        "plugin": self.name,
                        "event_data": str(event)
                    }
                )
                return

            # Check if this event originated from us to prevent loops
            if isinstance(event, RecordingEvent) and event.context and event.context.source_plugin == self.name:
                logger.debug(
                    "Skipping event that originated from us",
                    extra={
                        "plugin": self.name,
                        "recording_id": recording_id
                    }
                )
                return

            # Simulate processing delay if configured
            if self._event_delay > 0:
                await asyncio.sleep(self._event_delay)

            # Create and emit response event
            response_event = RecordingEvent(
                recording_timestamp=datetime.utcnow().isoformat(),
                recording_id=recording_id,
                event="example.processed",
                data={
                    "recording_id": recording_id,
                    "status": "processed",
                    "timestamp": datetime.utcnow().isoformat()
                },
                context=EventContext(
                    correlation_id=str(uuid.uuid4()),
                    source_plugin=self.name
                )
            )

            await self.event_bus.publish(response_event)

            logger.info(
                "Processed recording ended event",
                extra={
                    "plugin": self.name,
                    "recording_id": recording_id
                }
            )

        except Exception as e:
            logger.error(
                "Failed to handle recording ended event",
                extra={
                    "plugin": self.name,
                    "error": str(e),
                    "event_data": str(event)
                },
                exc_info=True
            )
            raise

    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        try:
            # Unsubscribe from all events
            for event_name in self._subscribed_events:
                if event_name == "recording.started":
                    await self.event_bus.unsubscribe(event_name, self._handle_recording_started)
                elif event_name == "recording.ended":
                    await self.event_bus.unsubscribe(event_name, self._handle_recording_ended)

            self._subscribed_events.clear()
            self._initialized = False

            logger.info(
                "Plugin shutdown complete",
                extra={"plugin": self.name}
            )

        except Exception as e:
            logger.error(
                "Error during plugin shutdown",
                extra={
                    "plugin": self.name,
                    "error": str(e)
                },
                exc_info=True
            )
            raise
