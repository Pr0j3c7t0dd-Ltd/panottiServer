"""Example plugin implementation."""

import asyncio
from datetime import datetime
from collections.abc import Callable, Coroutine
from typing import Any

from app.models.recording.events import (
    RecordingEndRequest,
    RecordingEvent,
    RecordingStartRequest,
    EventContext
)
from app.plugins.base import EventType, PluginBase, PluginConfig
from app.plugins.events.bus import EventBus
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Type alias for event handler
EventHandler = Callable[[EventType], Coroutine[Any, Any, None]]


class Plugin(PluginBase):
    """Example plugin implementation."""

    def __init__(self, config: PluginConfig, event_bus: EventBus | None = None) -> None:
        """Initialize plugin."""
        super().__init__(config, event_bus)

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "example"

    async def _initialize(self) -> None:
        """Subscribe to events"""
        if self.event_bus is None:
            return

        self.logger.info("Subscribing to events")
        await self._register_handlers()

        self.logger.info(
            "Example plugin initialized",
            extra={
                "subscriptions": [
                    "recording.started",
                    "recording.ended",
                ]
            },
        )

    async def _shutdown(self) -> None:
        """Unsubscribe from events"""
        if self.event_bus is None:
            return

        self.logger.info("Unsubscribing from events")
        await self.unsubscribe("recording.started", self._handle_recording_started)
        await self.unsubscribe("recording.ended", self._handle_recording_ended)

        self.logger.info("Example plugin shutdown")

    async def _register_handlers(self) -> None:
        """Register event handlers."""
        await self.subscribe("recording.started", self._handle_recording_started)
        await self.subscribe("recording.ended", self._handle_recording_ended)

    async def _handle_recording_started(self, event: EventType) -> None:
        """Handle recording started event."""
        try:
            # Extract recording ID and data
            if isinstance(event, (RecordingEvent, RecordingStartRequest, RecordingEndRequest)):
                recording_id = event.recording_id
                event_data = event.data if hasattr(event, 'data') else {}
            elif isinstance(event, dict):
                recording_id = event.get('recording_id')
                event_data = event.get('data', {})
            else:
                self.logger.error("Invalid event type", extra={"event_type": type(event).__name__})
                return

            if not recording_id:
                self.logger.error("No recording ID in event")
                return

            # Example processing
            self.logger.info(
                "Recording started",
                extra={
                    "plugin": self.name,
                    "recording_id": recording_id,
                    "event": "example.recording_started"
                }
            )

            # Emit enriched event
            if self.event_bus:
                from datetime import datetime
                
                event_data = {
                    "recording_id": recording_id,
                    "example": {
                        "status": "started",
                        "timestamp": datetime.utcnow().isoformat(),
                        "details": {
                            "plugin_name": self.name,
                            "event_type": "recording_started"
                        }
                    },
                    # Preserve original recording data
                    "recording": event_data.get("recording", {})
                }

                await self.event_bus.publish(RecordingEvent(
                    recording_timestamp=datetime.utcnow().isoformat(),
                    recording_id=recording_id,
                    event="example.recording_started",
                    data=event_data,
                    context=EventContext(
                        source_plugin=self.name,
                        metadata=event_data
                    )
                ))

        except Exception as e:
            self.logger.error(
                "Failed to handle recording started",
                extra={
                    "plugin": self.name,
                    "error": str(e),
                    "recording_id": recording_id if "recording_id" in locals() else "unknown"
                },
                exc_info=True
            )

    async def _handle_recording_ended(self, event: EventType) -> None:
        """Handle recording ended event.

        Args:
            event: Recording ended event data
        """
        try:
            # Extract recording ID from event
            if isinstance(event, dict):
                recording_id = event.get("recording_id")
            else:
                recording_id = event.recording_id

            if not recording_id:
                self.logger.error(
                    "No recording ID in event",
                    extra={
                        "plugin": self.name,
                        "event": str(event)
                    }
                )
                return

            # Check if this event originated from us to prevent loops
            if isinstance(event, RecordingEvent) and event.context and event.context.source_plugin == self.name:
                self.logger.debug(
                    "Skipping event that originated from us",
                    extra={
                        "plugin": self.name,
                        "recording_id": recording_id
                    }
                )
                return

            self.logger.info(
                "Recording ended",
                extra={
                    "plugin": self.name,
                    "recording_id": recording_id,
                    "event": "recording.ended"
                }
            )

            # Example processing logic here
            # Do not republish the event as it's not necessary
            
        except asyncio.CancelledError:
            self.logger.warning(
                "Task cancelled",
                extra={
                    "plugin": self.name,
                    "handler": "_handle_recording_ended"
                }
            )
            raise
        except Exception as e:
            self.logger.error(
                "Failed to handle recording ended",
                extra={
                    "plugin": self.name,
                    "error": str(e),
                    "recording_id": recording_id if 'recording_id' in locals() else None
                },
                exc_info=True
            )
