from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any

from app.models.recording.events import (
    RecordingEndRequest,
    RecordingEvent,
    RecordingStartRequest,
)
from app.plugins.base import PluginBase
from app.plugins.events.models import Event

EventData = (
    dict[str, Any] | RecordingEvent | RecordingStartRequest | RecordingEndRequest
)
EventHandler = Callable[[EventData], Coroutine[Any, Any, None]]


class Plugin(PluginBase):
    """Example plugin demonstrating event cascade with flattened payloads"""

    async def _initialize(self) -> None:
        """Subscribe to events"""
        if self.event_bus is None:
            self.logger.warning("Event bus not available, skipping event subscriptions")
            return

        self.logger.info("Subscribing to events")
        await self.event_bus.subscribe(
            "recording.started", self.handle_recording_started
        )
        await self.event_bus.subscribe("recording.ended", self.handle_recording_ended)

        self.logger.info(
            "Example plugin initialized",
            extra={
                "subscribed_events": [
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
        await self.event_bus.unsubscribe(
            "recording.started", self.handle_recording_started
        )
        await self.event_bus.unsubscribe("recording.ended", self.handle_recording_ended)

        self.logger.info("Example plugin shutdown")

    async def handle_recording_started(self, event: Event) -> None:
        """Handle recording started event"""
        self.logger.info(
            "Recording started",
            extra={
                "recording_id": event.data.get("recording_id"),
                "correlation_id": event.context.correlation_id,
            },
        )
        await self.emit_event(
            "example.recording_started",
            {
                "recording_id": event.data.get("recording_id"),
                "message": "Example plugin received recording started event",
            },
            correlation_id=event.context.correlation_id,
        )

    async def handle_recording_ended(self, event: Event) -> None:
        """Handle recording ended event"""
        self.logger.info(
            "Recording ended",
            extra={
                "recording_id": event.data.get("recording_id"),
                "correlation_id": event.context.correlation_id,
            },
        )
        await self.emit_event(
            "example.recording_ended",
            {
                "recording_id": event.data.get("recording_id"),
                "message": "Example plugin received recording ended event",
            },
            correlation_id=event.context.correlation_id,
        )

    async def emit_example_event(self, recording_id: str) -> None:
        """Emit an example event"""
        if not self.event_bus:
            return

        event_data: dict[str, Any] = {
            "type": "example.event",
            "recording_id": recording_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self.event_bus.emit(event_data)
