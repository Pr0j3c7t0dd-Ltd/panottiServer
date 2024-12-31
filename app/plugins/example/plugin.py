"""Example plugin implementation."""

from collections.abc import Callable, Coroutine
from typing import Any

from app.models.recording.events import (
    RecordingEndRequest,
    RecordingEvent,
    RecordingStartRequest,
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
        recording_id = (
            event.recording_id
            if isinstance(
                event, RecordingEvent | RecordingStartRequest | RecordingEndRequest
            )
            else event["recording_id"]
            if isinstance(event, dict)
            else None
        )
        if not recording_id:
            self.logger.error("No recording ID in event")
            return

        self.logger.info(
            "Recording started",
            extra={
                "recording_id": recording_id,
                "event_type": type(event).__name__,
            },
        )

    async def _handle_recording_ended(self, event: EventType) -> None:
        """Handle recording ended event."""
        recording_id = (
            event.recording_id
            if isinstance(
                event, RecordingEvent | RecordingStartRequest | RecordingEndRequest
            )
            else event["recording_id"]
            if isinstance(event, dict)
            else None
        )
        if not recording_id:
            self.logger.error("No recording ID in event")
            return

        self.logger.info(
            "Recording ended",
            extra={
                "recording_id": recording_id,
                "event_type": type(event).__name__,
            },
        )
