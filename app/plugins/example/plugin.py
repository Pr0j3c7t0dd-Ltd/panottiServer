from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any

from app.models.event import RecordingEndRequest, RecordingEvent, RecordingStartRequest
from app.plugins.base import PluginBase

EventData = (
    dict[str, Any] | RecordingEvent | RecordingStartRequest | RecordingEndRequest
)
EventHandler = Callable[[EventData], Coroutine[Any, Any, None]]


class Plugin(PluginBase):
    """Example plugin demonstrating event cascade with flattened payloads"""

    async def _initialize(self) -> None:
        """Initialize plugin"""
        if not self.event_bus:
            return

        # Subscribe to events in the cascade
        await self.event_bus.subscribe("recording.ended", self.handle_event)
        await self.event_bus.subscribe("noise_reduction.completed", self.handle_event)
        await self.event_bus.subscribe("transcription.completed", self.handle_event)
        await self.event_bus.subscribe("meeting_notes.completed", self.handle_event)
        await self.event_bus.subscribe(
            "desktop_notification.completed", self.handle_event
        )

        self.logger.info(
            "Example plugin initialized",
            extra={
                "subscribed_events": [
                    "recording.ended",
                    "noise_reduction.completed",
                    "transcription.completed",
                    "meeting_notes.completed",
                    "desktop_notification.completed",
                ]
            },
        )

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        if not self.event_bus:
            return

        # Unsubscribe from all events
        await self.event_bus.unsubscribe("recording.ended", self.handle_event)
        await self.event_bus.unsubscribe("noise_reduction.completed", self.handle_event)
        await self.event_bus.unsubscribe("transcription.completed", self.handle_event)
        await self.event_bus.unsubscribe("meeting_notes.completed", self.handle_event)
        await self.event_bus.unsubscribe(
            "desktop_notification.completed", self.handle_event
        )

        self.logger.info("Example plugin shutdown")

    async def handle_event(self, event_data: EventData) -> None:
        """Handle incoming events"""
        try:
            if isinstance(event_data, dict):
                recording_id = event_data.get("recording_id", "unknown")
                event_type = event_data.get("type", "unknown")
                status = event_data.get("status", "unknown")
            else:
                # Handle RecordingEvent, RecordingStartRequest, RecordingEndRequest
                recording_id = getattr(event_data, "recording_id", "unknown")
                event_type = getattr(event_data, "type", "unknown")
                status = getattr(event_data, "status", "unknown")

            self.logger.info(
                "Handling event",
                extra={
                    "recording_id": recording_id,
                    "event_type": event_type,
                    "status": status,
                },
            )

            # Process event based on type
            if event_type == "recording.start":
                await self._handle_recording_start(recording_id)
            elif event_type == "recording.end":
                await self._handle_recording_end(recording_id)
            elif event_type == "recording.complete":
                await self._handle_recording_complete(recording_id)

        except Exception as e:
            self.logger.error(
                f"Failed to handle event: {e}",
                extra={
                    "recording_id": (
                        recording_id if "recording_id" in locals() else "unknown"
                    )
                },
                exc_info=True,
            )

    async def _handle_recording_start(self, recording_id: str) -> None:
        """Handle recording start event"""
        if not self.event_bus:
            return

        event_data = {
            "type": "example.recording_started",
            "recording_id": recording_id,
            "status": "started",
            "timestamp": datetime.utcnow().isoformat(),
        }
        await self.event_bus.emit(event_data)

    async def _handle_recording_end(self, recording_id: str) -> None:
        """Handle recording end event"""
        if not self.event_bus:
            return

        event_data = {
            "type": "example.recording_ended",
            "recording_id": recording_id,
            "status": "ended",
            "timestamp": datetime.utcnow().isoformat(),
        }
        await self.event_bus.emit(event_data)

    async def _handle_recording_complete(self, recording_id: str) -> None:
        """Handle recording complete event"""
        if not self.event_bus:
            return

        event_data = {
            "type": "example.recording_completed",
            "recording_id": recording_id,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat(),
        }
        await self.event_bus.emit(event_data)

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
