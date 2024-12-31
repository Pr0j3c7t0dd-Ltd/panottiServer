from datetime import datetime

from app.plugins.base import PluginBase
from app.plugins.events.models import Event, EventContext, EventPriority


class Plugin(PluginBase):
    """Example plugin demonstrating event cascade with flattened payloads"""

    async def _initialize(self) -> None:
        """Initialize plugin"""
        # Subscribe to events in the cascade
        self.event_bus.subscribe("recording.ended", self.handle_recording_ended)
        self.event_bus.subscribe(
            "noise_reduction.completed", self.handle_noise_reduction
        )
        self.event_bus.subscribe("transcription.completed", self.handle_transcription)
        self.event_bus.subscribe("meeting_notes.completed", self.handle_meeting_notes)
        self.event_bus.subscribe(
            "desktop_notification.completed", self.handle_notification
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
        # Unsubscribe from all events
        self.event_bus.unsubscribe("recording.ended", self.handle_recording_ended)
        self.event_bus.unsubscribe(
            "noise_reduction.completed", self.handle_noise_reduction
        )
        self.event_bus.unsubscribe("transcription.completed", self.handle_transcription)
        self.event_bus.unsubscribe("meeting_notes.completed", self.handle_meeting_notes)
        self.event_bus.unsubscribe(
            "desktop_notification.completed", self.handle_notification
        )

        self.logger.info("Example plugin shutdown")

    async def handle_recording_ended(self, event: Event) -> None:
        """Handle recording ended event - Start of cascade"""
        self.logger.info(
            "Recording ended event received",
            extra={
                "recording_id": event.payload.get("recording_id"),
                "event_title": event.payload.get("event_title"),
                "correlation_id": event.context.correlation_id,
            },
        )

        # Example of creating a flattened event payload
        await self.emit_example_event(
            name="example.recording_processed",
            recording_id=event.payload.get("recording_id"),
            event_title=event.payload.get("event_title"),
            event_provider=event.payload.get("event_provider"),
            correlation_id=event.context.correlation_id,
            status="processing",
        )

    async def handle_noise_reduction(self, event: Event) -> None:
        """Handle noise reduction completed event"""
        self.logger.info(
            "Noise reduction completed event received",
            extra={
                "recording_id": event.payload.get("recording_id"),
                "noise_reduced_audio_path": event.payload.get(
                    "noise_reduced_audio_path"
                ),
                "noise_reduction_status": event.payload.get("noise_reduction_status"),
            },
        )

        await self.emit_example_event(
            name="example.noise_reduced",
            recording_id=event.payload["recording_id"],
            noise_reduced_audio_path=event.payload.get("noise_reduced_audio_path"),
            correlation_id=event.context.correlation_id,
            status="processing",
        )

    async def handle_transcription(self, event: Event) -> None:
        """Handle transcription completed event"""
        self.logger.info(
            "Transcription completed event received",
            extra={
                "recording_id": event.payload.get("recording_id"),
                "merged_transcript_path": event.payload.get("merged_transcript_path"),
                "transcription_status": event.payload.get("transcription_status"),
            },
        )

        await self.emit_example_event(
            name="example.transcribed",
            recording_id=event.payload["recording_id"],
            merged_transcript_path=event.payload.get("merged_transcript_path"),
            correlation_id=event.context.correlation_id,
            status="processing",
        )

    async def handle_meeting_notes(self, event: Event) -> None:
        """Handle meeting notes completed event"""
        self.logger.info(
            "Meeting notes completed event received",
            extra={
                "recording_id": event.payload.get("recording_id"),
                "meeting_notes_path": event.payload.get("meeting_notes_path"),
                "meeting_notes_status": event.payload.get("meeting_notes_status"),
            },
        )

        await self.emit_example_event(
            name="example.notes_generated",
            recording_id=event.payload["recording_id"],
            meeting_notes_path=event.payload.get("meeting_notes_path"),
            correlation_id=event.context.correlation_id,
            status="processing",
        )

    async def handle_notification(self, event: Event) -> None:
        """Handle desktop notification completed event - End of cascade"""
        self.logger.info(
            "Desktop notification completed event received",
            extra={
                "recording_id": event.payload.get("recording_id"),
                "notification_status": event.payload.get("notification_status"),
            },
        )

        await self.emit_example_event(
            name="example.cascade_completed",
            recording_id=event.payload["recording_id"],
            correlation_id=event.context.correlation_id,
            status="completed",
        )

    async def emit_example_event(
        self, name: str, recording_id: str, correlation_id: str, status: str, **kwargs
    ) -> None:
        """Helper method to emit events with consistent structure"""
        # Create base payload with common fields
        payload = {
            # Recording identifiers
            "recording_id": recording_id,
            "recording_timestamp": datetime.utcnow().isoformat(),
            # Processing status
            "status": status,
            # Add any additional fields from kwargs
            **kwargs,
        }

        event = Event(
            name=name,
            payload=payload,
            context=EventContext(
                correlation_id=correlation_id, source_plugin=self.name
            ),
            priority=EventPriority.LOW,
        )

        await self.event_bus.emit(event)
