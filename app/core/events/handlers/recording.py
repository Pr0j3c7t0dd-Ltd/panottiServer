"""Recording event handlers."""

from typing import Any

from app.utils.logging_config import get_logger

logger = get_logger(__name__)


async def handle_recording_started(event_data: Any) -> None:
    """Handle recording.started event.

    This handler is responsible for:
    1. Converting the event data to a RecordingEvent
    2. Saving the event to the database
    3. Logging the start of recording
    """
    from app.models.recording.events import (
        RecordingEvent,  # Import inside function to avoid circular import
    )

    try:
        # Convert event data to RecordingEvent if needed
        if not isinstance(event_data, RecordingEvent):
            event_data = RecordingEvent(**event_data)

        # Save event to database
        await event_data.save()

        logger.info(
            "Recording started",
            extra={
                "recording_id": event_data.recording_id,
                "event_id": event_data.event_id,
                "system_audio": event_data.system_audio_path,
                "microphone_audio": event_data.microphone_audio_path,
            },
        )
    except Exception as e:
        logger.error(
            "Error handling recording.started event",
            extra={
                "error": str(e),
                "event_data": str(event_data),
            },
            exc_info=True,
        )
        raise


async def handle_recording_ended(event_data: Any) -> None:
    """Handle recording.ended event.

    This handler is responsible for:
    1. Converting the event data to a RecordingEvent
    2. Saving the event to the database
    3. Logging the end of recording
    """
    from app.models.recording.events import (
        RecordingEvent,  # Import inside function to avoid circular import
    )

    try:
        # Convert event data to RecordingEvent if needed
        if not isinstance(event_data, RecordingEvent):
            event_data = RecordingEvent(**event_data)

        # Save event to database
        await event_data.save()

        logger.info(
            "Recording ended",
            extra={
                "recording_id": event_data.recording_id,
                "event_id": event_data.event_id,
                "system_audio": event_data.system_audio_path,
                "microphone_audio": event_data.microphone_audio_path,
            },
        )
    except Exception as e:
        logger.error(
            "Error handling recording.ended event",
            extra={
                "error": str(e),
                "event_data": str(event_data),
            },
            exc_info=True,
        )
