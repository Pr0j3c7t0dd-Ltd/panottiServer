import json
import logging
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, validator

from .database import get_db

logger = logging.getLogger(__name__)

# Constants for timestamp formats
COMPACT_TIMESTAMP_LENGTH = 14
ISO_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f",  # With microseconds
    "%Y-%m-%dT%H:%M:%S.%fZ",  # With microseconds and Z
    "%Y-%m-%dT%H:%M:%S",  # Without microseconds
    "%Y-%m-%dT%H:%M:%SZ",  # Without microseconds with Z
    "%Y-%m-%dT%H:%M:%S%z",  # With timezone offset
]


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string in various formats to datetime object.

    Supported formats:
    - ISO 8601 with microseconds and optional timezone
      (e.g., "2024-12-27T14:24:46.123456Z")
    - ISO 8601 without microseconds (e.g., "2024-12-27T14:24:46Z")
    - ISO 8601 with timezone offset (e.g., "2024-12-27T14:24:46+00:00")
    - Numeric timestamp (e.g., "1703686789" or "1703686789.123456")
    - Compact format (e.g., "20241227143602" for YYYYMMDDHHMMSS)
    """
    # Try compact format first (YYYYMMDDHHMMSS)
    if len(timestamp_str) == COMPACT_TIMESTAMP_LENGTH and timestamp_str.isdigit():
        try:
            return datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
        except ValueError:
            pass

    # First, try parsing as ISO format with various patterns
    # Remove trailing Z if present and try parsing
    clean_ts = timestamp_str.rstrip("Z")

    for fmt in ISO_FORMATS:
        try:
            return datetime.strptime(clean_ts, fmt)
        except ValueError:
            continue

    # Try parsing as numeric timestamp
    try:
        ts = float(timestamp_str)
        return datetime.fromtimestamp(ts)
    except ValueError as e:
        raise ValueError(f"Unable to parse timestamp: {timestamp_str}") from e


class EventMetadata(BaseModel):
    """Event metadata model.

    Attributes:
        event_provider_id: Unique identifier for the event provider
        event_title: Title of the event
        event_provider: Name of the event provider
        event_attendees: List of event attendees
        system_label: Label for system audio
        microphone_label: Label for microphone audio
        recording_started: Recording start timestamp
        recording_ended: Recording end timestamp
    """

    event_provider_id: str | None = None
    event_title: str | None = None
    event_provider: str | None = None
    event_attendees: list[str] | None = None
    system_label: str | None = None
    microphone_label: str | None = None
    recording_started: str | None = None
    recording_ended: str | None = None

    def to_db_format(self) -> dict[str, Any]:
        """Convert event to database format."""
        return self.dict(exclude_none=True)


class RecordingEvent(BaseModel):
    """Base class for recording events.

    Attributes:
        recording_timestamp: ISO8601 formatted timestamp
        recording_id: Unique identifier for the recording
        system_audio_path: Path to system audio file
        microphone_audio_path: Path to microphone audio file
        event: Type of recording event
        metadata: Additional event metadata
    """

    recording_timestamp: str
    recording_id: str
    system_audio_path: str | None = None
    microphone_audio_path: str | None = None
    event: Literal["Recording Started", "Recording Ended"]
    metadata: dict[str, Any] | EventMetadata | None = None

    def _get_metadata_field(self, field: str) -> Any:
        """Safely get a field from metadata whether it's a dict or EventMetadata."""
        if self.metadata is None:
            return None
        if isinstance(self.metadata, EventMetadata):
            return getattr(self.metadata, field, None)
        return self.metadata.get(field)

    def save(self) -> None:
        """Save the event to the database"""
        # Convert metadata to JSON if needed
        if self.metadata is None:
            metadata_json = None
        elif isinstance(self.metadata, EventMetadata):
            metadata_json = json.dumps(self.metadata.dict())
        else:
            metadata_json = json.dumps(self.metadata)

        # Extract metadata fields safely
        event_title = self._get_metadata_field("event_title")
        event_provider_id = self._get_metadata_field("event_provider_id")
        event_provider = self._get_metadata_field("event_provider")

        # Handle event_attendees specially
        attendees = self._get_metadata_field("event_attendees")
        event_attendees = json.dumps(attendees) if attendees else None

        system_label = self._get_metadata_field("system_label")
        microphone_label = self._get_metadata_field("microphone_label")
        recording_started = self._get_metadata_field("recording_started")
        recording_ended = self._get_metadata_field("recording_ended")

        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO events (
                    recording_id,
                    recording_timestamp,
                    event_title,
                    event_provider_id,
                    event_provider,
                    event_attendees,
                    system_label,
                    microphone_label,
                    recording_started,
                    recording_ended,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.recording_id,
                    self.recording_timestamp,
                    event_title,
                    event_provider_id,
                    event_provider,
                    event_attendees,
                    system_label,
                    microphone_label,
                    recording_started,
                    recording_ended,
                    metadata_json,
                ),
            )
            conn.commit()

    @classmethod
    def get_by_recording_id(cls, recording_id: str) -> list[dict[str, Any]]:
        """Retrieve all events for a specific recording."""
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT * FROM events WHERE recording_id = ?", (recording_id,)
            )
            events = [dict(row) for row in cursor.fetchall()]
            return events

    @classmethod
    def from_timestamp(cls, timestamp_str: str) -> str:
        """Parse timestamp string to datetime and return ISO format."""
        dt = parse_timestamp(timestamp_str)
        if not isinstance(dt, datetime):
            raise ValueError(f"Invalid timestamp format: {timestamp_str}")
        return dt.isoformat()


class RecordingStartRequest(BaseModel):
    """Request model for starting a recording session.

    Attributes:
        recording_timestamp: ISO8601 formatted timestamp of the event
        recording_id: Unique identifier for the recording session
        event: Type of recording event (always "Recording Started")
        metadata: Optional metadata about the recording
    """

    timestamp: str | None = None
    recording_timestamp: str | None = None
    recording_id: str
    event: Literal["Recording Started"] = Field(default="Recording Started")
    metadata: dict[str, Any] | None = None
    system_audio_path: str | None = None
    microphone_audio_path: str | None = None

    @validator("recording_timestamp", pre=True, always=True)
    def set_recording_timestamp(self, value: str | None, values: dict[str, Any]) -> str:
        """Use timestamp field if recording_timestamp is not provided"""
        if value is None:
            timestamp = str(values.get("timestamp", ""))
            if not timestamp:
                return datetime.utcnow().isoformat()
            return timestamp
        try:
            return parse_timestamp(value).isoformat()
        except ValueError as e:
            logger.error(f"Timestamp validation error for value '{value}': {e!s}")
            raise ValueError(str(e)) from e

    def to_event(self) -> RecordingEvent:
        """Convert request to RecordingEvent"""
        recording_timestamp = (
            self.recording_timestamp or self.timestamp or datetime.utcnow().isoformat()
        )

        return RecordingEvent(
            recording_id=self.recording_id,
            recording_timestamp=recording_timestamp,
            system_audio_path=self.system_audio_path,
            microphone_audio_path=self.microphone_audio_path,
            event=self.event,
            metadata=self.metadata,
        )


class RecordingEndRequest(BaseModel):
    """Request model for ending a recording session.

    Attributes:
        recording_timestamp: ISO8601 formatted timestamp of the event
        recording_id: Unique identifier for the recording session
        event: Type of recording event (always "Recording Ended")
        metadata: Metadata about the recording
        system_audio_path: Path to system audio file
        microphone_audio_path: Path to microphone audio file
    """

    timestamp: str | None = None
    recording_timestamp: str | None = None
    recording_id: str
    system_audio_path: str
    microphone_audio_path: str
    event: Literal["Recording Ended"] = Field(default="Recording Ended")
    metadata: dict[str, Any]

    @validator("recording_timestamp", pre=True, always=True)
    def set_recording_timestamp(self, value: str | None, values: dict[str, Any]) -> str:
        """Use timestamp field if recording_timestamp is not provided"""
        if value is None:
            timestamp = str(values.get("timestamp", ""))
            if not timestamp:
                return datetime.utcnow().isoformat()
            return timestamp
        try:
            return parse_timestamp(value).isoformat()
        except ValueError as e:
            logger.error(f"Timestamp validation error for value '{value}': {e!s}")
            raise ValueError(str(e)) from e

    def to_event(self) -> RecordingEvent:
        """Convert request to RecordingEvent"""
        recording_timestamp = (
            self.recording_timestamp or self.timestamp or datetime.utcnow().isoformat()
        )

        return RecordingEvent(
            recording_id=self.recording_id,
            recording_timestamp=recording_timestamp,
            system_audio_path=self.system_audio_path,
            microphone_audio_path=self.microphone_audio_path,
            event=self.event,
            metadata=self.metadata,
        )
