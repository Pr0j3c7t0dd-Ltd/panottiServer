"""Recording event models."""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.models.database import DatabaseManager
from app.plugins.events.models import EventContext

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
    """Parse timestamp string in various formats to datetime object."""
    # Try compact format first (YYYYMMDDHHMMSS)
    if len(timestamp_str) == COMPACT_TIMESTAMP_LENGTH and timestamp_str.isdigit():
        try:
            return datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
        except ValueError:
            pass

    # First, try parsing as ISO format with various patterns
    clean_ts = timestamp_str.rstrip("Z")
    for fmt in ISO_FORMATS:
        try:
            return datetime.strptime(clean_ts, fmt)
        except ValueError:
            continue

    # Try parsing as numeric timestamp
    try:
        return datetime.fromtimestamp(float(timestamp_str))
    except ValueError:
        raise ValueError(f"Unable to parse timestamp: {timestamp_str}")


class EventMetadata(BaseModel):
    """Event metadata model."""

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
        return json.loads(self.model_dump_json())


class RecordingEvent(BaseModel):
    """Base class for recording events."""

    recording_timestamp: str
    recording_id: str
    system_audio_path: str | None = None
    microphone_audio_path: str | None = None
    event: Literal["Recording Started", "Recording Ended"]
    metadata: dict[str, Any] | EventMetadata | None = None
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plugin_id: str = Field(default="recording_plugin")
    name: str = Field(default="")  # Required by event store
    data: dict[str, Any] = Field(default_factory=dict)  # Required by event store
    context: EventContext = Field(default_factory=lambda: EventContext(correlation_id=str(uuid.uuid4())))

    @field_validator("name", mode="before")
    @classmethod
    def set_name_from_event(cls, v: str, values: dict[str, Any]) -> str:
        """Set the name field based on the event type."""
        if not v and "event" in values:
            return values["event"]
        return v

    @field_validator("data", mode="before")
    @classmethod
    def set_data(cls, v: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
        """Set the data field with relevant event information."""
        if not v:
            data = {
                "recording_id": values.get("recording_id", ""),
                "recording_timestamp": values.get("recording_timestamp", ""),
                "system_audio_path": values.get("system_audio_path"),
                "microphone_audio_path": values.get("microphone_audio_path"),
            }
            if "metadata" in values and values["metadata"]:
                data["metadata"] = values["metadata"]
            return data
        return v

    def _get_metadata_field(self, field: str) -> Any:
        """Safely get a field from metadata whether it's a dict or EventMetadata."""
        if isinstance(self.metadata, EventMetadata):
            return getattr(self.metadata, field, None)
        elif isinstance(self.metadata, dict):
            return self.metadata.get(field)
        return None

    async def save(self) -> None:
        """Save the event to the database."""
        db = DatabaseManager.get_instance()
        await db.insert(
            """
            INSERT INTO recording_events (
                recording_id,
                event_type,
                event_timestamp,
                system_audio_path,
                microphone_audio_path,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self.recording_id,
                self.event,
                self.recording_timestamp,
                self.system_audio_path,
                self.microphone_audio_path,
                json.dumps(self.metadata) if self.metadata else None,
            ),
        )
        await db.commit()

    @classmethod
    async def get_by_recording_id(cls, recording_id: str) -> list["RecordingEvent"]:
        """Retrieve all events for a specific recording."""
        db = DatabaseManager.get_instance()
        rows = await db.fetch_all(
            """
            SELECT * FROM recording_events
            WHERE recording_id = ?
            ORDER BY event_timestamp
            """,
            (recording_id,),
        )
        return [
            cls(
                recording_id=row["recording_id"],
                event=row["event_type"],
                recording_timestamp=row["event_timestamp"],
                system_audio_path=row["system_audio_path"],
                microphone_audio_path=row["microphone_audio_path"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            for row in rows
        ]

    @classmethod
    def from_timestamp(cls, timestamp_str: str) -> str:
        """Parse timestamp string to datetime and return ISO format."""
        try:
            dt = parse_timestamp(timestamp_str)
            return dt.isoformat()
        except ValueError as e:
            logger.error(f"Error parsing timestamp: {e}")
            return datetime.utcnow().isoformat()


class RecordingStartRequest(BaseModel):
    """Request model for starting a recording session."""

    timestamp: str | None = None
    recording_timestamp: str | None = None
    recording_id: str
    event: Literal["Recording Started"] = Field(default="Recording Started")
    metadata: dict[str, Any] | None = None
    system_audio_path: str | None = None
    microphone_audio_path: str | None = None

    @field_validator("recording_timestamp", mode="before")
    # pylint: disable=no-self-argument  # Pydantic validators use cls instead of self
    @classmethod
    def set_recording_timestamp(cls, value: str | None, info: Any) -> str:
        """Use timestamp field if recording_timestamp is not provided."""
        data = info.data
        if value is None and data.get("timestamp"):
            return RecordingEvent.from_timestamp(data["timestamp"])
        elif value:
            return RecordingEvent.from_timestamp(value)
        return datetime.utcnow().isoformat()

    def to_event(self) -> RecordingEvent:
        """Convert request to RecordingEvent."""
        return RecordingEvent(
            recording_timestamp=self.recording_timestamp or self.timestamp,
            recording_id=self.recording_id,
            system_audio_path=self.system_audio_path,
            microphone_audio_path=self.microphone_audio_path,
            event=self.event,
            metadata=self.metadata,
            name=self.event,  # Set name to event type
            data={  # Populate data field
                "recording_id": self.recording_id,
                "recording_timestamp": self.recording_timestamp or self.timestamp,
                "system_audio_path": self.system_audio_path,
                "microphone_audio_path": self.microphone_audio_path,
                "metadata": self.metadata,
            },
        )


class RecordingEndRequest(BaseModel):
    """Request model for ending a recording session."""
    timestamp: str | None = None
    recording_timestamp: str | None = Field(None, alias="recordingTimestamp")
    recording_id: str = Field(..., alias="recordingId")
    system_audio_path: str = Field(..., alias="systemAudioPath")
    microphone_audio_path: str = Field(..., alias="microphoneAudioPath")
    event: Literal["Recording Ended"] = Field(default="Recording Ended")
    metadata: dict[str, Any]

    @field_validator("recording_timestamp", mode="before")
    # pylint: disable=no-self-argument  # Pydantic validators use cls instead of self
    def set_recording_timestamp(cls, value: str | None, info: Any) -> str | None:
        """Use timestamp field if recording_timestamp is not provided."""
        data = info.data
        if value is None and "timestamp" in data:
            return data["timestamp"]
        return value

    def to_event(self) -> RecordingEvent:
        """Convert request to RecordingEvent."""
        return RecordingEvent(
            recording_timestamp=self.recording_timestamp or self.timestamp,
            recording_id=self.recording_id,
            system_audio_path=self.system_audio_path,
            microphone_audio_path=self.microphone_audio_path,
            event=self.event,
            metadata=self.metadata,
            name=self.event,  # Set name to event type
            data={  # Populate data field
                "recording_id": self.recording_id,
                "recording_timestamp": self.recording_timestamp or self.timestamp,
                "system_audio_path": self.system_audio_path,
                "microphone_audio_path": self.microphone_audio_path,
                "metadata": self.metadata,
            },
        )
