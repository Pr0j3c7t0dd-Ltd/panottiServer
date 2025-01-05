"""Recording event models."""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, ValidationInfo, field_validator

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
    """Parse timestamp string into datetime object."""
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
    except ValueError as err:
        raise ValueError(f"Unable to parse timestamp: {timestamp_str}") from err


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
        return dict(json.loads(self.model_dump_json()))


T = TypeVar("T")


class RecordingEvent(BaseModel):
    """Base class for recording events."""

    recording_timestamp: str
    recording_id: str
    system_audio_path: str | None = None
    microphone_audio_path: str | None = None
    event: Literal[
        "recording.started",
        "recording.ended",
        "noise_reduction.completed",
        "transcription.completed",
        "transcription.error",
        "meeting_notes.completed",
        "meeting_notes.error",
        "desktop_notification.completed",
    ]
    metadata: dict[str, Any] | EventMetadata | None = None
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plugin_id: str = Field(default="recording_plugin")
    data: dict[str, Any] = Field(default_factory=dict)  # Required by event store
    context: EventContext = Field(
        default_factory=lambda: EventContext(correlation_id=str(uuid.uuid4()))
    )
    output_file: str | None = None  # For plugin completion events
    status: str | None = None  # For plugin completion events

    @field_validator("data")
    @classmethod
    def set_data(cls, v: dict[str, Any] | None, info: ValidationInfo) -> dict[str, Any]:
        """Set the data field with relevant event information."""
        if v is None or not v:
            data: dict[str, Any] = {
                "recording_id": info.data.get("recording_id", ""),
                "recording_timestamp": info.data.get("recording_timestamp", ""),
                "system_audio_path": info.data.get("system_audio_path"),
                "microphone_audio_path": info.data.get("microphone_audio_path"),
            }
            if info.data.get("metadata"):
                data["metadata"] = info.data["metadata"]
            return data
        return v

    @field_validator("recording_timestamp")
    @classmethod
    def set_recording_timestamp(cls, v: str | None, info: ValidationInfo) -> str:
        """Use timestamp field if recording_timestamp is not provided."""
        if v is None:
            timestamp = info.data.get("recording_timestamp") or info.data.get(
                "timestamp"
            )
            if timestamp is None:
                raise ValueError("recording_timestamp is required")
            return str(timestamp)
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

        # Save to recording_events table
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

        # For recording.ended events, update the recordings table
        if self.event == "recording.ended":
            await db.execute(
                """
                INSERT INTO recordings (
                    recording_id,
                    status,
                    system_audio_path,
                    microphone_audio_path,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(recording_id) DO UPDATE SET
                    status = 'completed',
                    system_audio_path = excluded.system_audio_path,
                    microphone_audio_path = excluded.microphone_audio_path,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    self.recording_id,
                    "completed",
                    self.system_audio_path,
                    self.microphone_audio_path,
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
    event: Literal["recording.started"] = Field(default="recording.started")
    metadata: dict[str, Any] | None = None
    system_audio_path: str | None = None
    microphone_audio_path: str | None = None

    @field_validator("recording_timestamp")
    @classmethod
    def set_recording_timestamp(cls, value: str | None, info: ValidationInfo) -> str:
        """Use timestamp field if recording_timestamp is not provided."""
        if value is None and "timestamp" in info.data:
            return RecordingEvent.from_timestamp(info.data["timestamp"])
        elif value:
            return RecordingEvent.from_timestamp(value)
        return datetime.utcnow().isoformat()

    def to_event(self) -> RecordingEvent:
        """Convert request to RecordingEvent."""
        timestamp = self.recording_timestamp or self.timestamp
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        return RecordingEvent(
            recording_timestamp=timestamp,
            recording_id=self.recording_id,
            system_audio_path=self.system_audio_path,
            microphone_audio_path=self.microphone_audio_path,
            event=self.event,
            metadata=self.metadata,
            data={  # Populate data field
                "recording_id": self.recording_id,
                "recording_timestamp": timestamp,
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
    event: Literal["recording.ended"] = Field(default="recording.ended")
    metadata: dict[str, Any]

    @field_validator("event")
    @classmethod
    def normalize_event(cls, value: str) -> str:
        """Normalize event name to recording.ended."""
        if value == "Recording Ended":
            return "recording.ended"
        return value

    @field_validator("recording_timestamp")
    @classmethod
    def set_recording_timestamp(cls, value: str | None, info: ValidationInfo) -> str:
        """Use timestamp field if recording_timestamp is not provided."""
        if value is None and "timestamp" in info.data:
            timestamp = info.data["timestamp"]
            return (
                str(timestamp)
                if timestamp is not None
                else datetime.utcnow().isoformat()
            )
        return str(value) if value is not None else datetime.utcnow().isoformat()

    def to_event(self) -> RecordingEvent:
        """Convert request to RecordingEvent."""
        timestamp = self.recording_timestamp or self.timestamp
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        return RecordingEvent(
            recording_timestamp=timestamp,
            recording_id=self.recording_id,
            system_audio_path=self.system_audio_path,
            microphone_audio_path=self.microphone_audio_path,
            event=self.event,
            metadata=self.metadata,
            data={
                "recording_id": self.recording_id,
                "recording_timestamp": timestamp,
                "system_audio_path": self.system_audio_path,
                "microphone_audio_path": self.microphone_audio_path,
                "metadata": self.metadata,
            },
        )
