import json
import logging
from datetime import datetime
from typing import Any, Literal, Dict, List, Optional, ClassVar

from pydantic import BaseModel, Field, validator

from .database import get_db

logger = logging.getLogger(__name__)

# Constants for timestamp formats
COMPACT_TIMESTAMP_LENGTH: ClassVar[int] = 14
ISO_FORMATS: ClassVar[List[str]] = [
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
        except ValueError as e:
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
    event_provider_id: Optional[str] = None
    event_title: Optional[str] = None
    event_provider: Optional[str] = None
    event_attendees: Optional[List[str]] = None
    system_label: Optional[str] = None
    microphone_label: Optional[str] = None
    recording_started: Optional[str] = None
    recording_ended: Optional[str] = None

    def to_db_format(self) -> Dict[str, Any]:
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
    system_audio_path: Optional[str] = None
    microphone_audio_path: Optional[str] = None
    event: Literal["Recording Started", "Recording Ended"]
    metadata: Optional[Dict[str, Any] | EventMetadata] = None

    def save(self) -> None:
        """Save the event to the database"""
        logger.debug("Starting save operation for RecordingEvent")
        metadata_json = None
        if self.metadata:
            if isinstance(self.metadata, EventMetadata):
                metadata_json = json.dumps(self.metadata.to_db_format())
            else:
                metadata_json = json.dumps(self.metadata)

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO events (
                    recording_id,
                    recording_timestamp,
                    event_type,
                    system_audio_path,
                    microphone_audio_path,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.recording_id,
                    self.recording_timestamp,
                    self.event,
                    self.system_audio_path,
                    self.microphone_audio_path,
                    metadata_json,
                ),
            )
            conn.commit()

    @classmethod
    def get_by_recording_id(cls, recording_id: str) -> List[Dict[str, Any]]:
        """Retrieve all events for a specific recording."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM events 
                WHERE json_extract(data, "$.recordingId") = ? 
                ORDER BY recording_timestamp
                """,
                (recording_id,),
            )
            return cursor.fetchall()

    @classmethod
    def from_timestamp(cls, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime.

        Supported formats:
        - ISO 8601 with microseconds and timezone
        - ISO 8601 without microseconds
        - ISO 8601 with timezone offset
        """
        return parse_timestamp(timestamp_str)


class RecordingStartRequest(BaseModel):
    """Request model for starting a recording session.

    Attributes:
        recording_timestamp: ISO8601 formatted timestamp of the event 
        recording_id: Unique identifier for the recording session
        event: Type of recording event (always "Recording Started")
        metadata: Optional metadata about the recording
    """
    timestamp: Optional[str] = None
    recording_timestamp: Optional[str] = None
    recording_id: str
    event: Literal["Recording Started"] = Field(default="Recording Started")
    metadata: Optional[Dict[str, Any]] = None
    system_audio_path: Optional[str] = None
    microphone_audio_path: Optional[str] = None

    @validator("recording_timestamp", pre=True, always=True)
    def set_recording_timestamp(self, value: Optional[str], values: Dict[str, Any]) -> str:
        """Use timestamp field if recording_timestamp is not provided"""
        if value is None:
            return values.get("timestamp", "")
        try:
            return parse_timestamp(value).isoformat()
        except ValueError as e:
            logger.error(f"Timestamp validation error for value '{value}': {e!s}")
            raise ValueError(str(e)) from e

    def to_event(self) -> RecordingEvent:
        """Convert request to RecordingEvent"""
        return RecordingEvent(
            recording_timestamp=self.recording_timestamp or self.timestamp,
            recording_id=self.recording_id,
            event=self.event,
            metadata=self.metadata,
            system_audio_path=self.system_audio_path,
            microphone_audio_path=self.microphone_audio_path,
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
    timestamp: Optional[str] = None
    recording_timestamp: Optional[str] = None
    recording_id: str
    system_audio_path: str
    microphone_audio_path: str
    event: Literal["Recording Ended"] = Field(default="Recording Ended")
    metadata: Dict[str, Any]

    @validator("recording_timestamp", pre=True, always=True)
    def set_recording_timestamp(self, value: Optional[str], values: Dict[str, Any]) -> str:
        """Use timestamp field if recording_timestamp is not provided"""
        if value is None:
            return values.get("timestamp", "")
        try:
            return parse_timestamp(value).isoformat()
        except ValueError as e:
            logger.error(f"Timestamp validation error for value '{value}': {e!s}")
            raise ValueError(str(e)) from e

    def to_event(self) -> RecordingEvent:
        """Convert request to RecordingEvent"""
        return RecordingEvent(
            recording_timestamp=self.recording_timestamp or self.timestamp,
            recording_id=self.recording_id,
            event=self.event,
            metadata=self.metadata,
            system_audio_path=self.system_audio_path,
            microphone_audio_path=self.microphone_audio_path,
        )
