import json
import logging
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, validator

from .database import get_db

logger = logging.getLogger(__name__)


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string in various formats to datetime object.

    Supported formats:
    - ISO 8601 with microseconds and optional timezone (e.g., "2024-12-27T14:24:46.123456Z")
    - ISO 8601 without microseconds (e.g., "2024-12-27T14:24:46Z")
    - ISO 8601 with timezone offset (e.g., "2024-12-27T14:24:46+00:00")
    - Numeric timestamp (e.g., "1703686789" or "1703686789.123456")
    - Compact format (e.g., "20241227143602" for YYYYMMDDHHMMSS)
    """
    # Try compact format first (YYYYMMDDHHMMSS)
    if len(timestamp_str) == 14 and timestamp_str.isdigit():
        try:
            return datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
        except ValueError:
            pass

    # First, try parsing as ISO format with various patterns
    iso_formats = [
        "%Y-%m-%dT%H:%M:%S.%f",  # With microseconds
        "%Y-%m-%dT%H:%M:%S.%fZ",  # With microseconds and Z
        "%Y-%m-%dT%H:%M:%S",  # Without microseconds
        "%Y-%m-%dT%H:%M:%SZ",  # Without microseconds with Z
        "%Y-%m-%dT%H:%M:%S%z",  # With timezone offset
    ]

    # Remove trailing Z if present and try parsing
    clean_ts = timestamp_str.rstrip("Z")

    for fmt in iso_formats:
        try:
            return datetime.strptime(clean_ts, fmt)
        except ValueError:
            continue

    # Try parsing as numeric timestamp
    try:
        ts = float(timestamp_str)
        return datetime.fromtimestamp(ts)
    except ValueError:
        pass

    raise ValueError(
        "Invalid timestamp format. Expected one of:\n"
        "1. ISO 8601 with microseconds: '2024-12-27T14:24:46.123456Z'\n"
        "2. ISO 8601 without microseconds: '2024-12-27T14:24:46Z'\n"
        "3. ISO 8601 with timezone: '2024-12-27T14:24:46+00:00'\n"
        "4. Numeric timestamp: '1703686789' or '1703686789.123456'\n"
        "5. Compact format: '20241227143602' (YYYYMMDDHHMMSS)"
    )


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
        return {
            "event_provider_id": self.event_provider_id,
            "event_title": self.event_title,
            "event_provider": self.event_provider,
            "event_attendees": (
                json.dumps(self.event_attendees) if self.event_attendees else None
            ),
            "system_label": self.system_label,
            "microphone_label": self.microphone_label,
            "recording_started": self.recording_started,
            "recording_ended": self.recording_ended,
        }


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
    event: Literal["Recording Started"] | Literal["Recording Ended"]
    metadata: dict[str, Any] | EventMetadata | None = None

    def save(self) -> None:
        """Save the event to the database"""
        logger.debug("Starting save operation for RecordingEvent")
        data = self.model_dump()
        logger.debug(f"Event data: {data}")
        metadata = {}

        # For RecordingEndRequest, extract metadata
        if self.event == "Recording Ended" and self.metadata:
            logger.debug("Processing metadata for Recording Ended event")
            logger.debug(f"Metadata type: {type(self.metadata)}")
            logger.debug(f"Metadata content: {self.metadata}")

            try:
                if isinstance(self.metadata, dict):
                    metadata = self.metadata
                    # Try to create EventMetadata instance
                    try:
                        logger.debug(f"Creating EventMetadata from dict: {metadata}")
                        metadata_model = EventMetadata(**metadata)
                        metadata = metadata_model.model_dump(exclude_none=True)
                        logger.debug(f"Successfully created metadata model: {metadata}")
                    except Exception as e:
                        logger.error(f"Failed to create EventMetadata from dict: {e}")
                elif isinstance(self.metadata, EventMetadata):
                    metadata = self.metadata.model_dump(exclude_none=True)

                logger.debug(f"Final processed metadata: {metadata}")

            except Exception as e:
                logger.error(f"Error processing metadata: {e}")
                metadata = {}

        # Also process metadata for Recording Started events
        elif self.event == "Recording Started" and self.metadata:
            logger.debug("Processing metadata for Recording Started event")
            logger.debug(f"Metadata type: {type(self.metadata)}")
            logger.debug(f"Metadata content: {self.metadata}")

            try:
                if isinstance(self.metadata, dict):
                    metadata = self.metadata
                    # Try to create EventMetadata instance
                    try:
                        logger.debug(f"Creating EventMetadata from dict: {metadata}")
                        metadata_model = EventMetadata(**metadata)
                        metadata = metadata_model.model_dump(exclude_none=True)
                        logger.debug(f"Successfully created metadata model: {metadata}")
                    except Exception as e:
                        logger.error(f"Failed to create EventMetadata from dict: {e}")
                elif isinstance(self.metadata, EventMetadata):
                    metadata = self.metadata.model_dump(exclude_none=True)

                logger.debug(f"Final processed metadata: {metadata}")

            except Exception as e:
                logger.error(f"Error processing metadata: {e}")
                metadata = {}

        # Handle empty string for systemAudioPath and microphoneAudioPath
        system_audio_path = (
            self.system_audio_path if self.system_audio_path != "" else None
        )
        microphone_audio_path = (
            self.microphone_audio_path if self.microphone_audio_path != "" else None
        )

        with get_db() as db:
            with db.get_connection() as conn:
                cursor = conn.cursor()

                # Log the values being inserted
                insert_values = (
                    self.event,
                    self.recording_timestamp,
                    self.recording_id,
                    metadata.get("event_title"),
                    metadata.get("event_provider_id"),
                    metadata.get("event_provider"),
                    (
                        json.dumps(metadata.get("event_attendees"))
                        if metadata.get("event_attendees")
                        else None
                    ),
                    metadata.get("system_label"),
                    metadata.get("microphone_label"),
                    metadata.get("recording_started"),
                    metadata.get("recording_ended"),
                    json.dumps(data),
                    system_audio_path,
                    microphone_audio_path,
                )
                logger.debug(f"Database insert values: {insert_values}")

                cursor.execute(
                    """
                    INSERT INTO events (
                        type,
                        recording_timestamp,
                        recording_id,
                        event_title,
                        event_provider_id,
                        event_provider,
                        event_attendees,
                        system_label,
                        microphone_label,
                        recording_started,
                        recording_ended,
                        metadata_json,
                        system_audio_path,
                        microphone_audio_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    insert_values,
                )
                conn.commit()

            logger.info(
                "Event saved to database",
                extra={
                    "event_type": self.event,
                    "recording_id": self.recording_id,
                    "timestamp": self.recording_timestamp,
                    "metadata": metadata,
                },
            )

    @classmethod
    def get_by_recording_id(cls, recording_id: str):
        """Retrieve all events for a specific recording"""
        with get_db() as db:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT * FROM events WHERE json_extract(data, "$.recordingId") = ? ORDER BY recording_timestamp',
                    (recording_id,),
                )
                return [json.loads(row[3]) for row in cursor.fetchall()]

    @classmethod
    def from_timestamp(cls, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime.

        Supported formats:
        - ISO 8601 with microseconds and timezone
        - ISO 8601 without microseconds
        - ISO 8601 with timezone offset
        """
        COMPACT_LENGTH = 14  # Constant for magic number

        # Try compact format first (YYYYMMDDHHMMSS)
        if len(timestamp_str) == COMPACT_LENGTH and timestamp_str.isdigit():
            try:
                return datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            except ValueError:
                pass

        # First, try parsing as ISO format with various patterns
        iso_formats = [
            "%Y-%m-%dT%H:%M:%S.%f",  # With microseconds
            "%Y-%m-%dT%H:%M:%S.%fZ",  # With microseconds and Z
            "%Y-%m-%dT%H:%M:%S",  # Without microseconds
            "%Y-%m-%dT%H:%M:%SZ",  # Without microseconds with Z
            "%Y-%m-%dT%H:%M:%S%z",  # With timezone offset
        ]

        # Remove trailing Z if present and try parsing
        clean_ts = timestamp_str.rstrip("Z")

        for fmt in iso_formats:
            try:
                return datetime.strptime(clean_ts, fmt)
            except ValueError:
                continue

        # Try parsing as numeric timestamp
        try:
            ts = float(timestamp_str)
            return datetime.fromtimestamp(ts)
        except ValueError:
            pass

        raise ValueError(
            "Invalid timestamp format. Expected one of:\n"
            "1. ISO 8601 with microseconds: '2024-12-27T14:24:46.123456Z'\n"
            "2. ISO 8601 without microseconds: '2024-12-27T14:24:46Z'\n"
            "3. ISO 8601 with timezone: '2024-12-27T14:24:46+00:00'\n"
            "4. Numeric timestamp: '1703686789' or '1703686789.123456'\n"
            "5. Compact format: '20241227143602' (YYYYMMDDHHMMSS)"
        )


class RecordingStartRequest(BaseModel):
    """Request model for starting a recording session.

    Attributes:
        recording_timestamp: ISO8601 formatted timestamp of the event (e.g., "2024-12-27T14:24:46Z")
        recording_id: Unique identifier for the recording session
        event: Type of recording event (always "Recording Started")
        metadata: Optional metadata about the recording
    """

    timestamp: str | None = None  # For backward compatibility
    recording_timestamp: str | None = None
    recording_id: str
    event: Literal["Recording Started"] = Field(default="Recording Started")
    metadata: dict[str, Any] | None = None
    system_audio_path: str | None = None
    microphone_audio_path: str | None = None

    @validator("recording_timestamp", pre=True, always=True)
    def set_recording_timestamp(cls, v, values):
        """Use timestamp field if recording_timestamp is not provided"""
        if v is None:
            v = values.get("timestamp")
        if v is None:
            raise ValueError("Either timestamp or recording_timestamp must be provided")
        try:
            dt = parse_timestamp(v)
            # Format to standard UTC format without microseconds
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError as e:
            logger.error(f"Timestamp validation error for value '{v}': {e!s}")
            raise ValueError(str(e))

    def to_event(self):
        """Convert request to RecordingEvent"""
        logger.debug("Converting RecordingStartRequest to RecordingEvent")
        logger.debug(f"Original request data: {self.model_dump()}")

        data = self.model_dump(exclude={"timestamp"})  # Exclude the legacy field
        if self.metadata:
            try:
                logger.debug(f"Processing metadata: {self.metadata}")
                metadata_dict = (
                    self.metadata
                    if isinstance(self.metadata, dict)
                    else (
                        self.metadata.dict()
                        if hasattr(self.metadata, "dict")
                        else dict(self.metadata)
                    )
                )
                try:
                    metadata_model = EventMetadata(**metadata_dict)
                    data["metadata"] = metadata_model.model_dump(exclude_none=True)
                    logger.debug(
                        f"Successfully created metadata model: {data['metadata']}"
                    )
                except Exception as e:
                    logger.error(f"Failed to create EventMetadata from dict: {e}")
                    data["metadata"] = metadata_dict
            except Exception as e:
                logger.error(f"Error processing metadata: {e}")

        logger.debug(f"Final event data: {data}")
        return RecordingEvent(**data)


class RecordingEndRequest(BaseModel):
    """Request model for ending a recording session.

    Attributes:
        recording_timestamp: ISO8601 formatted timestamp of the event (e.g., "2024-12-27T14:24:46Z")
        recording_id: Unique identifier for the recording session
        event: Type of recording event (always "Recording Ended")
        metadata: Metadata about the recording
        system_audio_path: Path to system audio file
        microphone_audio_path: Path to microphone audio file
    """

    timestamp: str | None = None  # For backward compatibility
    recording_timestamp: str | None = None
    recording_id: str
    system_audio_path: str
    microphone_audio_path: str
    event: Literal["Recording Ended"] = Field(default="Recording Ended")
    metadata: dict[str, Any]

    @validator("recording_timestamp", pre=True, always=True)
    def set_recording_timestamp(cls, v, values):
        """Use timestamp field if recording_timestamp is not provided"""
        if v is None:
            v = values.get("timestamp")
        if v is None:
            raise ValueError("Either timestamp or recording_timestamp must be provided")
        try:
            dt = parse_timestamp(v)
            # Format to standard UTC format without microseconds
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError as e:
            logger.error(f"Timestamp validation error for value '{v}': {e!s}")
            raise ValueError(str(e))

    def to_event(self):
        """Convert request to RecordingEvent"""
        logger.debug("Converting RecordingEndRequest to RecordingEvent")
        logger.debug(f"Original request data: {self.model_dump()}")

        data = self.model_dump(exclude={"timestamp"})  # Exclude the legacy field
        if self.metadata:
            try:
                logger.debug(f"Processing metadata: {self.metadata}")
                metadata_dict = (
                    self.metadata
                    if isinstance(self.metadata, dict)
                    else (
                        self.metadata.dict()
                        if hasattr(self.metadata, "dict")
                        else dict(self.metadata)
                    )
                )
                try:
                    metadata_model = EventMetadata(**metadata_dict)
                    data["metadata"] = metadata_model.model_dump(exclude_none=True)
                    logger.debug(
                        f"Successfully created metadata model: {data['metadata']}"
                    )
                except Exception as e:
                    logger.error(f"Failed to create EventMetadata from dict: {e}")
                    data["metadata"] = metadata_dict
            except Exception as e:
                logger.error(f"Error processing metadata: {e}")

        logger.debug(f"Final event data: {data}")
        return RecordingEvent(**data)
