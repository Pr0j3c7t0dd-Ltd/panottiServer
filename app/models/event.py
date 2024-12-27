from datetime import datetime
from typing import Optional, Literal, Union, List
from pydantic import BaseModel, Field, validator
import json
from .database import get_db
import logging
import re

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
            return datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
        except ValueError:
            pass

    # First, try parsing as ISO format with various patterns
    iso_formats = [
        '%Y-%m-%dT%H:%M:%S.%f',  # With microseconds
        '%Y-%m-%dT%H:%M:%S.%fZ',  # With microseconds and Z
        '%Y-%m-%dT%H:%M:%S',  # Without microseconds
        '%Y-%m-%dT%H:%M:%SZ',  # Without microseconds with Z
        '%Y-%m-%dT%H:%M:%S%z',  # With timezone offset
    ]
    
    # Remove trailing Z if present and try parsing
    clean_ts = timestamp_str.rstrip('Z')
    
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
    """Model for event metadata.
    
    Attributes:
        eventProviderId: Provider-specific event ID
        eventTitle: Title of the event
        eventProvider: Name of the event provider
        eventAttendees: List of attendee email addresses
        systemLabel: Label for the system audio source
        microphoneLabel: Label for the microphone audio source
        recordingStarted: ISO8601 formatted timestamp of when the recording started
        recordingEnded: ISO8601 formatted timestamp of when the recording ended
    """
    eventProviderId: Optional[str] = None
    eventTitle: Optional[str] = None
    eventProvider: Optional[str] = None
    eventAttendees: Optional[List[str]] = None
    systemLabel: Optional[str] = None
    microphoneLabel: Optional[str] = None
    recordingStarted: Optional[str] = None
    recordingEnded: Optional[str] = None

    def to_db_format(self) -> dict:
        """Convert the model to a format suitable for database storage"""
        return self.model_dump(exclude_none=True)

class RecordingEvent(BaseModel):
    """Base model for recording events.
    
    Attributes:
        recording_timestamp: ISO8601 formatted timestamp of the event
        recordingId: Unique identifier for the recording session
        systemAudioPath: Optional path to the system audio recording file
        microphoneAudioPath: Optional path to the microphone audio recording file
        event: Type of recording event
        metadata: Additional metadata about the recording
    """
    recording_timestamp: str
    recordingId: str
    systemAudioPath: Optional[str] = None
    microphoneAudioPath: Optional[str] = None
    event: Union[Literal["Recording Started"], Literal["Recording Ended"]]
    metadata: Optional[Union[dict, EventMetadata]] = None

    def save(self):
        """Save the event to the database"""
        logger.debug("Starting save operation for RecordingEvent")
        data = self.model_dump()
        logger.debug(f"Event data: {data}")
        metadata = {}
        
        # For RecordingEndRequest, extract metadata
        if self.event == "Recording Ended" and self.metadata:
            logger.debug(f"Processing metadata for Recording Ended event")
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
            logger.debug(f"Processing metadata for Recording Started event")
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
        system_audio_path = self.systemAudioPath if self.systemAudioPath != "" else None
        microphone_audio_path = self.microphoneAudioPath if self.microphoneAudioPath != "" else None

        with get_db() as db:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Log the values being inserted
                insert_values = (
                    self.event,
                    self.recording_timestamp,
                    self.recordingId,
                    metadata.get('eventTitle'),
                    metadata.get('eventProviderId'),
                    metadata.get('eventProvider'),
                    json.dumps(metadata.get('eventAttendees')) if metadata.get('eventAttendees') else None,
                    metadata.get('systemLabel'),
                    metadata.get('microphoneLabel'),
                    metadata.get('recordingStarted'),
                    metadata.get('recordingEnded'),
                    json.dumps(data),
                    system_audio_path,
                    microphone_audio_path
                )
                logger.debug(f"Database insert values: {insert_values}")
                
                cursor.execute('''
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
                ''', insert_values)
                conn.commit()
                
            logger.info(
                "Event saved to database",
                extra={
                    "event_type": self.event,
                    "recording_id": self.recordingId,
                    "timestamp": self.recording_timestamp,
                    "metadata": metadata
                }
            )

    @classmethod
    def get_by_recording_id(cls, recording_id: str):
        """Retrieve all events for a specific recording"""
        with get_db() as db:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT * FROM events WHERE json_extract(data, "$.recordingId") = ? ORDER BY recording_timestamp',
                    (recording_id,)
                )
                return [json.loads(row[3]) for row in cursor.fetchall()]

class RecordingStartRequest(BaseModel):
    """Request model for starting a recording session.
    
    Attributes:
        recording_timestamp: ISO8601 formatted timestamp of the event (e.g., "2024-12-27T14:24:46Z")
        recordingId: Unique identifier for the recording session
        event: Type of recording event (always "Recording Started")
        metadata: Optional metadata about the recording
    """
    recording_timestamp: str
    recordingId: str
    event: Literal["Recording Started"] = Field(default="Recording Started")
    metadata: Optional[dict] = None
    systemAudioPath: Optional[str] = None
    microphoneAudioPath: Optional[str] = None

    @validator('recording_timestamp')
    def validate_timestamp(cls, v):
        """Validate and format timestamp to ISO 8601 UTC format"""
        try:
            dt = parse_timestamp(v)
            # Format to standard UTC format without microseconds
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        except ValueError as e:
            logger.error(f"Timestamp validation error for value '{v}': {str(e)}")
            raise ValueError(str(e))

    def to_event(self):
        """Convert request to RecordingEvent"""
        logger.debug("Converting RecordingStartRequest to RecordingEvent")
        logger.debug(f"Original request data: {self.model_dump()}")
        
        data = self.model_dump()
        if self.metadata:
            try:
                logger.debug(f"Processing metadata: {self.metadata}")
                metadata_dict = self.metadata if isinstance(self.metadata, dict) else self.metadata.dict() if hasattr(self.metadata, 'dict') else dict(self.metadata)
                try:
                    metadata_model = EventMetadata(**metadata_dict)
                    data['metadata'] = metadata_model.model_dump(exclude_none=True)
                    logger.debug(f"Successfully created metadata model: {data['metadata']}")
                except Exception as e:
                    logger.error(f"Failed to create EventMetadata from dict: {e}")
                    data['metadata'] = metadata_dict
            except Exception as e:
                logger.error(f"Error processing metadata: {e}")
        
        logger.debug(f"Final event data: {data}")
        return RecordingEvent(**data)

class RecordingEndRequest(BaseModel):
    """Request model for ending a recording session.
    
    Attributes:
        recording_timestamp: ISO8601 formatted timestamp of the event (e.g., "2024-12-27T14:24:46Z")
        recordingId: Unique identifier for the recording session
        event: Type of recording event (always "Recording Ended")
        metadata: Metadata about the recording
        systemAudioPath: Path to system audio file
        microphoneAudioPath: Path to microphone audio file
    """
    recording_timestamp: str
    recordingId: str
    systemAudioPath: str
    microphoneAudioPath: str
    event: Literal["Recording Ended"] = Field(default="Recording Ended")
    metadata: dict

    @validator('recording_timestamp')
    def validate_timestamp(cls, v):
        """Validate and format timestamp to ISO 8601 UTC format"""
        try:
            dt = parse_timestamp(v)
            # Format to standard UTC format without microseconds
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        except ValueError as e:
            logger.error(f"Timestamp validation error for value '{v}': {str(e)}")
            raise ValueError(str(e))

    def to_event(self):
        """Convert request to RecordingEvent"""
        logger.debug("Converting RecordingEndRequest to RecordingEvent")
        logger.debug(f"Original request data: {self.model_dump()}")
        
        data = self.model_dump()
        if self.metadata:
            try:
                logger.debug(f"Processing metadata: {self.metadata}")
                metadata_dict = self.metadata if isinstance(self.metadata, dict) else self.metadata.dict() if hasattr(self.metadata, 'dict') else dict(self.metadata)
                try:
                    metadata_model = EventMetadata(**metadata_dict)
                    data['metadata'] = metadata_model.model_dump(exclude_none=True)
                    logger.debug(f"Successfully created metadata model: {data['metadata']}")
                except Exception as e:
                    logger.error(f"Failed to create EventMetadata from dict: {e}")
                    data['metadata'] = metadata_dict
            except Exception as e:
                logger.error(f"Error processing metadata: {e}")
        
        logger.debug(f"Final event data: {data}")
        return RecordingEvent(**data)
