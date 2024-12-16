from datetime import datetime
from typing import Optional, Literal, Union, List
from pydantic import BaseModel, Field
import json
from .database import get_db
import logging

logger = logging.getLogger(__name__)

class EventMetadata(BaseModel):
    """Model for event metadata.
    
    Attributes:
        eventProviderId: Provider-specific event ID
        eventTitle: Title of the event
        eventProvider: Name of the event provider
        recordingDateTime: ISO8601 formatted timestamp of when the recording was made
    """
    eventProviderId: Optional[str] = None
    eventTitle: Optional[str] = None
    eventProvider: Optional[str] = None
    recordingDateTime: str
    eventAttendees: Optional[List[str]] = None

    def to_db_format(self) -> dict:
        """Convert the model to a format suitable for database storage"""
        return self.model_dump(exclude_none=True)

class RecordingEvent(BaseModel):
    """Base model for recording events.
    
    Attributes:
        timestamp: ISO8601 formatted timestamp of the event
        recordingId: Unique identifier for the recording session
        systemAudioPath: Optional path to the system audio recording file
        microphoneAudioPath: Optional path to the microphone audio recording file
        event: Type of recording event
        metadata: Additional metadata about the recording
    """
    timestamp: str
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
        
        with get_db() as db:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Log the values being inserted
                insert_values = (
                    self.event,
                    self.timestamp,
                    self.recordingId,
                    self.systemAudioPath,
                    self.microphoneAudioPath,
                    metadata.get('recordingDateTime'),
                    metadata.get('eventTitle'),
                    metadata.get('eventProviderId'),
                    metadata.get('eventProvider'),
                    json.dumps(metadata.get('eventAttendees')) if metadata.get('eventAttendees') else None,
                    json.dumps(data)
                )
                logger.debug(f"Database insert values: {insert_values}")
                
                cursor.execute('''
                    INSERT INTO events (
                        type, 
                        timestamp, 
                        recording_id,
                        system_audio_path,
                        microphone_audio_path,
                        recording_datetime,
                        event_title,
                        event_provider_id,
                        event_provider,
                        event_attendees,
                        metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', insert_values)
                conn.commit()
                
            logger.info(
                "Event saved to database",
                extra={
                    "event_type": self.event,
                    "recording_id": self.recordingId,
                    "timestamp": self.timestamp,
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
                    'SELECT * FROM events WHERE json_extract(data, "$.recordingId") = ? ORDER BY timestamp',
                    (recording_id,)
                )
                return [json.loads(row[3]) for row in cursor.fetchall()]

class RecordingStartRequest(BaseModel):
    """Request model for starting a recording session.
    
    Attributes:
        timestamp: ISO8601 formatted timestamp of the event
        recordingId: Unique identifier for the recording session
        event: Type of recording event (always "Recording Started")
        metadata: Optional metadata about the recording
    """
    timestamp: str
    recordingId: str
    event: Literal["Recording Started"] = Field(default="Recording Started")
    metadata: Optional[dict] = None
    systemAudioPath: Optional[str] = None
    microphoneAudioPath: Optional[str] = None

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
        timestamp: ISO8601 formatted timestamp of the event
        recordingId: Unique identifier for the recording session
        event: Type of recording event (always "Recording Ended")
        metadata: Metadata about the recording
        systemAudioPath: Path to system audio file
        microphoneAudioPath: Path to microphone audio file
    """
    timestamp: str
    recordingId: str
    systemAudioPath: str
    microphoneAudioPath: str
    event: Literal["Recording Ended"] = Field(default="Recording Ended")
    metadata: dict

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
