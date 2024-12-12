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
        recordingDateTime: ISO8601 formatted timestamp of when the recording was made
        eventTitle: Optional title of the event
        eventProviderId: Optional provider-specific event ID
        eventProvider: Optional name of the event provider
        eventAttendees: Optional list of event attendees
    """
    recordingDateTime: str
    eventTitle: Optional[str] = None
    eventProviderId: Optional[str] = None
    eventProvider: Optional[str] = None
    eventAttendees: Optional[List[str]] = None

    def to_db_format(self) -> dict:
        """Convert the model to a format suitable for database storage"""
        data = self.model_dump(exclude_none=True)
        if self.eventAttendees is not None:
            data['eventAttendees'] = json.dumps(self.eventAttendees)
        return data

class RecordingEvent(BaseModel):
    """Base model for recording events.
    
    Attributes:
        event: Type of recording event
        timestamp: ISO8601 formatted timestamp of the event
        recordingId: Unique identifier for the recording session
    """
    event: Union[Literal["Recording Started"], Literal["Recording Ended"]]
    timestamp: str
    recordingId: str

    def save(self):
        """Save the event to the database"""
        data = self.model_dump()
        logger.log(
            logger.getEffectiveLevel(),
            "Saving event to database",
            extra={
                "event_type": self.event,
                "recording_id": self.recordingId,
                "data": data
            }
        )
        
        with get_db().get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO events (type, timestamp, data) VALUES (?, ?, ?)',
                (self.event, self.timestamp, json.dumps(data))
            )
            return cursor.lastrowid

    @classmethod
    def get_by_recording_id(cls, recording_id: str):
        """Retrieve all events for a specific recording"""
        with get_db().get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM events WHERE json_extract(data, "$.recordingId") = ? ORDER BY timestamp',
                (recording_id,)
            )
            return [json.loads(row[3]) for row in cursor.fetchall()]

class RecordingStartRequest(BaseModel):
    """Request model for starting a recording session."""
    event: Literal["Recording Started"] = Field(default="Recording Started")
    timestamp: str
    recordingId: str

    def to_event(self):
        """Convert request to RecordingEvent"""
        return RecordingEvent(**self.model_dump())

class RecordingEndRequest(BaseModel):
    """Request model for ending a recording session.
    
    Attributes:
        event: Type of recording event (always "Recording Ended")
        timestamp: ISO8601 formatted timestamp of the event
        recordingId: Unique identifier for the recording session
        metadata: Additional metadata about the recording
    """
    event: Literal["Recording Ended"] = Field(default="Recording Ended")
    timestamp: str
    recordingId: str
    metadata: EventMetadata

    def to_event(self):
        """Convert request to RecordingEvent"""
        data = self.model_dump()
        data['metadata'] = self.metadata.to_db_format()
        return RecordingEvent(**data)
