from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel
import json
from .database import get_db
import logging

logger = logging.getLogger(__name__)

class RecordingEvent(BaseModel):
    """
    Base model for recording events.
    
    Attributes:
        event: Type of recording event
        timestamp: ISO8601 formatted timestamp of the event
        recordingId: Unique identifier for the recording session
    """
    event: str
    timestamp: str
    recordingId: str

    def save(self):
        """Save the event to the database"""
        data = self.dict()
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
            results = [cls(**json.loads(row['data'])) for row in cursor.fetchall()]
            
            logger.log(
                logger.getEffectiveLevel(),
                "Retrieved events from database",
                extra={
                    "recording_id": recording_id,
                    "event_count": len(results),
                    "events": [result.dict() for result in results]
                }
            )
            return results

class RecordingMetadata(BaseModel):
    """
    Model for recording metadata.
    
    Attributes:
        recordingDateTime: ISO8601 formatted timestamp of when the recording was made
    """
    recordingDateTime: str

class RecordingStartRequest(RecordingEvent):
    """
    Request model for starting a recording session.
    """
    event: Literal["Recording Started"]

class RecordingEndRequest(RecordingEvent):
    """
    Request model for ending a recording session.
    
    Attributes:
        event: Type of recording event (always "Recording Ended")
        timestamp: ISO8601 formatted timestamp of the event
        recordingId: Unique identifier for the recording session
        metadata: Additional metadata about the recording
    """
    event: Literal["Recording Ended"]
    metadata: RecordingMetadata
