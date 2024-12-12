from datetime import datetime
from typing import Optional, Literal
import json
from .database import get_db
import logging

logger = logging.getLogger(__name__)

class RecordingEvent:
    """
    Base model for recording events.
    
    Attributes:
        event: Type of recording event
        timestamp: ISO8601 formatted timestamp of the event
        recordingId: Unique identifier for the recording session
    """
    def __init__(self, event: Literal["Recording Started", "Recording Ended"], timestamp: str, recordingId: str):
        self.event = event
        self.timestamp = timestamp
        self.recordingId = recordingId

    def save(self):
        """Save the event to the database"""
        data = self.__dict__
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
                    "events": [result.__dict__ for result in results]
                }
            )
            return results

class RecordingMetadata:
    """
    Model for recording metadata.
    
    Attributes:
        recordingDateTime: ISO8601 formatted timestamp of when the recording was made
    """
    def __init__(self, recordingDateTime: str):
        self.recordingDateTime = recordingDateTime

class RecordingStartRequest(RecordingEvent):
    """
    Request model for starting a recording session.
    """
    def __init__(self, timestamp: str, recordingId: str):
        super().__init__("Recording Started", timestamp, recordingId)

class RecordingEndRequest(RecordingEvent):
    """
    Request model for ending a recording session.
    
    Attributes:
        event: Type of recording event (always "Recording Ended")
        timestamp: ISO8601 formatted timestamp of the event
        recordingId: Unique identifier for the recording session
        metadata: Additional metadata about the recording
    """
    def __init__(self, timestamp: str, recordingId: str, metadata: RecordingMetadata):
        super().__init__("Recording Ended", timestamp, recordingId)
        self.metadata = metadata
