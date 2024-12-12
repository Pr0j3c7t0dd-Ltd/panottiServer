from datetime import datetime
from typing import Optional, Literal, Union
from pydantic import BaseModel, Field
import json
from .database import get_db
import logging

logger = logging.getLogger(__name__)

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
            results = [cls(**json.loads(row['data'])) for row in cursor.fetchall()]
            
            logger.log(
                logger.getEffectiveLevel(),
                "Retrieved events from database",
                extra={
                    "recording_id": recording_id,
                    "event_count": len(results),
                    "events": [result.model_dump() for result in results]
                }
            )
            return results

class RecordingMetadata(BaseModel):
    """
    Model for recording metadata.
    
    Attributes:
        recordingDateTime: ISO8601 formatted timestamp of when the recording was made
        systemAudioPath: Path to the system audio recording file
        microphoneAudioPath: Path to the microphone audio recording file
    """
    recordingDateTime: str
    systemAudioPath: str
    microphoneAudioPath: str

class RecordingStartRequest(BaseModel):
    """
    Request model for starting a recording session.
    """
    event: Literal["Recording Started"] = Field(default="Recording Started")
    timestamp: str
    recordingId: str

    def to_event(self) -> RecordingEvent:
        return RecordingEvent(
            event=self.event,
            timestamp=self.timestamp,
            recordingId=self.recordingId
        )

class RecordingEndRequest(BaseModel):
    """
    Request model for ending a recording session.
    
    Attributes:
        event: Type of recording event (always "Recording Ended")
        timestamp: ISO8601 formatted timestamp of the event
        recordingId: Unique identifier for the recording session
        metadata: Additional metadata about the recording
    """
    event: Literal["Recording Ended"] = Field(default="Recording Ended")
    timestamp: str
    recordingId: str
    metadata: RecordingMetadata

    def to_event(self) -> RecordingEvent:
        return RecordingEvent(
            event=self.event,
            timestamp=self.timestamp,
            recordingId=self.recordingId
        )
