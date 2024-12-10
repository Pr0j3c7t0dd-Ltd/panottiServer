from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel
import json
from .database import get_db

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
        with get_db().get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO events (type, timestamp, data) VALUES (?, ?, ?)',
                (self.event, self.timestamp, json.dumps(self.dict()))
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
            return [cls(**json.loads(row['data'])) for row in cursor.fetchall()]

class RecordingStartRequest(RecordingEvent):
    """
    Request model for starting a recording session.
    """
    event: Literal["Recording Started"]

class RecordingEndRequest(RecordingEvent):
    """
    Request model for ending a recording session.
    """
    event: Literal["Recording Ended"]
    systemAudioPath: str
    MicrophoneAudioPath: str
