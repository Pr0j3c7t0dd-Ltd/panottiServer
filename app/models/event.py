from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel

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
