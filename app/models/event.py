from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class RecordingEvent(BaseModel):
    """
    Model for recording events.
    
    Attributes:
        session_id: Unique identifier for the recording session
        timestamp: ISO8601 formatted timestamp of the event
    """
    session_id: str
    timestamp: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Add specific request classes for start and end recording events
class RecordingStartRequest(RecordingEvent):
    """
    Request model for starting a recording session.
    Inherits from RecordingEvent.
    """
    pass

class RecordingEndRequest(RecordingEvent):
    """
    Request model for ending a recording session.
    Inherits from RecordingEvent.
    """
    pass
