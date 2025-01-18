"""Request models for recording endpoints."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class RecordingEndRequest(BaseModel):
    """Request model for ending a recording."""

    recording_id: str = Field(
        ..., description="Unique identifier of the recording to end"
    )
    end_time: datetime = Field(
        default_factory=datetime.utcnow, description="Time when the recording ended"
    )
    metadata: Optional[dict] = Field(
        default=None, description="Optional metadata about the recording"
    )
    status: str = Field(
        default="completed",
        description="Status of the recording (completed, failed, etc.)",
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if the recording failed"
    )
