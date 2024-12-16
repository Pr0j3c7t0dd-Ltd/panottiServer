from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class EventPriority(str, Enum):
    """Event priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class EventContext(BaseModel):
    """Context information for events"""
    correlation_id: str = Field(..., description="Unique ID for tracing related events")
    source_plugin: str = Field(..., description="Name of the plugin that generated the event")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Event(BaseModel):
    """Base event model"""
    name: str = Field(..., description="Event name")
    payload: Dict[str, Any] = Field(..., description="Event data")
    context: EventContext = Field(..., description="Event context")
    priority: EventPriority = Field(default=EventPriority.NORMAL)

class EventError(BaseModel):
    """Model for event processing errors"""
    event_id: int
    error_message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    retry_count: int = Field(default=0)
    plugin_name: Optional[str] = None
