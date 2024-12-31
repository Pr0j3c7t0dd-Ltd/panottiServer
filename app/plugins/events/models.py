import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventPriority(str, Enum):
    """Event priority levels"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EventContext(BaseModel):
    """Context information for an event"""

    correlation_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_plugin: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Event(BaseModel):
    """Base event model"""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plugin_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    data: dict[str, Any]
    context: EventContext
    priority: EventPriority = Field(default=EventPriority.NORMAL)
    payload: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(
        cls,
        name: str,
        data: dict[str, Any],
        correlation_id: str,
        source_plugin: str | None = None,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> "Event":
        """Create a new event with default context"""
        return cls(
            name=name,
            data=data,
            context=EventContext(
                correlation_id=correlation_id, source_plugin=source_plugin
            ),
            priority=priority,
        )


def get_event_name(event: Any) -> str:
    """Get event name from event object."""
    if hasattr(event, "event"):
        return event.event
    elif isinstance(event, dict):
        return event.get("event", "unknown")
    else:
        return "unknown"


def get_event_id(event: Any) -> str:
    """Get event ID from event object."""
    if hasattr(event, "recording_id"):
        return event.recording_id
    elif isinstance(event, dict):
        return event.get("recording_id", "unknown")
    else:
        return "unknown"


class EventError(BaseModel):
    """Model for event processing errors"""

    event_id: int
    error_message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    stack_trace: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
