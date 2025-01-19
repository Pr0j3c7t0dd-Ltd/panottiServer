"""Core event models."""

from datetime import datetime, UTC
from enum import Enum
from typing import Any, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .types import EventContext


class EventPriority(str, Enum):
    """Event priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class Event(BaseModel):
    """Base event model."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    plugin_id: str = Field(default="system")
    name: str = Field(...)  # Required field
    data: dict[str, Any] = Field(default_factory=dict)
    context: EventContext = Field(default_factory=EventContext)
    priority: EventPriority = Field(default=EventPriority.NORMAL)
    payload: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("event_id", mode="before")
    @classmethod
    def validate_event_id(cls, value: Any) -> str:
        """Ensure event has an ID."""
        if value is None:
            return str(uuid4())
        return str(value)

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, value: Any) -> str:
        """Ensure event has a name."""
        if value is None:
            return cls.__name__.lower()
        return str(value)

    @field_validator("plugin_id", mode="before")
    @classmethod
    def validate_plugin_id(cls, value: Any) -> str:
        """Ensure event has a plugin ID."""
        if value is None:
            return "system"
        return str(value)

    @field_validator("context", mode="before")
    @classmethod
    def validate_context(cls, value: Any) -> EventContext:
        """Ensure event has a context."""
        if value is None:
            return EventContext()
        if isinstance(value, dict):
            return EventContext(**value)
        return cast(EventContext, value)

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
    """Get event name from event object.

    Args:
        event: Any object that might contain event information

    Returns:
        str: The event name or 'unknown' if not found
    """
    if hasattr(event, "event"):
        return str(event.event)
    elif isinstance(event, dict):
        return str(event.get("event", "unknown"))
    return "unknown"


def get_event_id(event: Any) -> str:
    """Get event ID from event object.

    Args:
        event: Any object that might contain event ID information

    Returns:
        str: The event ID or 'unknown' if not found
    """
    if hasattr(event, "recording_id"):
        return str(event.recording_id)
    elif isinstance(event, dict):
        return str(event.get("recording_id", "unknown"))
    return "unknown"


class EventError(BaseModel):
    """Model for event processing errors"""

    event_id: int
    error_message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stack_trace: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
