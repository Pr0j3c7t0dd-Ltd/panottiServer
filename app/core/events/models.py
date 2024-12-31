"""Core event models and interfaces."""

from enum import Enum
from typing import Any

from pydantic import BaseModel


class EventPriority(str, Enum):
    """Priority levels for events."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventContext(BaseModel):
    """Context information for an event."""

    source: str
    timestamp: float
    metadata: dict[str, Any] | None = None


class Event(BaseModel):
    """Base event model."""

    id: str
    type: str
    priority: EventPriority
    context: EventContext
    data: dict[str, Any]
