import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field


class EventPriority(str, Enum):
    """Event priority levels"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EventContext(PydanticBaseModel):
    """Context information for events"""

    correlation_id: str = Field(..., description="Unique ID for tracing related events")
    source_plugin: str = Field(
        ..., description="Name of the plugin that generated the event"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Event(PydanticBaseModel):
    """Base event model"""

    name: str = Field(..., description="Event name")
    payload: dict[str, Any] = Field(..., description="Event data")
    context: EventContext = Field(..., description="Event context")
    priority: EventPriority = Field(default=EventPriority.NORMAL)
    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique event ID"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_plugin_id(self) -> str:
        """Internal method to get the source plugin ID"""
        if not isinstance(self.context, EventContext):
            raise ValueError("Context must be an instance of EventContext")
        return self.context.source_plugin

    @property
    def plugin_id(self) -> str:
        """Get the source plugin ID from the context"""
        return self._get_plugin_id()


class EventError(PydanticBaseModel):
    """Model for event processing errors"""

    event_id: int
    error_message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    stack_trace: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
