"""Core event types."""

import uuid
from datetime import datetime, UTC
from typing import Any, Protocol

from pydantic import BaseModel, Field


class EventContext(BaseModel):
    """Event context model."""

    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventHandler(Protocol):
    """Event handler protocol."""

    async def __call__(self, event_data: Any) -> None:
        """Handle an event."""
        ...
