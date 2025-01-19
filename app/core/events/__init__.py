"""Core event system interfaces and implementations."""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Protocol

from app.models.recording.events import (
    RecordingEndRequest,
    RecordingEvent,
    RecordingStartRequest,
)

from .bus import EventBus as ConcreteEventBus
from .models import Event, EventContext, EventPriority
from .persistence import EventProcessingStatus, EventStore

# Type for any event data
EventData = (
    dict[str, Any] | RecordingEvent | RecordingStartRequest | RecordingEndRequest
)


class EventBus(Protocol):
    """Event bus interface."""

    @abstractmethod
    async def subscribe(
        self, event_type: str, callback: Callable[[EventData], Any]
    ) -> None:
        """Subscribe to events of a given type."""
        pass

    @abstractmethod
    async def unsubscribe(
        self, event_type: str, callback: Callable[[EventData], Any]
    ) -> None:
        """Unsubscribe from events of a given type."""
        pass

    @abstractmethod
    async def publish(self, event: EventData) -> None:
        """Publish an event to all subscribers."""
        pass

    @abstractmethod
    async def emit(self, event: EventData) -> None:
        """Emit an event (alias for publish)."""
        await self.publish(event)


__all__ = [
    "EventBus",  # Protocol
    "ConcreteEventBus",  # Implementation
    "Event",
    "EventContext",
    "EventPriority",
    "EventStore",
    "EventProcessingStatus",
]
