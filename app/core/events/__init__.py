"""Core event system interfaces."""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Protocol

from app.models.recording.events import (
    RecordingEndRequest,
    RecordingEvent,
    RecordingStartRequest,
)

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
