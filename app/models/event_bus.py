"""Event bus interface and implementation."""
from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol, TypeVar, Dict, Union

from .event import RecordingEvent, RecordingStartRequest, RecordingEndRequest

# Type for any event data
EventData = Union[Dict[str, Any], RecordingEvent, RecordingStartRequest, RecordingEndRequest]

class EventBus(Protocol):
    """Event bus interface."""

    @abstractmethod
    async def subscribe(self, event_type: str, callback: Callable[[EventData], Any]) -> None:
        """Subscribe to events of a given type."""
        pass

    @abstractmethod
    async def unsubscribe(self, event_type: str, callback: Callable[[EventData], Any]) -> None:
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
