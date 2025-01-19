"""Core event system interfaces and implementations."""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Protocol

from .bus import EventBus as ConcreteEventBus
from .handlers import handle_recording_ended, handle_recording_started
from .models import Event, EventPriority
from .persistence import EventProcessingStatus, EventStore
from .types import EventContext

# Type for any event data
EventData = Any  # Simplified to avoid circular import


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


async def register_core_handlers(event_bus: ConcreteEventBus) -> None:
    """Register core event handlers with the event bus."""
    await event_bus.subscribe("recording.started", handle_recording_started)
    await event_bus.subscribe("recording.ended", handle_recording_ended)
