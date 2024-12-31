"""Event bus implementation."""

import asyncio
import uuid
from asyncio import Task
from collections.abc import Awaitable, Callable
from typing import Any, overload

from app.core.events import EventBus as EventBusProtocol, EventData
from app.models.recording.events import RecordingEndRequest, RecordingEvent, RecordingStartRequest
from app.plugins.events.models import Event, EventPriority
from app.plugins.events.persistence import EventStore
from app.utils.logging_config import get_logger


@overload
def get_event_name(event: Event) -> str: ...


@overload
def get_event_name(
    event: RecordingEvent | RecordingStartRequest | RecordingEndRequest
) -> str: ...


@overload
def get_event_name(event: dict[str, Any]) -> str: ...


def get_event_name(event: Event | EventData) -> str:
    """Get event name from different event types"""
    if isinstance(event, Event):
        return event.name
    elif isinstance(
        event, RecordingEvent | RecordingStartRequest | RecordingEndRequest
    ):
        return event.event
    elif isinstance(event, dict):
        return str(event.get("name", "unknown"))
    return "unknown"


@overload
def get_event_id(event: Event) -> str: ...


@overload
def get_event_id(
    event: RecordingEvent | RecordingStartRequest | RecordingEndRequest
) -> str: ...


@overload
def get_event_id(event: dict[str, Any]) -> str: ...


def get_event_id(event: Event | EventData) -> str:
    """Get event ID from different event types"""
    if isinstance(event, Event):
        return event.event_id
    elif isinstance(
        event, RecordingEvent | RecordingStartRequest | RecordingEndRequest
    ):
        return event.recording_id
    elif isinstance(event, dict):
        return str(event.get("event_id", str(uuid.uuid4())))
    return str(uuid.uuid4())


class EventBus(EventBusProtocol):
    """Concrete implementation of the event bus."""

    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self._subscribers: dict[str, list[Callable[[EventData], Any]]] = {}
        self.plugins: dict[str, Any] = {}
        self.tasks: list[Task[Any]] = []
        self.logger = get_logger("event_bus")
        self.logger.info("Event bus initialized")

    async def publish(self, event: Any) -> None:
        """Publish an event to all subscribers."""
        event_name = get_event_name(event)
        event_type = type(event).__name__

        try:
            # Store the event
            await self.event_store.store_event(event)

            # Get subscribers for this event
            subscribers = self._subscribers.get(event_name, [])
            
            # Notify subscribers
            self.logger.debug(
                "Publishing event",
                extra={
                    "event_name": event_name,
                    "event_type": event_type,
                    "subscriber_count": len(subscribers),
                },
            )
            for subscriber in subscribers:
                try:
                    await subscriber(event)
                except Exception as e:
                    self.logger.error(
                        "Error in subscriber",
                        extra={
                            "error": str(e),
                            "subscriber": subscriber.__name__,
                            "event_name": event_name,
                            "event_type": event_type,
                        },
                        exc_info=True,
                    )
        except Exception as e:
            self.logger.error(
                "Error publishing event",
                extra={
                    "error": str(e),
                    "event_name": event_name,
                    "event_type": event_type,
                },
                exc_info=True,
            )
            raise

    async def emit(self, event: EventData) -> None:
        """Emit an event to all registered handlers"""
        await self.publish(event)

    async def subscribe(
        self,
        event_type: str,
        callback: Callable[[EventData], Any],
    ) -> None:
        """Subscribe a handler to an event"""
        self.logger.info(
            "Subscribing to event",
            extra={
                "event_name": event_type,
                "subscriber": callback.__name__,
            },
        )
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    async def unsubscribe(
        self,
        event_type: str,
        callback: Callable[[EventData], Any],
    ) -> None:
        """Unsubscribe a handler from an event"""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(callback)
            self.logger.debug(
                "Handler unsubscribed",
                extra={"event_name": event_type, "handler": callback.__qualname__},
            )

    async def wait_for_pending_events(self) -> None:
        """Wait for all pending event processing to complete"""
        if self.tasks:
            self.logger.debug(f"Waiting for {len(self.tasks)} pending events")
            await asyncio.gather(*self.tasks)
            self.logger.debug("All pending events processed")
