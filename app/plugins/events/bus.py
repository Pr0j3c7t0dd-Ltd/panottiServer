"""Event bus implementation."""

import asyncio
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

from app.models.recording.events import (
    RecordingEndRequest,
    RecordingEvent,
    RecordingStartRequest,
)
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Type alias for event handler functions
EventHandler = Callable[
    [dict[str, Any] | RecordingEvent | RecordingStartRequest | RecordingEndRequest],
    Coroutine[Any, Any, None],
]


class EventBus:
    """Event bus for handling event subscriptions and publishing."""

    def __init__(self) -> None:
        """Initialize event bus."""
        self._subscribers: defaultdict[str, list[EventHandler]] = defaultdict(list)
        self._lock = asyncio.Lock()
        logger.info("Event bus initialized")

    async def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """Subscribe to an event.

        Args:
            event_name: Name of the event to subscribe to
            handler: Async function to handle the event
        """
        async with self._lock:
            if handler not in self._subscribers[event_name]:
                self._subscribers[event_name].append(handler)
                logger.info(
                    "Subscribing to event",
                    extra={
                        "event_name": event_name,
                        "subscriber": handler.__name__,
                    },
                )

    async def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        """Unsubscribe from an event.

        Args:
            event_name: Name of the event to unsubscribe from
            handler: Handler function to remove
        """
        async with self._lock:
            if handler in self._subscribers[event_name]:
                self._subscribers[event_name].remove(handler)
                logger.info(
                    "Unsubscribed from event",
                    extra={
                        "event_name": event_name,
                        "subscriber": handler.__name__,
                    },
                )

    async def publish(
        self,
        event: (
            dict[str, Any]
            | RecordingEvent
            | RecordingStartRequest
            | RecordingEndRequest
        ),
    ) -> None:
        """Publish an event to all subscribers.

        Args:
            event: Event data to publish
        """
        event_name = (
            event.name
            if hasattr(event, "name")
            else (event.event if hasattr(event, "event") else event.get("name"))
        )
        if not event_name:
            logger.error("Event has no name", extra={"event": str(event)})
            return

        handlers = self._subscribers.get(event_name, [])
        if not handlers:
            logger.debug(
                "No subscribers for event",
                extra={"event_name": event_name},
            )
            return

        tasks = []
        for handler in handlers:
            try:
                task = asyncio.create_task(handler(event))
                tasks.append(task)
            except Exception as e:
                logger.error(
                    "Error creating task for event handler",
                    extra={
                        "event_name": event_name,
                        "handler": handler.__name__,
                        "error": str(e),
                    },
                )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def emit(
        self,
        event: (
            dict[str, Any]
            | RecordingEvent
            | RecordingStartRequest
            | RecordingEndRequest
        ),
    ) -> None:
        """Emit an event (alias for publish).

        Args:
            event: Event data to emit
        """
        await self.publish(event)
