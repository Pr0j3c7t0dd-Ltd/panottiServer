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
        self._processed_events: set[str] = set()
        self._pending_tasks: set[asyncio.Task] = set()
        logger.info("Event bus initialized")

    def _cleanup_task(self, task: asyncio.Task) -> None:
        """Remove task from pending tasks set.
        
        Args:
            task: Task to cleanup
        """
        self._pending_tasks.discard(task)

    async def _handle_task(self, handler: EventHandler, event: Any) -> None:
        """Handle a single event handler task.
        
        Args:
            handler: Event handler to run
            event: Event data to pass to handler
        """
        try:
            await handler(event)
        except Exception as e:
            logger.error(
                f"Error in event handler {handler.__name__}",
                extra={
                    "error": str(e),
                    "handler": handler.__name__,
                },
                exc_info=True
            )

    async def _is_event_processed(self, event_id: str) -> bool:
        """Check if an event has already been processed.
        
        Args:
            event_id: ID of the event to check
            
        Returns:
            bool: True if event was already processed
        """
        async with self._lock:
            return event_id in self._processed_events

    async def _mark_event_processed(self, event_id: str) -> None:
        """Mark an event as processed.
        
        Args:
            event_id: ID of the event to mark
        """
        async with self._lock:
            self._processed_events.add(event_id)

    def _get_event_name(self, event: dict[str, Any] | RecordingEvent | RecordingStartRequest | RecordingEndRequest) -> str | None:
        """Get event name from event field.
        
        Args:
            event: Event data to get name from
            
        Returns:
            str | None: Event name or None if not found
        """
        if isinstance(event, dict):
            return event.get("event")
        elif hasattr(event, "event"):
            return event.event
        return None

    def _get_event_id(self, event: dict[str, Any] | RecordingEvent | RecordingStartRequest | RecordingEndRequest) -> str | None:
        """Get event ID from event field.
        
        Args:
            event: Event data to get ID from
            
        Returns:
            str | None: Event ID or None if not found
        """
        if isinstance(event, dict):
            return event.get("id")
        elif hasattr(event, "id"):
            return event.id
        return None

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
                    "Subscribed to event",
                    extra={
                        "event_name": event_name,
                        "handler": handler.__name__,
                        "current_subscriptions": {
                            k: [h.__name__ for h in v]
                            for k, v in self._subscribers.items()
                        },
                    },
                )
            else:
                logger.warning(
                    "Handler already subscribed to event",
                    extra={
                        "event_name": event_name,
                        "handler": handler.__name__,
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
        event: dict[str, Any] | RecordingEvent | RecordingStartRequest | RecordingEndRequest,
    ) -> None:
        """Publish an event to all subscribers.

        Args:
            event: Event data to publish
        """
        try:
            event_name = self._get_event_name(event)
            event_id = self._get_event_id(event)

            # Skip if event was already processed
            if event_id and await self._is_event_processed(event_id):
                logger.debug(
                    "Skipping duplicate event",
                    extra={
                        "event_name": event_name,
                        "event_id": event_id,
                    },
                )
                return

            if not event_name:
                logger.error("Invalid event: no event name found", extra={"event": event})
                return

            handlers = self._subscribers.get(event_name, [])
            if not handlers:
                logger.debug(
                    "No handlers for event",
                    extra={"event_name": event_name, "event": event},
                )
                return

            logger.debug(
                "Publishing event",
                extra={
                    "event_name": event_name,
                    "num_handlers": len(handlers),
                    "handler_names": [h.__name__ for h in handlers],
                    "current_subscriptions": {
                        k: [h.__name__ for h in v]
                        for k, v in self._subscribers.items()
                    },
                },
            )

            # Create and track tasks
            tasks = []
            for handler in handlers:
                task = asyncio.create_task(self._handle_task(handler, event))
                task.add_done_callback(self._cleanup_task)
                self._pending_tasks.add(task)
                tasks.append(task)
                logger.debug(
                    "Created task for handler",
                    extra={
                        "event_name": event_name,
                        "handler": handler.__name__,
                    },
                )

            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Mark event as processed after successful handling
            if event_id:
                await self._mark_event_processed(event_id)

            logger.debug(
                "Finished processing event",
                extra={
                    "event_name": event_name,
                    "num_tasks_completed": len(tasks),
                },
            )

        except Exception as e:
            logger.error(
                "Error publishing event",
                extra={"error": str(e), "event": event},
                exc_info=True,
            )

    async def shutdown(self) -> None:
        """Gracefully shutdown the event bus by canceling pending tasks."""
        # Cancel all pending tasks
        for task in self._pending_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete or be cancelled
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
        
        self._pending_tasks.clear()
        logger.info("Event bus shutdown complete")
