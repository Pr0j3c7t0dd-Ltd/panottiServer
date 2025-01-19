"""Event bus implementation."""

import asyncio
import traceback
import uuid
from collections import defaultdict
from datetime import datetime, UTC
from typing import Any

from app.utils.logging_config import get_logger

from .types import EventHandler

logger = get_logger(__name__)

# Use Any for event types to break circular dependencies
EventData = Any  # Generic type for event data


class EventBus:
    """Event bus for handling event subscriptions and publishing."""

    def __init__(self) -> None:
        """Initialize event bus."""
        self._subscribers: defaultdict[str, list[EventHandler]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._processed_events: dict[str, datetime] = {}  # event_id -> timestamp
        self._pending_tasks: set[asyncio.Task] = set()
        self._cleanup_events_task = (
            None  # Will be initialized when event loop is available
        )
        self._req_id = str(uuid.uuid4())
        logger.info(
            "Event bus initialized",
            extra={"req_id": self._req_id, "component": "event_bus"},
        )

    async def start(self) -> None:
        """Start the event bus background tasks."""
        if self._cleanup_events_task is None:
            self._cleanup_events_task = asyncio.create_task(self._cleanup_old_events())
            logger.debug(
                "Started event cleanup task",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "task_id": id(self._cleanup_events_task),
                },
            )

    async def stop(self) -> None:
        """Stop the event bus background tasks."""
        if self._cleanup_events_task is not None:
            self._cleanup_events_task.cancel()
            try:
                await self._cleanup_events_task
            except asyncio.CancelledError:
                pass
            self._cleanup_events_task = None
            logger.debug(
                "Stopped event cleanup task",
                extra={"req_id": self._req_id, "component": "event_bus"},
            )

    async def _cleanup_old_events(self) -> None:
        """Periodically clean up old processed events."""
        while True:
            try:
                await asyncio.sleep(3600)  # Clean up every hour
                now = datetime.now(UTC)
                async with self._lock:
                    # Remove events older than 1 hour
                    old_events = [
                        event_id
                        for event_id, timestamp in self._processed_events.items()
                        if (now - timestamp).total_seconds() > 3600
                    ]
                    for event_id in old_events:
                        del self._processed_events[event_id]

                    if old_events:
                        logger.debug(
                            "Cleaned up old events",
                            extra={
                                "req_id": self._req_id,
                                "component": "event_bus",
                                "removed_count": len(old_events),
                                "remaining_count": len(self._processed_events),
                            },
                        )
            except Exception as e:
                logger.error(
                    "Error cleaning up old events",
                    extra={
                        "req_id": self._req_id,
                        "component": "event_bus",
                        "error": str(e),
                    },
                )

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
            logger.debug(
                "Starting event handler execution",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "handler": handler.__name__,
                    "handler_module": handler.__module__,
                    "handler_qualname": handler.__qualname__,
                    "handler_id": id(handler),
                    "event_type": type(event).__name__,
                    "event_data": str(event),
                    "event_id": getattr(event, "event_id", None)
                    or getattr(event, "id", None),
                    "event_name": self._get_event_name(event),
                    "handler_class": handler.__self__.__class__.__name__
                    if hasattr(handler, "__self__")
                    else None,
                    "handler_instance_id": id(handler.__self__)
                    if hasattr(handler, "__self__")
                    else None,
                    "stack_trace": "".join(traceback.format_stack()),
                },
            )

            # Add pre-execution state check
            if hasattr(handler, "__self__"):
                logger.debug(
                    "Handler instance state",
                    extra={
                        "req_id": self._req_id,
                        "component": "event_bus",
                        "handler": handler.__name__,
                        "handler_class": handler.__self__.__class__.__name__,
                        "handler_instance_vars": vars(handler.__self__),
                    },
                )

            # Create a task for the handler and don't wait for it
            handler_task = asyncio.create_task(handler(event))
            self._pending_tasks.add(handler_task)

            def handler_done_callback(task: asyncio.Task) -> None:
                try:
                    self._pending_tasks.discard(task)
                    exc = task.exception()
                    if exc:
                        logger.error(
                            f"Error in event handler {handler.__name__}",
                            extra={
                                "req_id": self._req_id,
                                "component": "event_bus",
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                                "handler": handler.__name__,
                                "handler_module": handler.__module__,
                                "handler_qualname": handler.__qualname__,
                                "handler_id": id(handler),
                                "event_type": type(event).__name__,
                                "event_data": str(event),
                                "event_id": getattr(event, "event_id", None)
                                or getattr(event, "id", None),
                                "event_name": self._get_event_name(event),
                                "stack_trace": traceback.format_exc(),
                            },
                            exc_info=True,
                        )
                    else:
                        logger.debug(
                            "Event handler execution completed",
                            extra={
                                "req_id": self._req_id,
                                "component": "event_bus",
                                "handler": handler.__name__,
                                "handler_module": handler.__module__,
                                "handler_qualname": handler.__qualname__,
                                "handler_id": id(handler),
                                "event_type": type(event).__name__,
                                "event_data": str(event),
                                "event_id": getattr(event, "event_id", None)
                                or getattr(event, "id", None),
                                "event_name": self._get_event_name(event),
                                "execution_status": "success",
                            },
                        )
                except Exception as e:
                    logger.error(
                        "Error in handler callback",
                        extra={
                            "req_id": self._req_id,
                            "component": "event_bus",
                            "error": str(e),
                            "handler": handler.__name__,
                            "event_id": getattr(event, "event_id", None)
                            or getattr(event, "id", None),
                        },
                    )

            handler_task.add_done_callback(handler_done_callback)

        except Exception as e:
            logger.error(
                f"Error setting up event handler {handler.__name__}",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "handler": handler.__name__,
                    "handler_module": handler.__module__,
                    "handler_qualname": handler.__qualname__,
                    "handler_id": id(handler),
                    "event_type": type(event).__name__,
                    "event_data": str(event),
                    "event_id": getattr(event, "event_id", None)
                    or getattr(event, "id", None),
                    "event_name": self._get_event_name(event),
                    "stack_trace": traceback.format_exc(),
                },
                exc_info=True,
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
            self._processed_events[event_id] = datetime.now(UTC)

    def _get_event_name(self, event: Any) -> str | None:
        """Get event name from event field.

        Args:
            event: Event data to get name from

        Returns:
            str | None: Event name or None if not found
        """
        event_name = None
        try:
            if isinstance(event, dict):
                # Try both 'event' and 'name' fields for dict events
                event_name = event.get("event") or event.get("name")
            elif hasattr(event, "event"):
                event_name = event.event
            elif hasattr(event, "name"):
                event_name = event.name

            logger.debug(
                "Getting event name",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "event_type": type(event).__name__,
                    "event_name": event_name,
                    "event_data": str(event),
                    "is_dict": isinstance(event, dict),
                    "has_event_attr": hasattr(event, "event"),
                    "has_name_attr": hasattr(event, "name"),
                },
            )
            return event_name
        except Exception as e:
            logger.error(
                "Failed to get event name",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "error": str(e),
                    "event_type": type(event).__name__,
                    "event_data": str(event),
                },
                exc_info=True,
            )
            return None

    def _get_event_id(self, event: Any) -> str:
        """Get event ID from event field.

        Args:
            event: Event data to get ID from

        Returns:
            str: Event ID, generated if not found
        """
        try:
            # Try to get existing ID
            if isinstance(event, dict):
                # Try different ID fields in order of preference
                for id_field in ["id", "event_id", "recording_id"]:
                    if event_id := event.get(id_field):
                        event_type = event.get("event", "unknown")
                        source = event.get("source_plugin", "unknown")
                        # Use event type and source only for uniqueness
                        return f"{event_id}_{event_type}_{source}"
                return str(uuid.uuid4())
            elif hasattr(event, "event_id") and event.event_id:
                return event.event_id
            elif hasattr(event, "recording_id"):
                event_type = getattr(event, "event", "unknown")
                source = getattr(event, "source_plugin", "unknown")
                # Use recording ID, event type and source for uniqueness
                return f"{event.recording_id}_{event_type}_{source}"

            # Generate new ID if none found
            return str(uuid.uuid4())
        except Exception as e:
            logger.error(
                "Failed to get event ID, generating new one",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "event_type": type(event).__name__,
                    "event_data": str(event),
                },
                exc_info=True,
            )
            return str(uuid.uuid4())

    def _should_process_event(self, event: Any) -> bool:
        """Check if an event should be processed based on its source.

        Args:
            event: Event to check

        Returns:
            bool: True if event should be processed
        """
        try:
            # Get source plugin
            if isinstance(event, dict):
                source = event.get("source_plugin")
            elif hasattr(event, "context") and event.context:
                source = event.context.source_plugin
            else:
                source = None

            # Get event type
            if isinstance(event, dict):
                event_type = event.get("event")
            elif hasattr(event, "event"):
                event_type = event.event
            else:
                event_type = None

            logger.debug(
                "Checking event source",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "source_plugin": source,
                    "event_type": event_type,
                    "event_data": str(event),
                },
            )

            return True
        except Exception as e:
            logger.error(
                "Error checking event source",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "error": str(e),
                    "event_data": str(event),
                },
                exc_info=True,
            )
            return True

    async def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """Subscribe to an event.

        Args:
            event_name: Name of the event to subscribe to
            handler: Async function to handle the event
        """
        if not event_name:
            raise ValueError("Event name cannot be empty")
        if not handler:
            raise ValueError("Handler cannot be None")

        async with self._lock:
            logger.debug(
                "Attempting to subscribe handler",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "event_name": event_name,
                    "handler": handler.__name__,
                    "handler_module": handler.__module__,
                    "current_handlers": [
                        {"name": h.__name__, "module": h.__module__}
                        for h in self._subscribers.get(event_name, [])
                    ],
                    "all_subscriptions": {
                        k: [{"name": h.__name__, "module": h.__module__} for h in v]
                        for k, v in self._subscribers.items()
                    },
                },
            )

            if handler not in self._subscribers[event_name]:
                self._subscribers[event_name].append(handler)
                logger.debug(
                    "Added event subscription",
                    extra={
                        "req_id": self._req_id,
                        "component": "event_bus",
                        "event_name": event_name,
                        "handler": handler.__qualname__,
                        "subscriber_count": len(self._subscribers[event_name]),
                    },
                )
            else:
                logger.warning(
                    "Handler already subscribed to event",
                    extra={
                        "req_id": self._req_id,
                        "component": "event_bus",
                        "event_name": event_name,
                        "handler": handler.__name__,
                        "handler_module": handler.__module__,
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
                logger.debug(
                    "Removed event subscription",
                    extra={
                        "req_id": self._req_id,
                        "component": "event_bus",
                        "event_name": event_name,
                        "handler": handler.__qualname__,
                        "subscriber_count": len(self._subscribers[event_name]),
                    },
                )

    async def publish(self, event: Any) -> None:
        """Publish an event to all subscribers.

        Args:
            event: Event data to publish
        """
        try:
            event_dict = event if isinstance(event, dict) else getattr(event, "__dict__", {}) if event else {}
        except Exception:
            event_dict = {}

        logger.debug(
            "BEGIN Event Publishing",
            extra={
                "req_id": self._req_id,
                "component": "event_bus",
                "event_type": type(event).__name__,
                "raw_event": str(event),
                "event_dict": event_dict,
                "subscriber_count": len(
                    self._subscribers.get(self._get_event_name(event) or "", [])
                ),
            },
        )

        event_name = self._get_event_name(event)
        if not event_name:
            logger.error(
                "No event name found in event data",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "event_type": type(event).__name__,
                    "event_data": str(event),
                },
            )
            return

        # Get subscribers for this event
        handlers = self._subscribers.get(event_name, [])
        logger.debug(
            "Found event subscribers",
            extra={
                "req_id": self._req_id,
                "component": "event_bus",
                "event_name": event_name,
                "subscriber_count": len(handlers),
                "handlers": [
                    {
                        "name": handler.__name__,
                        "module": handler.__module__,
                        "qualname": handler.__qualname__,
                        "id": id(handler),
                        "class": handler.__self__.__class__.__name__
                        if hasattr(handler, "__self__")
                        else None,
                        "instance_id": id(handler.__self__)
                        if hasattr(handler, "__self__")
                        else None,
                    }
                    for handler in handlers
                ],
            },
        )

        if not handlers:
            logger.warning(
                "No subscribers found for event",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "event_name": event_name,
                    "event_type": type(event).__name__,
                    "event_data": str(event),
                    "all_subscriptions": {
                        name: [h.__name__ for h in hs]
                        for name, hs in self._subscribers.items()
                    },
                },
            )
            return

        # Create tasks for each handler but don't wait for them
        for handler in handlers:
            task = asyncio.create_task(self._handle_task(handler, event))
            task.add_done_callback(self._cleanup_task)
            self._pending_tasks.add(task)

            logger.debug(
                "Created handler task",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "event_name": event_name,
                    "handler": handler.__name__,
                    "handler_module": handler.__module__,
                    "handler_id": id(handler),
                    "task_id": id(task),
                    "pending_tasks": len(self._pending_tasks),
                },
            )

        logger.debug(
            "All handler tasks created",
            extra={
                "req_id": self._req_id,
                "component": "event_bus",
                "event_name": event_name,
                "pending_tasks": len(self._pending_tasks),
            },
        )

    async def shutdown(self) -> None:
        """Gracefully shutdown the event bus by canceling pending tasks."""
        logger.info(
            "Starting event bus shutdown",
            extra={
                "req_id": self._req_id,
                "component": "event_bus",
                "pending_tasks": len(self._pending_tasks),
            },
        )

        # First stop the cleanup task
        if self._cleanup_events_task and not self._cleanup_events_task.done():
            self._cleanup_events_task.cancel()
            try:
                await self._cleanup_events_task
            except asyncio.CancelledError:
                pass

        # Cancel all pending tasks
        pending_tasks = list(self._pending_tasks)
        if pending_tasks:
            for task in pending_tasks:
                if not task.done():
                    task.cancel()

            # Wait for all tasks to complete or be cancelled with a timeout
            try:
                await asyncio.wait(pending_tasks, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout waiting for tasks to cancel",
                    extra={
                        "req_id": self._req_id,
                        "component": "event_bus",
                        "pending_tasks": len(pending_tasks),
                    },
                )
            except Exception as e:
                logger.error(
                    "Error during task cancellation",
                    extra={
                        "req_id": self._req_id,
                        "component": "event_bus",
                        "error": str(e),
                    },
                    exc_info=True,
                )

        self._pending_tasks.clear()
        self._subscribers.clear()
        self._processed_events.clear()

        logger.info(
            "Event bus shutdown complete",
            extra={"req_id": self._req_id, "component": "event_bus"},
        )
