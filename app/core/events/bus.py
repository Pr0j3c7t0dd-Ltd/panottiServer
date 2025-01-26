"""Event bus implementation."""

import asyncio
import inspect
import traceback
import uuid
from collections import defaultdict
from datetime import UTC, datetime
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
        self._shutting_down = False
        logger.info(
            "Event bus initialized",
            extra={"req_id": self._req_id, "component": "event_bus"},
        )

    async def start(self) -> None:
        """Start the event bus background tasks."""
        self._shutting_down = False
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
        self._shutting_down = True
        
        # Wait for pending tasks to complete
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()
            
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

    async def _cleanup_old_events(self, run_once: bool = False) -> None:
        """Periodically clean up old processed events.
        
        Args:
            run_once: If True, run cleanup once and return (for testing)
        """
        while True:
            try:
                if not run_once:
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
                
                if run_once:
                    break
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
                    "handler": getattr(handler, "__name__", str(handler))
                    if handler
                    else None,
                    "handler_module": getattr(handler, "__module__", None)
                    if handler
                    else None,
                    "handler_qualname": getattr(handler, "__qualname__", None)
                    if handler
                    else None,
                    "handler_id": id(handler) if handler else None,
                    "event_type": type(event).__name__,
                    "event_data": str(event),
                    "event_id": getattr(event, "event_id", None)
                    or getattr(event, "id", None),
                    "event_name": self._get_event_name(event),
                    "stack_trace": "".join(traceback.format_stack()),
                },
            )

            # Add pre-execution state check
            handler_info = {
                "req_id": self._req_id,
                "component": "event_bus",
                "handler": getattr(handler, "__name__", str(handler)),
            }

            # Check if it's a bound method and add instance info if available
            if inspect.ismethod(handler) and hasattr(handler, "__self__"):
                instance = handler.__self__
                handler_info.update(
                    {
                        "handler_class": instance.__class__.__name__,
                        "handler_instance_vars": vars(instance),
                    }
                )

            logger.debug(
                "Handler instance state",
                extra=handler_info,
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
                            f"Error in event handler {getattr(handler, '__name__', str(handler))}",
                            extra={
                                "req_id": self._req_id,
                                "component": "event_bus",
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                                "handler": getattr(handler, "__name__", str(handler))
                                if handler
                                else None,
                                "handler_module": getattr(handler, "__module__", None)
                                if handler
                                else None,
                                "handler_qualname": getattr(
                                    handler, "__qualname__", None
                                )
                                if handler
                                else None,
                                "handler_id": id(handler) if handler else None,
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
                                "handler": getattr(handler, "__name__", str(handler))
                                if handler
                                else None,
                                "handler_module": getattr(handler, "__module__", None)
                                if handler
                                else None,
                                "handler_qualname": getattr(
                                    handler, "__qualname__", None
                                )
                                if handler
                                else None,
                                "handler_id": id(handler) if handler else None,
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
                            "handler": getattr(handler, "__name__", str(handler))
                            if handler
                            else None,
                            "event_id": getattr(event, "event_id", None)
                            or getattr(event, "id", None),
                        },
                    )

            handler_task.add_done_callback(handler_done_callback)

        except Exception as e:
            logger.error(
                f"Error setting up event handler {getattr(handler, '__name__', str(handler))}",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "handler": getattr(handler, "__name__", str(handler))
                    if handler
                    else None,
                    "handler_module": getattr(handler, "__module__", None)
                    if handler
                    else None,
                    "handler_qualname": getattr(handler, "__qualname__", None)
                    if handler
                    else None,
                    "handler_id": id(handler) if handler else None,
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
                    "handler": getattr(handler, "__name__", str(handler))
                    if handler
                    else None,
                    "handler_module": getattr(handler, "__module__", None)
                    if handler
                    else None,
                    "current_handlers": [
                        {
                            "name": getattr(h, "__name__", str(h)),
                            "module": getattr(h, "__module__", None),
                        }
                        for h in self._subscribers.get(event_name, [])
                    ],
                    "all_subscriptions": {
                        k: [
                            {
                                "name": getattr(h, "__name__", str(h)),
                                "module": getattr(h, "__module__", None),
                            }
                            for h in v
                        ]
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
                        "handler": getattr(handler, "__qualname__", str(handler))
                        if handler
                        else None,
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
                        "handler": getattr(handler, "__name__", str(handler))
                        if handler
                        else None,
                        "handler_module": getattr(handler, "__module__", None)
                        if handler
                        else None,
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
                        "handler": getattr(handler, "__qualname__", str(handler))
                        if handler
                        else None,
                        "subscriber_count": len(self._subscribers[event_name]),
                    },
                )

    def _get_handler_info(self, handler: Any) -> dict:
        """Get handler information for logging.

        Args:
            handler: Event handler to get info for

        Returns:
            dict: Handler information
        """
        try:
            if not handler:
                return {
                    "name": "None",
                    "module": None,
                    "qualname": None,
                    "id": None,
                    "class": None,
                    "instance_id": None,
                }
            return {
                "name": getattr(handler, "__name__", str(handler)),
                "module": getattr(handler, "__module__", None),
                "qualname": getattr(handler, "__qualname__", None),
                "id": id(handler),
                "class": getattr(getattr(handler, "__self__", None), "__class__", None)
                if getattr(handler, "__self__", None)
                else None,
                "instance_id": id(getattr(handler, "__self__", None))
                if hasattr(handler, "__self__")
                else None,
            }
        except Exception as e:
            return {"name": str(handler), "error": str(e)}

    async def publish(self, event: Any) -> None:
        """Publish an event to all subscribers.

        Args:
            event: Event data to publish
        """
        try:
            event_dict = (
                event
                if isinstance(event, dict)
                else getattr(event, "__dict__", {})
                if event
                else {}
            )
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
                "handlers": [self._get_handler_info(handler) for handler in handlers],
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
                        name: [getattr(h, "__name__", str(h)) for h in hs if h]
                        for name, hs in self._subscribers.items()
                    },
                },
            )
            return

        # Create tasks for each handler but don't wait for them
        for handler in handlers:
            if not handler:
                logger.warning(
                    "Skipping None handler",
                    extra={
                        "req_id": self._req_id,
                        "component": "event_bus",
                        "event_name": event_name,
                    },
                )
                continue

            try:
                task = asyncio.create_task(self._handle_task(handler, event))
                task.add_done_callback(self._cleanup_task)
                self._pending_tasks.add(task)

                logger.debug(
                    "Created handler task",
                    extra={
                        "req_id": self._req_id,
                        "component": "event_bus",
                        "event_name": event_name,
                        "handler": getattr(handler, "__name__", str(handler))
                        if handler
                        else None,
                        "handler_module": getattr(handler, "__module__", None)
                        if handler
                        else None,
                        "handler_id": id(handler) if handler else None,
                        "task_id": id(task),
                        "pending_tasks": len(self._pending_tasks),
                    },
                )
            except Exception as e:
                logger.error(
                    "Failed to create handler task",
                    extra={
                        "req_id": self._req_id,
                        "component": "event_bus",
                        "event_name": event_name,
                        "handler": getattr(handler, "__name__", str(handler))
                        if handler
                        else None,
                        "error": str(e),
                    },
                    exc_info=True,
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
        if self._shutting_down:
            return

        self._shutting_down = True
        logger.info(
            "Starting event bus shutdown",
            extra={
                "req_id": self._req_id,
                "component": "event_bus",
                "pending_tasks": len(self._pending_tasks),
            },
        )

        try:
            # First stop accepting new events and clear state under lock
            try:
                async with asyncio.timeout(10.0):
                    async with self._lock:
                        self._subscribers.clear()
                        self._processed_events.clear()
            except asyncio.TimeoutError:
                logger.warning("Timeout acquiring lock during shutdown")

            # Then stop the cleanup task
            if self._cleanup_events_task and not self._cleanup_events_task.done():
                self._cleanup_events_task.cancel()
                try:
                    await asyncio.wait_for(self._cleanup_events_task, timeout=10.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    logger.warning("Cleanup task did not complete cleanly")
                finally:
                    self._cleanup_events_task = None

            # Cancel all pending tasks
            pending_tasks = list(self._pending_tasks)
            if pending_tasks:
                # Cancel tasks in batches to avoid overwhelming the event loop
                batch_size = 50
                for i in range(0, len(pending_tasks), batch_size):
                    batch = pending_tasks[i : i + batch_size]
                    for task in batch:
                        if not task.done():
                            task.cancel()

                    # Wait for batch to complete with timeout
                    try:
                        await asyncio.wait(batch, timeout=10.0)
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Timeout waiting for task batch to cancel",
                            extra={
                                "req_id": self._req_id,
                                "component": "event_bus",
                                "batch_start": i,
                                "batch_size": len(batch),
                            },
                        )
                    except Exception as e:
                        logger.error(
                            "Error during task batch cancellation",
                            extra={
                                "req_id": self._req_id,
                                "component": "event_bus",
                                "error": str(e),
                                "batch_start": i,
                                "batch_size": len(batch),
                            },
                            exc_info=True,
                        )

            # Clear remaining state
            self._pending_tasks.clear()

            logger.info(
                "Event bus shutdown complete",
                extra={"req_id": self._req_id, "component": "event_bus"},
            )

        except Exception as e:
            logger.error(
                "Error during event bus shutdown",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "error": str(e),
                },
                exc_info=True,
            )
            raise
        finally:
            self._shutting_down = False
