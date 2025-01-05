"""Event bus implementation."""

import asyncio
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any
from datetime import datetime
import uuid
import traceback

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
        self._processed_events: dict[str, datetime] = {}  # event_id -> timestamp
        self._pending_tasks: set[asyncio.Task] = set()
        self._cleanup_events_task = None  # Will be initialized when event loop is available
        self._req_id = str(uuid.uuid4())
        logger.info(
            "Event bus initialized",
            extra={
                "req_id": self._req_id,
                "component": "event_bus"
            }
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
                    "task_id": id(self._cleanup_events_task)
                }
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
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus"
                }
            )

    async def _cleanup_old_events(self) -> None:
        """Periodically clean up old processed events."""
        while True:
            try:
                await asyncio.sleep(3600)  # Clean up every hour
                now = datetime.utcnow()
                async with self._lock:
                    # Remove events older than 1 hour
                    old_events = [
                        event_id for event_id, timestamp in self._processed_events.items()
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
                                "remaining_count": len(self._processed_events)
                            }
                        )
            except Exception as e:
                logger.error(
                    "Error cleaning up old events",
                    extra={
                        "req_id": self._req_id,
                        "component": "event_bus",
                        "error": str(e)
                    }
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
                    "event_id": getattr(event, "event_id", None) or getattr(event, "id", None),
                    "stack_trace": "".join(traceback.format_stack())
                }
            )
            await handler(event)
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
                    "event_id": getattr(event, "event_id", None) or getattr(event, "id", None),
                    "stack_trace": "".join(traceback.format_stack())
                }
            )
        except Exception as e:
            logger.error(
                f"Error in event handler {handler.__name__}",
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
                    "event_id": getattr(event, "event_id", None) or getattr(event, "id", None),
                    "stack_trace": "".join(traceback.format_stack())
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
            self._processed_events[event_id] = datetime.utcnow()

    def _get_event_name(self, event: dict[str, Any] | RecordingEvent | RecordingStartRequest | RecordingEndRequest) -> str | None:
        """Get event name from event field.
        
        Args:
            event: Event data to get name from
            
        Returns:
            str | None: Event name or None if not found
        """
        event_name = None
        try:
            if isinstance(event, dict):
                event_name = event.get("event")
            elif isinstance(event, (RecordingEvent, RecordingStartRequest, RecordingEndRequest)):
                event_name = event.event
            elif hasattr(event, "event"):
                event_name = event.event

            logger.debug(
                "Getting event name",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "event_type": type(event).__name__,
                    "event_name": event_name,
                    "event_data": str(event),
                    "is_dict": isinstance(event, dict),
                    "is_recording_event": isinstance(event, RecordingEvent),
                    "has_event_attr": hasattr(event, "event")
                }
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
                    "event_data": str(event)
                },
                exc_info=True
            )
            return None

    def _get_event_id(self, event: dict[str, Any] | RecordingEvent | RecordingStartRequest | RecordingEndRequest) -> str:
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
            elif isinstance(event, (RecordingEvent, RecordingStartRequest, RecordingEndRequest)):
                # Use event_id if available, otherwise generate a unique one
                if hasattr(event, "event_id") and event.event_id:
                    return event.event_id
                source = getattr(event.context, "source_plugin", "unknown") if hasattr(event, "context") else "unknown"
                # Use recording ID, event type and source for uniqueness
                return f"{event.recording_id}_{event.event}_{source}"
            elif hasattr(event, "event_id") and getattr(event, "event_id"):
                return getattr(event, "event_id")
            elif hasattr(event, "recording_id"):
                event_type = getattr(event, "event", "unknown")
                source = getattr(event, "source_plugin", "unknown")
                # Use recording ID, event type and source for uniqueness
                return f"{getattr(event, 'recording_id')}_{event_type}_{source}"
            
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
                    "event_data": str(event)
                },
                exc_info=True
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
                    "event_data": str(event)
                }
            )

            return True
        except Exception as e:
            logger.error(
                "Error checking event source",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "error": str(e),
                    "event_data": str(event)
                },
                exc_info=True
            )
            return True

    async def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """Subscribe to an event.

        Args:
            event_name: Name of the event to subscribe to
            handler: Async function to handle the event
        """
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
                        {
                            "name": h.__name__,
                            "module": h.__module__
                        } for h in self._subscribers.get(event_name, [])
                    ],
                    "all_subscriptions": {
                        k: [{"name": h.__name__, "module": h.__module__} for h in v]
                        for k, v in self._subscribers.items()
                    }
                }
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
                        "subscriber_count": len(self._subscribers[event_name])
                    }
                )
            else:
                logger.warning(
                    "Handler already subscribed to event",
                    extra={
                        "req_id": self._req_id,
                        "component": "event_bus",
                        "event_name": event_name,
                        "handler": handler.__name__,
                        "handler_module": handler.__module__
                    }
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
                        "subscriber_count": len(self._subscribers[event_name])
                    }
                )

    async def publish(
        self,
        event: dict[str, Any] | RecordingEvent | RecordingStartRequest | RecordingEndRequest,
    ) -> None:
        """Publish event to all subscribers.
        
        Args:
            event: Event data to publish
        """
        # Log full event details at start
        logger.debug(
            "BEGIN Event Publishing",
            extra={
                "req_id": self._req_id,
                "component": "event_bus",
                "event_type": type(event).__name__,
                "raw_event": str(event),
                "event_dict": event.dict() if hasattr(event, "dict") else event,
                "event_dir": dir(event) if not isinstance(event, dict) else None,
                "event_module": event.__class__.__module__ if not isinstance(event, dict) else None,
                "processed_events": list(self._processed_events.keys()),
                "subscriber_count": sum(len(handlers) for handlers in self._subscribers.values()),
                "all_subscriptions": {
                    k: [{"name": h.__name__, "module": h.__module__} for h in v]
                    for k, v in self._subscribers.items()
                },
                "stack_trace": "".join(traceback.format_stack())
            }
        )

        # Check if event should be processed
        if not self._should_process_event(event):
            logger.warning(
                "Event processing skipped by source check",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "event_data": str(event),
                    "event_type": type(event).__name__,
                    "source": (event.context.source_plugin if hasattr(event, "context") else 
                             event.get("source_plugin") if isinstance(event, dict) else None),
                    "stack_trace": "".join(traceback.format_stack())
                }
            )
            return

        event_name = self._get_event_name(event)
        if not event_name:
            logger.error(
                "Could not determine event name",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "event_type": type(event).__name__,
                    "event_data": str(event),
                    "event_module": event.__class__.__module__ if not isinstance(event, dict) else None,
                    "event_attrs": dir(event) if not isinstance(event, dict) else None,
                    "stack_trace": "".join(traceback.format_stack())
                }
            )
            return

        event_id = self._get_event_id(event)
        logger.debug(
            "Event details determined",
            extra={
                "req_id": self._req_id,
                "component": "event_bus",
                "event_name": event_name,
                "event_id": event_id,
                "event_type": type(event).__name__,
                "event_data": str(event),
                "source": (event.context.source_plugin if hasattr(event, "context") else 
                          event.get("source_plugin") if isinstance(event, dict) else None),
                "stack_trace": "".join(traceback.format_stack())
            }
        )

        # Check if event was already processed
        async with self._lock:
            is_processed = await self._is_event_processed(event_id)
            logger.debug(
                "Event processing status check",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "event_id": event_id,
                    "event_name": event_name,
                    "is_processed": is_processed,
                    "processed_events_count": len(self._processed_events),
                    "processed_events": list(self._processed_events.keys()),
                    "stack_trace": "".join(traceback.format_stack())
                }
            )
            
            if is_processed:
                logger.warning(
                    f"Event {event_id} already processed, skipping",
                    extra={
                        "req_id": self._req_id,
                        "component": "event_bus",
                        "event_id": event_id,
                        "event_name": event_name,
                        "event_data": str(event),
                        "source": (event.context.source_plugin if hasattr(event, "context") else 
                                 event.get("source_plugin") if isinstance(event, dict) else None),
                        "stack_trace": "".join(traceback.format_stack())
                    }
                )
                return

            # Mark event as processed
            await self._mark_event_processed(event_id)
            logger.debug(
                "Marked event as processed",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "event_id": event_id,
                    "event_name": event_name,
                    "processed_events_count": len(self._processed_events),
                    "processed_events": list(self._processed_events.keys()),
                    "stack_trace": "".join(traceback.format_stack())
                }
            )

        handlers = self._subscribers.get(event_name, [])
        if not handlers:
            logger.warning(
                "No handlers found for event",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "event_name": event_name,
                    "event_id": event_id,
                    "available_subscriptions": {
                        k: [h.__name__ for h in v]
                        for k, v in self._subscribers.items()
                    },
                    "stack_trace": "".join(traceback.format_stack())
                }
            )
            return

        # Create tasks for each handler
        tasks = []
        for handler in handlers:
            logger.debug(
                "Creating handler task",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "event_name": event_name,
                    "event_id": event_id,
                    "handler": handler.__name__,
                    "handler_module": handler.__module__,
                    "handler_qualname": handler.__qualname__,
                    "handler_id": id(handler),
                    "stack_trace": "".join(traceback.format_stack())
                }
            )
            task = asyncio.create_task(self._handle_task(handler, event))
            task.add_done_callback(self._cleanup_task)
            self._pending_tasks.add(task)
            tasks.append(task)

        # Wait for all handlers to complete
        if tasks:
            logger.debug(
                "Waiting for handler tasks",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "event_name": event_name,
                    "event_id": event_id,
                    "num_tasks": len(tasks),
                    "handlers": [
                        {
                            "name": h.__name__,
                            "module": h.__module__,
                            "qualname": h.__qualname__,
                            "handler_id": id(h)
                        } for h in handlers
                    ],
                    "stack_trace": "".join(traceback.format_stack())
                }
            )
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug(
                "Handler tasks completed",
                extra={
                    "req_id": self._req_id,
                    "component": "event_bus",
                    "event_name": event_name,
                    "event_id": event_id,
                    "num_completed": len(tasks),
                    "stack_trace": "".join(traceback.format_stack())
                }
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
        logger.info(
            "Event bus shutdown complete",
            extra={
                "req_id": self._req_id,
                "component": "event_bus"
            }
        )
