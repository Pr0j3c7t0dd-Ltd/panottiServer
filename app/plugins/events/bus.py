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
        self.handlers: dict[str, list[Callable[[EventData], Any]]] = {}
        self.plugins: dict[str, Any] = {}
        self.tasks: list[Task[Any]] = []
        self.logger = get_logger("event_bus")
        self.logger.info("Event bus initialized")

    async def publish(self, event: EventData) -> None:
        """Publish an event to all registered handlers"""
        try:
            # Convert dict to Event if needed
            event_obj = Event(**event) if isinstance(event, dict) else event

            # Store event
            event_id = await self.event_store.store_event(event_obj)
            self.logger.info(
                "Event published",
                extra={
                    "event_id": event_id,
                    "event_name": get_event_name(event_obj),
                    "event_priority": event_obj.priority,
                    "correlation_id": event_obj.context.correlation_id,
                    "source_plugin": event_obj.context.source_plugin,
                },
            )

            # Get handlers for this event
            event_name = get_event_name(event_obj)
            handlers = self.handlers.get(event_name, [])
            if not handlers:
                self.logger.warning(
                    "No handlers registered for event",
                    extra={"event_name": event_name, "event_id": event_id},
                )
                return

            # Process event based on priority
            if event_obj.priority == EventPriority.CRITICAL:
                # Process critical events immediately and wait for completion
                await asyncio.gather(
                    *[self._execute_handler(handler, event) for handler in handlers]
                )
            else:
                # Process other events asynchronously
                for handler in handlers:
                    task = asyncio.create_task(self._execute_handler(handler, event))
                    self.tasks.append(task)
                    task.add_done_callback(lambda t: self.tasks.remove(t))

        except Exception as e:
            self.logger.error(
                "Error publishing event",
                extra={
                    "error": str(e),
                    "event_name": get_event_name(event),
                },
                exc_info=True,
            )
            raise

    async def _execute_handler(
        self, handler: Callable[[EventData], Any], event: EventData
    ) -> None:
        """Process a single event with error handling"""
        try:
            self.logger.debug(
                "Processing event",
                extra={
                    "event_name": get_event_name(event),
                    "handler": handler.__qualname__,
                },
            )
            await handler(event)
            await self.event_store.mark_processed(get_event_id(event))
            self.logger.debug(
                "Event processed successfully",
                extra={
                    "event_name": get_event_name(event),
                    "handler": handler.__qualname__,
                },
            )
        except Exception as e:
            error_msg = str(e)
            self.logger.error(
                "Error processing event",
                extra={
                    "event_name": get_event_name(event),
                    "handler": handler.__qualname__,
                    "error": error_msg,
                },
            )
            await self.event_store.mark_processed(
                get_event_id(event), success=False, error=error_msg
            )

    async def emit(self, event: EventData) -> None:
        """Emit an event to all registered handlers"""
        await self.publish(event)

    async def subscribe(
        self,
        event_type: str,
        callback: Callable[[EventData], Any],
    ) -> None:
        """Subscribe a handler to an event"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(callback)
        self.logger.debug(
            "Handler subscribed",
            extra={"event_name": event_type, "handler": callback.__qualname__},
        )

    async def unsubscribe(
        self,
        event_type: str,
        callback: Callable[[EventData], Any],
    ) -> None:
        """Unsubscribe a handler from an event"""
        if event_type in self.handlers:
            self.handlers[event_type].remove(callback)
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
