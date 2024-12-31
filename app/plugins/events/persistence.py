from datetime import datetime

from app.plugins.events.models import Event
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class EventProcessingStatus:
    """Status of event processing"""

    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"


class EventStore:
    """A simple in-memory event store for persisting and retrieving events."""

    def __init__(self):
        self._events: dict[str, list[Event]] = {}
        self._status: dict[str, dict] = {}  # event_id -> {status, timestamp, error}

    async def store_event(self, event: Event) -> str:
        """Store an event in memory and return its ID."""
        if event.plugin_id not in self._events:
            self._events[event.plugin_id] = []
        self._events[event.plugin_id].append(event)

        # Initialize event status
        self._status[event.event_id] = {
            "status": EventProcessingStatus.PENDING,
            "timestamp": datetime.utcnow(),
            "error": None,
        }

        logger.debug(
            "Stored event",
            extra={
                "event_id": event.event_id,
                "plugin_id": event.plugin_id,
                "event_name": event.name,
            },
        )
        return event.event_id

    async def mark_processed(
        self, event_id: str, success: bool = True, error: str = None
    ) -> None:
        """Mark an event as processed or failed."""
        if event_id not in self._status:
            logger.warning(
                "Attempted to mark unknown event as processed",
                extra={"event_id": event_id},
            )
            return

        self._status[event_id].update(
            {
                "status": (
                    EventProcessingStatus.PROCESSED
                    if success
                    else EventProcessingStatus.FAILED
                ),
                "timestamp": datetime.utcnow(),
                "error": error,
            }
        )

        logger.debug(
            "Updated event status",
            extra={
                "event_id": event_id,
                "status": self._status[event_id]["status"],
                "error": error if error else None,
            },
        )

    async def get_events(self, plugin_id: str) -> list[Event]:
        """Retrieve all events for a given plugin."""
        return self._events.get(plugin_id, [])

    async def get_event(self, event_id: str) -> Event | None:
        """Retrieve a specific event by its ID."""
        for events in self._events.values():
            for event in events:
                if event.event_id == event_id:
                    return event
        return None

    async def get_event_status(self, event_id: str) -> dict | None:
        """Get the processing status of an event."""
        return self._status.get(event_id)

    async def clear_events(self, plugin_id: str) -> None:
        """Clear all events for a given plugin."""
        if plugin_id in self._events:
            self._events[plugin_id] = []
            logger.debug(f"Cleared events for plugin {plugin_id}")
