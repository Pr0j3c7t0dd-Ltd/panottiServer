from typing import Dict, List, Optional
from app.plugins.events.models import Event
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

class EventStore:
    """A simple in-memory event store for persisting and retrieving events."""
    
    def __init__(self):
        self._events: Dict[str, List[Event]] = {}
        
    async def store_event(self, event: Event) -> None:
        """Store an event in memory."""
        if event.plugin_id not in self._events:
            self._events[event.plugin_id] = []
        self._events[event.plugin_id].append(event)
        logger.debug(f"Stored event {event.event_id} for plugin {event.plugin_id}")
        
    async def get_events(self, plugin_id: str) -> List[Event]:
        """Retrieve all events for a given plugin."""
        return self._events.get(plugin_id, [])
        
    async def get_event(self, event_id: str) -> Optional[Event]:
        """Retrieve a specific event by its ID."""
        for events in self._events.values():
            for event in events:
                if event.event_id == event_id:
                    return event
        return None
        
    async def clear_events(self, plugin_id: str) -> None:
        """Clear all events for a given plugin."""
        if plugin_id in self._events:
            self._events[plugin_id] = []
            logger.debug(f"Cleared events for plugin {plugin_id}")
