"""Event bus implementation."""
import asyncio
from typing import Any, Callable, Dict, List, Optional

from app.plugins.events.models import EventContext, EventPriority


class EventBus:
    """Event bus implementation for plugin communication."""

    def __init__(self):
        """Initialize the event bus."""
        self._handlers: Dict[str, List[Callable]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Callback function to handle the event
        """
        async with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    async def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler to remove
        """
        async with self._lock:
            if event_type in self._handlers:
                self._handlers[event_type].remove(handler)

    async def publish(
        self,
        event_type: str,
        data: Any,
        context: Optional[EventContext] = None
    ) -> None:
        """Publish an event to all subscribers.
        
        Args:
            event_type: Type of event being published
            data: Event data
            context: Optional event context
        """
        if context is None:
            context = EventContext(metadata={}, priority=EventPriority.NORMAL)

        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data, context)
                else:
                    handler(data, context)
            except Exception as e:
                # Log error but continue processing other handlers
                print(f"Error in event handler: {e}")
