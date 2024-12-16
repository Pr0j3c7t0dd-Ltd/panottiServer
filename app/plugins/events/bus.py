import asyncio
from typing import Callable, Awaitable, Dict, List
from app.plugins.events.models import Event, EventPriority
from app.plugins.events.persistence import EventStore
from app.utils.logging_config import get_logger

EventHandler = Callable[[Event], Awaitable[None]]

class EventBus:
    """Central event bus for plugin communication"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.handlers: Dict[str, List[EventHandler]] = {}
        self.logger = get_logger("event_bus")
        self._tasks: List[asyncio.Task] = []
        self.logger.info("Event bus initialized")
        
    async def publish(self, event: Event) -> None:
        """Publish an event to all registered handlers"""
        try:
            # Store event
            event_id = await self.event_store.store_event(event)
            self.logger.info(
                "Event published",
                extra={
                    "event_id": event_id,
                    "event_name": event.name,
                    "event_priority": event.priority.value,
                    "correlation_id": event.context.correlation_id,
                    "source_plugin": event.context.source_plugin
                }
            )
            
            # Get handlers for this event
            handlers = self.handlers.get(event.name, [])
            if not handlers:
                self.logger.warning(
                    "No handlers registered for event",
                    extra={
                        "event_name": event.name,
                        "event_id": event_id
                    }
                )
                return
                
            # Process event based on priority
            if event.priority == EventPriority.CRITICAL:
                # Process critical events immediately and wait for completion
                self.logger.debug(
                    "Processing critical event synchronously",
                    extra={
                        "event_id": event_id,
                        "handler_count": len(handlers)
                    }
                )
                await asyncio.gather(
                    *[self._process_event(handler, event, event_id) for handler in handlers]
                )
            else:
                # Process other events asynchronously
                self.logger.debug(
                    "Processing event asynchronously",
                    extra={
                        "event_id": event_id,
                        "handler_count": len(handlers)
                    }
                )
                for handler in handlers:
                    task = asyncio.create_task(
                        self._process_event(handler, event, event_id)
                    )
                    self._tasks.append(task)
                    task.add_done_callback(self._tasks.remove)
                    
        except Exception as e:
            self.logger.error(
                "Error publishing event",
                extra={
                    "event_name": event.name,
                    "error": str(e),
                    "correlation_id": event.context.correlation_id
                }
            )
            raise
            
    async def _process_event(
        self, 
        handler: EventHandler, 
        event: Event, 
        event_id: int
    ) -> None:
        """Process a single event with error handling"""
        try:
            self.logger.debug(
                "Processing event",
                extra={
                    "event_id": event_id,
                    "event_name": event.name,
                    "handler": handler.__qualname__
                }
            )
            await handler(event)
            await self.event_store.mark_processed(event_id, success=True)
            self.logger.debug(
                "Event processed successfully",
                extra={
                    "event_id": event_id,
                    "event_name": event.name,
                    "handler": handler.__qualname__
                }
            )
        except Exception as e:
            error_msg = str(e)
            self.logger.error(
                "Error processing event",
                extra={
                    "event_id": event_id,
                    "event_name": event.name,
                    "handler": handler.__qualname__,
                    "error": error_msg
                }
            )
            await self.event_store.mark_processed(
                event_id, 
                success=False, 
                error=error_msg
            )
            
    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """Subscribe to an event"""
        if event_name not in self.handlers:
            self.handlers[event_name] = []
        self.handlers[event_name].append(handler)
        self.logger.info(
            "Event handler registered",
            extra={
                "event_name": event_name,
                "handler": handler.__qualname__
            }
        )
        
    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        """Unsubscribe from an event"""
        if event_name in self.handlers:
            try:
                self.handlers[event_name].remove(handler)
                self.logger.info(
                    "Event handler unregistered",
                    extra={
                        "event_name": event_name,
                        "handler": handler.__qualname__
                    }
                )
            except ValueError:
                self.logger.warning(
                    "Handler not found for event",
                    extra={
                        "event_name": event_name,
                        "handler": handler.__qualname__
                    }
                )
                
    async def wait_for_pending_events(self) -> None:
        """Wait for all pending event processing to complete"""
        if self._tasks:
            self.logger.info(
                "Waiting for pending events",
                extra={"pending_count": len(self._tasks)}
            )
            await asyncio.gather(*self._tasks)
