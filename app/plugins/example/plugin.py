import asyncio
from app.plugins.base import PluginBase
from app.plugins.events.models import Event, EventContext, EventPriority
from datetime import datetime

# pylint: disable=too-few-public-methods
class Plugin(PluginBase):
    """Example plugin demonstrating basic functionality"""
    
    async def _initialize(self) -> None:
        """Initialize plugin"""
        # Subscribe to events
        self.event_bus.subscribe("example.event", self.handle_event)
        self.event_bus.subscribe("example.status", self.handle_status)
        self.logger.info(
            "Subscribed to events",
            extra={
                "event_types": ["example.event", "example.status"],
                "check_interval": self.get_config("check_interval", 60)
            }
        )
        
        # Schedule periodic task
        asyncio.create_task(self._periodic_task())
        
    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        # Unsubscribe from events
        self.event_bus.unsubscribe("example.event", self.handle_event)
        self.event_bus.unsubscribe("example.status", self.handle_status)
        self.logger.info(
            "Unsubscribed from events",
            extra={"event_types": ["example.event", "example.status"]}
        )
        
    async def handle_event(self, event: Event) -> None:
        """Handle incoming events"""
        self.logger.info(
            "Received event",
            extra={
                "event_name": event.name,
                "correlation_id": event.context.correlation_id,
                "source_plugin": event.context.source_plugin
            }
        )
        self.logger.debug(
            "Event details",
            extra={
                "event_name": event.name,
                "event_payload": event.payload,
                "event_context": event.context.dict()
            }
        )
        
        # Process event
        await asyncio.sleep(1)  # Simulate processing
        self.logger.info(
            "Processed event",
            extra={
                "event_name": event.name,
                "correlation_id": event.context.correlation_id
            }
        )
        
    async def handle_status(self, event: Event) -> None:
        """Handle status events"""
        self.logger.info(
            "Received status update",
            extra={
                "correlation_id": event.context.correlation_id,
                "source_plugin": event.context.source_plugin,
                "status": event.payload.get("status", "unknown")
            }
        )
        
    async def _periodic_task(self) -> None:
        """Example periodic task"""
        check_interval = self.get_config("check_interval", 60)
        self.logger.info(
            "Starting periodic task",
            extra={"check_interval": check_interval}
        )
        
        while self.is_initialized:
            try:
                # Generate a unique correlation ID with timestamp
                correlation_id = f"periodic-status-{datetime.utcnow().isoformat()}"
                
                # Create and publish event
                event = Event(
                    name="example.status",
                    payload={"status": "healthy"},
                    context=EventContext(
                        correlation_id=correlation_id,
                        source_plugin=self.name
                    ),
                    priority=EventPriority.LOW
                )
                
                self.logger.debug(
                    "Publishing status event",
                    extra={
                        "event_name": event.name,
                        "correlation_id": correlation_id,
                        "plugin": self.name
                    }
                )
                await self.event_bus.publish(event)
                
            except Exception as e:
                self.logger.error(
                    "Error in periodic task",
                    extra={
                        "error": str(e),
                        "plugin_name": self.name
                    }
                )
                
            await asyncio.sleep(check_interval)  # Run every minute
