"""Example plugin that demonstrates plugin functionality."""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from pytz import UTC

from app.core.events import Event, EventContext, EventPriority
from app.core.plugins import PluginBase, PluginConfig
from app.utils.logging_config import get_logger

logger = get_logger("app.plugins.example.plugin")

EventData = dict[str, Any]


class ExamplePlugin(PluginBase):
    """Example plugin that demonstrates how to create a plugin."""

    def __init__(self, config: PluginConfig, event_bus: Any = None) -> None:
        """Initialize the example plugin.

        Args:
            config: Plugin configuration
            event_bus: Event bus for subscribing to events
        """
        super().__init__(config, event_bus)
        self._executor: ThreadPoolExecutor | None = None
        self._processing_lock = threading.Lock()

    async def _initialize(self) -> None:
        """Initialize the plugin."""
        if not self.event_bus:
            logger.warning("No event bus available for plugin")
            return

        try:
            logger.debug(
                "Starting example plugin initialization", extra={"plugin": self.name}
            )

            # Initialize thread pool for processing
            max_workers = self.get_config("max_concurrent_tasks", 4)
            self._executor = ThreadPoolExecutor(max_workers=max_workers)

            # Subscribe to events
            await self.event_bus.subscribe(
                "recording.ended", self._handle_recording_ended
            )

            logger.info("Example plugin initialization complete")

        except Exception as e:
            logger.error(
                "Failed to initialize example plugin",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise

    async def _shutdown(self) -> None:
        """Shutdown the plugin."""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
            logger.info("Example plugin shutdown complete")
        except Exception as e:
            logger.error(
                "Error during example plugin shutdown",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise

    async def _handle_recording_ended(self, event_data: EventData) -> None:
        """Handle recording.ended event.

        This is an example event handler that demonstrates how to process events.

        Args:
            event_data: Event data containing both event data and context
        """
        try:
            # Initialize context as None
            context = None

            # Extract context from event
            context = (
                event_data.get("context")
                if isinstance(event_data, dict)
                else getattr(event_data, "context", None)
            )
            if not context:
                logger.warning("No context found in event")
                context = EventContext(correlation_id=str(uuid.uuid4()))

            logger.debug(
                "Handling recording.ended event",
                extra={
                    "event_id": getattr(context, "event_id", None),
                    "event_type": getattr(context, "event_type", None),
                },
            )

            # Example processing
            debug_mode = self.get_config("debug_mode", False)
            example_setting = self.get_config("example_setting")
            recording_id = event_data.get("data", {}).get("recording", {}).get("id")

            # Example of emitting a completion event with preserved event chain
            if self.event_bus:
                completion_event = Event.create(
                    name="example.completed",
                    data={
                        "recording": event_data.get("data", {}).get("recording", {}),
                        "example": {
                            "status": "completed",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "recording_id": recording_id,
                            "config": {
                                "debug_mode": debug_mode,
                                "example_setting": example_setting,
                            },
                        },
                        "metadata": event_data.get("data", {}).get("metadata", {}),
                        "context": {
                            "correlation_id": getattr(
                                event_data, "correlation_id", str(uuid.uuid4())
                            ),
                            "source_plugin": self.name,
                            "metadata": event_data.get("data", {}).get("metadata", {}),
                        },
                    },
                    correlation_id=getattr(
                        event_data, "correlation_id", str(uuid.uuid4())
                    ),
                    source_plugin=self.name,
                    priority=EventPriority.NORMAL,
                )
                await self.event_bus.publish(completion_event)

            logger.info(
                "Example plugin processed recording.ended event",
                extra={
                    "event_id": getattr(context, "event_id", None),
                    "debug_mode": debug_mode,
                    "example_setting": example_setting,
                },
            )

        except Exception as e:
            logger.error(
                "Error handling recording.ended event",
                extra={
                    "event_id": getattr(context, "event_id", None) if context else None,  # type: ignore
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            # Example of emitting an error event with preserved event chain
            if self.event_bus:
                error_event = Event.create(
                    name="example.error",
                    data={
                        # Preserve original event data
                        "recording": event_data.get("data", {}).get("recording", {}),
                        # Add current event data
                        "example": {
                            "status": "error",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "error": str(e),
                            "debug_mode": debug_mode,
                            "example_setting": example_setting,
                            "event_id": getattr(context, "event_id", None),
                        },
                        # Preserve metadata
                        "metadata": event_data.get("metadata", {})
                        or event_data.get("data", {}).get("metadata", {}),
                    },
                    correlation_id=getattr(context, "correlation_id", None)
                    or str(uuid.uuid4()),
                    source_plugin=self.name,
                    priority=EventPriority.NORMAL,
                )
                await self.event_bus.publish(error_event)
            raise

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "example"
