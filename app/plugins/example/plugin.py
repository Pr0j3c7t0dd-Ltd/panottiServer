"""Example plugin to demonstrate plugin development."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.models.recording.events import (
    RecordingEndRequest,
    RecordingEvent,
    RecordingStartRequest,
)
from app.plugins.base import PluginBase, PluginConfig
from app.plugins.events.models import EventContext

logger = logging.getLogger(__name__)

EventData = (
    dict[str, Any] | RecordingEvent | RecordingStartRequest | RecordingEndRequest
)


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
                "Starting example plugin initialization",
                extra={"plugin": self.name}
            )

            # Initialize thread pool for processing
            max_workers = self.config.config.get("max_concurrent_tasks", 4)
            self._executor = ThreadPoolExecutor(max_workers=max_workers)

            # Subscribe to events
            await self.event_bus.subscribe(
                "recording.ended",
                self._handle_recording_ended
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

    async def _handle_recording_ended(
        self, event_data: EventData, context: EventContext
    ) -> None:
        """Handle recording.ended event.
        
        This is an example event handler that demonstrates how to process events.
        
        Args:
            event_data: Event data
            context: Event context
        """
        try:
            logger.debug(
                "Handling recording.ended event",
                extra={
                    "event_id": context.event_id,
                    "event_type": context.event_type,
                },
            )

            # Example processing
            debug_mode = self.config.config.get("debug_mode", False)
            example_setting = self.config.config.get("example_setting")

            logger.info(
                "Example plugin processed recording.ended event",
                extra={
                    "event_id": context.event_id,
                    "debug_mode": debug_mode,
                    "example_setting": example_setting,
                },
            )

        except Exception as e:
            logger.error(
                "Error handling recording.ended event",
                extra={
                    "event_id": context.event_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "example"
