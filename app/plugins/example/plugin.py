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

    async def _handle_recording_ended(self, event_data: Event | dict) -> None:
        """Handle recording.ended event.

        This is an example event handler that demonstrates how to process events.

        Args:
            event_data: Event data containing both event data and context
        """
        try:
            # Extract data handling both dict and Event object formats
            if isinstance(event_data, dict):
                recording_id = event_data.get("recording_id")
                current_event = event_data.get("current_event", {})
                recording_data = current_event.get("recording", {})
                audio_paths = recording_data.get("audio_paths", {})
                mic_path = audio_paths.get("microphone")
                sys_path = audio_paths.get("system")
                metadata = event_data.get("metadata", {})
                correlation_id = str(metadata.get("correlation_id", uuid.uuid4()))
            else:
                recording_id = event_data.data.get("recording_id")
                mic_path = event_data.data.get("microphone_audio_path")
                sys_path = event_data.data.get("system_audio_path")
                metadata = event_data.data.get("metadata", {})
                correlation_id = str(getattr(event_data.context, "correlation_id", uuid.uuid4()))

            if not recording_id:
                logger.error("No recording_id found in event data")
                return

            # Process the event
            logger.info(
                "Processing recording ended event",
                extra={
                    "recording_id": recording_id,
                    "mic_path": mic_path,
                    "sys_path": sys_path
                }
            )

            # Emit completion event
            if self.event_bus:
                completed_event = Event.create(
                    name="example.completed",
                    data={
                        "recording": {
                            "status": "completed",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "recording_id": recording_id,
                            "audio_paths": {
                                "system": sys_path,
                                "microphone": mic_path
                            }
                        },
                        "metadata": metadata,
                        "context": {
                            "correlation_id": correlation_id,
                            "source_plugin": self.name,
                            "metadata": metadata
                        }
                    },
                    correlation_id=correlation_id,
                    source_plugin=self.name,
                    priority=EventPriority.NORMAL
                )
                await self.event_bus.publish(completed_event)

        except Exception as e:
            logger.error(
                "Error handling recording.ended event",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "example"
