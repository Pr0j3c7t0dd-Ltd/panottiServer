"""Desktop notifier plugin for sending system notifications."""

import os
import subprocess
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from app.core.events import ConcreteEventBus as EventBus, Event, EventPriority
from app.core.plugins import PluginBase, PluginConfig
from app.models.database import DatabaseManager
from app.models.recording.events import EventContext, RecordingEvent
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

EventData = dict[str, Any] | RecordingEvent | Event


class DesktopNotifierPlugin(PluginBase):
    """Plugin for sending desktop notifications when meeting notes are completed"""

    def __init__(self, config: PluginConfig, event_bus: EventBus | None = None) -> None:
        super().__init__(config, event_bus)
        self._executor: ThreadPoolExecutor | None = None
        self._processing_lock = threading.Lock()
        self._db_initialized = False

    async def _initialize(self) -> None:
        """Initialize plugin."""
        if not self.event_bus:
            logger.warning("No event bus available for plugin")
            return

        try:
            logger.debug(
                "Initializing desktop notifier plugin", extra={"plugin": self.name}
            )

            # Initialize database
            self.db = await DatabaseManager.get_instance()
            await self._init_database()

            # Subscribe to meeting notes completed events
            await self.event_bus.subscribe(
                "meeting_notes_local.completed", self.handle_meeting_notes_completed
            )
            await self.event_bus.subscribe(
                "meeting_notes_remote.completed", self.handle_meeting_notes_completed
            )

            logger.info(
                "Desktop notifier plugin initialized",
                extra={
                    "plugin": self.name,
                    "subscribed_events": [
                        "meeting_notes_local.completed",
                        "meeting_notes_remote.completed",
                    ],
                    "handler": "handle_meeting_notes_completed",
                },
            )

        except Exception as e:
            logger.error(
                "Failed to initialize plugin",
                extra={"plugin": self.name, "error": str(e)},
                exc_info=True,
            )
            raise

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        if not self.event_bus:
            return

        # Unsubscribe from events
        await self.event_bus.unsubscribe(
            "meeting_notes_local.completed", self.handle_meeting_notes_completed
        )
        await self.event_bus.unsubscribe(
            "meeting_notes_remote.completed", self.handle_meeting_notes_completed
        )

        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)

        logger.info("Desktop notifier plugin shutdown", extra={"plugin": self.name})

    async def handle_meeting_notes_completed(self, event_data: EventData) -> None:
        """Handle meeting notes completed event"""
        try:
            logger.debug(
                "Raw meeting notes completed event received",
                extra={
                    "plugin": self.name,
                    "event_type": type(event_data).__name__,
                    "event_data": str(event_data),
                },
            )

            # Initialize recording_id with default value
            recording_id = "unknown"

            # Extract data based on event type
            if isinstance(event_data, dict):
                data = event_data
            elif isinstance(event_data, (Event, RecordingEvent)):
                data = event_data.data if hasattr(event_data, "data") else {}
            else:
                logger.error(
                    "Unsupported event type",
                    extra={
                        "plugin": self.name,
                        "event_type": type(event_data).__name__,
                    },
                )
                return

            # Extract recording ID and output path, with fallbacks
            recording_id = data.get("recording_id") or data.get("data", {}).get(
                "recording_id"
            )

            output_path = (
                data.get("output_path")
                or data.get("notes_path")
                or data.get("data", {}).get("output_path")
                or data.get("data", {}).get("notes_path")
                or data.get("current_event", {})
                .get("meeting_notes", {})
                .get("output_path")
            )

            if not output_path:
                logger.warning(
                    "No output path in event data",
                    extra={
                        "plugin": self.name,
                        "event_data": str(data),
                        "recording_id": recording_id,
                    },
                )
                return

            logger.info(
                "Processing meeting notes completion",
                extra={
                    "plugin": self.name,
                    "recording_id": recording_id,
                    "output_path": output_path,
                },
            )

            # Send notification
            await self._send_notification(recording_id, output_path)

            # Auto-open if configured
            config_dict = self.config.config or {}
            auto_open = config_dict.get("auto_open_notes", False)
            if auto_open:
                await self._open_notes_file(output_path)

            # Record notification in database
            if self.db:
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO notifications
                        (recording_id, notification_type)
                        VALUES (?, ?)
                        """,
                        (recording_id, "meeting_notes_complete"),
                    )
                    conn.commit()

            # Emit completion event
            if self.event_bus:
                completion_event = Event.create(
                    name="desktop_notification.completed",
                    data={
                        # Preserve original event data
                        "recording": data.get("data", {}).get("recording", {}),
                        "noise_reduction": data.get("data", {}).get("noise_reduction", {}),
                        "transcription": data.get("data", {}).get("transcription", {}),
                        "meeting_notes": data.get("data", {}).get("meeting_notes", {}),
                        # Add current event data
                        "desktop_notification": {
                            "status": "completed",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "notification_type": "terminal-notifier",
                            "settings": {"auto_open_notes": auto_open},
                            "recording_id": recording_id,
                            "output_path": output_path,
                        }
                    },
                    correlation_id=(
                        event_data.context.correlation_id
                        if isinstance(event_data, (Event, RecordingEvent)) and hasattr(event_data, 'context') and hasattr(event_data.context, 'correlation_id')
                        else data.get('context', {}).get('correlation_id')
                        if isinstance(event_data, dict)
                        else str(uuid.uuid4())
                    ),
                    source_plugin=self.__class__.__name__,
                    priority=EventPriority.NORMAL
                )
                logger.debug(
                    "Publishing completion event",
                    extra={
                        "plugin": self.name,
                        "event_name": completion_event.name,
                        "recording_id": recording_id,
                        "output_path": output_path,
                    },
                )
                await self.event_bus.publish(completion_event)

        except Exception as e:
            error_msg = f"Failed to handle meeting notes completion: {e!s}"
            logger.error(
                error_msg,
                extra={
                    "plugin": self.name,
                    "recording_id": recording_id  # type: ignore
                    if "recording_id" in locals()
                    else "unknown",
                    "error": str(e),
                },
                exc_info=True,
            )

            if self.event_bus and "recording_id" in locals():
                # Emit error event
                error_event = Event.create(
                    name="desktop_notification.error",
                    data={
                        # Preserve original event data
                        "recording": data.get("data", {}).get("recording", {}),
                        "noise_reduction": data.get("data", {}).get("noise_reduction", {}),
                        "transcription": data.get("data", {}).get("transcription", {}),
                        "meeting_notes": data.get("data", {}).get("meeting_notes", {}),
                        # Add current event data
                        "desktop_notification": {
                            "status": "error",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "error": str(e),
                            "recording_id": recording_id
                        }
                    },
                    correlation_id=(
                        event_data.context.correlation_id
                        if isinstance(event_data, (Event, RecordingEvent)) and hasattr(event_data, 'context') and hasattr(event_data.context, 'correlation_id')
                        else data.get('context', {}).get('correlation_id')
                        if isinstance(event_data, dict)
                        else str(uuid.uuid4())
                    ),
                    source_plugin=self.__class__.__name__,
                    priority=EventPriority.NORMAL
                )
                await self.event_bus.publish(error_event)

    async def _send_notification(self, recording_id: str, notes_path: str) -> None:
        """Send desktop notification"""
        try:
            title = "Meeting Notes Ready"
            message = f"Meeting notes for recording {recording_id} are ready"

            # Use terminal-notifier on macOS
            if os.uname().sysname == "Darwin":
                subprocess.run(
                    [
                        "terminal-notifier",
                        "-title",
                        title,
                        "-message",
                        message,
                        "-open",
                        f"file://{notes_path}",
                    ],
                    check=True,
                )
            else:
                # Fallback for other platforms
                logger.warning(
                    "Desktop notifications not implemented for this platform",
                    extra={"plugin": self.name, "platform": os.uname().sysname},
                )
        except subprocess.CalledProcessError as e:
            logger.error(
                "Failed to send notification",
                extra={
                    "plugin": self.name,
                    "error": str(e),
                    "recording_id": recording_id,
                },
            )
            raise

    async def _open_notes_file(self, notes_path: str) -> None:
        """Open notes file with default application"""
        try:
            if os.uname().sysname == "Darwin":
                subprocess.run(["open", notes_path], check=True)
            else:
                logger.warning(
                    "Auto-open not implemented for this platform",
                    extra={"plugin": self.name, "platform": os.uname().sysname},
                )
        except subprocess.CalledProcessError as e:
            logger.error(
                "Failed to open notes file",
                extra={"plugin": self.name, "error": str(e), "notes_path": notes_path},
            )
            raise

    async def _init_database(self) -> None:
        """Initialize database tables"""
        if not self.db or self._db_initialized:
            return

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id TEXT NOT NULL,
                    notification_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
            self._db_initialized = True
