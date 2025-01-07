"""Desktop notifier plugin for sending system notifications."""

import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import uuid
from typing import Any

from app.plugins.base import PluginBase, PluginConfig
from app.plugins.events.bus import EventBus
from app.plugins.events.models import Event
from app.models.recording.events import RecordingEvent, EventContext
from app.models.database import DatabaseManager
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
                "Initializing desktop notifier plugin",
                extra={
                    "plugin": self.name
                }
            )

            # Initialize database
            self.db = await DatabaseManager.get_instance()

            # Subscribe to meeting notes completed event only
            await self.event_bus.subscribe(
                "meeting_notes.completed",
                self.handle_meeting_notes_completed
            )

            logger.info(
                "Desktop notifier plugin initialized",
                extra={
                    "plugin": self.name,
                    "subscribed_events": ["meeting_notes.completed"],
                    "handler": "handle_meeting_notes_completed"
                }
            )

        except Exception as e:
            logger.error(
                "Failed to initialize plugin",
                extra={
                    "plugin": self.name,
                    "error": str(e)
                },
                exc_info=True
            )
            raise

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        if not self.event_bus:
            return

        # Unsubscribe from events
        await self.event_bus.unsubscribe(
            "meeting_notes.completed",
            self.handle_meeting_notes_completed
        )

        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)

        logger.info(
            "Desktop notifier plugin shutdown",
            extra={
                "plugin": self.name
            }
        )

    async def handle_meeting_notes_completed(self, event_data: EventData) -> None:
        """Handle meeting notes completed event"""
        try:
            logger.debug(
                "Received meeting notes completed event",
                extra={
                    "plugin": self.name,
                    "event_data": str(event_data),
                    "event_data_type": type(event_data).__name__
                }
            )

            if isinstance(event_data, dict):
                recording_id = event_data.get("recording_id", "unknown")
                output_path = event_data.get("output_path")

                logger.debug(
                    "Extracted event data",
                    extra={
                        "plugin": self.name,
                        "recording_id": recording_id,
                        "output_path": output_path
                    }
                )

                if output_path:
                    # Send notification
                    await self._send_notification(recording_id, output_path)

                    # Auto-open if configured
                    config_dict = self.config.config or {}
                    auto_open = config_dict.get("auto_open_notes", False)
                    logger.debug(
                        "Auto-open setting",
                        extra={
                            "plugin": self.name,
                            "auto_open": auto_open,
                            "config": str(config_dict)
                        }
                    )
                    
                    if auto_open:
                        await self._open_notes_file(output_path)

                    # Record notification in database
                    with self.db.get_connection() as conn:
                        conn.execute(
                            """
                            INSERT INTO notifications 
                            (recording_id, notification_type) 
                            VALUES (?, ?)
                            """,
                            (recording_id, "meeting_notes_complete")
                        )
                        conn.commit()

                    # Emit completion event with proper event structure
                    completion_event = Event(
                        name="desktop_notification.completed",
                        data={
                            "recording_id": recording_id,
                            "output_path": output_path,
                            "current_event": {
                                "desktop_notification": {
                                    "status": "completed",
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "notification_type": "terminal-notifier",
                                    "settings": {
                                        "auto_open_notes": auto_open
                                    }
                                }
                            },
                            "event_history": {
                                "meeting_notes": event_data.get("data", {}).get("current_event", {}).get("meeting_notes", {}),
                                "transcription": event_data.get("data", {}).get("event_history", {}).get("transcription", {}),
                                "recording": event_data.get("data", {}).get("event_history", {}).get("recording", {})
                            }
                        },
                        correlation_id=str(uuid.uuid4()),
                        source_plugin=self.name,
                        metadata=event_data.get("metadata", {})
                    )

                    logger.debug(
                        "Publishing completion event",
                        extra={
                            "plugin": self.name,
                            "event": str(completion_event)
                        }
                    )
                    await self.event_bus.publish(completion_event)
                else:
                    logger.warning(
                        "No output path in event data",
                        extra={
                            "plugin": self.name,
                            "event_data": str(event_data)
                        }
                    )

        except Exception as e:
            error_msg = f"Failed to handle meeting notes completion: {str(e)}"
            logger.error(
                error_msg,
                extra={
                    "plugin": self.name,
                    "recording_id": recording_id if "recording_id" in locals() else "unknown",
                    "error": str(e)
                },
                exc_info=True
            )
            
            if self.event_bus and "recording_id" in locals():
                # Emit error event with preserved chain
                event = RecordingEvent(
                    recording_timestamp=datetime.utcnow().isoformat(),
                    recording_id=recording_id,
                    event="desktop_notification.error",
                    data={
                        "recording_id": recording_id,
                        "desktop_notification": {
                            "status": "error",
                            "timestamp": datetime.utcnow().isoformat(),
                            "error": str(e)
                        },
                        # Preserve previous event data
                        "meeting_notes": event_data.get("data", {}).get("current_event", {}).get("meeting_notes", {}),
                        "transcription": event_data.get("data", {}).get("event_history", {}).get("transcription", {}),
                        "recording": event_data.get("data", {}).get("event_history", {}).get("recording", {})
                    },
                    context=EventContext(
                        correlation_id=str(uuid.uuid4()),
                        source_plugin=self.name
                    )
                )
                await self.event_bus.publish(event)

    async def _send_notification(self, recording_id: str, notes_path: str) -> None:
        """Send desktop notification"""
        try:
            title = "Meeting Notes Ready"
            message = f"Meeting notes for recording {recording_id} are ready"
            
            # Use terminal-notifier on macOS
            if os.uname().sysname == "Darwin":
                subprocess.run([
                    "terminal-notifier",
                    "-title", title,
                    "-message", message,
                    "-open", f"file://{notes_path}"
                ], check=True)
            else:
                # Fallback for other platforms
                logger.warning(
                    "Desktop notifications not implemented for this platform",
                    extra={
                        "plugin": self.name,
                        "platform": os.uname().sysname
                    }
                )
        except subprocess.CalledProcessError as e:
            logger.error(
                "Failed to send notification",
                extra={
                    "plugin": self.name,
                    "error": str(e),
                    "recording_id": recording_id
                }
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
                    extra={
                        "plugin": self.name,
                        "platform": os.uname().sysname
                    }
                )
        except subprocess.CalledProcessError as e:
            logger.error(
                "Failed to open notes file",
                extra={
                    "plugin": self.name,
                    "error": str(e),
                    "notes_path": notes_path
                }
            )
            raise

    async def _init_database(self) -> None:
        """Initialize database tables."""
        if not self.db:
            return

        # Create tables using the connection from our db instance
        with self.db.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id TEXT NOT NULL,
                    notification_type TEXT NOT NULL,
                    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (recording_id) REFERENCES recordings(recording_id)
                )
            """)
            conn.commit()

    def _update_task_status(
        self, notes_id: str, status: str, error_message: str | None = None
    ) -> None:
        """Update the status of a notification task in the database"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            update_values = [
                status,
                error_message,
                datetime.utcnow().isoformat(),
                notes_id,
            ]
            cursor.execute(
                """
                UPDATE desktop_notification_tasks
                SET status = ?, error_message = ?, updated_at = ?
                WHERE notes_id = ?
            """,
                update_values,
            )
            conn.commit()

    def _create_task_record(self, notes_id: str, notes_path: str) -> None:
        """Create a new notification task record"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO desktop_notification_tasks
                (notes_id, status, notes_path)
                VALUES (?, ?, ?)
            """,
                (notes_id, "pending", notes_path),
            )
            conn.commit()
