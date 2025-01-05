import logging
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any
import uuid

from app.models.database import DatabaseManager
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


class DesktopNotifierPlugin(PluginBase):
    """Plugin for sending desktop notifications when meeting notes are completed"""

    def __init__(self, config: PluginConfig, event_bus: Any = None) -> None:
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

            # Subscribe to transcription completed event
            await self.event_bus.subscribe(
                "transcription.completed",
                self.handle_transcription_completed
            )

            logger.info(
                "Desktop notifier plugin initialized",
                extra={
                    "plugin": self.name,
                    "subscribed_events": ["transcription.completed"],
                    "handler": "handle_transcription_completed"
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
            "transcription.completed",
            self.handle_transcription_completed
        )

        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)

    async def handle_meeting_notes_completed(self, event_data: EventData) -> None:
        """Handle meeting notes completed event"""
        try:
            if isinstance(event_data, dict):
                recording_id = event_data.get("recording_id", "unknown")
                current_event = event_data.get("current_event", {})
                meeting_notes = current_event.get("meeting_notes", {})
                output_paths = meeting_notes.get("output_paths", {})
                notes_file_path = output_paths.get("notes")

                # Send notification and emit completion event
                event_data = {
                    "recording_id": recording_id,
                    "event_type": "desktop_notification.completed",
                    "current_event": {
                        "desktop_notification": {
                            "status": "completed",
                            "timestamp": datetime.utcnow().isoformat(),
                            "notification_type": "terminal-notifier",
                            "settings": {
                                "auto_open_notes": bool(self.config.config.get("auto_open_notes", False))
                            }
                        }
                    },
                    "event_history": {
                        "meeting_notes": event_data.get("current_event", {}),
                        "transcription": event_data.get("event_history", {}).get("transcription", {}),
                        "noise_reduction": event_data.get("event_history", {}).get("noise_reduction", {}),
                        "recording": event_data.get("event_history", {}).get("recording", {})
                    }
                }

                event = RecordingEvent(
                    recording_timestamp=datetime.utcnow().isoformat(),
                    recording_id=recording_id,
                    event="desktop_notification.completed",
                    data=event_data,
                    context=EventContext(
                        correlation_id=str(uuid.uuid4()),
                        source_plugin=self.name
                    )
                )
                await self.event_bus.publish(event)

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
                        "meeting_notes": event_data.data.get("meeting_notes", {}),
                        "transcription": event_data.data.get("transcription", {}),
                        "noise_reduction": event_data.data.get("noise_reduction", {}),
                        "recording": event_data.data.get("recording", {})
                    },
                    context=event_data.context if hasattr(event_data, "context") else None
                )
                await self.event_bus.publish(event)

    async def _send_notification(self, recording_id: str, output_path: str) -> None:
        """Send notification"""
        abs_path = os.path.abspath(output_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File not found: {abs_path}")

        # Send notification
        subprocess.run(
            [
                "terminal-notifier",
                "-title",
                "Meeting Notes Ready",
                "-message",
                "Your meeting notes are ready to view.",
                "-sound",
                "default",
            ],
            check=False,
        )

    async def _open_notes_file(self, output_path: str) -> None:
        """Open notes file"""
        subprocess.run(["open", output_path], check=True)

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

    async def handle_transcription_completed(self, event_data: EventData) -> None:
        """Handle transcription completed event."""
        try:
            if isinstance(event_data, dict):
                recording_id = event_data.get("recording_id", "unknown")
                current_event = event_data.get("current_event", {})
                transcription = current_event.get("transcription", {})
                output_paths = transcription.get("output_paths", {})
                transcript_path = output_paths.get("transcript")

                if transcript_path:
                    # Send notification
                    await self._send_notification(recording_id, transcript_path)

                    # Auto-open if configured
                    config_dict = self.config.config or {}
                    if config_dict.get("auto_open_notes", False):
                        await self._open_notes_file(transcript_path)

                    # Record notification in database
                    with self.db.get_connection() as conn:
                        conn.execute(
                            """
                            INSERT INTO notifications 
                            (recording_id, notification_type) 
                            VALUES (?, ?)
                            """,
                            (recording_id, "transcription_complete")
                        )
                        conn.commit()

        except Exception as e:
            logger.error(
                "Failed to handle transcription completion",
                extra={
                    "plugin": self.name,
                    "recording_id": recording_id if "recording_id" in locals() else "unknown",
                    "error": str(e)
                },
                exc_info=True
            )
            raise
