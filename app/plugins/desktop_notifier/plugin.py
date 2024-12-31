import logging
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from app.models.database import DatabaseManager
from app.models.recording.events import (
    RecordingEndRequest,
    RecordingEvent,
    RecordingStartRequest,
)
from app.plugins.base import PluginBase, PluginConfig

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
        """Initialize plugin"""
        if not self.event_bus:
            return

        # Initialize database table
        await self._init_database()

        # Initialize thread pool executor
        config_dict: dict[str, Any] = self.config.config or {}
        max_workers: int = config_dict.get("max_concurrent_tasks", 4)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Subscribe to meeting notes completed event
        await self.event_bus.subscribe(
            "meeting_notes.completed", self.handle_meeting_notes_completed
        )

        self.logger.info(
            "DesktopNotifierPlugin initialized successfully",
            extra={
                "plugin": "desktop_notifier",
                "max_workers": max_workers,
                "auto_open_notes": bool(config_dict.get("auto_open_notes", False)),
                "db_initialized": self._db_initialized,
            },
        )

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        if not self.event_bus:
            return

        # Unsubscribe from events
        await self.event_bus.unsubscribe(
            "meeting_notes.completed", self.handle_meeting_notes_completed
        )

        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)

    async def handle_meeting_notes_completed(self, event_data: EventData) -> None:
        """Handle meeting notes completed event"""
        try:
            if isinstance(event_data, dict):
                recording_id = event_data.get("recording_id", "unknown")
                output_path = event_data.get("output_path")
            else:
                # Handle RecordingEvent, RecordingStartRequest, RecordingEndRequest
                recording_id = getattr(event_data, "recording_id", "unknown")
                output_path = getattr(event_data, "output_path", None)

            if not output_path or not os.path.exists(output_path):
                logger.error(
                    "Meeting notes file not found",
                    extra={
                        "recording_id": recording_id,
                        "output_path": output_path,
                    },
                )
                return

            # Send notification
            await self._send_notification(recording_id, output_path)

            # Auto-open notes if configured
            if self.config.config and self.config.config.get("auto_open_notes"):
                await self._open_notes_file(output_path)

            # Emit completion event
            if self.event_bus:
                notification_data = {
                    "type": "desktop_notification.completed",
                    "recording_id": recording_id,
                    "status": "completed",
                    "output_path": output_path,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                await self.event_bus.emit(notification_data)

        except Exception as e:
            logger.error(
                f"Failed to handle meeting notes completion: {e}",
                extra={
                    "recording_id": (
                        recording_id if "recording_id" in locals() else "unknown"
                    )
                },
                exc_info=True,
            )

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
        """Initialize database table for tracking notification state"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS desktop_notification_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    notes_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    notes_path TEXT NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()
        self._db_initialized = True

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
