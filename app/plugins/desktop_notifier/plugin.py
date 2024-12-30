import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import logging
import subprocess

from app.plugins.base import PluginBase
from app.plugins.events.models import Event, EventContext, EventPriority
from app.models.database import DatabaseManager

logger = logging.getLogger(__name__)

class DesktopNotifierPlugin(PluginBase):
    """Plugin for sending desktop notifications when meeting notes are completed"""
    
    def __init__(self, config, event_bus=None):
        super().__init__(config, event_bus)
        self._executor = None
        self._processing_lock = threading.Lock()
        self._db_initialized = False
        
    async def _initialize(self) -> None:
        """Initialize plugin"""
        # Initialize database table
        await self._init_database()
        
        # Initialize thread pool executor
        max_workers = getattr(self.config, "max_concurrent_tasks", 4)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Subscribe to meeting notes completed event
        self.event_bus.subscribe("meeting_notes.completed", self.handle_meeting_notes_completed)
        
        self.logger.info(
            "DesktopNotifierPlugin initialized successfully",
            extra={
                "plugin": "desktop_notifier",
                "max_workers": max_workers,
                "auto_open_notes": getattr(self.config, "auto_open_notes", False),
                "db_initialized": self._db_initialized
            }
        )
        
    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        # Unsubscribe from events
        self.event_bus.unsubscribe("meeting_notes.completed", self.handle_meeting_notes_completed)
        
        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)
            
        self.logger.info("Desktop notifier plugin shutdown")
        
    async def _init_database(self) -> None:
        """Initialize database table for tracking notification state"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS desktop_notification_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    notes_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    notes_path TEXT NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        self._db_initialized = True
        
    def _update_task_status(self, notes_id: str, status: str, 
                           error_message: Optional[str] = None) -> None:
        """Update the status of a notification task in the database"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            update_values = [
                status,
                error_message,
                datetime.utcnow().isoformat(),
                notes_id
            ]
            cursor.execute('''
                UPDATE desktop_notification_tasks
                SET status = ?, error_message = ?, updated_at = ?
                WHERE notes_id = ?
            ''', update_values)
            conn.commit()

    def _create_task_record(self, notes_id: str, notes_path: str) -> None:
        """Create a new notification task record"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO desktop_notification_tasks 
                (notes_id, status, notes_path)
                VALUES (?, ?, ?)
            ''', (notes_id, "pending", notes_path))
            conn.commit()

    def notify_and_open(self, title: str, message: str, file_path: str) -> None:
        """Send notification and optionally open file"""
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File not found: {abs_path}")

        # Send notification
        subprocess.run([
            'terminal-notifier',
            '-title', title,
            '-message', message,
            '-sound', 'default'
        ])

        # Open file if configured
        auto_open = getattr(self.config, "auto_open_notes", False)
        if auto_open:
            os.system(f'open "{abs_path}"')

    def _process_notification(self, notes_id: str, notes_path: str, 
                            original_event: Event) -> None:
        """Process notification in worker thread"""
        try:
            self.logger.info(
                "Processing notification",
                extra={
                    "notes_id": notes_id,
                    "notes_path": notes_path,
                    "correlation_id": original_event.context.correlation_id
                }
            )

            # Update status to processing
            self._update_task_status(notes_id, "processing")

            # Send notification and optionally open file
            self.notify_and_open(
                "Meeting Notes Ready",
                "Your meeting notes are ready to view.",
                notes_path
            )

            # Update status to completed
            self._update_task_status(notes_id, "completed")

            # Emit completion event
            completion_event = Event(
                type="desktop_notification.completed",
                payload=original_event.payload,
                context=EventContext(
                    correlation_id=original_event.context.correlation_id,
                    priority=EventPriority.LOW
                )
            )
            
            # Schedule event emission in event loop
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    self.event_bus.emit(completion_event)
                )
            )

        except Exception as e:
            error_msg = str(e)
            self.logger.error(
                "Error processing notification",
                extra={
                    "notes_id": notes_id,
                    "error": error_msg,
                    "correlation_id": original_event.context.correlation_id
                },
                exc_info=True
            )
            self._update_task_status(notes_id, "error", error_msg)

    async def handle_meeting_notes_completed(self, event: Event) -> None:
        """Handle meeting notes completed event"""
        try:
            # Extract data from event payload
            recording_id = event.payload.get("recording_id")
            notes_path = event.payload.get("meeting_notes_path")
            
            if not notes_path or not os.path.exists(notes_path):
                raise ValueError(f"Invalid notes path: {notes_path}")

            self.logger.info(
                "Received meeting notes completion",
                extra={
                    "recording_id": recording_id,
                    "correlation_id": event.context.correlation_id,
                    "notes_path": notes_path
                }
            )

            # Create task record
            self._create_task_record(recording_id, notes_path)

            # Process in thread pool
            self._executor.submit(
                self._process_notification,
                recording_id,
                notes_path,
                event
            )

        except Exception as e:
            self.logger.error(
                "Error handling meeting notes completion",
                extra={
                    "error": str(e),
                    "event_id": event.id,
                    "correlation_id": event.context.correlation_id,
                    "payload": event.payload
                },
                exc_info=True
            )
