import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from sqlite3 import Connection
from typing import Any, Optional


class DatabaseManager:
    _instance: Optional["DatabaseManager"] = None
    _lock = threading.Lock()
    _local = threading.local()

    def __init__(self) -> None:
        if DatabaseManager._instance is not None:
            raise RuntimeError("Use DatabaseManager.get_instance() instead")

        # Get the project root directory (two levels up from this file)
        root_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        # Ensure the data directory exists in the root
        data_dir = root_dir / "data"
        data_dir.mkdir(exist_ok=True)

        self.db_path = str(data_dir / "panotti.db")
        self._init_db()

    def __enter__(self) -> Connection:
        return self.get_connection()

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any | None,
    ) -> None:
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection

    @classmethod
    def get_instance(cls) -> "DatabaseManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _init_db(self) -> None:
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create events table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    recording_timestamp DATETIME NOT NULL,
                    recording_id TEXT NOT NULL,
                    event_title TEXT,
                    event_provider_id TEXT,
                    event_provider TEXT,
                    event_attendees TEXT,  -- JSON array
                    system_label TEXT,     -- Label for system audio source
                    microphone_label TEXT, -- Label for microphone audio source
                    recording_started TEXT, -- ISO8601 timestamp when recording started
                    recording_ended TEXT,   -- ISO8601 timestamp when recording ended
                    metadata_json TEXT,    -- Full JSON payload
                    system_audio_path TEXT,
                    microphone_audio_path TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Add any other tables as needed
            conn.commit()

    def get_active_recordings(self) -> dict[str, str]:
        """Get all active recordings (started but not ended)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT json_extract(data, '$.recordingId') as recording_id,
                       json_extract(data, '$.timestamp') as timestamp
                FROM events
                WHERE type = 'Recording Started'
                AND recording_id NOT IN (
                    SELECT DISTINCT json_extract(data, '$.recordingId')
                    FROM events
                    WHERE type = 'Recording Ended'
                )
            """
            )
            return {row["recording_id"]: row["timestamp"] for row in cursor.fetchall()}

    def get_connection(self, name: str = "default") -> Connection:
        """Get a database connection by name."""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def close_connections(self) -> None:
        """Close all connections - useful for cleanup"""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection


@contextmanager
def get_db():
    db = DatabaseManager.get_instance()
    try:
        yield db
    finally:
        if hasattr(db._local, "connection"):
            db._local.connection.close()
            del db._local.connection
