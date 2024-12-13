import sqlite3
import threading
from contextlib import contextmanager
import os
from pathlib import Path

class DatabaseManager:
    _instance = None
    _lock = threading.Lock()
    _local = threading.local()

    def __init__(self):
        if DatabaseManager._instance is not None:
            raise RuntimeError("Use DatabaseManager.get_instance() instead")
        
        # Get the project root directory (two levels up from this file)
        root_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        # Ensure the data directory exists in the root
        data_dir = root_dir / 'data'
        data_dir.mkdir(exist_ok=True)
        
        self.db_path = str(data_dir / 'panotti.db')
        self._init_db()

    def __enter__(self):
        return self.get_connection()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            del self._local.connection

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _init_db(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    recording_id TEXT NOT NULL,
                    recording_datetime TEXT,
                    event_title TEXT,
                    event_provider_id TEXT,
                    event_provider TEXT,
                    event_attendees TEXT,  -- JSON array
                    metadata_json TEXT,    -- Full JSON payload
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add any other tables as needed
            conn.commit()

    def get_active_recordings(self):
        """Get all active recordings (started but not ended)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT json_extract(data, '$.recordingId') as recording_id,
                       json_extract(data, '$.timestamp') as timestamp
                FROM events 
                WHERE type = 'Recording Started'
                AND recording_id NOT IN (
                    SELECT DISTINCT json_extract(data, '$.recordingId')
                    FROM events
                    WHERE type = 'Recording Ended'
                )
            ''')
            return {row['recording_id']: row['timestamp'] 
                    for row in cursor.fetchall()}

    def get_connection(self):
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def close_connections(self):
        """Close all connections - useful for cleanup"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            del self._local.connection

@contextmanager
def get_db():
    db = DatabaseManager.get_instance()
    try:
        yield db
    finally:
        if hasattr(db._local, 'connection'):
            db._local.connection.close()
            del db._local.connection
