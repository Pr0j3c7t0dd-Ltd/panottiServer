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

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _init_db(self):
        """Initialize the database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add any other tables as needed
            conn.commit()

    @contextmanager
    def get_connection(self):
        """Thread-safe connection management"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.row_factory = sqlite3.Row
        
        try:
            yield self._local.connection
        except Exception as e:
            self._local.connection.rollback()
            raise e
        else:
            self._local.connection.commit()

    def close_connections(self):
        """Close all connections - useful for cleanup"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            del self._local.connection

# Global accessor function
def get_db():
    return DatabaseManager.get_instance()
