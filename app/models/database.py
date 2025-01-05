from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
import uuid
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from sqlite3 import Connection
from typing import Any, TypeVar, cast

T = TypeVar("T")

logger = logging.getLogger(__name__)

class DatabaseManager:
    _instance = None
    _lock = asyncio.Lock()
    _local = threading.local()
    _executor = ThreadPoolExecutor(max_workers=4)

    def __init__(self) -> None:
        if DatabaseManager._instance is not None:
            raise RuntimeError("Use DatabaseManager.get_instance() instead")

        # Get the project root directory (two levels up from this file)
        root_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        # Ensure the data directory exists in the root
        data_dir = root_dir / "data"
        data_dir.mkdir(exist_ok=True)

        self.db_path = str(data_dir / "panotti.db")
        self._req_id = str(uuid.uuid4())
        self._init_db()

    def __enter__(self) -> Connection:
        return self.get_connection()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection

    @classmethod
    async def get_instance(cls) -> 'DatabaseManager':
        """Get singleton instance of DatabaseManager."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance.initialize()
            return cls._instance

    async def initialize(self) -> None:
        """Initialize database connection."""
        logger.info(
            "Initializing database",
            extra={
                "req_id": self._req_id,
                "db_path": self.db_path
            }
        )

        try:
            # Initialize connection in thread pool
            def _init() -> None:
                conn = self.get_connection()
                conn.execute("PRAGMA foreign_keys = ON")
                
            await asyncio.get_event_loop().run_in_executor(self._executor, _init)
            logger.info(
                "Database initialized successfully",
                extra={
                    "req_id": self._req_id,
                    "db_path": self.db_path
                }
            )
        except Exception as e:
            logger.error(
                "Failed to initialize database",
                extra={
                    "req_id": self._req_id,
                    "db_path": self.db_path,
                    "error": str(e)
                }
            )
            raise

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self.get_connection() as conn:
            # Create json_patch function
            conn.create_function(
                "json_patch",
                2,
                lambda old, patch: json.dumps(
                    {**(json.loads(old) if old else {}), **(json.loads(patch))}
                ),
            )

            # Create recording_events table with all fields
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recording_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_timestamp TEXT NOT NULL,
                    metadata TEXT,  -- JSON object with event metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Create recordings table to track all recordings
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recordings (
                    recording_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,  -- 'active', 'completed', etc.
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Create plugin_tasks table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS plugin_tasks (
                    recording_id TEXT NOT NULL,
                    plugin_name TEXT NOT NULL,
                    status TEXT NOT NULL,  -- 'processing', 'completed', 'failed'
                    input_paths TEXT,  -- Comma-separated list of input file paths
                    output_paths TEXT,  -- Comma-separated list of output file paths
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (recording_id, plugin_name),
                    FOREIGN KEY (recording_id) REFERENCES recordings(recording_id)
                )
                """
            )

            # Create indexes
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_recording_events_recording_id
                ON recording_events(recording_id)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_plugin_tasks_recording_id
                ON plugin_tasks(recording_id)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_plugin_tasks_plugin_name
                ON plugin_tasks(plugin_name)
                """
            )

            # Run any pending migrations
            self._run_migrations(conn)

            conn.commit()

    def _run_migrations(self, conn: Connection) -> None:
        """Run any pending database migrations."""
        logger.info(
            "Running database migrations",
            extra={
                "req_id": self._req_id,
                "db_path": self.db_path
            }
        )

        try:
            # Create migrations table if it doesn't exist
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Get list of applied migrations
            applied = {row[0] for row in conn.execute("SELECT name FROM migrations")}

            # Get migrations directory
            migrations_dir = Path(__file__).parent / "migrations"
            migrations_dir.mkdir(exist_ok=True)

            # Run any new migrations in order
            for migration_file in sorted(migrations_dir.glob("*.sql")):
                if migration_file.stem not in applied:
                    logger.info(
                        "Applying migration",
                        extra={
                            "req_id": self._req_id,
                            "migration": migration_file.name
                        }
                    )
                    
                    with migration_file.open() as f:
                        conn.executescript(f.read())
                    conn.execute(
                        "INSERT INTO migrations (name) VALUES (?)", (migration_file.stem,)
                    )
                    conn.commit()

            logger.info(
                "Database migrations complete",
                extra={
                    "req_id": self._req_id,
                    "applied_count": len(applied)
                }
            )
        except Exception as e:
            logger.error(
                "Failed to apply migrations",
                extra={
                    "req_id": self._req_id,
                    "error": str(e)
                }
            )
            raise

    def get_connection(self, name: str = "default") -> sqlite3.Connection:
        """Get a database connection by name.

        Args:
            name: Connection name identifier, defaults to "default"

        Returns:
            sqlite3.Connection: SQLite database connection object
        """
        if not hasattr(self._local, "connection"):
            connection = sqlite3.connect(self.db_path, check_same_thread=False)
            # Enable foreign keys
            connection.execute("PRAGMA foreign_keys = ON")
            # Row factory for dictionary-like access
            connection.row_factory = sqlite3.Row
            self._local.connection = connection
        return cast(sqlite3.Connection, self._local.connection)

    async def execute(self, sql: str, parameters: tuple = ()) -> None:
        """Execute a SQL query asynchronously."""
        logger.info(
            "Executing SQL query",
            extra={
                "req_id": self._req_id,
                "sql": sql,
                "parameters": parameters
            }
        )

        def _execute() -> None:
            conn = self.get_connection()
            conn.execute(sql, parameters)
            conn.commit()

        await asyncio.get_event_loop().run_in_executor(self._executor, _execute)

    async def execute_fetchall(
        self, sql: str, parameters: tuple = ()
    ) -> list[sqlite3.Row]:
        """Execute a SQL query and fetch all results asynchronously."""

        logger.info(
            "Executing SQL query and fetching results",
            extra={
                "req_id": self._req_id,
                "sql": sql,
                "parameters": parameters
            }
        )

        def _execute_fetchall() -> list[sqlite3.Row]:
            conn = self.get_connection()
            cursor = conn.execute(sql, parameters)
            return cursor.fetchall()

        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _execute_fetchall
        )

    async def insert(self, sql: str, parameters: tuple = ()) -> None:
        """Insert a record into the database."""

        logger.info(
            "Inserting record into database",
            extra={
                "req_id": self._req_id,
                "sql": sql,
                "parameters": parameters
            }
        )

        def _insert() -> None:
            conn = self.get_connection()
            conn.execute(sql, parameters)
            conn.commit()

        await asyncio.get_event_loop().run_in_executor(self._executor, _insert)

    async def fetch_one(self, sql: str, parameters: tuple = ()) -> sqlite3.Row | None:
        """Fetch a single row from the database.

        Args:
            sql: SQL query to execute
            parameters: Query parameters

        Returns:
            sqlite3.Row | None: Single row result or None if no results
        """

        logger.info(
            "Fetching single row from database",
            extra={
                "req_id": self._req_id,
                "sql": sql,
                "parameters": parameters
            }
        )

        def _fetch_one() -> sqlite3.Row | None:
            conn = self.get_connection()
            cursor = conn.execute(sql, parameters)
            result = cursor.fetchone()
            return cast(sqlite3.Row | None, result)

        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _fetch_one
        )

    async def fetch_all(self, sql: str, parameters: tuple = ()) -> list[sqlite3.Row]:
        """Fetch all records from the database."""

        logger.info(
            "Fetching all records from database",
            extra={
                "req_id": self._req_id,
                "sql": sql,
                "parameters": parameters
            }
        )

        def _fetch_all() -> list[sqlite3.Row]:
            conn = self.get_connection()
            cursor = conn.execute(sql, parameters)
            return cursor.fetchall()

        return await asyncio.get_event_loop().run_in_executor(
            self._executor, _fetch_all
        )

    async def commit(self) -> None:
        """Commit the current transaction asynchronously."""

        logger.info(
            "Committing transaction",
            extra={
                "req_id": self._req_id
            }
        )

        def _commit() -> None:
            conn = self.get_connection()
            conn.commit()

        await asyncio.get_event_loop().run_in_executor(self._executor, _commit)

    async def close(self) -> None:
        """Close all database connections."""
        try:
            # Close thread pool executor
            self._executor.shutdown(wait=True)
            
            # Close any remaining connections
            if hasattr(self._local, "connection"):
                def _close() -> None:
                    if hasattr(self._local, "connection"):
                        self._local.connection.close()
                        del self._local.connection
                
                await asyncio.get_event_loop().run_in_executor(None, _close)
                
            logger.info(
                "Database connections closed",
                extra={
                    "req_id": self._req_id,
                    "db_path": self.db_path
                }
            )
        except Exception as e:
            logger.error(
                "Error closing database connections",
                extra={
                    "req_id": self._req_id,
                    "error": str(e)
                }
            )
            raise

    def close_connections(self) -> None:
        """Close all connections - useful for cleanup."""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection

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


@contextmanager
def get_db() -> Generator[Connection, None, None]:
    """Get a database connection from the manager."""
    db = DatabaseManager.get_instance()
    try:
        yield db.get_connection()
    finally:
        if hasattr(db._local, "connection"):
            db._local.connection.close()
            del db._local.connection


async def get_db_async() -> DatabaseManager:
    """Get a database manager instance asynchronously."""
    return DatabaseManager.get_instance()
