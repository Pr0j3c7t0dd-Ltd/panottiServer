import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import sqlite3
import pytest
import logging
from app.models.database import DatabaseManager

class TestDatabaseManager(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create mock cursor and connection
        self.mock_cursor = MagicMock()
        self.mock_conn = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor
        
        # Make close a synchronous operation since it's called in cleanup
        self.mock_conn.close = MagicMock()
        self.mock_conn.commit = MagicMock()  # Changed from AsyncMock to MagicMock since it runs in thread pool
        
        # Create database manager instance with synchronous close
        self.db_manager = DatabaseManager()
        self.db_manager.close_connections = MagicMock()  # Make this synchronous
        self.db_manager._local.connection = self.mock_conn
        
        # Set up mock cursor methods
        mock_row = {"value": "test value"}
        self.mock_cursor.fetchone = AsyncMock(return_value=mock_row)
        self.mock_cursor.fetchall = AsyncMock(return_value=[mock_row])
        self.mock_cursor.execute = AsyncMock(return_value=self.mock_cursor)
        
        # Set up mock connection
        self.mock_conn.cursor = AsyncMock(return_value=self.mock_cursor)
        self.mock_conn.commit = MagicMock()  # Changed from AsyncMock to MagicMock since it runs in thread pool
        
        # Reset singleton instance
        DatabaseManager._instance = None
        
        # Mock database initialization
        with patch("app.models.database.DatabaseManager._create_connection") as mock_create_conn:
            mock_create_conn.return_value = self.mock_conn
            self.db_manager = await DatabaseManager.get_instance_async()
            self.db_manager._conn = self.mock_conn
            
    async def asyncTearDown(self):
        if hasattr(self, 'db_manager'):
            # Reset close_connections to original before teardown
            if hasattr(self.db_manager, '_original_close_connections'):
                self.db_manager.close_connections = self.db_manager._original_close_connections
            await self.db_manager.close()
            
    async def test_singleton_pattern(self):
        instance1 = await DatabaseManager.get_instance_async()
        instance2 = await DatabaseManager.get_instance_async()
        self.assertIs(instance1, instance2)
        
    async def test_initialization_error(self):
        DatabaseManager._instance = None
        
        async def raise_error(*args, **kwargs):
            raise sqlite3.Error("Test error")
            
        with patch("app.models.database.DatabaseManager._create_connection") as mock_create_conn:
            mock_create_conn.side_effect = raise_error
            with pytest.raises(sqlite3.Error):
                await DatabaseManager.get_instance_async()
                
    async def test_migration_error(self):
        DatabaseManager._instance = None
        
        async def raise_error(*args, **kwargs):
            raise Exception("Migration error")
            
        with patch("app.models.database.DatabaseManager._create_connection") as mock_create_conn, \
             patch("app.models.database.DatabaseManager._run_migrations") as mock_run_migrations:
            mock_create_conn.return_value = self.mock_conn
            mock_run_migrations.side_effect = raise_error
            with pytest.raises(Exception):
                await DatabaseManager.get_instance_async()
                
    async def test_execute_with_error(self):
        async def raise_error(*args, **kwargs):
            raise sqlite3.Error("Test error")
            
        self.mock_cursor.execute.side_effect = raise_error
        with pytest.raises(sqlite3.Error):
            await self.db_manager.execute("SELECT * FROM test")
            
    async def test_fetch_operations(self):
        # Test fetch_one
        row = await self.db_manager.fetch_one("SELECT * FROM test")
        self.assertEqual(row["value"], "test value")
        
        # Test fetch_all
        rows = await self.db_manager.fetch_all("SELECT * FROM test")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], "test value")
        
    async def test_cleanup_with_error(self):
        def raise_error(*args, **kwargs):
            raise Exception("Cleanup error")

        # Mock the module logger and close_connections
        with patch("app.models.database.logger.error") as mock_logger, \
             patch("asyncio.timeout") as mock_timeout:
            
            # Set up the mock timeout context manager
            mock_timeout.return_value.__aenter__ = AsyncMock()
            mock_timeout.return_value.__aexit__ = AsyncMock(return_value=False)  # Don't suppress exceptions
            
            # Set up executor
            self.db_manager._executor = MagicMock()
            self.db_manager._executor.shutdown = MagicMock()
            
            # Set up _req_id and _shutting_down
            self.db_manager._req_id = None
            self.db_manager._shutting_down = False
            
            # Set up lock context manager
            self.db_manager._lock = AsyncMock()
            self.db_manager._lock.__aenter__ = AsyncMock()
            self.db_manager._lock.__aexit__ = AsyncMock(return_value=False)  # Don't suppress exceptions
            
            # Save original close_connections and replace with mock
            self.db_manager._original_close_connections = self.db_manager.close_connections
            self.db_manager.close_connections = MagicMock(side_effect=raise_error)
            
            # Call close and verify error was logged
            try:
                await self.db_manager.close()
            except Exception as e:
                assert str(e) == "Cleanup error"
                mock_logger.assert_called_once_with(
                    "Error during database cleanup",
                    extra={"req_id": None, "error": "Cleanup error"}
                )
            else:
                pytest.fail("Expected exception was not raised")
            
    async def test_close_connections(self):
        await self.db_manager.close()
        self.mock_conn.close.assert_called_once()
        
    async def test_commit_and_close(self):
        # Mock get_connection to return our mock connection
        def get_mock_conn(*args, **kwargs):
            return self.mock_conn
        self.db_manager.get_connection = MagicMock(side_effect=get_mock_conn)
        
        # Mock run_in_executor to run the function synchronously
        async def mock_run_in_executor(executor, func, *args):
            return func(*args)
        
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(side_effect=mock_run_in_executor)
            await self.db_manager.commit()
            
        self.mock_conn.commit.assert_called_once()
        
    async def test_get_db_async_context_manager(self):
        from app.models.database import get_db_async
        with patch("app.models.database.DatabaseManager.get_instance_async", new_callable=AsyncMock) as mock_get_instance:
            mock_get_instance.return_value = self.db_manager
            async with get_db_async() as conn:
                self.assertIsNotNone(conn)
            
    async def test_get_db_context_manager(self):
        from app.models.database import get_db
        
        # Set up the mock connection
        self.mock_conn.close = MagicMock()
        
        # Mock get_connection to return our mock connection
        def get_mock_conn(*args, **kwargs):
            return self.mock_conn
        self.db_manager.get_connection = MagicMock(side_effect=get_mock_conn)
        
        with patch("app.models.database.DatabaseManager.get_instance", return_value=self.db_manager):
            async with get_db() as conn:
                self.assertIsNotNone(conn)
                self.assertEqual(conn, self.mock_conn)
            self.mock_conn.close.assert_called_once() 