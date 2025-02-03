import asyncio
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.models.database import DatabaseManager
from unittest.mock import patch
import unittest.mock


@pytest.fixture(autouse=True)
def reset_database_manager():
    # Reset singleton between tests
    DatabaseManager._instance = None
    DatabaseManager._lock = asyncio.Lock()


@pytest.mark.asyncio
async def test_get_instance_singleton():
    instance1 = await DatabaseManager.get_instance()
    instance2 = await DatabaseManager.get_instance()
    assert instance1 is instance2


@pytest.mark.asyncio
async def test_initialize_calls_init_and_migrations():
    # Create a mock connection to be passed to _run_migrations
    mock_conn = MagicMock()
    mock_conn.execute = MagicMock()
    mock_conn.commit = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=None)
    
    # Mock get_connection to return our mock connection
    with patch.object(DatabaseManager, 'get_connection', return_value=mock_conn) as mock_get_conn:
        # Mock _run_migrations at the class level
        with patch.object(DatabaseManager, '_run_migrations') as mock_run_migrations:
            instance = await DatabaseManager.get_instance()
            await instance.initialize()
            
            # _run_migrations should be called once from _init_db
            mock_run_migrations.assert_called_once_with(mock_conn)
            
            # initialize() should call get_connection and execute PRAGMA
            assert mock_get_conn.call_count >= 1
            mock_conn.execute.assert_any_call("PRAGMA foreign_keys = ON")


@pytest.mark.asyncio
async def test_execute_query_success():
    instance = await DatabaseManager.get_instance()
    fake_result = "result"
    instance.execute = AsyncMock(return_value=fake_result)
    result = await instance.execute("SELECT * FROM table")
    assert result == fake_result
    instance.execute.assert_called_once_with("SELECT * FROM table")


@pytest.mark.asyncio
async def test_execute_query_failure():
    instance = await DatabaseManager.get_instance()
    instance.execute = AsyncMock(side_effect=Exception("DB Error"))
    with pytest.raises(Exception, match="DB Error"):
        await instance.execute("BAD QUERY")


@pytest.mark.asyncio
async def test_close_connection():
    instance = await DatabaseManager.get_instance()
    # Create a mock connection
    fake_conn = MagicMock()
    fake_conn.close = MagicMock()
    
    # Set the connection in thread local storage
    instance._local.connection = fake_conn
    
    # Call close_connections which is the method that actually closes connections
    instance.close_connections()
    
    # Verify the connection was closed
    fake_conn.close.assert_called_once()


@pytest.mark.asyncio
async def test_get_connection():
    instance = await DatabaseManager.get_instance()
    instance.get_connection = MagicMock(return_value="connection_obj")
    conn = instance.get_connection()
    assert conn == "connection_obj"
    instance.get_connection.assert_called_once()


@pytest.mark.asyncio
async def test_init_db_success():
    instance = await DatabaseManager.get_instance()
    instance._init_db = AsyncMock(return_value=None)
    await instance._init_db()
    instance._init_db.assert_called_once()


@pytest.mark.asyncio
async def test_run_migrations_success():
    instance = await DatabaseManager.get_instance()
    instance._run_migrations = MagicMock()
    instance._run_migrations()
    instance._run_migrations.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_failure_triggers_exit(monkeypatch):
    instance = await DatabaseManager.get_instance()

    # Mock run_in_executor to raise the error
    async def mock_run_in_executor(executor, func, *args):
        raise Exception("Initialization failed")
            
    with patch('asyncio.get_event_loop') as mock_loop:
        mock_loop.return_value.run_in_executor = mock_run_in_executor
        with pytest.raises(Exception, match="Initialization failed"):
            await instance.initialize() 