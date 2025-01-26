"""Common test fixtures and configuration."""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv

from app.core.events.bus import EventBus
from app.core.plugins import PluginBase, PluginConfig

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set test environment
os.environ["API_KEY"] = "test_api_key"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture(autouse=True)
def mock_threadpool():
    """Mock ThreadPoolExecutor to prevent shutdown issues."""
    with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
        # Configure executor mock
        mock_executor_instance = MagicMock()
        mock_executor.return_value = mock_executor_instance
        mock_executor_instance._shutdown = False

        # Create a real Future object for the submit method to return
        def submit_side_effect(*args, **kwargs):
            future = asyncio.Future()
            future.set_result(None)
            return future

        mock_executor_instance.submit.side_effect = submit_side_effect
        yield mock_executor_instance


@pytest.fixture(autouse=True)
def mock_asyncio_sleep():
    """Mock asyncio.sleep globally."""

    async def immediate_sleep(*args, **kwargs):
        return None

    with patch("asyncio.sleep", side_effect=immediate_sleep):
        yield


@pytest.fixture(autouse=True)
def mock_event_bus():
    """Mock event bus fixture"""
    mock_bus = AsyncMock(spec=EventBus)
    mock_bus.start = AsyncMock()
    mock_bus.stop = AsyncMock()
    mock_bus.publish = AsyncMock()
    mock_bus._cleanup_task = None

    with patch("app.main.event_bus", mock_bus):
        yield mock_bus


@pytest.fixture(autouse=True)
def mock_db():
    """Mock database fixture"""
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.close = AsyncMock()
    mock_db.initialize = AsyncMock()
    mock_db._init_db = MagicMock()
    mock_db._run_migrations = MagicMock()
    mock_db.get_connection = MagicMock()

    with patch("app.models.database.DatabaseManager") as mock_manager:
        mock_manager._instance = None
        mock_manager._lock = asyncio.Lock()
        mock_manager.get_instance = AsyncMock(return_value=mock_db)
        yield mock_db


@pytest.fixture(autouse=True)
def mock_sqlite():
    """Mock sqlite3 to prevent actual database operations."""
    mock_conn = MagicMock()
    mock_conn.execute = MagicMock()
    mock_conn.executescript = MagicMock()
    mock_conn.commit = MagicMock()
    mock_conn.close = MagicMock()

    with patch("sqlite3.connect", return_value=mock_conn):
        yield mock_conn


@pytest.fixture(autouse=True)
def load_test_env():
    """Load test environment variables before each test"""
    # Get the project root directory
    root_dir = Path(__file__).parent.parent

    # Load test environment variables
    test_env_path = root_dir / ".env.test"
    if test_env_path.exists():
        load_dotenv(test_env_path)
    else:
        pytest.fail(f"Test environment file not found: {test_env_path}")

    # Create test directories
    test_dirs = [
        os.getenv("RECORDINGS_DIR", "data/test_recordings"),
        os.getenv("TRANSCRIPTS_DIR", "data/test_transcripts"),
        os.getenv("MEETING_NOTES_DIR", "data/test_meeting_notes"),
        os.getenv("CLEANED_AUDIO_DIR", "data/test_cleaned_audio"),
    ]

    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    yield


# Test implementation of PluginBase for testing
class _TestPluginImpl(PluginBase):
    async def _initialize(self) -> None:
        pass

    async def _shutdown(self) -> None:
        pass


@pytest.fixture
def test_plugin_impl(plugin_config, event_bus):
    return _TestPluginImpl(plugin_config, event_bus)


@pytest.fixture
def plugin_config():
    return PluginConfig(
        name="test_plugin",
        version="1.0.0",
        enabled=True,
        config={"test_key": "test_value"},
    )


@pytest.fixture
def event_bus():
    return AsyncMock(spec=EventBus)
