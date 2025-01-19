"""Common test fixtures and configuration."""

import asyncio
import os
import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set test environment
os.environ["API_KEY"] = "test_api_key"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


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

    # Optional: Clean up test directories after tests
    # Uncomment if you want to clean up after tests
    # for dir_path in test_dirs:
    #     shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def mock_event_bus():
    """Mock event bus fixture"""
    from app.plugins.events.bus import EventBus

    event_bus = EventBus()
    event_bus.subscribe = AsyncMock()
    event_bus.unsubscribe = AsyncMock()
    event_bus.publish = AsyncMock()
    return event_bus


@pytest.fixture
def mock_db():
    """Mock database fixture"""
    with patch("app.models.database.DatabaseManager") as mock:
        db_instance = MagicMock()
        mock.get_instance.return_value = db_instance
        yield db_instance
