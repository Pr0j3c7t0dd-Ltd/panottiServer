"""Common test fixtures and configuration."""

import asyncio
import os
from collections.abc import Generator

import pytest

# Set test environment
os.environ["API_KEY"] = "test_api_key"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
