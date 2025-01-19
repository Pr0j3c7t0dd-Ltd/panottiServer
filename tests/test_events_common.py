"""Common test utilities for event tests."""

from unittest.mock import AsyncMock


class TestEvent:
    """Test event class."""
    def __init__(self, name: str, data: dict, event_id: str | None = None):
        self.name = name
        self.data = data
        self.event_id = event_id
    
    def __str__(self) -> str:
        return f"TestEvent(name={self.name}, data={self.data}, event_id={self.event_id})"


def create_mock_handler(name: str = "mock_handler") -> AsyncMock:
    """Create a mock handler with required attributes."""
    handler = AsyncMock()
    handler.__name__ = name
    handler.__qualname__ = f"test_event_bus.{name}"
    handler.__module__ = "test_event_bus"
    return handler 