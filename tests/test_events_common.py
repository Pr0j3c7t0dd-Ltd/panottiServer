"""Common test utilities for event tests."""

from typing import Any
from unittest.mock import AsyncMock


def create_test_event(
    name: str, data: dict | None = None, event_id: str | None = None
) -> dict[str, Any]:
    """Create a test event dictionary.

    Args:
        name: Event name
        data: Event data
        event_id: Optional event ID

    Returns:
        dict: Event dictionary
    """
    event = {
        "name": name,
        "data": data or {"data": "value"},
    }
    if event_id:
        event["event_id"] = event_id
    return event


def create_mock_handler(name: str = "mock_handler") -> AsyncMock:
    """Create a mock handler with required attributes."""
    handler = AsyncMock()
    handler.__name__ = name
    handler.__qualname__ = f"test_event_bus.{name}"
    handler.__module__ = "test_event_bus"
    return handler
