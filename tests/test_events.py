"""Tests for the core events module."""

import pytest
from unittest.mock import AsyncMock, Mock

from app.core.events import (
    ConcreteEventBus,
    Event,
    EventBus,
    register_core_handlers,
)
from app.core.events.handlers import handle_recording_ended, handle_recording_started


class MockEventBus:
    """Mock implementation of EventBus protocol for testing."""
    def __init__(self):
        self.subscribe = AsyncMock()
        self.unsubscribe = AsyncMock()
        self.publish = AsyncMock()
        self.emit = AsyncMock()


@pytest.mark.asyncio(loop_scope="function")
async def test_event_bus_protocol():
    """Test that EventBus protocol methods are properly defined."""
    mock_bus = MockEventBus()
    mock_callback = Mock()
    mock_event = Event(name="test.event", data={})

    # Test subscribe
    await mock_bus.subscribe("test.event", mock_callback)
    mock_bus.subscribe.assert_called_once_with("test.event", mock_callback)

    # Test unsubscribe
    await mock_bus.unsubscribe("test.event", mock_callback)
    mock_bus.unsubscribe.assert_called_once_with("test.event", mock_callback)

    # Test publish
    await mock_bus.publish(mock_event)
    mock_bus.publish.assert_called_once_with(mock_event)

    # Test emit
    await mock_bus.emit(mock_event)
    mock_bus.emit.assert_called_once_with(mock_event)


@pytest.mark.asyncio(loop_scope="function")
async def test_register_core_handlers():
    """Test registration of core event handlers."""
    mock_bus = ConcreteEventBus()
    mock_bus.subscribe = AsyncMock()

    await register_core_handlers(mock_bus)

    # Verify that both core handlers are registered
    assert mock_bus.subscribe.call_count == 2
    mock_bus.subscribe.assert_any_call("recording.started", handle_recording_started)
    mock_bus.subscribe.assert_any_call("recording.ended", handle_recording_ended) 