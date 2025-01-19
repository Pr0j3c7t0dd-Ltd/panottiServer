"""Tests for the core events module."""

import pytest
from typing import Any, Callable, cast, runtime_checkable, Protocol
from unittest.mock import AsyncMock, Mock, call

from app.core.events import (
    ConcreteEventBus,
    Event,
    register_core_handlers,
)
from app.core.events.handlers import handle_recording_ended, handle_recording_started


@runtime_checkable
class EventBus(Protocol):
    """Event bus interface."""

    async def subscribe(
        self, event_type: str, callback: Callable[[Any], Any]
    ) -> None:
        """Subscribe to events of a given type."""
        pass

    async def unsubscribe(
        self, event_type: str, callback: Callable[[Any], Any]
    ) -> None:
        """Unsubscribe from events of a given type."""
        pass

    async def publish(self, event: Any) -> None:
        """Publish an event to all subscribers."""
        pass

    async def emit(self, event: Any) -> None:
        """Emit an event (alias for publish)."""
        await self.publish(event)


class MinimalEventBus:
    """Minimal concrete implementation of EventBus protocol."""
    async def subscribe(
        self, event_type: str, callback: Callable[[Any], Any]
    ) -> None:
        """Subscribe to events of a given type."""
        pass

    async def unsubscribe(
        self, event_type: str, callback: Callable[[Any], Any]
    ) -> None:
        """Unsubscribe from events of a given type."""
        pass

    async def publish(self, event: Any) -> None:
        """Publish an event to all subscribers."""
        pass

    async def emit(self, event: Any) -> None:
        """Use the default implementation from the protocol."""
        await self.publish(event)


@pytest.mark.asyncio(loop_scope="function")
async def test_event_bus_protocol():
    """Test that EventBus protocol methods are properly defined."""
    bus = MinimalEventBus()
    assert isinstance(bus, EventBus)
    event_bus = cast(EventBus, bus)
    mock_callback = Mock()
    mock_event = Event(name="test.event", data={})

    # Test subscribe
    await event_bus.subscribe("test.event", mock_callback)

    # Test unsubscribe
    await event_bus.unsubscribe("test.event", mock_callback)

    # Test publish
    await event_bus.publish(mock_event)

    # Test emit
    bus.publish = AsyncMock()
    await event_bus.emit(mock_event)
    bus.publish.assert_called_once_with(mock_event)


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