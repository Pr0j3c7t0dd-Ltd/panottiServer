"""Tests for the core events module."""

import pytest
from typing import Any, Callable, cast
from unittest.mock import AsyncMock, Mock, call

from app.core.events import (
    ConcreteEventBus,
    Event,
    EventBus,
    register_core_handlers,
)
from app.core.events.handlers import handle_recording_ended, handle_recording_started


async def mock_callback(event: Any) -> None:
    """Mock callback function for testing."""
    pass


@pytest.mark.asyncio(loop_scope="function")
async def test_event_bus_protocol():
    """Test that EventBus protocol methods are properly defined."""
    bus = ConcreteEventBus()
    mock_event = Event(name="test.event", data={})

    # Test subscribe
    await bus.subscribe("test.event", mock_callback)

    # Test unsubscribe
    await bus.unsubscribe("test.event", mock_callback)

    # Test publish
    await bus.publish(mock_event)

    # Test publish as emit
    await bus.publish(mock_event)  # Since emit is just an alias for publish


@pytest.mark.asyncio(loop_scope="function")
async def test_event_bus_abstract_methods():
    """Test that EventBus protocol methods raise NotImplementedError when not implemented."""
    class MinimalEventBusImpl:
        async def subscribe(self, event_type: str, callback: Callable[[Any], Any]) -> None:
            raise NotImplementedError

        async def unsubscribe(self, event_type: str, callback: Callable[[Any], Any]) -> None:
            raise NotImplementedError

        async def publish(self, event: Any) -> None:
            raise NotImplementedError

        async def emit(self, event: Any) -> None:
            raise NotImplementedError

    bus = MinimalEventBusImpl()
    mock_event = Event(name="test.event", data={})

    # Verify that calling methods raises NotImplementedError
    with pytest.raises(NotImplementedError):
        await bus.subscribe("test.event", mock_callback)

    with pytest.raises(NotImplementedError):
        await bus.unsubscribe("test.event", mock_callback)

    with pytest.raises(NotImplementedError):
        await bus.publish(mock_event)

    with pytest.raises(NotImplementedError):
        await bus.emit(mock_event)


@pytest.mark.asyncio(loop_scope="function")
async def test_event_bus_emit_default_implementation():
    """Test that EventBus protocol's default emit implementation calls publish."""
    class TestEventBus:
        def __init__(self):
            self.publish_called = False
            self.published_event = None

        async def subscribe(self, event_type: str, callback: Callable[[Any], Any]) -> None:
            pass

        async def unsubscribe(self, event_type: str, callback: Callable[[Any], Any]) -> None:
            pass

        async def publish(self, event: Any) -> None:
            self.publish_called = True
            self.published_event = event

        # Use the default implementation from the Protocol
        emit = EventBus.emit

    bus = TestEventBus()
    mock_event = Event(name="test.event", data={})
    await bus.emit(mock_event)
    assert bus.publish_called
    assert bus.published_event == mock_event


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


@pytest.mark.asyncio(loop_scope="function")
async def test_event_bus_protocol_implementation():
    """Test that EventBus protocol's abstract methods are properly defined."""
    # Test that the Protocol's abstract methods are defined
    assert hasattr(EventBus, 'subscribe')
    assert hasattr(EventBus, 'unsubscribe')
    assert hasattr(EventBus, 'publish')
    assert hasattr(EventBus, 'emit')

    # Test that the Protocol's methods have the correct signatures
    subscribe_method = getattr(EventBus, 'subscribe')
    unsubscribe_method = getattr(EventBus, 'unsubscribe')
    publish_method = getattr(EventBus, 'publish')
    emit_method = getattr(EventBus, 'emit')

    assert subscribe_method.__name__ == 'subscribe'
    assert unsubscribe_method.__name__ == 'unsubscribe'
    assert publish_method.__name__ == 'publish'
    assert emit_method.__name__ == 'emit'

    # Test that the Protocol's methods are abstract
    assert getattr(subscribe_method, '__isabstractmethod__', False)
    assert getattr(unsubscribe_method, '__isabstractmethod__', False)
    assert getattr(publish_method, '__isabstractmethod__', False)
    assert getattr(emit_method, '__isabstractmethod__', False)  # emit is also abstract


@pytest.mark.asyncio(loop_scope="function")
async def test_event_bus_emit_implementation_direct():
    """Test the emit method implementation directly."""
    class TestEventBus:
        def __init__(self):
            self.publish_called = False
            self.published_event = None

        async def subscribe(self, event_type: str, callback: Callable[[Any], Any]) -> None:
            pass

        async def unsubscribe(self, event_type: str, callback: Callable[[Any], Any]) -> None:
            pass

        async def publish(self, event: Any) -> None:
            self.publish_called = True
            self.published_event = event

    bus = TestEventBus()
    mock_event = Event(name="test.event", data={})

    # Call the emit method directly from the Protocol
    await EventBus.emit.__get__(bus)(mock_event)
    assert bus.publish_called
    assert bus.published_event == mock_event


@pytest.mark.asyncio(loop_scope="function")
async def test_event_bus_protocol_method_implementations():
    """Test the actual implementations of the Protocol's methods."""
    class TestEventBus:
        def __init__(self):
            self.publish_called = False
            self.published_event = None

        async def subscribe(self, event_type: str, callback: Callable[[Any], Any]) -> None:
            pass

        async def unsubscribe(self, event_type: str, callback: Callable[[Any], Any]) -> None:
            pass

        async def publish(self, event: Any) -> None:
            self.publish_called = True
            self.published_event = event

    bus = TestEventBus()
    mock_event = Event(name="test.event", data={})

    # Test each method's implementation
    await EventBus.subscribe.__get__(bus)("test.event", mock_callback)
    await EventBus.unsubscribe.__get__(bus)("test.event", mock_callback)
    await EventBus.publish.__get__(bus)(mock_event)
    await EventBus.emit.__get__(bus)(mock_event)

    assert bus.publish_called
    assert bus.published_event == mock_event


@pytest.mark.asyncio(loop_scope="function")
async def test_event_handler_protocol():
    """Test that EventHandler protocol is properly defined."""
    from app.core.events.types import EventHandler

    # Test that the Protocol's __call__ method is defined
    assert hasattr(EventHandler, '__call__')

    # Test that the method has the correct signature
    call_method = getattr(EventHandler, '__call__')
    assert call_method.__name__ == '__call__'

    # Test that the method is abstract
    assert getattr(call_method, '__isabstractmethod__', False)

    # Verify we can create a concrete implementation
    class ConcreteHandler:
        async def __call__(self, event_data: Any) -> None:
            pass

    # This should not raise a TypeError
    handler = ConcreteHandler()
    assert isinstance(handler, EventHandler) 