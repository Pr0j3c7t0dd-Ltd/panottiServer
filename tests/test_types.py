"""Tests for core event types."""

import pytest
from datetime import datetime, UTC
from uuid import UUID

from app.core.events.types import EventContext, EventHandler


def test_event_context_defaults():
    """Test EventContext model default values."""
    context = EventContext()
    
    # Test correlation_id is a valid UUID
    assert UUID(context.correlation_id)
    
    # Test timestamp is current time in UTC
    now = datetime.now(UTC)
    assert context.timestamp.tzinfo == UTC
    assert abs((context.timestamp - now).total_seconds()) < 1
    
    # Test metadata defaults to empty dict
    assert context.metadata == {}


def test_event_context_custom_values():
    """Test EventContext model with custom values."""
    custom_id = "test-id"
    custom_time = datetime.now(UTC)
    custom_metadata = {"key": "value"}
    
    context = EventContext(
        correlation_id=custom_id,
        timestamp=custom_time,
        metadata=custom_metadata
    )
    
    assert context.correlation_id == custom_id
    assert context.timestamp == custom_time
    assert context.metadata == custom_metadata


@pytest.mark.asyncio
async def test_event_handler_protocol():
    """Test EventHandler protocol implementation."""
    # Test that the Protocol's __call__ method is defined
    assert hasattr(EventHandler, '__call__')

    # Test that the method has the correct signature
    call_method = getattr(EventHandler, '__call__')
    assert call_method.__name__ == '__call__'

    # Test that the method is abstract
    assert getattr(call_method, '__isabstractmethod__', False)

    # Test concrete implementation
    class ConcreteHandler:
        async def __call__(self, event_data: dict) -> None:
            self.called = True

    # Create instance and verify it can be called
    handler = ConcreteHandler()
    await handler({"test": "data"})
    assert handler.called

    # Test that the protocol is runtime checkable
    assert isinstance(handler, EventHandler) 