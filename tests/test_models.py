"""Tests for core event models."""

from uuid import UUID

from app.core.events.models import Event, EventPriority, get_event_name
from app.core.events.types import EventContext


def test_event_priority_value_int():
    """Test EventPriority value_int property."""
    assert EventPriority.LOW.value_int == 0
    assert EventPriority.NORMAL.value_int == 1
    assert EventPriority.HIGH.value_int == 2
    assert EventPriority.CRITICAL.value_int == 3


def test_event_validators():
    """Test Event model validators."""
    # Test validate_event_id
    event = Event(name="test.event", event_id=None)
    assert UUID(event.event_id)  # Verify it's a valid UUID

    # Test validate_event_id with existing value
    event = Event(name="test.event", event_id="123")
    assert event.event_id == "123"

    # Test validate_name
    event = Event(name="event")
    assert event.name == "event"

    # Test validate_name with None
    event = Event(name=None)
    assert event.name == "event"  # Uses class name

    # Test validate_name with existing value
    event = Event(name="custom.event")
    assert event.name == "custom.event"

    # Test validate_plugin_id
    event = Event(name="test.event")
    assert event.plugin_id == "system"

    # Test validate_plugin_id with None
    event = Event(name="test.event", plugin_id=None)
    assert event.plugin_id == "system"

    # Test validate_plugin_id with existing value
    event = Event(name="test.event", plugin_id="custom_plugin")
    assert event.plugin_id == "custom_plugin"

    # Test validate_context with None
    event = Event(name="test.event", context=None)
    assert isinstance(event.context, EventContext)

    # Test validate_context with existing EventContext
    existing_context = EventContext(correlation_id="123")
    event = Event(name="test.event", context=existing_context)
    assert event.context == existing_context

    # Test validate_context with dict
    event = Event(
        name="test.event",
        context={"correlation_id": "123", "metadata": {"source_plugin": "test"}},
    )
    assert isinstance(event.context, EventContext)
    assert event.context.correlation_id == "123"
    assert event.context.metadata["source_plugin"] == "test"


def test_event_create_classmethod():
    """Test Event.create classmethod."""
    event = Event.create(
        name="test.event",
        data={"key": "value"},
        correlation_id="123",
        source_plugin="test_plugin",
        priority=EventPriority.HIGH,
    )

    assert event.name == "test.event"
    assert event.data == {"key": "value"}
    assert event.context.correlation_id == "123"
    assert event.plugin_id == "system"  # plugin_id remains default
    assert event.priority == EventPriority.HIGH


def test_get_event_name():
    """Test get_event_name function."""

    # Test with object having event attribute
    class TestEvent:
        event = "test.event"

    assert get_event_name(TestEvent()) == "test.event"

    # Test with dict
    assert get_event_name({"event": "test.event"}) == "test.event"

    # Test with unknown
    assert get_event_name(None) == "unknown"
