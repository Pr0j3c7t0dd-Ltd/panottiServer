"""Tests for the event persistence module."""

from datetime import datetime

import pytest

from app.core.events import Event
from app.core.events.persistence import EventProcessingStatus, EventStore


@pytest.mark.asyncio
async def test_event_store_initialization():
    """Test EventStore initialization."""
    store = EventStore()
    assert store._events == {}
    assert store._status == {}


@pytest.mark.asyncio
async def test_store_event():
    """Test storing events in the EventStore."""
    store = EventStore()
    event = Event(name="test.event", data={})

    # Test storing event
    event_id = await store.store_event(event)
    assert event_id == event.event_id
    assert "system" in store._events
    assert event in store._events["system"]

    # Test storing event with plugin_id
    plugin_event = Event(name="plugin.event", data={}, plugin_id="test_plugin")
    await store.store_event(plugin_event)
    assert "test_plugin" in store._events
    assert plugin_event in store._events["test_plugin"]

    # Verify status is set correctly
    assert event.event_id in store._status
    status = store._status[event.event_id]
    assert status["status"] == EventProcessingStatus.PENDING
    assert isinstance(status["timestamp"], datetime)
    assert status["error"] is None


@pytest.mark.asyncio
async def test_mark_processed():
    """Test marking events as processed or failed."""
    store = EventStore()
    event = Event(name="test.event", data={})
    await store.store_event(event)

    # Test marking as processed
    await store.mark_processed(event.event_id)
    status = store._status[event.event_id]
    assert status["status"] == EventProcessingStatus.PROCESSED
    assert status["error"] is None

    # Test marking as failed
    error_msg = "Test error"
    await store.mark_processed(event.event_id, success=False, error=error_msg)
    status = store._status[event.event_id]
    assert status["status"] == EventProcessingStatus.FAILED
    assert status["error"] == error_msg

    # Test marking unknown event
    await store.mark_processed("unknown_id")  # Should not raise exception


@pytest.mark.asyncio
async def test_get_events():
    """Test retrieving events from the EventStore."""
    store = EventStore()
    event1 = Event(name="test.event1", data={}, plugin_id="plugin1")
    event2 = Event(name="test.event2", data={}, plugin_id="plugin1")

    await store.store_event(event1)
    await store.store_event(event2)

    # Test getting events for existing plugin
    events = await store.get_events("plugin1")
    assert len(events) == 2
    assert event1 in events
    assert event2 in events

    # Test getting events for non-existent plugin
    events = await store.get_events("non_existent")
    assert events == []


@pytest.mark.asyncio
async def test_get_event():
    """Test retrieving a specific event by ID."""
    store = EventStore()
    event1 = Event(name="test.event1", data={})
    event2 = Event(name="test.event2", data={})

    await store.store_event(event1)
    await store.store_event(event2)

    # Test getting existing event
    retrieved_event = await store.get_event(event1.event_id)
    assert retrieved_event == event1

    # Test getting non-existent event
    retrieved_event = await store.get_event("non_existent")
    assert retrieved_event is None


@pytest.mark.asyncio
async def test_get_event_status():
    """Test retrieving event status."""
    store = EventStore()
    event = Event(name="test.event", data={})
    await store.store_event(event)

    # Test getting existing event status
    status = await store.get_event_status(event.event_id)
    assert status is not None
    assert status["status"] == EventProcessingStatus.PENDING

    # Test getting non-existent event status
    status = await store.get_event_status("non_existent")
    assert status is None


@pytest.mark.asyncio
async def test_clear_events():
    """Test clearing events for a plugin."""
    store = EventStore()
    event1 = Event(name="test.event1", data={}, plugin_id="plugin1")
    event2 = Event(name="test.event2", data={}, plugin_id="plugin2")

    await store.store_event(event1)
    await store.store_event(event2)

    # Test clearing events for plugin1
    await store.clear_events("plugin1")
    assert len(await store.get_events("plugin1")) == 0
    assert len(await store.get_events("plugin2")) == 1

    # Test clearing events for non-existent plugin
    await store.clear_events("non_existent")  # Should not raise exception
