"""Unit tests for EventBus."""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timedelta, UTC
from unittest.mock import patch, AsyncMock

from app.core.events.bus import EventBus


@pytest.fixture(scope="module")
def event_loop():
    """Create an event loop for each test module."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def event_bus(event_loop):
    """Create and start an event bus instance."""
    # Mock sleep to return immediately
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        mock_sleep.return_value = None
        
        # Create and start bus
        bus = EventBus()
        await bus.start()
        
        # Disable cleanup task
        if bus._cleanup_events_task:
            bus._cleanup_events_task.cancel()
            try:
                await bus._cleanup_events_task
            except asyncio.CancelledError:
                pass
            bus._cleanup_events_task = None
        
        yield bus
        
        # Cleanup
        await bus.stop()
        await bus.shutdown()


@pytest.mark.asyncio
async def test_cleanup_old_events(event_bus):
    """Test that old events are cleaned up correctly."""
    now = datetime.now(UTC)
    old_event_id = "old_event"
    recent_event_id = "recent_event"

    event_bus._processed_events = {
        old_event_id: now - timedelta(hours=2),
        recent_event_id: now - timedelta(minutes=30),
    }

    # Run cleanup once for testing
    await event_bus._cleanup_old_events(run_once=True)
    
    assert old_event_id not in event_bus._processed_events
    assert recent_event_id in event_bus._processed_events


@pytest.mark.asyncio
async def test_event_id_extraction(event_bus):
    """Test event ID extraction from different event types."""
    event_dict = {
        "id": "test_id",
        "event": "test_event",
        "source_plugin": "test_source"
    }
    assert event_bus._get_event_id(event_dict) == "test_id_test_event_test_source"

    class TestEvent:
        def __init__(self):
            self.recording_id = "rec_id"
            self.event = "test_event"
            self.source_plugin = "test_source"

    event_obj = TestEvent()
    assert event_bus._get_event_id(event_obj) == "rec_id_test_event_test_source"

    empty_event = {}
    event_id = event_bus._get_event_id(empty_event)
    assert len(event_id) == 36

    class BrokenEvent:
        @property
        def event_id(self):
            raise Exception("Test error")

    broken_event = BrokenEvent()
    event_id = event_bus._get_event_id(broken_event)
    assert len(event_id) == 36


@pytest.mark.asyncio
async def test_event_processing_decision(event_bus):
    """Test event processing decision logic."""
    event_dict = {"source_plugin": "test_source"}
    assert event_bus._should_process_event(event_dict) is True

    class EventContext:
        def __init__(self):
            self.source_plugin = "test_source"

    class EventWithContext:
        def __init__(self):
            self.context = EventContext()

    event_obj = EventWithContext()
    assert event_bus._should_process_event(event_obj) is True

    empty_event = {}
    assert event_bus._should_process_event(empty_event) is True


@pytest.mark.asyncio
async def test_event_publishing_error_handling(event_bus):
    """Test error handling during event publishing."""
    await event_bus.publish({})

    class BrokenEvent:
        def to_dict(self):
            raise Exception("Test error")
        
        def __getattribute__(self, name):
            if name == "__dict__":
                raise Exception("Test error")
            return super().__getattribute__(name)

    await event_bus.publish(BrokenEvent())
    await event_bus.publish(None)


@pytest.mark.asyncio
async def test_handler_error_handling(event_bus):
    """Test error handling in event handlers."""
    async def failing_handler(event):
        raise Exception("Test handler error")

    event_name = "test_event"
    await event_bus.subscribe(event_name, failing_handler)
    await event_bus.publish({"event": event_name, "id": "test_id"})
    assert not event_bus._shutting_down


@pytest.mark.asyncio
async def test_shutdown_behavior(event_bus):
    """Test event bus shutdown behavior."""
    async def long_handler(event):
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    await event_bus.subscribe("test_event", long_handler)
    await event_bus.publish({"event": "test_event", "id": "test_id"})
    await event_bus.stop()
    assert event_bus._shutting_down
    assert not event_bus._pending_tasks
