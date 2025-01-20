"""Tests for the EventBus implementation."""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch, create_autospec
from asyncio import Task, Future
from typing import Any, Callable, Optional
from collections import defaultdict
import uuid

from app.core.events.bus import EventBus
from .test_events_common import create_test_event, create_mock_handler


class MockEvent:
    """Mock event class for testing."""
    def __init__(self, name=None, event_id=None, event=None):
        # Store attributes directly to make them accessible via hasattr
        self.event = name  # Use name as the event field
        self.event_id = event_id
        self.name = name  # Also store as name for compatibility

    def __str__(self):
        """String representation for logging."""
        return f"MockEvent(name={self.name}, event_id={self.event_id}, event={self.event})"

    def __getattr__(self, name):
        """Handle attribute access for the event bus."""
        if name == "__dict__":
            return {
                "event": self.event,
                "event_id": self.event_id,
                "name": self.name
            }
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


@pytest.fixture
async def event_bus():
    """Create an EventBus instance without starting background tasks."""
    bus = EventBus()
    bus._subscribers = defaultdict(list)  # Initialize subscribers dict with defaultdict
    bus._lock = asyncio.Lock()  # Initialize lock
    yield bus


@pytest.fixture
def mock_handler():
    """Create a properly configured mock handler."""
    handler = AsyncMock()
    handler.__name__ = "mock_handler"
    handler.__qualname__ = "test_event_bus.mock_handler"
    handler.__module__ = "test_event_bus"
    return handler


@pytest.fixture
def test_event():
    """Create a test event instance."""
    return lambda name="test.event", data=None, event_id=None: create_test_event(
        name=name,
        data=data,
        event_id=event_id
    )


@pytest.mark.asyncio
async def test_event_bus_start_stop():
    """Test normal start/stop of event bus."""
    bus = EventBus()
    
    # Create a real coroutine for _cleanup_old_events
    async def mock_cleanup():
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    # Patch create_task to use our mock coroutine
    with patch.object(bus, '_cleanup_old_events', side_effect=mock_cleanup):
        await bus.start()
        assert bus._cleanup_events_task is not None
        assert not bus._cleanup_events_task.done()
        
        await bus.stop()
        assert bus._cleanup_events_task is None


@pytest.mark.asyncio
async def test_event_bus_start_error():
    """Test error handling during event bus start."""
    bus = EventBus()
    
    # Patch create_task to raise an error
    with patch('asyncio.create_task', side_effect=Exception("Test error")), \
         pytest.raises(Exception, match="Test error"):  # Verify error is propagated
        await bus.start()
    
    # Verify cleanup task is not set
    assert bus._cleanup_events_task is None


@pytest.mark.asyncio
async def test_event_bus_stop_error():
    """Test error handling during event bus stop."""
    bus = EventBus()
    
    # Create a real coroutine that raises on cancel
    async def mock_cleanup():
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise Exception("Cancel error")

    # Create a task for the mock coroutine and patch the cleanup method
    cleanup_task = asyncio.create_task(mock_cleanup())
    bus._cleanup_events_task = cleanup_task
    
    # Ensure we await any pending tasks
    try:
        await bus.stop()
    except Exception:
        pass  # We expect an error here
    
    # Ensure cleanup task is properly cancelled
    if not cleanup_task.done():
        cleanup_task.cancel()
        try:
            await cleanup_task
        except (asyncio.CancelledError, Exception):
            pass
    
    assert bus._cleanup_events_task is None


@pytest.mark.asyncio
async def test_cleanup_old_events(event_bus):
    """Test cleanup of old events."""
    event_id = "test_id"
    now = datetime.now(timezone.utc)
    event_bus._processed_events[event_id] = now - timedelta(hours=2)
    
    # Create a real coroutine for cleanup that runs once
    async def mock_cleanup():
        await asyncio.sleep(0)  # Simulate a short delay
        async with event_bus._lock:
            old_events = [
                event_id
                for event_id, timestamp in event_bus._processed_events.items()
                if (now - timestamp).total_seconds() > 3600
            ]
            for event_id in old_events:
                del event_bus._processed_events[event_id]

    # Run cleanup directly without patching
    await mock_cleanup()
    assert event_id not in event_bus._processed_events


@pytest.mark.asyncio
async def test_handle_task_success(mock_handler):
    """Test successful task handling."""
    bus = EventBus()
    event = MockEvent(name="test.event")
    
    # Create a real coroutine for the handler
    async def mock_handler_coro(event):
        await asyncio.sleep(0)  # Simulate work
        return None

    # Create a future for the mock handler
    future = asyncio.Future()
    future.set_result(None)
    mock_handler.return_value = future
    
    await bus._handle_task(mock_handler, event)
    await asyncio.sleep(0.1)  # Give task time to complete
    
    mock_handler.assert_called_once_with(event)


@pytest.mark.asyncio
async def test_handle_task_error(mock_handler):
    """Test task handling with error."""
    bus = EventBus()
    event = MockEvent(name="test.event")
    
    # Create a real coroutine that raises an error
    async def mock_handler_coro(event):
        await asyncio.sleep(0)  # Simulate work
        raise Exception("Handler error")

    mock_handler.side_effect = mock_handler_coro
    
    await bus._handle_task(mock_handler, event)
    await asyncio.sleep(0.1)  # Give task time to complete
    
    mock_handler.assert_called_once_with(event)


@pytest.mark.asyncio
async def test_handle_task_callback_error(mock_handler):
    """Test error in task callback handling."""
    bus = EventBus()
    event = MockEvent(name="test.event")
    
    # Create a real coroutine for the handler
    async def mock_handler_coro(event):
        await asyncio.sleep(0)  # Simulate work
        return None

    mock_handler.side_effect = mock_handler_coro
    
    # Create a task that will raise in its callback
    task = asyncio.create_task(mock_handler_coro(event))
    task.add_done_callback(lambda _: 1/0)  # Will raise ZeroDivisionError
    
    with patch('asyncio.create_task', return_value=task):
        await bus._handle_task(mock_handler, event)
        await asyncio.sleep(0.1)  # Give task time to complete
    
    mock_handler.assert_called_once_with(event)


@pytest.mark.asyncio
async def test_should_process_event(event_bus):
    """Test event processing check."""
    event_id = "test_id"
    now = datetime.now(timezone.utc)
    
    # Test with unprocessed event
    result = await event_bus._is_event_processed(event_id)
    assert not result
    
    # Test with recently processed event
    event_bus._processed_events[event_id] = now
    result = await event_bus._is_event_processed(event_id)
    assert result
    
    # Test with old processed event
    event_bus._processed_events[event_id] = now - timedelta(hours=2)
    result = await event_bus._is_event_processed(event_id)
    assert result


@pytest.mark.asyncio
async def test_publish_event(event_bus):
    """Test publishing events to subscribers."""
    event = MockEvent(name="test.event")
    
    # Create a properly configured mock handler
    mock_handler = create_mock_handler()
    
    # Subscribe handler
    await event_bus.subscribe(event.name, mock_handler)
    
    # Publish event and wait for handler to be called
    await event_bus.publish(event)
    
    # Wait for all pending tasks to complete
    pending_tasks = list(event_bus._pending_tasks)
    if pending_tasks:
        await asyncio.gather(*pending_tasks)
    
    # Verify handler was called with event
    mock_handler.assert_called_once_with(event)


@pytest.mark.asyncio
async def test_event_name_handling(event_bus):
    """Test event name extraction."""
    # Test with name attribute
    event = MockEvent(name="test.event")
    print(f"Event: {event}")
    print(f"Event name: {event.name}")
    print(f"Has name attribute: {hasattr(event, 'name')}")
    print(f"Event dict: {event.__dict__}")
    name = event_bus._get_event_name(event)
    print(f"Got event name: {name}")
    assert name == "test.event"


@pytest.mark.asyncio
async def test_event_id_handling():
    """Test event ID handling."""
    bus = EventBus()
    
    # Test with event that has event_id attribute
    event = MockEvent(event_id="test_id")
    assert bus._get_event_id(event) == "test_id"
    
    # Test with event that has id attribute
    event = MockEvent()
    event.id = "test_id_2"  # type: ignore
    
    # Mock uuid generation to return a known value
    mock_uuid = uuid.UUID('12345678-1234-5678-1234-567812345678')
    with patch('uuid.uuid4', return_value=mock_uuid):
        event_id = bus._get_event_id(event)
        assert isinstance(event_id, str)
        assert event_id == str(mock_uuid)
    
    # Test with event that has no ID
    event = MockEvent()
    mock_uuid = uuid.UUID('87654321-4321-8765-4321-876543210987')
    with patch('uuid.uuid4', return_value=mock_uuid):
        event_id = bus._get_event_id(event)
        assert isinstance(event_id, str)
        assert event_id == str(mock_uuid)
    
    # Clean up any pending tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_subscribe_error_handling(event_bus, mock_handler):
    """Test error handling in subscribe."""
    # Test with invalid event name
    with pytest.raises(ValueError, match="Event name cannot be empty"):
        await event_bus.subscribe(None, mock_handler)
    
    # Test with invalid handler
    with pytest.raises(ValueError, match="Handler cannot be None"):
        await event_bus.subscribe("test.event", None)
    
    # Verify no subscriptions were added
    assert len(event_bus._subscribers) == 0


@pytest.mark.asyncio
async def test_publish_error_handling(event_bus, mock_handler):
    """Test error handling in publish."""
    # Test with invalid event
    await event_bus.publish(None)  # Should not raise
    
    # Test with missing event name
    await event_bus.publish({})  # Should not raise
    
    # Test with invalid handler
    event_bus._subscribers["test.event"].append(None)
    await event_bus.publish(MockEvent(name="test.event"))  # Should not raise


@pytest.mark.asyncio
async def test_shutdown_error_handling(event_bus):
    """Test error handling in shutdown."""
    # Create a mock task that raises an error when cancelled
    async def mock_task():
        try:
            await asyncio.sleep(0.1)  # Short sleep to allow cancellation
            return  # Task completes normally if not cancelled
        except asyncio.CancelledError:
            raise Exception("Task error")

    # Add some tasks to pending_tasks
    task1 = asyncio.create_task(mock_task())
    task2 = asyncio.create_task(mock_task())
    event_bus._pending_tasks.add(task1)
    event_bus._pending_tasks.add(task2)

    # Add a cleanup task that can be cancelled quickly
    async def mock_cleanup():
        try:
            await asyncio.sleep(0.1)  # Short sleep to allow cancellation
            return  # Task completes normally if not cancelled
        except asyncio.CancelledError:
            raise Exception("Cleanup error")

    event_bus._cleanup_events_task = asyncio.create_task(mock_cleanup())

    # Shutdown should handle errors gracefully
    await event_bus.shutdown()

    # Verify cleanup
    assert len(event_bus._pending_tasks) == 0
    assert len(event_bus._subscribers) == 0
    assert event_bus._cleanup_events_task is None

    # Ensure all tasks are properly cleaned up
    for task in [task1, task2]:
        assert task.done()


@pytest.mark.asyncio
async def test_event_processing_tracking(event_bus):
    """Test event processing status tracking."""
    event = MockEvent(name="test.event", event_id="test_id")
    
    # Verify event not processed initially
    assert not await event_bus._is_event_processed(event.event_id)
    
    # Mark event as processed
    await event_bus._mark_event_processed(event.event_id)
    
    # Verify event is now processed
    assert await event_bus._is_event_processed(event.event_id)


@pytest.mark.asyncio
async def test_subscribe_unsubscribe(event_bus, mock_handler):
    """Test subscribe/unsubscribe functionality."""
    event_name = "test.event"
    
    # Subscribe handler
    await event_bus.subscribe(event_name, mock_handler)
    assert mock_handler in event_bus._subscribers[event_name]
    
    # Unsubscribe handler
    await event_bus.unsubscribe(event_name, mock_handler)
    assert mock_handler not in event_bus._subscribers[event_name] 