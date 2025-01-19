"""Tests for the EventBus implementation."""

import asyncio
import pytest
from datetime import datetime, timedelta, UTC
from unittest.mock import AsyncMock, Mock, patch

from app.core.events.bus import EventBus
from .test_events_common import TestEvent, create_mock_handler


@pytest.fixture
async def event_bus():
    """Create and start an event bus instance."""
    bus = EventBus()
    await bus.start()
    yield bus
    # Ensure all tasks are cleaned up
    for task in bus._pending_tasks:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    await bus.stop()


@pytest.fixture
def test_event():
    """Create a test event instance."""
    return lambda name="test.event", data=None, event_id=None: TestEvent(
        name=name,
        data=data or {"data": "value"},
        event_id=event_id
    )


@pytest.mark.asyncio
async def test_event_bus_start_stop():
    """Test starting and stopping the event bus."""
    bus = EventBus()
    assert bus._cleanup_events_task is None
    
    await bus.start()
    assert bus._cleanup_events_task is not None
    assert not bus._cleanup_events_task.done()
    
    await bus.stop()
    assert bus._cleanup_events_task is None


@pytest.mark.asyncio
async def test_cleanup_old_events():
    """Test cleanup of old events."""
    bus = EventBus()
    
    # Add some old and new events
    old_time = datetime.now(UTC) - timedelta(hours=2)
    new_time = datetime.now(UTC)
    
    async with bus._lock:
        bus._processed_events = {
            "old_event": old_time,
            "new_event": new_time
        }
    
    # Run cleanup with mocked sleep
    cleanup_done = asyncio.Event()
    original_sleep = asyncio.sleep
    
    async def mock_sleep(_):
        await original_sleep(0)
        cleanup_done.set()
    
    with patch("asyncio.sleep", mock_sleep):
        cleanup_task = asyncio.create_task(bus._cleanup_old_events())
        try:
            await asyncio.wait_for(cleanup_done.wait(), timeout=1)
        finally:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
    
    # Verify old event was removed but new one remains
    assert "old_event" not in bus._processed_events
    assert "new_event" in bus._processed_events


@pytest.mark.asyncio
async def test_handle_task_success(event_bus, test_event):
    """Test successful event handler execution."""
    mock_handler = create_mock_handler()
    event = test_event()
    
    # Create a done event to track when the handler is complete
    handler_done = asyncio.Event()
    task_done = asyncio.Event()
    
    async def wrapped_handler(*args, **kwargs):
        try:
            await asyncio.sleep(0)  # Simulate some work
            handler_done.set()
        finally:
            task_done.set()
    
    mock_handler.side_effect = wrapped_handler
    
    # Add a callback to remove the task from pending tasks
    def cleanup_callback(task):
        event_bus._pending_tasks.discard(task)
    
    task = asyncio.create_task(event_bus._handle_task(mock_handler, event))
    task.add_done_callback(cleanup_callback)
    
    # Wait for both the handler and task cleanup
    await asyncio.wait_for(asyncio.gather(
        handler_done.wait(),
        task_done.wait()
    ), timeout=1)
    
    # Give the event loop a chance to run callbacks
    await asyncio.sleep(0)
    
    mock_handler.assert_called_once_with(event)
    assert len(event_bus._pending_tasks) == 0


@pytest.mark.asyncio
async def test_handle_task_error(event_bus, test_event):
    """Test event handler execution with error."""
    handler_done = asyncio.Event()
    task_done = asyncio.Event()
    
    async def failing_handler(_):
        try:
            raise ValueError("Test error")
        finally:
            handler_done.set()
            task_done.set()
    
    failing_handler.__name__ = "failing_handler"
    failing_handler.__qualname__ = "test_event_bus.failing_handler"
    failing_handler.__module__ = "test_event_bus"
    
    event = test_event()
    
    # Add a callback to remove the task from pending tasks
    def cleanup_callback(task):
        event_bus._pending_tasks.discard(task)
    
    task = asyncio.create_task(event_bus._handle_task(failing_handler, event))
    task.add_done_callback(cleanup_callback)
    
    # Wait for both the handler and task cleanup
    await asyncio.wait_for(asyncio.gather(
        handler_done.wait(),
        task_done.wait()
    ), timeout=1)
    
    # Give the event loop a chance to run callbacks
    await asyncio.sleep(0)
    
    assert len(event_bus._pending_tasks) == 0


@pytest.mark.asyncio
async def test_event_processing_tracking(event_bus):
    """Test event processing tracking."""
    event_id = "test_event_1"
    
    # Initially event should not be processed
    assert not await event_bus._is_event_processed(event_id)
    
    # Mark event as processed
    await event_bus._mark_event_processed(event_id)
    
    # Now event should be marked as processed
    assert await event_bus._is_event_processed(event_id)


@pytest.mark.asyncio
async def test_subscribe_unsubscribe(event_bus):
    """Test subscribing and unsubscribing handlers."""
    mock_handler = create_mock_handler()
    event_name = "test.event"
    
    # Subscribe handler
    await event_bus.subscribe(event_name, mock_handler)
    assert mock_handler in event_bus._subscribers[event_name]
    
    # Unsubscribe handler
    await event_bus.unsubscribe(event_name, mock_handler)
    assert mock_handler not in event_bus._subscribers[event_name]


@pytest.mark.asyncio
async def test_publish_event(event_bus, test_event):
    """Test publishing events to subscribers."""
    mock_handler1 = create_mock_handler("mock_handler1")
    mock_handler2 = create_mock_handler("mock_handler2")
    event_name = "test.event"
    
    # Create done events to track when handlers complete
    handler1_done = asyncio.Event()
    handler2_done = asyncio.Event()
    task1_done = asyncio.Event()
    task2_done = asyncio.Event()
    
    async def wrapped_handler1(*args, **kwargs):
        try:
            await asyncio.sleep(0)  # Simulate some work
            handler1_done.set()
        finally:
            task1_done.set()
    
    async def wrapped_handler2(*args, **kwargs):
        try:
            await asyncio.sleep(0)  # Simulate some work
            handler2_done.set()
        finally:
            task2_done.set()
    
    mock_handler1.side_effect = wrapped_handler1
    mock_handler2.side_effect = wrapped_handler2
    
    # Subscribe handlers
    await event_bus.subscribe(event_name, mock_handler1)
    await event_bus.subscribe(event_name, mock_handler2)
    
    # Publish event
    event = test_event()
    await event_bus.publish(event)
    
    # Wait for both handlers and tasks to complete
    await asyncio.wait_for(asyncio.gather(
        handler1_done.wait(),
        handler2_done.wait(),
        task1_done.wait(),
        task2_done.wait()
    ), timeout=1)
    
    # Give the event loop a chance to run callbacks
    await asyncio.sleep(0)
    
    # Verify handlers were called
    mock_handler1.assert_called_once_with(event)
    mock_handler2.assert_called_once_with(event)
    assert len(event_bus._pending_tasks) == 0


@pytest.mark.asyncio
async def test_duplicate_event_handling(event_bus, test_event):
    """Test handling of duplicate events."""
    mock_handler = create_mock_handler()
    event_name = "test.event"
    event_id = "test_id"

    # Create a done event to track when the handler is complete
    handler_done = asyncio.Event()
    task_done = asyncio.Event()
    call_count = 0

    async def wrapped_handler(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        try:
            await asyncio.sleep(0)  # Simulate some work
            handler_done.set()
        finally:
            task_done.set()

    mock_handler.side_effect = wrapped_handler

    await event_bus.subscribe(event_name, mock_handler)

    # Create two events with the same ID
    event1 = test_event(event_id=event_id)
    event2 = test_event(event_id=event_id)

    # First publish
    await event_bus.publish(event1)
    await asyncio.wait_for(asyncio.gather(
        handler_done.wait(),
        task_done.wait()
    ), timeout=1)

    # Reset events for second publish
    handler_done.clear()
    task_done.clear()

    # Second publish
    await event_bus.publish(event2)
    await asyncio.wait_for(asyncio.gather(
        handler_done.wait(),
        task_done.wait()
    ), timeout=1)

    # Handler should be called twice since EventBus does not handle duplicates
    assert call_count == 2


@pytest.mark.asyncio
async def test_shutdown(event_bus, test_event):
    """Test event bus shutdown."""
    # Create some pending tasks
    handler_done = asyncio.Event()
    task_done = asyncio.Event()
    mock_handler = create_mock_handler()
    
    async def slow_handler(*args, **kwargs):
        try:
            await asyncio.sleep(0.1)
            handler_done.set()
        finally:
            task_done.set()
    
    mock_handler.side_effect = slow_handler
    event_name = "test.event"
    
    await event_bus.subscribe(event_name, mock_handler)
    event = test_event()
    await event_bus.publish(event)
    
    # Wait for handler and task to complete
    await asyncio.wait_for(asyncio.gather(
        handler_done.wait(),
        task_done.wait()
    ), timeout=1)
    
    # Give the event loop a chance to run callbacks
    await asyncio.sleep(0)
    
    # Verify all tasks completed
    assert len(event_bus._pending_tasks) == 0
    mock_handler.assert_called_once_with(event) 