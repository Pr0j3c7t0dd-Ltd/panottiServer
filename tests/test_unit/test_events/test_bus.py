"""Tests for event bus."""

import asyncio
from datetime import datetime, timedelta, UTC
import pytest
import pytest_asyncio
import warnings
from unittest.mock import patch, AsyncMock, MagicMock

# Suppress specific warning about EventBus._cleanup_old_events coroutine
warnings.filterwarnings("ignore", message="coroutine 'EventBus._cleanup_old_events' was never awaited")

from app.core.events.bus import EventBus


@pytest.fixture
async def cleanup_tasks():
    yield
    # Get all tasks except current task
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        # Wait with timeout for tasks to cancel
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=1.0)
        except asyncio.TimeoutError:
            pass  # Some tasks may be stuck, continue cleanup


@pytest_asyncio.fixture
async def event_bus():
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
                await asyncio.wait_for(bus._cleanup_events_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            bus._cleanup_events_task = None
        
        yield bus
        
        # Cleanup
        await bus.stop()
        await bus.shutdown()
        
        # Ensure all tasks are cleaned up
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass


@pytest.mark.asyncio
async def test_start_error_handling():
    """Test error handling during event bus start."""
    # Create a new event bus instance
    bus = EventBus()
    
    # Mock create_task to raise an error
    with patch('asyncio.create_task', side_effect=Exception("Test error")):
        try:
            await bus.start()
        except Exception:
            pass
        assert bus._cleanup_events_task is None


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.asyncio
async def test_cleanup_old_events_error(event_bus, caplog):
    """Test that cleanup task handles errors gracefully."""
    # Create a mock lock with proper async context manager behavior
    class ErrorLock:
        async def __aenter__(self):
            raise Exception("Lock error")
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    original_lock = event_bus._lock
    event_bus._lock = ErrorLock()
    
    # Create and run the cleanup task
    cleanup_task = asyncio.create_task(event_bus._cleanup_old_events(run_once=True))
    
    try:
        await cleanup_task
    except Exception:
        pass  # Expected to raise
    finally:
        event_bus._lock = original_lock
        
        # Cancel task if still running
        if not cleanup_task.done():
            cleanup_task.cancel()
            try:
                await cleanup_task
            except (asyncio.CancelledError, Exception):
                pass
            
    # Verify error was logged
    error_records = [r for r in caplog.records if r.levelname == 'ERROR']
    assert len(error_records) == 1, "Expected one error log record"
    assert "Error cleaning up old events" in error_records[0].message
    assert "Lock error" in error_records[0].error, "Lock error not found in error details"


@pytest.mark.asyncio
async def test_event_name_extraction_edge_cases(event_bus):
    """Test edge cases for event name extraction."""
    # Test with None event
    assert event_bus._get_event_name(None) is None

    # Test with event object missing name
    class NoNameEvent:
        pass
    assert event_bus._get_event_name(NoNameEvent()) is None

    # Test with event object with error in name property
    class ErrorEvent:
        @property
        def event(self):
            raise Exception("Test error")
    assert event_bus._get_event_name(ErrorEvent()) is None

    # Test with dict missing event key
    assert event_bus._get_event_name({}) is None


@pytest.mark.asyncio
async def test_event_id_extraction_edge_cases(event_bus):
    """Test edge cases for event ID extraction."""
    # Test with None event
    event_id = event_bus._get_event_id(None)
    assert isinstance(event_id, str)
    assert len(event_id) == 36  # UUID length

    # Test with event object with error in to_dict
    class ErrorDictEvent:
        def to_dict(self):
            raise Exception("Test error")
    event_id = event_bus._get_event_id(ErrorDictEvent())
    assert isinstance(event_id, str)
    assert len(event_id) == 36


@pytest.mark.asyncio
async def test_event_processing_decision_edge_cases(event_bus, cleanup_tasks):
    """Test edge cases for event processing decisions."""
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        mock_sleep.return_value = None
        
        # Test with None event
        assert event_bus._should_process_event(None) is True

        # Test with event object with error in context access
        class ErrorContextEvent:
            @property
            def context(self):
                raise Exception("Test error")
        assert event_bus._should_process_event(ErrorContextEvent()) is True

        # Test with dict with error in source_plugin access
        mock_source_plugin = MagicMock()
        mock_source_plugin.side_effect = Exception("Test error")
        event_dict = {"context": {"source_plugin": mock_source_plugin}}
        assert event_bus._should_process_event(event_dict) is True


@pytest.mark.asyncio
async def test_subscribe_edge_cases(event_bus):
    """Test edge cases for event subscription."""
    # Test subscribing None handler
    with pytest.raises(ValueError, match="Handler cannot be None"):
        await event_bus.subscribe("test_event", None)

    # Test subscribing non-callable handler
    class NotCallable:
        pass
    not_callable = NotCallable()
    with pytest.raises(ValueError, match="Handler must be callable"):
        await event_bus.subscribe("test_event", not_callable)

    # Test subscribing to empty event name
    with pytest.raises(ValueError, match="Event name cannot be empty"):
        await event_bus.subscribe("", lambda x: None)


@pytest.mark.asyncio
async def test_unsubscribe_edge_cases(event_bus):
    """Test edge cases for event unsubscription."""
    # Test unsubscribing non-existent handler
    async def test_handler(event):
        pass
    
    # Should not raise exception
    await event_bus.unsubscribe("test_event", test_handler)


@pytest.mark.asyncio
async def test_handler_info_edge_cases(event_bus):
    """Test edge cases for handler info extraction."""
    # Test with None handler
    info = event_bus._get_handler_info(None)
    assert isinstance(info, dict)
    assert info["name"] == "None"
    assert info["module"] is None
    assert info["class"] is None
    assert info["id"] is None


@pytest.mark.asyncio
async def test_publish_validation(event_bus):
    """Test event publishing validation."""
    # Test publishing None event
    await event_bus.publish(None)

    # Test publishing invalid event type
    await event_bus.publish(123)


@pytest.mark.asyncio
async def test_publish_error_handling(event_bus):
    """Test error handling during event publishing."""
    # Test with event that raises error during processing
    class ErrorEvent:
        def __getattribute__(self, _):
            raise Exception("Test error")

    await event_bus.publish(ErrorEvent())


@pytest.mark.asyncio
async def test_event_processing_error(event_bus):
    """Test error handling during event processing."""
    # Create event that will cause processing error
    event = {"id": "test_id", "event": "test_event"}
    
    # Mock _is_event_processed to raise exception
    with patch.object(event_bus, '_is_event_processed', side_effect=Exception("Test error")):
        await event_bus.publish(event)


@pytest.mark.asyncio
async def test_shutdown_error(event_bus):
    """Test error handling during shutdown."""
    # Create a task that will raise error during shutdown
    async def error_task():
        raise Exception("Test error")
    
    task = asyncio.create_task(error_task())
    event_bus._pending_tasks.add(task)
    
    await event_bus.shutdown()
    assert len(event_bus._pending_tasks) == 0


@pytest.mark.asyncio
async def test_handler_callback_error(event_bus):
    """Test error handling in handler callback."""
    async def error_handler(event):
        raise Exception("Test error")

    # Subscribe error handler
    await event_bus.subscribe("test_event", error_handler)
    
    # Publish event to trigger handler
    event = {"event": "test_event", "id": "test_id"}
    await event_bus.publish(event)
    
    # Wait for handler to complete
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_handler_callback_cleanup_error(event_bus):
    """Test error handling in handler callback cleanup."""
    async def normal_handler(event):
        pass

    # Create a task that will raise during cleanup
    task = asyncio.create_task(normal_handler({"event": "test"}))
    event_bus._pending_tasks.add(task)
    
    # Force task to raise during cleanup
    with patch.object(task, 'exception', side_effect=Exception("Test error")):
        event_bus._cleanup_task(task)
    
    assert task not in event_bus._pending_tasks


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


@pytest.mark.asyncio
async def test_stop_with_pending_tasks(event_bus):
    """Test stopping event bus with pending tasks."""
    async def long_task():
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(long_task())
    event_bus._pending_tasks.add(task)
    await event_bus.stop()
    assert not event_bus._pending_tasks


@pytest.mark.asyncio
async def test_cleanup_task_error_handling(event_bus):
    """Test error handling in cleanup task."""
    task = MagicMock()
    task.exception.side_effect = Exception("Test error")
    event_bus._pending_tasks.add(task)
    event_bus._cleanup_task(task)
    assert task not in event_bus._pending_tasks


@pytest.mark.asyncio
async def test_handler_done_callback_error(event_bus):
    """Test error handling in handler done callback."""
    async def error_handler(event):
        raise Exception("Test error")

    # Create task and add to pending tasks
    task = asyncio.create_task(error_handler({"event": "test"}))
    event_bus._pending_tasks.add(task)
    
    # Wait for task to complete
    try:
        await task
    except Exception:
        pass

    # Let the event loop process callbacks
    await asyncio.sleep(0)
    
    # Add the task's done callback and call it directly
    event_bus._cleanup_task(task)
    
    # Let the event loop process callbacks again
    await asyncio.sleep(0)
    
    # Verify task is removed from pending tasks
    assert task not in event_bus._pending_tasks
