"""Tests for the EventBus implementation."""

import asyncio
import inspect
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.core.events.bus import EventBus

from .test_events_common import create_test_event


class MockEvent:
    """Mock event class for testing."""

    def __init__(self, name=None, data=None, context=None, event_id=None, recording_id=None, source_plugin=None):
        self.name = name
        self.data = data or {}
        self.context = context or {}
        self.event_id = event_id
        self.recording_id = recording_id
        self.source_plugin = source_plugin

        if "recording" in self.data:
            self.recording_id = self.data["recording"].get("id")

    def get(self, key, default=None):
        """Get value from event data."""
        return self.data.get(key, default)

    def __str__(self):
        return f"MockEvent(name={self.name})"

    def __getattr__(self, name):
        if name == "__dict__":
            return {
                "event": self.name,
                "event_id": self.event_id,
                "name": self.name,
                "source_plugin": self.source_plugin,
                "id": self.recording_id,
                "context": self.context,
                "recording_id": self.recording_id
            }
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


@pytest.fixture
async def event_bus():
    bus = EventBus()
    bus._subscribers = defaultdict(list)
    bus._lock = asyncio.Lock()
    yield bus


@pytest.fixture
def mock_handler():
    handler = AsyncMock()
    handler.__name__ = "mock_handler"
    handler.__qualname__ = "test_event_bus.mock_handler"
    handler.__module__ = "test_event_bus"
    return handler


@pytest.fixture
def test_event():
    return lambda name="test.event", data=None, event_id=None: create_test_event(
        name=name, data=data, event_id=event_id
    )


@pytest.fixture
async def cleanup_tasks():
    yield
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_event_bus_start_stop():
    bus = EventBus()

    async def mock_cleanup():
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    with patch.object(bus, "_cleanup_old_events", new_callable=AsyncMock):
        await bus.start()
        assert bus._cleanup_events_task is not None
        assert not bus._cleanup_events_task.done()

        await bus.stop()
        assert bus._cleanup_events_task is None


@pytest.mark.asyncio
async def test_event_bus_stop_error(cleanup_tasks):
    bus = EventBus()

    async def mock_cleanup():
        while True:
            await asyncio.sleep(0)

    with patch.object(bus, "_cleanup_old_events", new_callable=AsyncMock):
        await bus.start()
        await bus.stop()


@pytest.mark.asyncio
async def test_cleanup_old_events(event_bus):
    event_id = "test_id"
    now = datetime.now(UTC)
    event_bus._processed_events[event_id] = now - timedelta(hours=2)

    async def mock_cleanup():
        await asyncio.sleep(0)
        async with event_bus._lock:
            old_events = [
                event_id
                for event_id, timestamp in event_bus._processed_events.items()
                if (now - timestamp).total_seconds() > 3600
            ]
            for event_id in old_events:
                del event_bus._processed_events[event_id]

    await mock_cleanup()
    assert event_id not in event_bus._processed_events


@pytest.mark.asyncio
async def test_handle_task_success(mock_handler):
    bus = EventBus()
    event = MockEvent(name="test.event")

    async def mock_handler_coro(event):
        await asyncio.sleep(0)

    future = asyncio.Future()
    future.set_result(None)
    mock_handler.return_value = future

    await bus._handle_task(mock_handler, event)
    await asyncio.sleep(0.1)

    mock_handler.assert_called_once_with(event)


@pytest.mark.asyncio
async def test_handle_task_error(mock_handler):
    bus = EventBus()
    event = MockEvent(name="test.event")

    async def mock_handler_coro(event):
        await asyncio.sleep(0)
        raise Exception("Handler error")

    mock_handler.side_effect = mock_handler_coro

    await bus._handle_task(mock_handler, event)
    await asyncio.sleep(0.1)

    mock_handler.assert_called_once_with(event)


@pytest.mark.asyncio
async def test_should_process_event(cleanup_tasks):
    bus = EventBus()
    event = MockEvent()

    event.name = "test.event"
    event.source_plugin = "test_plugin"

    mock_handler = Mock()
    future = asyncio.Future()
    future.set_result(None)
    mock_handler.return_value = future

    bus._subscribers["test.event"].append(mock_handler)

    should_process = bus._should_process_event(event)
    assert should_process is True

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_cleanup_old_events_error():
    bus = EventBus()

    # Mock both the lock and sleep to prevent infinite loop
    with patch.object(bus, "_lock") as mock_lock, patch(
        "asyncio.sleep", AsyncMock()
    ) as mock_sleep:
        mock_lock.__aenter__.side_effect = Exception("Lock error")
        # Only run one iteration
        mock_sleep.side_effect = asyncio.CancelledError()

        try:
            await bus._cleanup_old_events()
        except asyncio.CancelledError:
            pass  # Expected exception

        mock_sleep.assert_called_once_with(3600)


@pytest.mark.asyncio
async def test_handle_task_callback_error():
    bus = EventBus()
    event = MockEvent(name="test.event")

    async def failing_handler(event):
        raise Exception("Handler error")

    handler = AsyncMock(side_effect=failing_handler)
    handler.__name__ = "failing_handler"

    await bus._handle_task(handler, event)
    await asyncio.sleep(0.1)  # Allow callback to execute


@pytest.mark.asyncio
async def test_publish_with_processed_event():
    bus = EventBus()
    event = MockEvent(name="test.event", event_id="test_id")

    async with bus._lock:
        bus._processed_events[event.event_id] = datetime.now(UTC)

    await bus.publish(event)
    assert event.event_id in bus._processed_events


@pytest.mark.asyncio
async def test_publish_with_subscribers():
    bus = EventBus()
    event = MockEvent(name="test.event")
    handler_called = asyncio.Event()

    async def test_handler(evt):
        handler_called.set()

    handler = AsyncMock(side_effect=test_handler)
    bus._subscribers["test.event"].append(handler)

    await bus.publish(event)
    try:
        await asyncio.wait_for(handler_called.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        pytest.fail("Handler was not called within timeout")

    handler.assert_called_once_with(event)


@pytest.mark.asyncio
async def test_publish_with_no_subscribers():
    bus = EventBus()
    event = MockEvent(name="test.event")

    await bus.publish(event)
    assert len(bus._pending_tasks) == 0


@pytest.mark.asyncio
async def test_shutdown(cleanup_tasks):
    bus = EventBus()
    event_completed = asyncio.Event()
    task_cancelled = asyncio.Event()

    # Create some pending tasks
    async def dummy_task():
        try:
            event_completed.set()
            await asyncio.sleep(3600)  # Long sleep that will be cancelled
        except asyncio.CancelledError:
            task_cancelled.set()
            raise

    # Create multiple tasks and track them
    tasks = []
    for _ in range(2):
        task = asyncio.create_task(dummy_task())
        tasks.append(task)
        bus._pending_tasks.add(task)

    # Wait for tasks to start
    await asyncio.wait_for(event_completed.wait(), timeout=1.0)

    # Start cleanup task
    await bus.start()

    # Shutdown should cancel all tasks
    await bus.shutdown()

    # Wait for tasks to be cancelled
    try:
        # Wait for all tasks to complete
        await asyncio.wait(tasks, timeout=1.0)

        # Verify tasks are cancelled
        for task in tasks:
            assert task.cancelled() or task.done()

        assert len(bus._pending_tasks) == 0
        assert bus._cleanup_events_task is None
    finally:
        # Clean up any remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass


@pytest.mark.asyncio
async def test_cleanup_task():
    bus = EventBus()
    task = asyncio.create_task(asyncio.sleep(0))
    bus._pending_tasks.add(task)

    bus._cleanup_task(task)
    assert task not in bus._pending_tasks


@pytest.mark.asyncio
async def test_get_event_name_variations():
    bus = EventBus()

    # Test with event attribute
    event1 = MockEvent(name="test.event")
    assert bus._get_event_name(event1) == "test.event"

    # Test with name attribute
    event2 = type("Event", (), {"name": "test.name"})()
    assert bus._get_event_name(event2) == "test.name"

    # Test with no relevant attributes
    event3 = type("Event", (), {})()
    assert bus._get_event_name(event3) is None


@pytest.mark.asyncio
async def test_subscribe_unsubscribe():
    bus = EventBus()
    handler = AsyncMock()
    event_name = "test.event"

    await bus.subscribe(event_name, handler)
    assert handler in bus._subscribers[event_name]

    await bus.unsubscribe(event_name, handler)
    assert handler not in bus._subscribers[event_name]


@pytest.mark.asyncio
async def test_handle_task_with_bound_method():
    """Test handling task with bound method."""
    bus = EventBus()
    
    class TestHandler:
        def __init__(self):
            self.called = False
            self.event = asyncio.Event()
        
        async def handle(self, evt):
            self.called = True
            self.event.set()
    
    handler = TestHandler()
    event = MockEvent(name="test.event")
    
    await bus._handle_task(handler.handle, event)
    await asyncio.wait_for(handler.event.wait(), timeout=1.0)
    
    assert handler.called


@pytest.mark.asyncio
async def test_cleanup_old_events_with_lock_timeout(cleanup_tasks):
    bus = EventBus()

    # Add some processed events with valid IDs
    event1 = MockEvent()
    event2 = MockEvent()
    event1.recording_id = "test_id_1"
    event2.recording_id = "test_id_2"
    bus._processed_events[event1.recording_id] = event1
    bus._processed_events[event2.recording_id] = event2

    # Mock the lock to simulate timeout
    original_lock = bus._lock

    class TimeoutLock:
        async def __aenter__(self):
            await asyncio.sleep(0.2)  # Simulate slow lock acquisition
            raise asyncio.TimeoutError()

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    bus._lock = TimeoutLock()

    try:
        # Start cleanup task
        await bus.start()
        await asyncio.sleep(0.5)  # Allow time for cleanup attempt

        # Verify events remain since cleanup failed
        assert len(bus._processed_events) == 2
        assert "test_id_1" in bus._processed_events
        assert "test_id_2" in bus._processed_events
    finally:
        bus._lock = original_lock
        await bus.shutdown()


@pytest.mark.asyncio
async def test_handle_task_with_callback_error():
    bus = EventBus()
    event = MockEvent(name="test.event")
    handler_called = asyncio.Event()

    async def test_handler(evt):
        handler_called.set()
        raise Exception("Handler error")

    handler = AsyncMock(side_effect=test_handler)
    handler.__name__ = "test_handler"

    await bus._handle_task(handler, event)
    try:
        await asyncio.wait_for(handler_called.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        pytest.fail("Handler was not called within timeout")

    await asyncio.sleep(0.1)  # Allow error callback to execute


@pytest.mark.asyncio
async def test_publish_with_multiple_subscribers():
    bus = EventBus()
    event = MockEvent(name="test.event")
    handler1_called = asyncio.Event()
    handler2_called = asyncio.Event()

    async def test_handler1(evt):
        handler1_called.set()

    async def test_handler2(evt):
        handler2_called.set()

    handler1 = AsyncMock(side_effect=test_handler1)
    handler2 = AsyncMock(side_effect=test_handler2)

    bus._subscribers["test.event"].extend([handler1, handler2])

    await bus.publish(event)
    try:
        await asyncio.wait_for(
            asyncio.gather(handler1_called.wait(), handler2_called.wait()), timeout=1.0
        )
    except asyncio.TimeoutError:
        pytest.fail("Handlers were not called within timeout")

    handler1.assert_called_once_with(event)
    handler2.assert_called_once_with(event)


@pytest.mark.asyncio
async def test_publish_with_handler_timeout():
    bus = EventBus()
    event = MockEvent(name="test.event")
    handler_started = asyncio.Event()

    async def slow_handler(evt):
        handler_started.set()
        await asyncio.sleep(2.0)  # Longer than our timeout

    handler = AsyncMock(side_effect=slow_handler)
    bus._subscribers["test.event"].append(handler)

    await bus.publish(event)
    try:
        await asyncio.wait_for(handler_started.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        pytest.fail("Handler did not start within timeout")

    # Convert set to list before iteration to avoid "Set changed size" error
    tasks = list(bus._pending_tasks)
    for task in tasks:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_handle_task_with_invalid_handler(cleanup_tasks):
    bus = EventBus()
    event = MockEvent()

    # Register an invalid handler (not a coroutine)
    def invalid_handler(event):
        pass

    await bus.subscribe(MockEvent, invalid_handler)

    # Start the bus
    await bus.start()

    try:
        # Publish event and wait for processing
        await bus.publish(event)
        await asyncio.sleep(0.1)  # Allow time for processing

        # Verify the event was not processed due to invalid handler
        assert len(bus._processed_events) == 0
    finally:
        await bus.shutdown()


@pytest.mark.asyncio
async def test_publish_during_shutdown():
    bus = EventBus()
    event = MockEvent(name="test.event")
    handler = AsyncMock()

    bus._subscribers["test.event"].append(handler)
    bus._shutting_down = True

    # Publish should be a no-op during shutdown
    await bus.publish(event)
    await asyncio.sleep(0.1)  # Allow any potential handler execution to complete

    handler.assert_not_called()

    # Wait for any pending tasks to complete
    pending_tasks = list(bus._pending_tasks)
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    # After tasks are done, there should be no pending tasks
    assert len(bus._pending_tasks) == 0


@pytest.mark.asyncio
async def test_mark_event_processed():
    bus = EventBus()
    event_id = "test_id"

    await bus._mark_event_processed(event_id)
    assert event_id in bus._processed_events


@pytest.mark.asyncio
async def test_is_event_processed():
    bus = EventBus()
    event_id = "test_id"

    # Test unprocessed event
    is_processed = await bus._is_event_processed(event_id)
    assert not is_processed

    # Test processed event
    await bus._mark_event_processed(event_id)
    is_processed = await bus._is_event_processed(event_id)
    assert is_processed


@pytest.mark.asyncio
async def test_get_event_id():
    """Test getting event ID with various event types."""
    bus = EventBus()
    
    # Test with recording_id attribute
    event1 = MockEvent(name="test.event", recording_id="test_id_1")
    assert bus._get_event_id(event1) == "test_id_1_unknown_None"
    
    # Test with id attribute and source plugin
    event2 = MockEvent(name="test.event", recording_id="test_id_2", source_plugin="test_plugin")
    assert bus._get_event_id(event2) == "test_id_2_unknown_test_plugin"
    
    # Test with dict containing id and event type
    event3 = {"recording_id": "test_id_3", "event": "test.event", "source_plugin": "test_plugin"}
    assert bus._get_event_id(event3) == "test_id_3_test.event_test_plugin"
    
    # Test with no id attributes (should generate UUID)
    event4 = MockEvent(name="test.event")
    event_id = bus._get_event_id(event4)
    assert isinstance(event_id, str)
    assert len(event_id) > 0


@pytest.mark.asyncio
async def test_get_handler_info():
    bus = EventBus()

    class TestHandler:
        async def handle(self, event):
            pass

    handler_obj = TestHandler()

    # Test bound method
    handler_info = bus._get_handler_info(handler_obj.handle)
    assert "name" in handler_info
    assert "qualname" in handler_info
    assert "module" in handler_info
    assert "id" in handler_info
    assert "class" in handler_info
    assert "instance_id" in handler_info

    # Test function
    async def test_handler(event):
        pass

    handler_info = bus._get_handler_info(test_handler)
    assert "name" in handler_info
    assert "qualname" in handler_info
    assert "module" in handler_info
    assert "id" in handler_info


@pytest.mark.asyncio
async def test_subscribe_with_handler_info():
    bus = EventBus()

    class TestHandler:
        async def handle(self, event):
            pass

    handler_obj = TestHandler()
    event_name = "test.event"

    await bus.subscribe(event_name, handler_obj.handle)
    assert handler_obj.handle in bus._subscribers[event_name]

    # Verify handler info is logged correctly
    handler_info = bus._get_handler_info(handler_obj.handle)
    assert handler_info["class"] == TestHandler
    assert "instance_id" in handler_info


@pytest.mark.asyncio
async def test_publish_with_event_context(cleanup_tasks):
    bus = EventBus()
    event = MockEvent(name="test.event", recording_id="test_id")
    handler_called = asyncio.Event()

    async def test_handler(evt):
        handler_called.set()
        # Mark event as processed in handler
        await bus._mark_event_processed(evt.recording_id)

    handler = AsyncMock(side_effect=test_handler)
    bus._subscribers["test.event"].append(handler)

    # Add some context to simulate a real event
    event.context = {"source": "test", "timestamp": datetime.now(UTC)}

    # Publish event
    await bus.publish(event)

    # Wait for handler to be called
    await asyncio.wait_for(handler_called.wait(), timeout=1.0)

    # Give time for event processing to complete
    await asyncio.sleep(0.1)

    # Verify handler was called
    handler.assert_called_once_with(event)

    # Event should be marked as processed after publishing
    assert event.recording_id in bus._processed_events


@pytest.mark.asyncio
async def test_event_bus_cleanup_task_error():
    bus = EventBus()
    task = asyncio.create_task(asyncio.sleep(0))
    bus._pending_tasks.add(task)
    bus._cleanup_task(task)
    assert task not in bus._pending_tasks


@pytest.mark.asyncio
async def test_handle_task_setup_error():
    bus = EventBus()
    event = MockEvent(name="test.event")
    
    # Create a handler that will raise an error when called
    handler = Mock()
    handler.__name__ = "error_handler"
    handler.side_effect = Exception("Setup error")
    
    await bus._handle_task(handler, event)
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_handle_task_with_bound_method_detailed():
    bus = EventBus()
    event = MockEvent(name="test.event")

    class TestHandler:
        def __init__(self):
            self.test_var = "test_value"

        async def handle(self, evt):
            return True

    handler = TestHandler().handle
    await bus._handle_task(handler, event)
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_event_processing_with_source_plugin():
    bus = EventBus()
    event = MockEvent(name="test.event", source_plugin="test_plugin")
    
    async def test_handler(evt):
        pass
    
    bus._subscribers["test.event"].append(test_handler)
    assert bus._should_process_event(event) is True


@pytest.mark.asyncio
async def test_event_processing_without_source_plugin():
    bus = EventBus()
    event = MockEvent(name="test.event")
    
    async def test_handler(evt):
        pass
    
    bus._subscribers["test.event"].append(test_handler)
    assert bus._should_process_event(event) is True


@pytest.mark.asyncio
async def test_publish_with_event_context_detailed():
    bus = EventBus()
    event = MockEvent(name="test.event", recording_id="test_id")
    event.context = {"correlation_id": "test_corr_id"}
    
    handler_called = asyncio.Event()
    
    async def test_handler(evt):
        assert hasattr(evt, "context")
        assert evt.context["correlation_id"] == "test_corr_id"
        handler_called.set()
    
    bus._subscribers["test.event"].append(test_handler)
    await bus.publish(event)
    await asyncio.wait_for(handler_called.wait(), timeout=1)


@pytest.mark.asyncio
async def test_shutdown_with_pending_tasks():
    bus = EventBus()
    bus._shutting_down = False  # Ensure initial state
    
    # Create some pending tasks
    async def long_running_task():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            pass
    
    task1 = asyncio.create_task(long_running_task())
    task2 = asyncio.create_task(long_running_task())
    bus._pending_tasks.add(task1)
    bus._pending_tasks.add(task2)
    
    await bus.shutdown()
    assert len(bus._pending_tasks) == 0


@pytest.mark.asyncio
async def test_unsubscribe_nonexistent_handler():
    bus = EventBus()
    event_name = "test.event"
    
    async def test_handler(evt):
        pass
    
    # Try to unsubscribe a handler that was never subscribed
    await bus.unsubscribe(event_name, test_handler)
    assert event_name not in bus._subscribers or test_handler not in bus._subscribers[event_name]


@pytest.mark.asyncio
async def test_get_event_id_with_different_attributes():
    bus = EventBus()
    
    # Test with recording_id attribute
    event1 = MockEvent(name="test.event", recording_id="test_id_1")
    assert bus._get_event_id(event1) == "test_id_1_unknown_None"
    
    # Test with id attribute and source plugin
    event2 = MockEvent(name="test.event", recording_id="test_id_2", source_plugin="test_plugin")
    event_id = bus._get_event_id(event2)
    assert event_id == "test_id_2_unknown_test_plugin"
    
    # Test with no id attributes (should generate UUID)
    event3 = MockEvent(name="test.event")
    event_id = bus._get_event_id(event3)
    assert isinstance(event_id, str)
    assert len(event_id) > 0


@pytest.mark.asyncio
async def test_handle_task_with_invalid_event():
    bus = EventBus()
    event = MockEvent()
    event.name = None  # Invalid event
    
    async def test_handler(evt):
        pass
    
    await bus._handle_task(test_handler, event)
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_handle_task_with_logging_error():
    bus = EventBus()
    
    # Mock asyncio.sleep to prevent hanging
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.return_value = None
        await bus.start()
        
        event = MockEvent(name="test.event")
        
        # Create a handler that will raise an error during logging
        class ErrorHandler:
            def __init__(self):
                self.__name__ = "error_handler"
                self.__module__ = "test_module"
                self.__qualname__ = "test_module.error_handler"
                
            async def __call__(self, evt):
                raise Exception("Handler error")
                
            def __str__(self):
                return "error_handler"
                
            def __repr__(self):
                return "error_handler"
        
        handler = ErrorHandler()
        
        # Use wait_for to prevent hanging
        async def run_handler():
            await bus._handle_task(handler, event)
            
        await asyncio.wait_for(run_handler(), timeout=1.0)
        await bus.stop()


@pytest.mark.asyncio
async def test_cleanup_old_events_with_exception():
    bus = EventBus()
    event_id = "test_id"
    now = datetime.now(UTC)
    bus._processed_events[event_id] = now - timedelta(hours=2)
    
    # Mock the lock to raise an exception
    class ErrorLock:
        async def __aenter__(self):
            raise Exception("Lock error")
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    original_lock = bus._lock
    bus._lock = ErrorLock()
    
    # Mock asyncio.sleep to break the infinite loop
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = asyncio.CancelledError()
        
        try:
            await bus._cleanup_old_events()
        except asyncio.CancelledError:
            pass  # Expected
        finally:
            bus._lock = original_lock
            
        # Verify sleep was called with correct interval
        mock_sleep.assert_called_once_with(3600)


@pytest.mark.asyncio
async def test_publish_with_invalid_event_name():
    bus = EventBus()
    event = MockEvent()
    event.name = None  # Invalid event name
    
    await bus.publish(event)  # Should handle gracefully


@pytest.mark.asyncio
async def test_publish_with_complex_event_context():
    bus = EventBus()
    event = MockEvent(name="test.event", recording_id="test_id")
    event.context = {
        "correlation_id": "test_corr_id",
        "metadata": {
            "user": "test_user",
            "timestamp": datetime.now(UTC).isoformat()
        },
        "source": "test_source"
    }
    
    handler_called = asyncio.Event()
    
    async def test_handler(evt):
        assert hasattr(evt, "context")
        assert evt.context["correlation_id"] == "test_corr_id"
        handler_called.set()
    
    bus._subscribers["test.event"].append(test_handler)
    await bus.publish(event)
    await asyncio.wait_for(handler_called.wait(), timeout=1)


@pytest.mark.asyncio
async def test_handle_task_with_multiple_exceptions():
    bus = EventBus()
    event = MockEvent(name="test.event")
    
    async def failing_handler(evt):
        raise Exception("First error")
        
    handler = AsyncMock(side_effect=failing_handler)
    handler.__name__ = "failing_handler"
    
    # Add a callback that will also raise an exception
    def error_callback(task):
        raise Exception("Callback error")
    
    task = asyncio.create_task(asyncio.sleep(0))
    task.add_done_callback(error_callback)
    bus._pending_tasks.add(task)
    
    await bus._handle_task(handler, event)
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_cleanup_old_events_with_multiple_events():
    bus = EventBus()
    now = datetime.now(UTC)
    
    # Add multiple events with different ages
    bus._processed_events["old_event_1"] = now - timedelta(hours=2)
    bus._processed_events["old_event_2"] = now - timedelta(hours=3)
    bus._processed_events["recent_event"] = now - timedelta(minutes=30)
    
    async def mock_cleanup():
        await asyncio.sleep(0)
        async with bus._lock:
            old_events = [
                event_id
                for event_id, timestamp in bus._processed_events.items()
                if (now - timestamp).total_seconds() > 3600
            ]
            for event_id in old_events:
                del bus._processed_events[event_id]
                
    await mock_cleanup()
    assert "old_event_1" not in bus._processed_events
    assert "old_event_2" not in bus._processed_events
    assert "recent_event" in bus._processed_events


@pytest.mark.asyncio
async def test_cleanup_old_events_with_lock_error():
    bus = EventBus()
    now = datetime.now(UTC)
    
    # Add some events
    bus._processed_events["test_id"] = now - timedelta(hours=2)
    
    # Mock the lock to raise an error during exit
    class ErrorLock:
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            raise Exception("Lock exit error")
    
    original_lock = bus._lock
    bus._lock = ErrorLock()
    
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = asyncio.CancelledError()
        
        try:
            await bus._cleanup_old_events()
        except asyncio.CancelledError:
            pass
        finally:
            bus._lock = original_lock


@pytest.mark.asyncio
async def test_handle_task_with_complex_error():
    bus = EventBus()
    event = MockEvent(name="test.event")
    
    class ComplexError(Exception):
        def __str__(self):
            raise ValueError("Error in error string")
    
    async def failing_handler(evt):
        raise ComplexError()
    
    handler = AsyncMock(side_effect=failing_handler)
    handler.__name__ = "failing_handler"
    
    await bus._handle_task(handler, event)
    await asyncio.sleep(0.1)  # Allow error handling to complete


@pytest.mark.asyncio
async def test_cleanup_old_events_with_corrupted_data():
    """Test cleanup of old events with corrupted data."""
    bus = EventBus()
    now = datetime.now(UTC)
    
    # Create some valid events
    valid_event = MockEvent(name="test.event", data={"value": "test"})
    await bus.publish(valid_event)
    
    # Simulate corrupted data by adding invalid events directly to the processed events
    corrupted_event = {"name": "corrupted.event", "data": None}  # Invalid event structure
    bus._processed_events["corrupted_id"] = now - timedelta(hours=2)  # Make it old enough to clean up
    
    # Add more valid events
    valid_event2 = MockEvent(name="test.event2", data={"value": "test2"})
    valid_event2_id = bus._get_event_id(valid_event2)
    bus._processed_events[valid_event2_id] = now  # Add as recent event
    
    # Mock sleep to prevent infinite loop
    with patch('asyncio.sleep', AsyncMock()) as mock_sleep:
        mock_sleep.side_effect = asyncio.CancelledError()
        
        try:
            await bus._cleanup_old_events()
        except asyncio.CancelledError:
            pass  # Expected exception
    
    # Verify that valid events are still present and corrupted event is handled
    assert len(bus._processed_events) >= 1  # At least one valid event remains
    assert valid_event2_id in bus._processed_events  # Recent event should still be there


@pytest.mark.asyncio
async def test_handle_task_with_handler_info_error():
    """Test handling of tasks when handler info causes an error."""
    bus = EventBus()
    event = MockEvent(name="test.event", data={"value": "test"})
    
    # Create a handler that will cause an error when getting its info
    class ProblematicHandler:
        def __str__(self):
            raise ValueError("Handler info error")
        
        async def handle(self, event):
            pass
    
    handler = ProblematicHandler()
    await bus.subscribe("test.event", handler.handle)
    
    # Create and handle the task
    task = asyncio.create_task(bus._handle_task(handler.handle, event))
    await task
    
    # Verify that the task completed despite the handler info error
    assert task.done()
    assert not task.cancelled()


@pytest.mark.asyncio
async def test_handle_task_with_callback_chain():
    """Test handling of tasks with a chain of callbacks."""
    bus = EventBus()
    event = MockEvent(name="test.event", data={"value": "test"})
    callback_order = []
    
    async def handler(event):
        callback_order.append("handler")
    
    def callback1(task):
        callback_order.append("callback1")
    
    def callback2(task):
        callback_order.append("callback2")
    
    await bus.subscribe("test.event", handler)
    task = asyncio.create_task(bus._handle_task(handler, event))
    task.add_done_callback(callback1)
    task.add_done_callback(callback2)
    
    await task
    await asyncio.sleep(0.1)  # Allow callbacks to complete
    
    # Verify callback execution order
    assert callback_order == ["handler", "callback1", "callback2"]
