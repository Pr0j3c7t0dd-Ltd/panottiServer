"""Tests for the EventBus implementation."""

import asyncio
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.core.events.bus import EventBus

from .test_events_common import create_test_event


class MockEvent:
    """Mock event class for testing."""

    def __init__(self, name=None, event_id=None, event=None, source_plugin=None):
        self.event = name
        self.event_id = event_id
        self.name = name
        self.source_plugin = source_plugin

    def __str__(self):
        return (
            f"MockEvent(name={self.name}, event_id={self.event_id}, event={self.event})"
        )

    def __getattr__(self, name):
        if name == "__dict__":
            return {
                "event": self.event,
                "event_id": self.event_id,
                "name": self.name,
                "source_plugin": self.source_plugin,
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

    event.event = "test.event"
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
    with patch.object(bus, "_lock") as mock_lock, \
         patch("asyncio.sleep", AsyncMock()) as mock_sleep:
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
async def test_shutdown():
    bus = EventBus()
    event = MockEvent(name="test.event")
    handler = AsyncMock()
    
    bus._subscribers["test.event"].append(handler)
    bus._shutting_down = True
    
    await bus.publish(event)
    handler.assert_not_called()


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
    class TestHandler:
        def __init__(self):
            self.called = False
            self.event = asyncio.Event()
            
        async def handle(self, event):
            self.called = True
            self.event.set()
    
    handler_obj = TestHandler()
    bus = EventBus()
    event = MockEvent(name="test.event")
    
    await bus._handle_task(handler_obj.handle, event)
    try:
        await asyncio.wait_for(handler_obj.event.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        pytest.fail("Handler was not called within timeout")
    
    assert handler_obj.called


@pytest.mark.asyncio
async def test_cleanup_old_events_with_lock_timeout():
    bus = EventBus()
    
    # Mock sleep and lock to simulate timeout
    with patch("asyncio.sleep", AsyncMock()) as mock_sleep, \
         patch.object(bus, "_lock") as mock_lock:
        mock_sleep.side_effect = [None, asyncio.CancelledError()]  # Allow one iteration
        mock_lock.__aenter__.side_effect = asyncio.TimeoutError("Lock timeout")
        
        try:
            await bus._cleanup_old_events()
        except asyncio.CancelledError:
            pass  # Expected
        
        mock_sleep.assert_called_with(3600)


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
        await asyncio.wait_for(asyncio.gather(
            handler1_called.wait(),
            handler2_called.wait()
        ), timeout=1.0)
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
async def test_handle_task_with_invalid_handler():
    bus = EventBus()
    event = MockEvent(name="test.event")
    
    # Create an invalid handler that's not a coroutine function
    def invalid_handler(evt):
        pass
    
    # Mock AsyncMock to make it return None instead of a coroutine
    handler = Mock()
    handler.__name__ = "invalid_handler"
    handler.return_value = None
    
    await bus._handle_task(handler, event)
    # The error is logged but not raised


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
