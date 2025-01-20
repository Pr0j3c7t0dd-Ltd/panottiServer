"""Tests for the EventBus implementation."""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch
from collections import defaultdict
import uuid

from app.core.events.bus import EventBus
from .test_events_common import create_test_event, create_mock_handler


class MockEvent:
    """Mock event class for testing."""
    def __init__(self, name=None, event_id=None, event=None, source_plugin=None):
        self.event = name
        self.event_id = event_id
        self.name = name
        self.source_plugin = source_plugin

    def __str__(self):
        return f"MockEvent(name={self.name}, event_id={self.event_id}, event={self.event})"

    def __getattr__(self, name):
        if name == "__dict__":
            return {
                "event": self.event,
                "event_id": self.event_id,
                "name": self.name,
                "source_plugin": self.source_plugin
            }
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


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
        name=name,
        data=data,
        event_id=event_id
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

    with patch.object(bus, '_cleanup_old_events', new_callable=AsyncMock):
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

    with patch.object(bus, '_cleanup_old_events', new_callable=AsyncMock):
        await bus.start()
        await bus.stop()


@pytest.mark.asyncio
async def test_cleanup_old_events(event_bus):
    event_id = "test_id"
    now = datetime.now(timezone.utc)
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
