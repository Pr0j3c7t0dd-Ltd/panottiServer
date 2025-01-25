from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from app.core.events import Event, EventContext, ConcreteEventBus
from app.core.plugins import PluginConfig
from app.plugins.example.plugin import ExamplePlugin
from tests.plugins.test_plugin_interface import BasePluginTest


class TestExamplePlugin(BasePluginTest):
    """Test suite for ExamplePlugin"""

    @pytest.fixture
    def plugin_config(self):
        """Example plugin specific config"""
        return PluginConfig(
            name="example",
            version="1.0.0",
            enabled=True,
            config={
                "max_concurrent_tasks": 2,
                "debug_mode": True,
                "example_setting": "test",
            },
        )

    @pytest.fixture
    def event_bus(self):
        """Event bus fixture with mocked methods"""
        bus = ConcreteEventBus()
        bus.subscribe = AsyncMock()
        bus.publish = AsyncMock()
        return bus

    @pytest.fixture
    def plugin(self, plugin_config, event_bus):
        """Example plugin instance"""
        return ExamplePlugin(plugin_config, event_bus)

    async def test_event_bus_methods(self, plugin, event_bus):
        """Test event bus integration methods"""
        test_event = {"name": "test_event", "data": {}}

        # Test subscribe/unsubscribe
        callback_called = False

        async def test_callback(event):
            nonlocal callback_called
            callback_called = True

        # Mock subscribe to store the callback
        callbacks = []
        async def mock_subscribe(event_name, callback):
            callbacks.append((event_name, callback))
        plugin.event_bus.subscribe = AsyncMock(side_effect=mock_subscribe)

        # Mock unsubscribe to remove the callback
        async def mock_unsubscribe(event_name, callback):
            callbacks.remove((event_name, callback))
        plugin.event_bus.unsubscribe = AsyncMock(side_effect=mock_unsubscribe)

        # Mock publish to call the stored callback
        async def mock_publish(event):
            for event_name, callback in callbacks:
                if event_name == event["name"]:
                    await callback(event)
        plugin.event_bus.publish = AsyncMock(side_effect=mock_publish)

        # Subscribe and publish
        await plugin.subscribe("test_event", test_callback)
        await plugin.publish(test_event)

        assert callback_called

        # Test unsubscribe
        callback_called = False
        await plugin.unsubscribe("test_event", test_callback)
        await plugin.publish(test_event)

        assert not callback_called

    async def test_example_plugin_initialization(self, plugin):
        """Test example plugin specific initialization"""
        with patch("app.plugins.example.plugin.ThreadPoolExecutor") as mock_executor:
            await plugin.initialize()

            # Verify thread pool created with correct workers
            mock_executor.assert_called_once_with(max_workers=2)

            # Verify event subscription
            plugin.event_bus.subscribe.assert_called_once_with(
                "recording.ended", plugin._handle_recording_ended
            )

    async def test_example_plugin_shutdown(self, plugin):
        """Test example plugin specific shutdown"""
        # Initialize plugin first
        with patch("app.plugins.example.plugin.ThreadPoolExecutor") as mock_executor:
            await plugin.initialize()
            instance = mock_executor.return_value

            # Test shutdown
            await plugin.shutdown()
            instance.shutdown.assert_called_once_with(wait=True)

    async def test_handle_recording_ended_no_context(self, plugin):
        """Test recording ended handler with missing context"""
        # Mock event data without context
        event = {"data": {"recording": {"id": "test_recording"}}}

        # Initialize plugin
        with patch("app.plugins.example.plugin.ThreadPoolExecutor"):
            await plugin.initialize()

            # Test event handling with missing context
            await plugin._handle_recording_ended(event)

            # Verify completion event was published
            assert plugin.event_bus.publish.call_count == 1
            completion_event = plugin.event_bus.publish.call_args.args[0]
            
            # Verify event structure
            assert completion_event.name == "example.completed"
            assert completion_event.data["context"]["source_plugin"] == "example"
            assert completion_event.data["metadata"] == {}

    async def test_handle_recording_ended_error(self, plugin):
        """Test recording ended handler error case"""
        # Mock event data with context and metadata
        event = {
            "name": "recording.ended",
            "data": {
                "recording": {
                    "id": "test_recording"
                },
                "metadata": {"test_key": "test_value"}
            },
            "correlation_id": "test_correlation_id",
            "source_plugin": "test_source",
            "context": {
                "correlation_id": "test_correlation_id",
                "source_plugin": "test_source",
                "metadata": {"test_key": "test_value"}
            }
        }

        # Set up the mock to track calls and raise exception on first call
        mock_publish = AsyncMock()
        mock_publish.side_effect = [Exception("Test error"), None]  # First call raises, second succeeds
        plugin.event_bus.publish = mock_publish

        # Initialize plugin
        with patch("app.plugins.example.plugin.ThreadPoolExecutor"):
            await plugin.initialize()

            # Test event handling with error
            with pytest.raises(Exception):
                await plugin._handle_recording_ended(event)

            # Verify both events were attempted to be published
            assert mock_publish.call_count == 2  # Both completion and error events
            error_event = mock_publish.call_args_list[1].args[0]  # Get the second call's args
            
            # Verify error event structure
            assert error_event.name == "example.error"
            assert error_event.data["recording"] == {"id": "test_recording"}
            assert error_event.data["example"]["status"] == "error"
            assert "Test error" in error_event.data["example"]["error"]
            assert error_event.data["metadata"] == {"test_key": "test_value"}
            assert error_event.context.source_plugin == "example"

    async def test_name_property(self, plugin):
        """Test name property returns correct value"""
        assert plugin.name == "example"
