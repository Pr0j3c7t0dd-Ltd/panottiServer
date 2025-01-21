from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from app.core.events import Event, EventContext
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
        """Mock event bus fixture"""
        mock_bus = MagicMock()
        mock_bus.publish = AsyncMock()
        mock_bus.subscribe = AsyncMock()
        mock_bus.unsubscribe = AsyncMock()
        mock_bus._pending_tasks = set()
        return mock_bus

    @pytest.fixture
    def plugin(self, plugin_config, event_bus):
        """Example plugin instance"""
        return ExamplePlugin(plugin_config, event_bus)

    async def test_example_plugin_initialization(self, plugin):
        """Test example plugin specific initialization"""
        with patch("app.plugins.example.plugin.ThreadPoolExecutor") as mock_executor:
            await plugin.initialize()

            # Verify thread pool created with correct workers
            mock_executor.assert_called_once_with(max_workers=2)

            # Verify event subscription
            assert plugin.event_bus is not None
            assert "recording.ended" in plugin.event_bus.subscriptions
            assert (
                plugin._handle_recording_ended
                in plugin.event_bus.subscriptions["recording.ended"]
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

    async def test_handle_recording_ended(self, plugin):
        """Test recording ended event handler"""
        # Mock event data with context and metadata
        event = Event.create(
            name="recording.ended",
            data={"recording_id": "test_recording"},
            correlation_id="test_correlation_id",
            source_plugin="test_source",
            context=EventContext(
                correlation_id="test_correlation_id",
                source_plugin="test_source",
                metadata={"test_key": "test_value"}
            )
        )

        # Initialize plugin
        with patch("app.plugins.example.plugin.ThreadPoolExecutor"):
            await plugin.initialize()

            # Test event handling
            await plugin._handle_recording_ended(event)

            # Verify completion event was published with correct metadata
            published_events = [call.args[0] for call in plugin.event_bus.publish.call_args_list]
            assert len(published_events) == 1
            completion_event = published_events[0]
            
            # Verify event structure
            assert completion_event.name == "example.completed"
            assert completion_event.correlation_id == "test_correlation_id"
            assert completion_event.source_plugin == "example"
            assert completion_event.context.correlation_id == "test_correlation_id"
            assert completion_event.context.source_plugin == "example"
            assert completion_event.context.metadata == {"test_key": "test_value"}

    async def test_handle_recording_ended_no_context(self, plugin):
        """Test recording ended handler with missing context"""
        # Mock event data without context
        event = {"data": {"recording_id": "test_recording"}}

        # Initialize plugin
        with patch("app.plugins.example.plugin.ThreadPoolExecutor"):
            await plugin.initialize()

            # Test event handling with missing context
            await plugin._handle_recording_ended(event)

            # Verify completion event was published with default metadata
            published_events = [call.args[0] for call in plugin.event_bus.publish.call_args_list]
            assert len(published_events) == 1
            completion_event = published_events[0]
            
            # Verify event structure
            assert completion_event.name == "example.completed"
            assert completion_event.source_plugin == "example"
            assert completion_event.context.source_plugin == "example"
            assert completion_event.context.metadata == {}

    async def test_handle_recording_ended_error(self, plugin):
        """Test recording ended handler error case"""
        # Mock event data with context and metadata
        event = Event.create(
            name="recording.ended",
            data={"recording_id": "test_recording"},
            correlation_id="test_correlation_id",
            source_plugin="test_source"
        )
        # Set additional context metadata after creation
        event.context.metadata = {"test_key": "test_value"}

        # Set up the mock to track calls and raise exception on first call
        mock_publish = AsyncMock()
        mock_publish.side_effect = [Exception("Test error"), None]
        plugin.event_bus.publish = mock_publish

        # Initialize plugin
        with patch("app.plugins.example.plugin.ThreadPoolExecutor"):
            await plugin.initialize()

            # Test event handling with error
            with pytest.raises(Exception):
                await plugin._handle_recording_ended(event)

            # Verify error event was published
            assert mock_publish.call_count == 2  # Both completion and error events
            error_event = mock_publish.call_args.args[0]
            
            # Verify error event structure
            assert error_event.name == "example.error"
            assert error_event.correlation_id == "test_correlation_id"
            assert error_event.source_plugin == "example"
            assert error_event.context.correlation_id == "test_correlation_id"
            assert error_event.context.source_plugin == "example"
            assert error_event.context.metadata == {"test_key": "test_value"}
            assert "Test error" in error_event.data["example"]["error"]

    async def test_name_property(self, plugin):
        """Test name property returns correct value"""
        assert plugin.name == "example"
