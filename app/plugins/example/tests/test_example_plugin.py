from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from app.plugins.base import PluginConfig
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
        
        # Store subscriptions
        mock_bus.subscriptions = {}
        
        async def mock_subscribe(event_type, callback):
            if event_type not in mock_bus.subscriptions:
                mock_bus.subscriptions[event_type] = []
            mock_bus.subscriptions[event_type].append(callback)
            
        async def mock_unsubscribe(event_type, callback):
            if event_type in mock_bus.subscriptions:
                mock_bus.subscriptions[event_type].remove(callback)
                
        async def mock_publish(event):
            event_type = event.get("name")
            if event_type in mock_bus.subscriptions:
                for callback in mock_bus.subscriptions[event_type]:
                    await callback(event)
        
        mock_bus.subscribe = mock_subscribe
        mock_bus.unsubscribe = mock_unsubscribe
        mock_bus.publish = mock_publish
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
            assert plugin._handle_recording_ended in plugin.event_bus.subscriptions["recording.ended"]

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
        # Mock event data
        event = {
            "context": {"event_id": "test_id", "event_type": "recording.ended"},
            "data": {"recording_id": "test_recording"},
        }

        # Initialize plugin
        with patch("app.plugins.example.plugin.ThreadPoolExecutor"):
            await plugin.initialize()

            # Test event handling
            await plugin._handle_recording_ended(event)

    async def test_handle_recording_ended_no_context(self, plugin):
        """Test recording ended handler with missing context"""
        event = {"data": {"recording_id": "test_recording"}}

        # Initialize plugin
        with patch("app.plugins.example.plugin.ThreadPoolExecutor"):
            await plugin.initialize()

            # Test event handling with missing context
            await plugin._handle_recording_ended(event)

    async def test_name_property(self, plugin):
        """Test name property returns correct value"""
        assert plugin.name == "example"
