import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from datetime import datetime

from app.plugins.base import PluginConfig
from app.plugins.cleanup_files.plugin import CleanupFilesPlugin
from app.core.events import Event, ConcreteEventBus as EventBus
from app.models.recording.events import RecordingEvent
from tests.plugins.test_plugin_interface import BasePluginTest


class TestCleanupFilesPlugin(BasePluginTest):
    """Test suite for CleanupFilesPlugin"""

    @pytest.fixture
    def plugin_config(self):
        """Cleanup files plugin specific config"""
        return PluginConfig(
            name="cleanup_files",
            version="1.0.0",
            enabled=True,
            config={
                "include_dirs": ["data", "temp"],
                "exclude_dirs": ["protected"],
                "cleanup_delay": 5
            }
        )

    @pytest.fixture
    def event_bus(self):
        """Mock event bus fixture"""
        event_bus = EventBus()
        event_bus.subscribe = AsyncMock()
        event_bus.unsubscribe = AsyncMock()
        event_bus.publish = AsyncMock()
        return event_bus

    @pytest.fixture
    def plugin(self, plugin_config, event_bus):
        """Cleanup files plugin instance"""
        return CleanupFilesPlugin(plugin_config, event_bus)

    async def test_cleanup_files_initialization(self, plugin):
        """Test cleanup files plugin specific initialization"""
        await plugin.initialize()
        
        # Verify thread pool initialization
        assert plugin._executor is not None
        
        # Verify event subscription
        plugin.event_bus.subscribe.assert_called_once_with(
            "desktop_notification.completed",
            plugin.handle_desktop_notification_completed
        )

    async def test_cleanup_files_shutdown(self, plugin):
        """Test cleanup files plugin specific shutdown"""
        await plugin.initialize()
        
        # Mock the executor's shutdown method
        plugin._executor.shutdown = MagicMock()
        
        await plugin.shutdown()
        
        # Verify event unsubscription
        plugin.event_bus.unsubscribe.assert_called_once_with(
            "desktop_notification.completed",
            plugin.handle_desktop_notification_completed
        )
        
        # Verify thread pool shutdown
        plugin._executor.shutdown.assert_called_once()

    async def test_handle_desktop_notification_completed_dict(self, plugin):
        """Test handling desktop notification completed with dict event"""
        event_data = {
            "recording_id": "test_recording",
            "data": {
                "recording_id": "test_recording"
            }
        }

        with patch.object(plugin, '_cleanup_files') as mock_cleanup:
            mock_cleanup.return_value = ["file1.txt", "file2.txt"]
            
            await plugin.initialize()
            await plugin.handle_desktop_notification_completed(event_data)
            
            mock_cleanup.assert_called_once_with("test_recording")
            
            # Verify completion event was published
            plugin.event_bus.publish.assert_called()
            publish_call = plugin.event_bus.publish.call_args
            assert publish_call is not None
            event = publish_call[0][0]
            assert isinstance(event, Event)
            assert event.name == "cleanup_files.completed"
            assert event.data["recording_id"] == "test_recording"
            assert event.data["cleaned_files"] == ["file1.txt", "file2.txt"]

    async def test_handle_desktop_notification_completed_event(self, plugin):
        """Test handling desktop notification completed with Event object"""
        event = Event(
            name="desktop_notification.completed",
            data={
                "recording_id": "test_recording"
            }
        )

        with patch.object(plugin, '_cleanup_files') as mock_cleanup:
            mock_cleanup.return_value = ["file1.txt"]
            
            await plugin.initialize()
            await plugin.handle_desktop_notification_completed(event)
            
            mock_cleanup.assert_called_once_with("test_recording")

    async def test_handle_desktop_notification_completed_no_recording_id(self, plugin):
        """Test handling desktop notification completed with missing recording id"""
        event_data = {
            "data": {}
        }

        with patch.object(plugin, '_cleanup_files') as mock_cleanup:
            await plugin.initialize()
            await plugin.handle_desktop_notification_completed(event_data)
            
            mock_cleanup.assert_not_called()

    async def test_cleanup_files_config(self, plugin):
        """Test cleanup files configuration"""
        assert len(plugin.include_dirs) == 2
        assert all(isinstance(d, Path) for d in plugin.include_dirs)
        assert str(plugin.include_dirs[0]) == "data"
        assert str(plugin.include_dirs[1]) == "temp"
        
        assert len(plugin.exclude_dirs) == 1
        assert all(isinstance(d, Path) for d in plugin.exclude_dirs)
        assert str(plugin.exclude_dirs[0]) == "protected"
        
        assert plugin.cleanup_delay == 5

    async def test_event_bus_methods(self, plugin):
        """Test event bus integration methods"""
        test_event = {"name": "test_event", "data": {}}

        # Test subscribe/unsubscribe
        callback_called = False

        async def test_callback(event):
            nonlocal callback_called
            callback_called = True

        # Mock event bus methods
        plugin.event_bus.subscribe = AsyncMock()
        plugin.event_bus.unsubscribe = AsyncMock()
        plugin.event_bus.publish = AsyncMock()

        await plugin.subscribe("test_event", test_callback)
        plugin.event_bus.subscribe.assert_called_once_with("test_event", test_callback)

        await plugin.publish(test_event)
        plugin.event_bus.publish.assert_called_once_with(test_event)

        await plugin.unsubscribe("test_event", test_callback)
        plugin.event_bus.unsubscribe.assert_called_once_with("test_event", test_callback)

        assert plugin.event_bus.subscribe.call_count == 1
        assert plugin.event_bus.publish.call_count == 1
        assert plugin.event_bus.unsubscribe.call_count == 1 