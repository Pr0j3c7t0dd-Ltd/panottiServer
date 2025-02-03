from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.events import ConcreteEventBus as EventBus
from app.core.events import Event
from app.core.plugins import PluginConfig
from app.plugins.cleanup_files.plugin import CleanupFilesPlugin
from tests.plugins.test_plugin_interface import BasePluginTest


@pytest.fixture(autouse=True)
def mock_asyncio_sleep():
    """Mock asyncio.sleep globally to prevent delays."""
    async def immediate_sleep(*args, **kwargs):
        return None
    
    with patch("asyncio.sleep", side_effect=immediate_sleep):
        yield


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
                "cleanup_delay": 5,
            },
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
            plugin.handle_desktop_notification_completed,
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
            plugin.handle_desktop_notification_completed,
        )

        # Verify thread pool shutdown
        plugin._executor.shutdown.assert_called_once()

    async def test_handle_desktop_notification_completed_dict(self, plugin):
        """Test handling desktop notification completed with dict event"""
        event_data = {
            "recording_id": "test_recording",
            "data": {"recording_id": "test_recording"},
        }

        with patch.object(plugin, "_cleanup_files") as mock_cleanup:
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
            assert event.data["cleanup_files"]["cleaned_files"] == [
                "file1.txt",
                "file2.txt",
            ]
            assert event.data["cleanup_files"]["status"] == "completed"
            assert isinstance(event.data["cleanup_files"]["timestamp"], str)
            assert len(event.data["cleanup_files"]["config"]["include_dirs"]) == 2
            assert len(event.data["cleanup_files"]["config"]["exclude_dirs"]) == 1
            assert event.data["cleanup_files"]["config"]["cleanup_delay"] == 5

    async def test_handle_desktop_notification_completed_event(self, plugin):
        """Test handling desktop notification completed with Event object"""
        event = Event(
            name="desktop_notification.completed",
            data={"recording_id": "test_recording"},
        )

        with patch.object(plugin, "_cleanup_files") as mock_cleanup:
            mock_cleanup.return_value = ["file1.txt"]

            await plugin.initialize()
            await plugin.handle_desktop_notification_completed(event)

            mock_cleanup.assert_called_once_with("test_recording")

    async def test_handle_desktop_notification_completed_no_recording_id(self, plugin):
        """Test handling desktop notification completed with missing recording id"""
        event_data = {"data": {}}

        with patch.object(plugin, "_cleanup_files") as mock_cleanup:
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
        plugin.event_bus.unsubscribe.assert_called_once_with(
            "test_event", test_callback
        )

        assert plugin.event_bus.subscribe.call_count == 1
        assert plugin.event_bus.publish.call_count == 1
        assert plugin.event_bus.unsubscribe.call_count == 1

    async def test_initialization_error(self, plugin_config):
        """Test error handling during initialization"""
        event_bus = EventBus()
        event_bus.subscribe = AsyncMock(side_effect=Exception("Subscribe failed"))
        plugin = CleanupFilesPlugin(plugin_config, event_bus)

        with pytest.raises(Exception) as exc_info:
            await plugin.initialize()
        assert str(exc_info.value) == "Subscribe failed"

    async def test_handle_desktop_notification_error(self, plugin):
        """Test error handling in desktop notification handler"""
        event_data = {
            "recording_id": "test_recording",
            "correlation_id": "test_correlation",
            "data": {"recording_id": "test_recording"},
            "metadata": {"test": "metadata"}
        }

        with patch.object(plugin, "_cleanup_files") as mock_cleanup:
            mock_cleanup.side_effect = Exception("Cleanup failed")
            
            await plugin.initialize()
            # Mock the logger to prevent stack_info error
            with patch("app.plugins.cleanup_files.plugin.logger") as mock_logger:
                await plugin.handle_desktop_notification_completed(event_data)

                # Verify error event was published
                plugin.event_bus.publish.assert_called()
                publish_call = plugin.event_bus.publish.call_args
                assert publish_call is not None
                event = publish_call[0][0]
                assert isinstance(event, Event)
                assert event.name == "cleanup_files.error"
                assert event.data["cleanup_files"]["status"] == "error"
                assert event.data["cleanup_files"]["error"] == "Cleanup failed"
                assert event.data["metadata"] == {"test": "metadata"}
                assert event.context.correlation_id == "test_correlation"

    @pytest.mark.asyncio
    async def test_cleanup_delay(self, plugin, tmp_path):
        """Test cleanup delay handling"""
        test_file = tmp_path / "test_recording_file.txt"
        test_file.write_text("test content")
        
        plugin.include_dirs = [tmp_path]
        plugin.cleanup_delay = 0.1  # Small delay for testing
        
        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = None
            cleaned_files = await plugin._cleanup_files("test_recording")
            
            mock_sleep.assert_called_once_with(0.1)
            assert str(test_file) in cleaned_files
            assert not test_file.exists()

    @pytest.mark.asyncio
    async def test_directory_scan_error(self, plugin, tmp_path):
        """Test directory scanning error handling"""
        non_existent_dir = tmp_path / "non_existent"
        plugin.include_dirs = [non_existent_dir]
        
        cleaned_files = await plugin._cleanup_files("test_recording")
        assert cleaned_files == []

    @pytest.mark.asyncio
    async def test_file_cleanup_implementation(self, plugin, tmp_path):
        """Test file cleanup implementation details"""
        # Create test directory structure
        include_dir = tmp_path / "include"
        exclude_dir = tmp_path / "include/exclude"
        include_dir.mkdir()
        exclude_dir.mkdir()
        
        # Create test files
        test_files = [
            include_dir / "test_recording_1.txt",
            include_dir / "test_recording_2.txt",
            exclude_dir / "test_recording_3.txt"
        ]
        for file in test_files:
            file.write_text("test content")
            
        plugin.include_dirs = [include_dir]
        plugin.exclude_dirs = [exclude_dir]
        
        cleaned_files = await plugin._cleanup_files("test_recording")
        
        # Verify only non-excluded files were cleaned
        assert len(cleaned_files) == 2
        assert str(test_files[0]) in cleaned_files
        assert str(test_files[1]) in cleaned_files
        assert str(test_files[2]) not in cleaned_files
        
        # Verify files were actually deleted
        assert not test_files[0].exists()
        assert not test_files[1].exists()
        assert test_files[2].exists()

    @pytest.mark.asyncio
    async def test_file_cleanup_error_handling(self, plugin, tmp_path):
        """Test error handling during file cleanup"""
        test_file = tmp_path / "test_recording_file.txt"
        test_file.write_text("test content")
        plugin.include_dirs = [tmp_path]
        
        with patch("pathlib.Path.unlink") as mock_unlink:
            mock_unlink.side_effect = PermissionError("Permission denied")
            
            cleaned_files = await plugin._cleanup_files("test_recording")
            assert cleaned_files == []
            assert test_file.exists()

    @pytest.mark.asyncio
    async def test_cleanup_error_event_publishing(self, plugin):
        """Test error event publishing during cleanup"""
        event_data = {
            "recording_id": "test_recording",
            "correlation_id": "test_correlation",
            "data": {
                "recording": {"id": "test_recording"},
                "noise_reduction": {"status": "completed"},
                "transcription": {"status": "completed"},
                "meeting_notes": {"status": "completed"},
                "desktop_notification": {"status": "completed"}
            },
            "metadata": {"test": "metadata"}
        }

        with patch.object(plugin, "_cleanup_files") as mock_cleanup:
            mock_cleanup.side_effect = Exception("Cleanup process failed")
            
            await plugin.initialize()
            # Mock the logger to prevent stack_info error
            with patch("app.plugins.cleanup_files.plugin.logger") as mock_logger:
                await plugin.handle_desktop_notification_completed(event_data)

                # Verify error event was published with all data
                plugin.event_bus.publish.assert_called()
                publish_call = plugin.event_bus.publish.call_args
                assert publish_call is not None
                event = publish_call[0][0]
                assert isinstance(event, Event)
                assert event.name == "cleanup_files.error"
                assert event.data["cleanup_files"]["status"] == "error"
                assert event.data["cleanup_files"]["error"] == "Cleanup process failed"
                assert event.data["recording"]["id"] == "test_recording"
                assert event.data["noise_reduction"]["status"] == "completed"
                assert event.data["transcription"]["status"] == "completed"
                assert event.data["meeting_notes"]["status"] == "completed"
                assert event.data["desktop_notification"]["status"] == "completed"
                assert event.data["metadata"] == {"test": "metadata"}
                assert event.context.correlation_id == "test_correlation"

    async def test_initialization_no_event_bus(self):
        """Test initialization without event bus"""
        config = PluginConfig(
            name="cleanup_files",
            version="1.0.0",
            enabled=True,
            config={
                "include_dirs": ["data", "temp"],
                "exclude_dirs": ["protected"],
                "cleanup_delay": 5,
            },
        )
        plugin = CleanupFilesPlugin(config, None)
        await plugin.initialize()
        # Should not raise any exceptions

    async def test_shutdown_no_event_bus(self):
        """Test shutdown without event bus"""
        config = PluginConfig(
            name="cleanup_files",
            version="1.0.0",
            enabled=True,
            config={
                "include_dirs": ["data", "temp"],
                "exclude_dirs": ["protected"],
                "cleanup_delay": 5,
            },
        )
        plugin = CleanupFilesPlugin(config, None)
        await plugin.shutdown()
        # Should not raise any exceptions

    async def test_shutdown_error(self, plugin):
        """Test error handling during shutdown"""
        await plugin.initialize()
        plugin.event_bus.unsubscribe = AsyncMock(side_effect=Exception("Unsubscribe failed"))
        
        # Should raise the unsubscribe error
        with pytest.raises(Exception) as exc_info:
            await plugin.shutdown()
        assert str(exc_info.value) == "Unsubscribe failed"
        
        # Verify unsubscribe was called
        plugin.event_bus.unsubscribe.assert_called_once_with(
            "desktop_notification.completed",
            plugin.handle_desktop_notification_completed,
        )

    async def test_handle_desktop_notification_unsupported_type(self, plugin):
        """Test handling desktop notification with unsupported event type"""
        class UnsupportedEvent:
            pass

        await plugin.initialize()
        await plugin.handle_desktop_notification_completed(UnsupportedEvent())
        plugin.event_bus.publish.assert_not_called()

    async def test_cleanup_files_error(self, plugin, tmp_path):
        """Test error handling in cleanup files"""
        test_file = tmp_path / "test_recording_file.txt"
        test_file.write_text("test content")
        plugin.include_dirs = [tmp_path]

        with patch("os.walk") as mock_walk:
            mock_walk.side_effect = Exception("Walk failed")
            cleaned_files = await plugin._cleanup_files("test_recording")
            assert cleaned_files == []

    async def test_directory_scan_excluded(self, plugin, tmp_path):
        """Test directory scanning with excluded directories"""
        # Create test directory structure
        include_dir = tmp_path / "include"
        exclude_dir = tmp_path / "include/exclude"
        include_dir.mkdir()
        exclude_dir.mkdir()

        # Create test files
        test_file = exclude_dir / "test_recording_file.txt"
        test_file.write_text("test content")

        plugin.include_dirs = [include_dir]
        plugin.exclude_dirs = [exclude_dir]

        cleaned_files = await plugin._cleanup_files("test_recording")
        assert cleaned_files == []
        assert test_file.exists()

    async def test_file_cleanup_error_with_retry(self, plugin, tmp_path):
        """Test file cleanup error handling"""
        test_file = tmp_path / "test_recording_file.txt"
        test_file.write_text("test content")
        plugin.include_dirs = [tmp_path]

        with patch("pathlib.Path.unlink") as mock_unlink:
            # Mock unlink to raise an error
            mock_unlink.side_effect = OSError("IO Error")
            cleaned_files = await plugin._cleanup_files("test_recording")
            
            # Verify error was handled gracefully
            assert cleaned_files == []  # No files should be marked as cleaned
            assert mock_unlink.call_count == 1  # Only one attempt should be made
            assert test_file.exists()  # File should still exist

    async def test_cleanup_files_general_error(self, plugin, tmp_path):
        """Test general error handling in cleanup files"""
        test_file = tmp_path / "test_recording_file.txt"
        test_file.write_text("test content")
        plugin.include_dirs = [tmp_path]

        with patch("os.walk") as mock_walk:
            mock_walk.side_effect = Exception("Walk failed")
            cleaned_files = await plugin._cleanup_files("test_recording")
            assert cleaned_files == []
            assert test_file.exists()  # File should not be deleted
