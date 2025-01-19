import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.plugins.base import PluginConfig
from app.plugins.desktop_notifier.plugin import DesktopNotifierPlugin
from app.core.events import Event
from app.models.recording.events import RecordingEvent
from tests.plugins.test_plugin_interface import BasePluginTest


class TestDesktopNotifierPlugin(BasePluginTest):
    """Test suite for DesktopNotifierPlugin"""

    @pytest.fixture
    def plugin_config(self):
        """Desktop notifier plugin specific config"""
        return PluginConfig(
            name="desktop_notifier",
            version="1.0.0",
            enabled=True,
            config={
                "auto_open_notes": True
            }
        )

    @pytest.fixture
    def plugin(self, plugin_config, event_bus):
        """Desktop notifier plugin instance"""
        return DesktopNotifierPlugin(plugin_config, event_bus)

    @pytest.fixture
    def mock_db(self):
        """Mock database manager"""
        with patch('app.models.database.DatabaseManager') as mock:
            db_instance = MagicMock()
            mock.get_instance.return_value = db_instance
            yield db_instance

    async def test_desktop_notifier_initialization(self, plugin, mock_db):
        """Test desktop notifier plugin specific initialization"""
        await plugin.initialize()
        
        # Verify database initialization
        assert plugin.db == mock_db
        
        # Verify event subscriptions
        plugin.event_bus.subscribe.assert_any_call(
            "meeting_notes_local.completed",
            plugin.handle_meeting_notes_completed
        )
        plugin.event_bus.subscribe.assert_any_call(
            "meeting_notes_remote.completed",
            plugin.handle_meeting_notes_completed
        )

    async def test_desktop_notifier_shutdown(self, plugin, mock_db):
        """Test desktop notifier plugin specific shutdown"""
        await plugin.initialize()
        await plugin.shutdown()
        
        # Verify event unsubscriptions
        plugin.event_bus.unsubscribe.assert_any_call(
            "meeting_notes_local.completed",
            plugin.handle_meeting_notes_completed
        )
        plugin.event_bus.unsubscribe.assert_any_call(
            "meeting_notes_remote.completed",
            plugin.handle_meeting_notes_completed
        )

    async def test_handle_meeting_notes_completed_dict(self, plugin, mock_db):
        """Test handling meeting notes completed with dict event"""
        event_data = {
            "recording_id": "test_recording",
            "output_path": "/path/to/notes.txt",
            "data": {
                "recording_id": "test_recording",
                "output_path": "/path/to/notes.txt"
            }
        }

        with patch.object(plugin, '_send_notification') as mock_send:
            with patch.object(plugin, '_open_notes_file') as mock_open:
                await plugin.initialize()
                await plugin.handle_meeting_notes_completed(event_data)
                
                mock_send.assert_called_once_with("test_recording", "/path/to/notes.txt")
                mock_open.assert_called_once_with("/path/to/notes.txt")

    async def test_handle_meeting_notes_completed_event(self, plugin, mock_db):
        """Test handling meeting notes completed with Event object"""
        event = Event(
            name="meeting_notes_local.completed",
            event="meeting_notes_local.completed",
            recording_id="test_recording",
            data={
                "recording_id": "test_recording",
                "output_path": "/path/to/notes.txt"
            }
        )

        with patch.object(plugin, '_send_notification') as mock_send:
            with patch.object(plugin, '_open_notes_file') as mock_open:
                await plugin.initialize()
                await plugin.handle_meeting_notes_completed(event)
                
                mock_send.assert_called_once_with("test_recording", "/path/to/notes.txt")
                mock_open.assert_called_once_with("/path/to/notes.txt")

    async def test_handle_meeting_notes_completed_no_output_path(self, plugin, mock_db):
        """Test handling meeting notes completed with missing output path"""
        event_data = {
            "recording_id": "test_recording",
            "data": {
                "recording_id": "test_recording"
            }
        }

        with patch.object(plugin, '_send_notification') as mock_send:
            await plugin.initialize()
            await plugin.handle_meeting_notes_completed(event_data)
            
            mock_send.assert_not_called()

    async def test_handle_meeting_notes_completed_error(self, plugin, mock_db):
        """Test handling meeting notes completed with error"""
        event_data = {
            "recording_id": "test_recording",
            "output_path": "/path/to/notes.txt"
        }

        with patch.object(plugin, '_send_notification') as mock_send:
            mock_send.side_effect = Exception("Test error")
            
            await plugin.initialize()
            await plugin.handle_meeting_notes_completed(event_data)
            
            # Verify error event was published
            assert any(
                call.args[0].event == "meeting_notes.error"
                for call in plugin.event_bus.publish.call_args_list
            ) 