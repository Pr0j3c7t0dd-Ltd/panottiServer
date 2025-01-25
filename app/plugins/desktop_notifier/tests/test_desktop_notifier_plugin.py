from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.events import Event
from app.core.plugins import PluginConfig
from app.plugins.desktop_notifier.plugin import DesktopNotifierPlugin
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
            config={"auto_open_notes": True},
        )

    @pytest.fixture
    def event_bus(self):
        """Mock event bus fixture with async methods"""
        mock_bus = AsyncMock()
        mock_bus.subscribe = AsyncMock()
        mock_bus.unsubscribe = AsyncMock()
        mock_bus.publish = AsyncMock()
        return mock_bus

    @pytest.fixture
    def plugin(self, plugin_config, event_bus):
        """Desktop notifier plugin instance"""
        return DesktopNotifierPlugin(plugin_config, event_bus)

    @pytest.fixture
    async def mock_db(self):
        """Mock database manager"""
        db_instance = MagicMock()
        with patch(
            "app.models.database.DatabaseManager.get_instance", new_callable=AsyncMock
        ) as mock:
            mock.return_value = db_instance
            yield db_instance

    async def test_desktop_notifier_initialization(self, plugin, mock_db):
        """Test desktop notifier plugin specific initialization"""
        await plugin.initialize()

        # Verify database initialization
        assert plugin.db == mock_db

        # Verify event subscriptions
        plugin.event_bus.subscribe.assert_any_call(
            "meeting_notes_local.completed", plugin.handle_meeting_notes_completed
        )
        plugin.event_bus.subscribe.assert_any_call(
            "meeting_notes_remote.completed", plugin.handle_meeting_notes_completed
        )

    async def test_desktop_notifier_shutdown(self, plugin, mock_db):
        """Test desktop notifier plugin specific shutdown"""
        await plugin.initialize()
        await plugin.shutdown()

        # Verify event unsubscriptions
        plugin.event_bus.unsubscribe.assert_any_call(
            "meeting_notes_local.completed", plugin.handle_meeting_notes_completed
        )
        plugin.event_bus.unsubscribe.assert_any_call(
            "meeting_notes_remote.completed", plugin.handle_meeting_notes_completed
        )

    async def test_handle_meeting_notes_completed_dict(self, plugin, mock_db):
        """Test handling meeting notes completed with dict event"""
        event_data = {
            "recording": {"recording_id": "test_recording"},
            "meeting_notes_local": {"output_path": "/path/to/notes.txt"},
            "metadata": {},
            "correlation_id": "test-correlation-id",
            "source_plugin": "test-plugin",
        }

        with patch.object(plugin, "_send_notification") as mock_send:
            with patch.object(plugin, "_open_notes_file") as mock_open:
                await plugin.initialize()
                await plugin.handle_meeting_notes_completed(event_data)

                mock_send.assert_called_once_with(
                    "test_recording", "/path/to/notes.txt"
                )
                mock_open.assert_called_once_with("/path/to/notes.txt")

    async def test_handle_meeting_notes_completed_event(self, plugin, mock_db):
        """Test handling meeting notes completed with Event object"""
        event = Event(
            name="meeting_notes_local.completed",
            data={
                "recording": {"recording_id": "test_recording"},
                "meeting_notes_local": {"output_path": "/path/to/notes.txt"},
            },
        )

        with patch.object(plugin, "_send_notification") as mock_send:
            with patch.object(plugin, "_open_notes_file") as mock_open:
                await plugin.initialize()
                await plugin.handle_meeting_notes_completed(event)

                mock_send.assert_called_once_with(
                    "test_recording", "/path/to/notes.txt"
                )
                mock_open.assert_called_once_with("/path/to/notes.txt")

    async def test_handle_meeting_notes_completed_no_output_path(self, plugin, mock_db):
        """Test handling meeting notes completed with missing output path"""
        event_data = {
            "recording_id": "test_recording",
            "data": {"recording_id": "test_recording"},
        }

        with patch.object(plugin, "_send_notification") as mock_send:
            await plugin.initialize()
            await plugin.handle_meeting_notes_completed(event_data)

            mock_send.assert_not_called()

    async def test_handle_meeting_notes_completed_error(self, plugin, mock_db):
        """Test handling meeting notes completed with error"""
        event_data = {
            "recording_id": "test_recording",
            "output_path": "/path/to/notes.txt",
            "context": {"correlation_id": "test_correlation_id"},
        }

        with patch.object(plugin, "_send_notification") as mock_send:
            mock_send.side_effect = Exception("Test error")

            await plugin.initialize()
            await plugin.handle_meeting_notes_completed(event_data)

            # Verify error event was published
            assert any(
                call.args[0].name == "desktop_notification.error"
                for call in plugin.event_bus.publish.call_args_list
            )

    async def test_event_bus_methods(self, plugin, event_bus):
        """Test event bus integration methods"""
        test_event = {"name": "test_event", "data": {}}

        # Test subscribe/unsubscribe
        callback_called = False

        async def test_callback(event):
            nonlocal callback_called
            callback_called = True

        await plugin.subscribe("test_event", test_callback)
        event_bus.subscribe.assert_called_once_with("test_event", test_callback)

        await plugin.publish(test_event)
        event_bus.publish.assert_called_once_with(test_event)

        # Test unsubscribe
        await plugin.unsubscribe("test_event", test_callback)
        event_bus.unsubscribe.assert_called_once_with("test_event", test_callback)
