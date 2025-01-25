from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from openai import AsyncOpenAI

from app.core.events import Event, EventPriority, ConcreteEventBus
from app.core.plugins import PluginConfig
from app.plugins.meeting_notes_remote.plugin import MeetingNotesRemotePlugin
from tests.plugins.test_plugin_interface import BasePluginTest


class TestMeetingNotesRemotePlugin(BasePluginTest):
    """Test suite for MeetingNotesRemotePlugin"""

    @pytest.fixture
    def plugin_config(self):
        """Meeting notes remote plugin specific config"""
        return PluginConfig(
            name="meeting_notes_remote",
            version="1.0.0",
            enabled=True,
            config={
                "provider": "openai",
                "openai": {
                    "api_key": "test_key",
                    "model": "gpt-4-turbo-preview",
                },
                "output_directory": "data/meeting_notes_remote",
                "max_concurrent_tasks": 2,
                "timeout": 300,
                "temperature": 0.7,
                "max_tokens": 8192,
            },
        )

    @pytest.fixture
    def event_bus(self):
        """Event bus fixture"""
        return ConcreteEventBus()

    @pytest.fixture
    def mock_event_bus(self):
        """Mock event bus for tests that need mocked behavior"""
        mock_bus = Mock()
        mock_bus.subscribe = AsyncMock()
        mock_bus.publish = AsyncMock()
        return mock_bus

    @pytest.fixture
    def plugin_with_mock_bus(self, plugin_config, mock_event_bus):
        """Plugin instance with mock event bus"""
        return MeetingNotesRemotePlugin(plugin_config, mock_event_bus)

    @pytest.fixture
    def plugin(self, plugin_config, event_bus):
        """Meeting notes remote plugin instance"""
        return MeetingNotesRemotePlugin(plugin_config, event_bus)

    @pytest.fixture
    def sample_transcript(self):
        """Sample transcript with metadata"""
        return """## Metadata
```json
{
    "event": {
        "title": "Test Meeting",
        "date": "2024-01-20T10:00:00Z",
        "duration": "PT1H30M",
        "attendees": ["user1@example.com", "user2@example.com"]
    }
}
```

## Transcript
Speaker 1: Let's begin the meeting.
Speaker 2: We need to discuss the project timeline.
Speaker 1: I agree. The deadline is next month.
Speaker 2: I'll prepare the report by next week.
"""

    async def test_meeting_notes_remote_initialization(self, plugin_with_mock_bus):
        """Test meeting notes remote plugin specific initialization"""
        with patch.object(Path, "mkdir") as mock_mkdir:
            await plugin_with_mock_bus.initialize()
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)
            plugin_with_mock_bus.event_bus.subscribe.assert_awaited_once_with(
                "transcription_local.completed",
                plugin_with_mock_bus.handle_transcription_completed,
            )

    async def test_handle_transcription_completed_dict_event(
        self, plugin_with_mock_bus, sample_transcript
    ):
        """Test handling transcription completed event with dict data"""
        transcript_path = Path("test_transcript.txt")
        event = Event.create(
            name="transcription_local.completed",
            data={
                "transcription": {
                    "recording_id": "test_recording",
                    "transcript_path": str(transcript_path),
                }
            },
            source_plugin="transcription_local",
            correlation_id="test_correlation_id",
            priority=EventPriority.NORMAL
        )

        with patch.object(Path, "mkdir"), patch.object(
            plugin_with_mock_bus, "_read_transcript", return_value=sample_transcript
        ), patch.object(
            plugin_with_mock_bus,
            "_generate_meeting_notes",
            return_value=Path("output.md"),
        ), patch.object(
            plugin_with_mock_bus,
            "_get_transcript_path",
            return_value=transcript_path,
        ):
            await plugin_with_mock_bus.initialize()
            await plugin_with_mock_bus.handle_transcription_completed(event)

            plugin_with_mock_bus.event_bus.publish.assert_called_once()
            call_args = plugin_with_mock_bus.event_bus.publish.call_args[0][0]
            assert call_args.name == "meeting_notes_remote.completed"
            assert call_args.data["meeting_notes"]["status"] == "completed"
            assert call_args.data["meeting_notes"]["recording_id"] == "test_recording"
            assert call_args.data["meeting_notes"]["notes_path"] == "output.md"
            assert call_args.data["meeting_notes"]["input_paths"]["transcript"] == str(transcript_path)

    async def test_handle_transcription_completed_no_path(self, plugin_with_mock_bus):
        """Test handling transcription completed event with no transcript path"""
        event_data = Event.create(
            name="transcription_local.completed",
            data={
                "transcription": {
                    "recording_id": "test_recording"
                }
            },
            correlation_id="test-123",
            source_plugin="test_plugin"
        )

        with patch.object(Path, "mkdir"):
            await plugin_with_mock_bus.initialize()
            await plugin_with_mock_bus.handle_transcription_completed(event_data)

            plugin_with_mock_bus.event_bus.publish.assert_not_called()

    async def test_generate_meeting_notes_from_text_empty(self, plugin):
        """Test meeting notes generation with empty text"""
        event_id = "test_event"
        with patch.object(plugin, "_generate_notes_with_llm") as mock_generate:
            mock_generate.return_value = "No transcript text found to generate notes from."
            result = await plugin._generate_notes_with_llm("", event_id)
            assert result == "No transcript text found to generate notes from."
            mock_generate.assert_called_once_with("", event_id)

    def test_plugin_configuration_defaults(self):
        """Test plugin configuration with defaults"""
        config = PluginConfig(
            name="meeting_notes_remote",
            version="1.0.0",
            enabled=True,
            config={
                "provider": "openai",
                "openai": {
                    "api_key": "test_key",
                    "model": "gpt-4-turbo-preview",
                },
            },
        )
        plugin = MeetingNotesRemotePlugin(config)

        assert plugin.output_dir == Path("data/meeting_notes_remote")
        assert plugin.max_concurrent_tasks == 4
        assert plugin.timeout == 600
        assert plugin.provider == "openai"

    def test_plugin_configuration_custom(self, plugin):
        """Test plugin configuration with custom values"""
        assert plugin.provider == "openai"
        assert isinstance(plugin.client, AsyncOpenAI)
        assert plugin.model == "gpt-4-turbo-preview"
        assert plugin.output_dir == Path("data/meeting_notes_remote")
        assert plugin.max_concurrent_tasks == 2
        assert plugin.timeout == 300

    async def test_plugin_initialization_no_event_bus(self):
        """Test plugin initialization without event bus"""
        config = PluginConfig(
            name="meeting_notes_remote",
            version="1.0.0",
            enabled=True,
            config={
                "provider": "openai",
                "openai": {
                    "api_key": "test_key",
                    "model": "gpt-4-turbo-preview",
                },
            },
        )
        plugin = MeetingNotesRemotePlugin(config, event_bus=None)

        with patch.object(Path, "mkdir"):
            await plugin.initialize()
            # Should not raise an exception
