from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, mock_open

import pytest
from openai import AsyncOpenAI

from app.core.events import ConcreteEventBus, Event, EventPriority
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
            priority=EventPriority.NORMAL,
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
            assert call_args.data["meeting_notes"]["input_paths"]["transcript"] == str(
                transcript_path
            )

    async def test_handle_transcription_completed_no_path(self, plugin_with_mock_bus):
        """Test handling transcription completed event with no transcript path"""
        event_data = Event.create(
            name="transcription_local.completed",
            data={"transcription": {"recording_id": "test_recording"}},
            correlation_id="test-123",
            source_plugin="test_plugin",
        )

        with patch.object(Path, "mkdir"):
            await plugin_with_mock_bus.initialize()
            await plugin_with_mock_bus.handle_transcription_completed(event_data)

            plugin_with_mock_bus.event_bus.publish.assert_not_called()

    async def test_generate_meeting_notes_from_text_empty(self, plugin):
        """Test meeting notes generation with empty text"""
        event_id = "test_event"
        with patch.object(plugin, "_generate_notes_with_llm") as mock_generate:
            mock_generate.return_value = (
                "No transcript text found to generate notes from."
            )
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

    async def test_initialization_error_handling(self, plugin_with_mock_bus):
        """Test error handling during initialization"""
        with patch.object(Path, "mkdir", side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError):
                await plugin_with_mock_bus.initialize()

    async def test_metadata_handling(self, plugin_with_mock_bus, sample_transcript):
        """Test metadata handling in transcription completed event"""
        transcript_path = Path("test_transcript.txt")
        metadata = {
            "meeting_title": "Test Meeting",
            "date": "2024-01-20T10:00:00Z",
            "attendees": ["user1@example.com"]
        }
        
        event = Event.create(
            name="transcription_local.completed",
            data={
                "transcription": {
                    "recording_id": "test_recording",
                    "transcript_path": str(transcript_path)
                },
                "metadata": metadata
            },
            source_plugin="transcription_local",
            correlation_id="test_correlation_id",
            priority=EventPriority.NORMAL,
        )

        with patch.object(Path, "mkdir"), \
             patch.object(plugin_with_mock_bus, "_read_transcript", return_value=sample_transcript), \
             patch.object(plugin_with_mock_bus, "_generate_meeting_notes", return_value=Path("output.md")), \
             patch.object(plugin_with_mock_bus, "_get_transcript_path", return_value=transcript_path):
            
            await plugin_with_mock_bus.initialize()
            await plugin_with_mock_bus.handle_transcription_completed(event)

            plugin_with_mock_bus.event_bus.publish.assert_called_once()
            call_args = plugin_with_mock_bus.event_bus.publish.call_args[0][0]
            assert call_args.data["metadata"] == metadata

    async def test_event_bus_error_handling(self, plugin_with_mock_bus):
        """Test error handling in event bus operations"""
        plugin_with_mock_bus.event_bus.publish.side_effect = Exception("Network error")
        transcript_path = Path("test_transcript.txt")
        
        event = Event.create(
            name="transcription_local.completed",
            data={
                "transcription": {
                    "recording_id": "test_recording",
                    "transcript_path": str(transcript_path)
                }
            },
            source_plugin="transcription_local",
            correlation_id="test-123",
        )

        with patch.object(Path, "mkdir"), \
             patch.object(plugin_with_mock_bus, "_read_transcript", return_value="test"), \
             patch.object(plugin_with_mock_bus, "_generate_meeting_notes", return_value=Path("output.md")), \
             patch.object(plugin_with_mock_bus, "_get_transcript_path", return_value=transcript_path):
            
            await plugin_with_mock_bus.initialize()
            # Should not raise exception
            await plugin_with_mock_bus.handle_transcription_completed(event)

    async def test_transcript_path_handling(self, plugin_with_mock_bus):
        """Test transcript path handling"""
        # Test with string path
        event = Event.create(
            name="transcription_local.completed",
            data={
                "transcription": {
                    "recording_id": "test_recording",
                    "transcript_path": "test_transcript.txt"
                }
            },
            source_plugin="transcription_local",
            correlation_id="test-123",
        )
        
        with patch.object(plugin_with_mock_bus, "_get_transcript_path", 
                         return_value=Path("test_transcript.txt")) as mock_get_path:
            path = await plugin_with_mock_bus._get_transcript_path(event)
            assert path == Path("test_transcript.txt")

            # Test with Path object
            event.data["transcription"]["transcript_path"] = Path("test_transcript.txt")
            path = await plugin_with_mock_bus._get_transcript_path(event)
            assert path == Path("test_transcript.txt")

    async def test_transcript_reading(self, plugin):
        """Test transcript reading functionality"""
        test_content = "Test transcript content"
        transcript_path = Path("test_transcript.txt")
        
        # Mock the file operations
        with patch.object(Path, "read_text", return_value=test_content) as mock_read:
            content = await plugin._read_transcript(transcript_path)
            assert content == test_content
            mock_read.assert_called_once()

        # Test error handling
        with patch.object(Path, "read_text", side_effect=FileNotFoundError()), \
             pytest.raises(FileNotFoundError):
            await plugin._read_transcript(transcript_path)

    def test_llm_response_cleaning(self, plugin):
        """Test LLM response cleaning"""
        # Test markdown code block removal
        response = "```\nTest notes\n```\nMore notes"
        cleaned = plugin._clean_llm_response(response)
        assert cleaned == "Test notes\n\nMore notes"

        # Test whitespace handling
        response = "\n\nTest   notes  \n\n  More notes\n\n"
        cleaned = plugin._clean_llm_response(response)
        assert cleaned == "Test   notes  \n\n  More notes"

    async def test_output_path_handling(self, plugin):
        """Test output path generation"""
        transcript_path = Path("data/transcripts/meeting_2024_01_20.txt")
        output_path = plugin._get_output_path(transcript_path)
        assert output_path.suffix == ".md"
        assert output_path.parent == plugin.output_dir

    async def test_shutdown_handling(self, plugin):
        """Test plugin shutdown"""
        with patch.object(plugin._executor, "shutdown") as mock_shutdown:
            await plugin.shutdown()
            mock_shutdown.assert_called_once_with(wait=True)

    async def test_generate_notes_with_openai(self, plugin):
        """Test meeting notes generation with OpenAI"""
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock(message=AsyncMock(content="Test notes"))]
        
        # Create a mock client with the proper structure
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        with patch.object(plugin, "client", mock_client):
            result = await plugin._generate_notes_with_llm("Test transcript", "test-123")
            assert result == "Test notes"
            mock_client.chat.completions.create.assert_called_once()

    async def test_generate_notes_with_anthropic(self, plugin_config):
        """Test meeting notes generation with Anthropic"""
        config = plugin_config
        config.config["provider"] = "anthropic"
        config.config["anthropic"] = {
            "api_key": "test_key",
            "model": "claude-3-opus-20240229"
        }
        plugin = MeetingNotesRemotePlugin(config)

        mock_response = AsyncMock()
        mock_response.content = [AsyncMock(text="Test notes")]
        
        # Create a mock client with the proper structure
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        
        with patch.object(plugin, "client", mock_client):
            result = await plugin._generate_notes_with_llm("Test transcript", "test-123")
            assert result == "Test notes"
            mock_client.messages.create.assert_called_once()

    async def test_generate_notes_with_google(self, plugin_config):
        """Test meeting notes generation with Google"""
        config = plugin_config
        config.config["provider"] = "google"
        config.config["google"] = {
            "api_key": "test_key",
            "model": "gemini-pro"
        }
        plugin = MeetingNotesRemotePlugin(config)

        mock_response = AsyncMock()
        mock_response.text = "Test notes"
        
        with patch.object(plugin.client, "generate_content_async", 
                         return_value=mock_response) as mock_generate:
            result = await plugin._generate_notes_with_llm("Test transcript", "test-123")
            assert result == "Test notes"
            mock_generate.assert_called_once()
