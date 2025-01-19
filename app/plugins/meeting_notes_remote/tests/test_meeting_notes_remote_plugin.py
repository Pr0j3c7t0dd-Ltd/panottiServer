import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from datetime import datetime

from app.plugins.base import PluginConfig
from app.plugins.meeting_notes_remote.plugin import MeetingNotesRemotePlugin
from app.plugins.events.models import Event, EventContext
from app.models.recording.events import RecordingEvent
from tests.plugins.test_plugin_interface import BasePluginTest
from openai import AsyncOpenAI


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
                "max_tokens": 4000
            }
        )

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

    async def test_meeting_notes_remote_initialization(self, plugin):
        """Test meeting notes remote plugin specific initialization"""
        with patch('os.makedirs') as mock_makedirs:
            await plugin.initialize()
            
            # Verify directory creation
            mock_makedirs.assert_called_with(plugin.output_dir, exist_ok=True)
            
            # Verify event subscription
            plugin.event_bus.subscribe.assert_called_once_with(
                "transcription_local.completed",
                plugin.handle_transcription_completed
            )

    async def test_meeting_notes_remote_shutdown(self, plugin):
        """Test meeting notes remote plugin specific shutdown"""
        await plugin.initialize()
        await plugin.shutdown()
        
        # Verify thread pool shutdown
        assert plugin._executor.shutdown.called

    async def test_handle_transcription_completed(self, plugin, sample_transcript):
        """Test handling transcription completed event"""
        transcript_path = Path("test_transcript.txt")
        event_data = {
            "recording_id": "test_recording",
            "transcript_path": str(transcript_path),
            "data": {
                "recording_id": "test_recording",
                "transcript_path": str(transcript_path)
            }
        }

        with patch.object(plugin, '_read_transcript') as mock_read:
            with patch.object(plugin, '_generate_meeting_notes') as mock_generate:
                mock_read.return_value = sample_transcript
                mock_generate.return_value = Path("output.md")
                
                await plugin.initialize()
                await plugin.handle_transcription_completed(event_data)
                
                mock_read.assert_called_once()
                mock_generate.assert_called_once()

    async def test_generate_meeting_notes_from_text(self, plugin, sample_transcript):
        """Test meeting notes generation from transcript text"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Generated notes"}}]
            }
            
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            mock_session.return_value.__aenter__.return_value.post.return_value = mock_context
            
            await plugin.initialize()
            result = await plugin._generate_meeting_notes_from_text(sample_transcript)
            
            assert result == "Generated notes"
            assert mock_session.return_value.__aenter__.return_value.post.called

    async def test_process_transcript(self, plugin, sample_transcript):
        """Test transcript processing"""
        recording_id = "test_recording"
        event = Event(
            name="transcription_local.completed",
            data={"recording_id": recording_id},
            context=EventContext(correlation_id="test_id")
        )

        with patch.object(plugin, '_generate_meeting_notes_from_text') as mock_generate:
            with patch.object(plugin, '_get_output_path') as mock_path:
                mock_generate.return_value = "Generated notes"
                mock_path.return_value = Path("output.md")
                
                await plugin.initialize()
                await plugin._process_transcript(recording_id, sample_transcript, event)
                
                mock_generate.assert_called_once_with(sample_transcript)
                assert plugin.event_bus.publish.called

    def test_get_output_path(self, plugin):
        """Test output path generation"""
        transcript_path = Path("data/transcripts/test.txt")
        output_path = plugin._get_output_path(transcript_path)
        
        assert isinstance(output_path, Path)
        assert output_path.suffix == ".md"
        assert output_path.parent == plugin.output_dir

    async def test_read_transcript(self, plugin):
        """Test transcript reading"""
        transcript_path = Path("test.txt")
        test_content = "Test transcript content"
        
        with patch("builtins.open", mock_open(read_data=test_content)):
            content = await plugin._read_transcript(transcript_path)
            assert content == test_content

    async def test_api_error_handling(self, plugin, sample_transcript):
        """Test API error handling"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text.return_value = "Internal Server Error"
            
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            mock_session.return_value.__aenter__.return_value.post.return_value = mock_context
            
            await plugin.initialize()
            with pytest.raises(Exception):
                await plugin._generate_meeting_notes_from_text(sample_transcript)

    def test_plugin_configuration(self, plugin):
        """Test plugin configuration parameters"""
        assert plugin.provider == "openai"
        assert isinstance(plugin.client, AsyncOpenAI)
        assert plugin.model == "gpt-4-turbo-preview"
        assert isinstance(plugin.output_dir, Path)
        assert str(plugin.output_dir) == "data/meeting_notes_remote"
        assert plugin.max_concurrent_tasks == 2
        assert plugin.timeout == 300 