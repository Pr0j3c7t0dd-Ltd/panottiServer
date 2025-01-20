from concurrent.futures import Future
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock, mock_open

import pytest
import aioresponses

from app.core.events import Event, EventContext
from app.plugins.base import PluginConfig
from app.plugins.meeting_notes_local.plugin import MeetingNotesLocalPlugin
from tests.plugins.test_plugin_interface import BasePluginTest


class TestMeetingNotesLocalPlugin(BasePluginTest):
    """Test suite for MeetingNotesLocalPlugin"""

    @pytest.fixture
    def event_bus(self):
        """Override event bus fixture to return a mock"""
        mock_bus = MagicMock()
        mock_bus.publish = AsyncMock()
        mock_bus._pending_tasks = set()
        mock_bus._callbacks = {}

        async def mock_subscribe(event_name, callback):
            if event_name not in mock_bus._callbacks:
                mock_bus._callbacks[event_name] = set()
            mock_bus._callbacks[event_name].add(callback)

        async def mock_unsubscribe(event_name, callback):
            if event_name in mock_bus._callbacks:
                mock_bus._callbacks[event_name].discard(callback)

        async def mock_publish(event):
            event_name = event.get("event") if isinstance(event, dict) else event.name
            if event_name in mock_bus._callbacks:
                for callback in mock_bus._callbacks[event_name]:
                    await callback(event)

        mock_bus.subscribe = mock_subscribe
        mock_bus.unsubscribe = mock_unsubscribe
        mock_bus.publish = mock_publish

        return mock_bus

    @pytest.fixture
    def plugin_config(self):
        """Meeting notes local plugin specific config"""
        return PluginConfig(
            name="meeting_notes_local",
            version="1.0.0",
            enabled=True,
            config={
                "ollama_url": "http://localhost:11434/api/generate",
                "model_name": "llama3.1:latest",
                "output_directory": "data/meeting_notes_local",
                "num_ctx": 128000,
                "max_concurrent_tasks": 2,
                "timeout": 300,
            },
        )

    @pytest.fixture
    def plugin(self, plugin_config, event_bus):
        """Meeting notes local plugin instance"""
        plugin = MeetingNotesLocalPlugin(plugin_config, event_bus)
        
        # Mock the executor
        mock_executor = MagicMock()
        future = Future()
        future.set_result(None)
        mock_executor.submit.return_value = future
        mock_executor.shutdown = MagicMock()
        plugin._executor = mock_executor
        
        return plugin

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

    @pytest.fixture
    def mock_aioresponse(self):
        with aioresponses.aioresponses() as m:
            yield m

    async def test_meeting_notes_initialization(self, plugin_config, event_bus):
        """Test meeting notes local plugin specific initialization"""
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            plugin = MeetingNotesLocalPlugin(plugin_config, event_bus)
            await plugin.initialize()
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

            # Verify event subscription
            assert len(event_bus._callbacks.get("transcription_local.completed", set())) == 1
            assert plugin.handle_transcription_completed in event_bus._callbacks["transcription_local.completed"]

    async def test_meeting_notes_shutdown(self, plugin):
        """Test meeting notes local plugin specific shutdown"""
        await plugin.initialize()
        await plugin.shutdown()

        # Verify thread pool shutdown
        plugin._executor.shutdown.assert_called_once()

    async def test_handle_transcription_completed(self, plugin, sample_transcript):
        """Test handling transcription completed event"""
        transcript_path = Path("test_transcript.txt")
        event_data = {
            "event": "transcription_local.completed",
            "recording_id": "test_recording",
            "transcript_path": str(transcript_path),
            "data": {
                "recording_id": "test_recording",
                "transcript_path": str(transcript_path),
            },
        }

        mock_read = AsyncMock(return_value=sample_transcript)
        mock_generate = AsyncMock(return_value=Path("output.md"))
        mock_publish = AsyncMock()

        with patch.object(plugin, "_read_transcript", mock_read), \
             patch.object(plugin, "_generate_meeting_notes", mock_generate), \
             patch.object(plugin.event_bus, "publish", mock_publish):
            await plugin.initialize()
            await plugin.handle_transcription_completed(event_data)

            assert mock_read.await_count == 0  # _read_transcript is not called directly
            assert mock_generate.await_count == 1
            assert mock_generate.call_args[0][0] == transcript_path
            assert mock_generate.call_args[0][2] == "test_recording"
            assert mock_publish.await_count > 0

    async def test_generate_meeting_notes_from_text(self, plugin, sample_transcript, mock_aioresponse):
        """Test meeting notes generation from transcript text"""
        mock_aioresponse.post(
            plugin.ollama_url,
            payload={"response": "Generated notes"},
            status=200
        )

        await plugin.initialize()
        result = await plugin._generate_meeting_notes_from_text(sample_transcript)

        assert result == "Generated notes"

    async def test_process_transcript(self, plugin, sample_transcript):
        """Test transcript processing"""
        recording_id = "test_recording"
        event = Event(
            name="transcription_local.completed",
            data={"recording_id": recording_id},
            context=EventContext(correlation_id="test_id"),
        )

        mock_generate = AsyncMock(return_value="Generated notes")
        mock_path = MagicMock(return_value=Path("output.md"))
        mock_publish = AsyncMock()

        with patch.object(plugin, "_generate_meeting_notes_from_text", mock_generate):
            with patch.object(plugin, "_get_output_path", mock_path):
                with patch.object(plugin.event_bus, "publish", mock_publish):
                    await plugin.initialize()
                    await plugin._process_transcript(recording_id, sample_transcript, event)

                    mock_generate.assert_awaited_once_with(sample_transcript)
                    assert mock_publish.await_count > 0

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

        # Create a future that returns the test content
        future = Future()
        future.set_result(test_content)
        plugin._executor.submit.return_value = future

        with patch("builtins.open", mock_open(read_data=test_content)):
            content = await plugin._read_transcript(transcript_path)
            assert content == test_content

    def test_plugin_configuration(self, plugin):
        """Test plugin configuration parameters"""
        assert plugin.ollama_url == "http://localhost:11434/api/generate"
        assert plugin.model == "llama3.1:latest"
        assert isinstance(plugin.output_dir, Path)
        assert str(plugin.output_dir) == "data/meeting_notes_local"
        assert plugin.num_ctx == 128000
        assert plugin.max_concurrent_tasks == 2
        assert plugin.timeout == 300

    async def test_event_bus_methods(self, plugin, event_bus):
        """Test event bus integration methods"""
        test_event = {"event": "test_event", "data": {}}

        # Test subscribe/unsubscribe
        callback_called = False

        async def test_callback(event):
            nonlocal callback_called
            callback_called = True

        await plugin.subscribe("test_event", test_callback)
        await plugin.publish(test_event)

        assert callback_called

        # Test unsubscribe
        callback_called = False
        await plugin.unsubscribe("test_event", test_callback)
        await plugin.publish(test_event)

        assert not callback_called
