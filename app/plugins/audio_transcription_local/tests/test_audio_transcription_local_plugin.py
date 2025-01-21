from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch
import os

import pytest

from app.core.plugins import PluginConfig
from app.plugins.audio_transcription_local.plugin import AudioTranscriptionLocalPlugin
from tests.plugins.test_plugin_interface import BasePluginTest


class TestAudioTranscriptionLocalPlugin(BasePluginTest):
    """Test suite for AudioTranscriptionLocalPlugin"""

    @pytest.fixture
    def plugin_config(self):
        """Audio transcription local plugin specific config"""
        return PluginConfig(
            name="audio_transcription_local",
            version="1.0.0",
            enabled=True,
            config={
                "whisper_model": "base",
                "output_directory": "data/transcripts",
                "max_concurrent_tasks": 2,
                "device": "cpu",
                "language": "en",
                "task": "transcribe",
                "initial_prompt": "Meeting transcript:",
                "word_timestamps": True,
                "temperature": 0.0,
                "condition_on_previous_text": True,
                "verbose": True,
            },
        )

    @pytest.fixture
    def mock_db(self):
        """Mock database manager"""
        mock = AsyncMock()
        mock.get_instance = AsyncMock(return_value=mock)
        mock.get_connection = MagicMock()
        mock.execute = AsyncMock()
        return mock

    @pytest.fixture
    def plugin(self, plugin_config, event_bus):
        """Audio transcription local plugin instance"""
        mock_db = AsyncMock()
        mock_db.get_instance = AsyncMock(return_value=mock_db)
        mock_db.get_connection = MagicMock()
        mock_db.execute = AsyncMock()

        with patch("pathlib.Path.mkdir") as mock_mkdir, patch(
            "app.models.database.DatabaseManager.get_instance", return_value=mock_db
        ):
            plugin = AudioTranscriptionLocalPlugin(plugin_config, event_bus)
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)
            return plugin

    @pytest.fixture
    def mock_wave_file(self):
        """Mock wave file"""
        mock = MagicMock()
        mock.getnchannels.return_value = 1
        mock.getsampwidth.return_value = 2
        mock.getframerate.return_value = 16000
        mock.getnframes.return_value = 16000
        return mock

    @pytest.fixture
    def mock_whisper(self):
        """Mock whisper model"""
        model = MagicMock()
        model.transcribe = AsyncMock()
        model.transcribe.return_value = (
            MagicMock(
                text="Test transcription",
                segments=[
                    {"start": 0, "end": 1, "text": "Test"},
                    {"start": 1, "end": 2, "text": "transcription"},
                ],
            ),
            None,
        )
        return model

    @pytest.fixture
    def event_bus(self):
        """Mock event bus fixture"""
        mock_bus = AsyncMock()
        mock_bus.subscribe = AsyncMock()
        mock_bus.unsubscribe = AsyncMock()
        mock_bus.publish = AsyncMock()
        return mock_bus

    async def test_transcription_initialization(self, plugin, mock_whisper):
        """Test audio transcription plugin specific initialization"""
        with patch.object(plugin, "_init_model") as mock_init_model:
            await plugin.initialize()

            # Verify model initialization was called
            mock_init_model.assert_called_once()

            # Verify event subscription
            plugin.event_bus.subscribe.assert_called_once_with(
                "noise_reduction.completed", plugin.handle_noise_reduction_completed
            )

    async def test_transcription_shutdown(self, plugin):
        """Test audio transcription plugin specific shutdown"""
        with patch.object(plugin, "_init_model"):
            await plugin.initialize()
            await plugin.shutdown()

            # Verify event bus unsubscribe
            plugin.event_bus.unsubscribe.assert_called_once_with(
                "noise_reduction.completed", plugin.handle_noise_reduction_completed
            )

    async def test_handle_noise_reduction_completed(self, plugin, mock_whisper):
        """Test handling noise reduction completed event"""
        # Test recording ID and paths
        recording_id = "test_recording"
        audio_path = "/path/to/audio.wav"
        transcript_path = str(Path("data/transcripts_local/test_recording_transcript.txt"))

        event_data = {
            "name": "noise_reduction.completed",
            "recording_id": recording_id,
            "output_path": audio_path,
            "data": {
                "recording_id": recording_id,
                "output_path": audio_path,
            },
        }

        # Create mock transcription results
        mock_segments = [
            MagicMock(text="Test", start=0.0, end=1.0),
            MagicMock(text="transcription", start=1.0, end=2.0)
        ]
        mock_transcript = MagicMock(text="Test transcription")

        # Set up the mocks
        with patch.object(plugin, "_process_audio", new_callable=AsyncMock) as mock_process_audio, \
             patch.object(plugin, "_init_model"), \
             patch.object(plugin, "_model", mock_whisper), \
             patch("builtins.open", mock_open()) as mock_file, \
             patch("os.path.exists") as mock_exists:

            # Configure the mocks
            mock_process_audio.return_value = (mock_segments, mock_transcript)
            mock_exists.return_value = True

            # Initialize the plugin
            await plugin.initialize()

            # Call the handler
            await plugin.handle_noise_reduction_completed(event_data)

            # Verify _process_audio was called with correct arguments
            mock_process_audio.assert_called_once_with(
                str(Path(audio_path)), "Microphone"
            )

            # Verify file operations
            mock_file.assert_called()

    async def test_handle_noise_reduction_completed_no_path(self, plugin):
        """Test handling noise reduction completed with missing path"""
        event_data = {
            "recording_id": "test_recording",
            "data": {"recording_id": "test_recording"},
        }

        with patch.object(plugin, "transcribe_audio") as mock_transcribe, patch.object(
            plugin, "_init_model"
        ):
            await plugin.initialize()
            await plugin.handle_noise_reduction_completed(event_data)

            mock_transcribe.assert_not_called()

    async def test_transcribe_audio(self, plugin, mock_whisper):
        """Test audio transcription functionality"""
        audio_path = "test.wav"
        output_path = os.path.join("test_output", "output.md")
        label = "Speaker"

        # Create a segment object with the required attributes
        class MockSegment:
            def __init__(self, text, start, end):
                self.text = text
                self.start = start
                self.end = end

        mock_segments = [
            MockSegment("Transcript content would go here", 0.0, 1.0)
        ]

        # Create a mock loop with run_in_executor
        mock_loop = AsyncMock()
        mock_loop.run_in_executor.return_value = (mock_segments, "en")

        with patch("wave.open") as mock_wave, patch(
            "builtins.open", mock_open()
        ) as mock_file, patch.object(plugin, "_init_model"), patch.object(
            plugin, "_model", mock_whisper
        ), patch("os.makedirs") as mock_makedirs, patch(
            "asyncio.get_running_loop", return_value=mock_loop
        ):
            mock_wave.return_value.__enter__.return_value = MagicMock(
                getnchannels=lambda: 1,
                getsampwidth=lambda: 2,
                getframerate=lambda: 16000,
                getnframes=lambda: 16000,
            )

            await plugin.initialize()
            result = await plugin.transcribe_audio(audio_path, output_path, label)

            assert result == Path(output_path)
            mock_makedirs.assert_called_once_with("test_output", exist_ok=True)
            mock_file.assert_called_with(output_path, "w", encoding="utf-8")
            mock_file().write.assert_any_call(f"# {label}'s Transcript\n\n")
            mock_file().write.assert_any_call("[00:00.000 - 00:01.000] Transcript content would go here\n")

            # Verify run_in_executor was called correctly
            mock_loop.run_in_executor.assert_called_once()
            executor_args = mock_loop.run_in_executor.call_args[0]
            assert executor_args[0] == plugin._executor  # First arg should be the executor
            assert callable(executor_args[1])  # Second arg should be the lambda function

    def test_plugin_configuration(self, plugin):
        """Test plugin configuration parameters"""
        config = plugin.config
        assert config.name == "audio_transcription_local"
        assert config.version == "1.0.0"
        assert config.enabled is True
        assert config.config["whisper_model"] == "base"
        assert config.config["output_directory"] == "data/transcripts"
        assert config.config["max_concurrent_tasks"] == 2
        assert config.config["device"] == "cpu"
        assert config.config["language"] == "en"
        assert config.config["task"] == "transcribe"
        assert config.config["initial_prompt"] == "Meeting transcript:"
        assert config.config["word_timestamps"] is True
        assert config.config["temperature"] == 0.0
        assert config.config["condition_on_previous_text"] is True
        assert config.config["verbose"] is True

    async def test_event_bus_methods(self, plugin, event_bus):
        """Test event bus integration methods"""
        test_event = {"name": "test_event", "data": {}}
        test_callback = AsyncMock()

        # Test subscribe
        await plugin.subscribe("test_event", test_callback)
        event_bus.subscribe.assert_awaited_once_with("test_event", test_callback)

        # Test publish
        await plugin.publish(test_event)
        event_bus.publish.assert_awaited_once_with(test_event)

        # Test unsubscribe
        await plugin.unsubscribe("test_event", test_callback)
        event_bus.unsubscribe.assert_awaited_once_with("test_event", test_callback)
