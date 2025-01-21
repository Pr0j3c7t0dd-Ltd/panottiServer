from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch, call
import os
import numpy as np
import concurrent.futures

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

    @pytest.mark.asyncio
    async def test_transcribe_audio(self):
        mock_whisper = MagicMock()
        mock_segments = [MagicMock()]
        mock_segments[0].start = 0.0
        mock_segments[0].end = 1.0
        mock_segments[0].text = "Transcript content"
        mock_segments[0].words = [
            MagicMock(
                start=0.0,
                end=0.5,
                word="Transcript",
                probability=0.9
            ),
            MagicMock(
                start=0.5,
                end=1.0,
                word="content",
                probability=0.9
            )
        ]

        mock_result = MagicMock()
        mock_result.text = "Test transcription"
        mock_result.segments = mock_segments

        mock_whisper.transcribe.return_value = mock_result

        # Create mock config
        mock_config = MagicMock()
        mock_config.model_size = "base"
        mock_config.compute_type = "float32"
        mock_config.device = "cpu"
        mock_config.num_workers = 1

        # Create a mock file context
        mock_file_handle = mock_open()

        with patch("faster_whisper.WhisperModel", return_value=mock_whisper), \
             patch("pathlib.Path.mkdir") as mock_mkdir, \
             patch("builtins.open", mock_file_handle) as mock_file, \
             patch("concurrent.futures.ThreadPoolExecutor") as mock_executor, \
             patch("av.open"), \
             patch("faster_whisper.audio.decode_audio", return_value=(np.zeros(16000), 16000)):

            # Create a Future object for the executor's submit method
            future = concurrent.futures.Future()
            future.set_result((mock_segments, {}))

            # Create another Future for the write_transcript function
            write_future = concurrent.futures.Future()
            write_future.set_result(None)

            # Set up the mock executor to return different futures for different calls
            mock_executor.submit.side_effect = [future, write_future]

            plugin = AudioTranscriptionLocalPlugin(config=mock_config)
            plugin._model = mock_whisper
            plugin._executor = mock_executor

            output_path = Path("test_output/output.md")
            result = await plugin.transcribe_audio("test.wav", str(output_path), label="Speaker")

            # Verify directory creation
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)

            # Verify file operations
            mock_file.assert_called_with(output_path, 'w')
            
            # Verify model call
            mock_whisper.transcribe.assert_called_once_with(
                "test.wav",
                condition_on_previous_text=False,
                word_timestamps=True,
                vad_filter=True,
                beam_size=5
            )

            # Verify result
            assert result == output_path

            # Verify file content
            write_calls = [call.args[0] for call in mock_file_handle().write.mock_calls]
            assert "# Audio Transcript\n\n" in write_calls
            assert "Speaker: Speaker\n\n" in write_calls
            assert "## Segments\n\n" in write_calls
            assert "[00:00.000 -> 00:01.000] Speaker: Transcript content\n" in write_calls

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
