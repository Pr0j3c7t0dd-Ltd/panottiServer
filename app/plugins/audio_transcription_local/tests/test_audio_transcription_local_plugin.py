from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch, call
import os
import numpy as np
import concurrent.futures
import asyncio

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
            "data": {
                "noise_reduction": {
                    "recording_id": recording_id,
                    "output_path": audio_path,
                },
                "recording": {
                    "recording_id": recording_id,
                },
                "metadata": {
                    "speaker_labels": {
                        "microphone": "Microphone",
                        "system": "System"
                    }
                }
            }
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
                str(Path(audio_path)), 
                "Microphone",
                {"speaker_labels": {"microphone": "Microphone", "system": "System"}}
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
    async def test_transcribe_audio(self, plugin, mock_whisper):
        """Test transcribing audio file"""
        # Test data
        audio_path = "test_input.wav"
        output_path = "test_output/output.md"
        label = "Speaker"
        metadata = {"test": "data"}

        # Mock segments and results
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

        # Mock the transcribe method to return the mock result
        mock_whisper.transcribe.return_value = (mock_segments, mock_result)

        # Set up mocks
        with patch.object(plugin, "_init_model"), \
             patch.object(plugin, "_model", mock_whisper), \
             patch("builtins.open", mock_open()) as mock_file, \
             patch("pathlib.Path.mkdir") as mock_mkdir, \
             patch("os.path.exists", return_value=True), \
             patch("asyncio.get_running_loop") as mock_loop:

            # Create a mock loop
            mock_loop_instance = MagicMock()
            mock_loop.return_value = mock_loop_instance

            # Create futures for both calls
            transcribe_future = asyncio.Future()
            transcribe_future.set_result((mock_segments, mock_result))

            write_future = asyncio.Future()
            write_future.set_result(None)

            # Mock the event loop's run_in_executor
            async def executor_side_effect(executor, func, *args):
                if isinstance(func, type(lambda: None)):  # Check if it's a lambda
                    # First call - transcription
                    mock_whisper.transcribe.return_value = (mock_segments, mock_result)
                    # Execute the lambda and await any coroutine it returns
                    result = func()
                    if asyncio.iscoroutine(result):
                        await result
                    return await transcribe_future
                else:
                    # Second call - file writing
                    func()  # Execute the file writing function
                    return await write_future

            mock_loop_instance.run_in_executor = AsyncMock(side_effect=executor_side_effect)

            # Call the method
            result = await plugin.transcribe_audio(audio_path, output_path, label, metadata)

            # Verify file operations
            mock_file.assert_called_with(Path(output_path), 'w')
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)

            # Verify model call
            mock_whisper.transcribe.assert_called_once_with(
                audio_path,
                condition_on_previous_text=False,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=100,
                ),
                beam_size=5
            )

            # Verify result
            assert result == Path(output_path)

            # Verify file content was written
            handle = mock_file()
            write_calls = [call[0][0] for call in handle.write.call_args_list]
            assert any("# Audio Transcript" in call for call in write_calls)
            assert any(f"Speaker: {label}" in call for call in write_calls)
            assert any("## Metadata" in call for call in write_calls)
            assert any("## Segments" in call for call in write_calls)

            # Verify run_in_executor was called twice
            assert mock_loop_instance.run_in_executor.call_count == 2

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
