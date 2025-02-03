import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch
import json

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
        transcript_path = str(
            Path("data/transcripts_local/test_recording_transcript.txt")
        )

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
                    "speaker_labels": {"microphone": "Microphone", "system": "System"}
                },
            },
        }

        # Create mock transcription results
        mock_segments = [
            MagicMock(text="Test", start=0.0, end=1.0),
            MagicMock(text="transcription", start=1.0, end=2.0),
        ]
        mock_transcript = MagicMock(text="Test transcription")

        # Set up the mocks
        with patch.object(
            plugin, "_process_audio", new_callable=AsyncMock
        ) as mock_process_audio, patch.object(plugin, "_init_model"), patch.object(
            plugin, "_model", mock_whisper
        ), patch("builtins.open", mock_open()) as mock_file, patch(
            "os.path.exists"
        ) as mock_exists:
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
                {"speaker_labels": {"microphone": "Microphone", "system": "System"}},
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
            MagicMock(start=0.0, end=0.5, word="Transcript", probability=0.9),
            MagicMock(start=0.5, end=1.0, word="content", probability=0.9),
        ]

        mock_result = MagicMock()
        mock_result.text = "Test transcription"
        mock_result.segments = mock_segments

        # Mock the transcribe method to return the mock result
        mock_whisper.transcribe.return_value = (mock_segments, mock_result)

        # Set up mocks
        with patch.object(plugin, "_init_model"), patch.object(
            plugin, "_model", mock_whisper
        ), patch("builtins.open", mock_open()) as mock_file, patch(
            "pathlib.Path.mkdir"
        ) as mock_mkdir, patch("os.path.exists", return_value=True), patch(
            "asyncio.get_running_loop"
        ) as mock_loop:
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

            mock_loop_instance.run_in_executor = AsyncMock(
                side_effect=executor_side_effect
            )

            # Call the method
            result = await plugin.transcribe_audio(
                audio_path, output_path, label, metadata
            )

            # Verify file operations
            mock_file.assert_called_with(Path(output_path), "w")
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
                beam_size=5,
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

    async def test_initialization_error(self, plugin_config, event_bus):
        """Test error handling during initialization"""
        with patch("app.models.database.DatabaseManager.get_instance") as mock_db_get:
            mock_db_get.side_effect = Exception("DB Error")
            plugin = AudioTranscriptionLocalPlugin(plugin_config, event_bus)
            
            with pytest.raises(Exception) as exc_info:
                await plugin.initialize()
            assert str(exc_info.value) == "DB Error"

    async def test_shutdown_error_handling(self, plugin):
        """Test error handling during shutdown"""
        # Mock the executor to raise an error on shutdown
        plugin._executor = MagicMock()
        plugin._executor.shutdown.side_effect = Exception("Shutdown Error")
        
        # Mock processing lock to test timeout
        plugin._processing_lock = MagicMock()
        plugin._processing_lock.locked.return_value = True
        
        # Mock the model
        plugin._model = MagicMock()
        
        # Mock the event bus
        plugin.event_bus = AsyncMock()
        plugin.event_bus.unsubscribe = AsyncMock()
        
        # Mock the shutdown event
        plugin._shutdown_event = AsyncMock()
        plugin._shutdown_event.set = MagicMock()
        
        await plugin.shutdown()
        # Verify the plugin handles the error gracefully and cleans up resources
        assert plugin._shutdown_event.set.called
        assert plugin.event_bus.unsubscribe.called

    async def test_handle_noise_reduction_invalid_event(self, plugin):
        """Test handling invalid noise reduction event data"""
        invalid_event = {"name": "noise_reduction.completed", "data": {}}
        await plugin.handle_noise_reduction_completed(invalid_event)
        
        # Verify no transcription events were emitted
        plugin.event_bus.publish.assert_not_called()

    async def test_database_initialization(self, plugin, mock_db):
        """Test database initialization"""
        plugin.db = mock_db
        
        # Mock the database connection and cursor
        mock_connection = MagicMock()
        mock_db.get_connection.return_value.__enter__.return_value = mock_connection
        
        # Mock successful execution
        mock_connection.execute.return_value = None
        
        # Mock the database schema - exactly matching the source file
        schema_sql = """
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id TEXT NOT NULL,
                    transcript TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (recording_id) REFERENCES recordings(recording_id)
                )
            """
        
        with patch("app.models.database.DatabaseManager.get_instance", return_value=mock_db):
            await plugin._init_database()
            mock_connection.execute.assert_called_with(schema_sql)
            mock_connection.commit.assert_called_once()
            
            # Test error handling with a new mock connection
            error_connection = MagicMock()
            error_connection.execute.side_effect = Exception("DB Error")
            mock_db.get_connection.return_value.__enter__.return_value = error_connection
            
            # The error should be handled gracefully
            try:
                await plugin._init_database()
            except Exception as e:
                assert str(e) == "DB Error"

    async def test_transcription_error_handling(self, plugin, mock_whisper):
        """Test error handling during transcription"""
        audio_path = "test.wav"
        output_path = "output.txt"
        label = "Speaker"
        
        # Mock transcription error
        mock_whisper.transcribe.side_effect = Exception("Transcription Error")
        
        with patch.object(plugin, "_model", mock_whisper), \
             patch("os.path.exists", return_value=True), \
             patch("wave.open") as mock_wave, \
             patch("asyncio.get_running_loop") as mock_loop:
            
            mock_wave.return_value.__enter__.return_value.getnchannels.return_value = 1
            mock_loop_instance = AsyncMock()
            mock_loop.return_value = mock_loop_instance
            
            # Mock the executor to raise an exception
            def executor_side_effect(*args, **kwargs):
                raise Exception("Transcription Error")
            
            mock_loop_instance.run_in_executor.side_effect = executor_side_effect
            
            try:
                result = await plugin.transcribe_audio(audio_path, output_path, label)
                assert result is None
            except Exception as e:
                assert str(e) == "Transcription Error"

    async def test_audio_processing_error(self, plugin):
        """Test error handling in audio processing"""
        audio_path = "invalid.wav"
        
        with patch("os.path.exists", return_value=False):
            result = await plugin._process_audio(audio_path, "Speaker")
            assert result is None
            
        with patch("os.path.exists", return_value=True), \
             patch("wave.open") as mock_wave:
            mock_wave.side_effect = Exception("Invalid WAV file")
            result = await plugin._process_audio(audio_path, "Speaker")
            assert result is None

    async def test_emit_transcription_event(self, plugin):
        """Test transcription event emission"""
        recording_id = "test_id"
        status = "completed"
        output_file = "output.txt"
        error = None
        
        # Mock the event bus
        plugin.event_bus = AsyncMock()
        plugin.event_bus.publish = AsyncMock()
        
        await plugin._emit_transcription_event(
            recording_id, status, output_file, error
        )
        
        plugin.event_bus.publish.assert_called_once()
        event = plugin.event_bus.publish.call_args[0][0]
        assert event.name == "transcription_local.completed"
        assert event.data["transcription"]["recording_id"] == recording_id
        assert event.data["transcription"]["status"] == status
        assert event.data["transcription"]["output_file"] == output_file
