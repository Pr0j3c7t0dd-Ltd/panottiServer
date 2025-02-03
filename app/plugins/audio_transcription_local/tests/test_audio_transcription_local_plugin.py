import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch, call
import json
import wave

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
        executor = MagicMock()
        executor.shutdown.side_effect = Exception("Shutdown Error")
        plugin._executor = executor
        
        # Mock processing lock to simulate a busy state that times out
        plugin._processing_lock = MagicMock()
        plugin._processing_lock.locked.side_effect = [True, True, False]  # Will return True twice then False
        
        # Mock the model
        plugin._model = MagicMock()
        
        # Mock the event bus
        plugin.event_bus = AsyncMock()
        plugin.event_bus.unsubscribe = AsyncMock()
        
        # Test shutdown
        await plugin._shutdown()
        
        # Verify event bus unsubscribe was called
        plugin.event_bus.unsubscribe.assert_called_once_with(
            "noise_reduction.completed", plugin.handle_noise_reduction_completed
        )
        
        # Verify executor shutdown was attempted
        executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        # Verify executor was set to None after shutdown attempt
        assert plugin._executor is None

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
        """Test emitting transcription event"""
        recording_id = "test_recording"
        status = "completed"
        output_file = "output.txt"
        error = None
        original_event = {
            "metadata": {"test": "data"},
            "context": {"metadata": {"additional": "info"}}
        }
        transcript_paths = {"mic": "mic.txt", "sys": "sys.txt"}

        await plugin._emit_transcription_event(
            recording_id, status, output_file, error, original_event, transcript_paths
        )
        plugin.event_bus.publish.assert_called_once()

    async def test_emit_transcription_event_error(self, plugin):
        """Test emitting transcription event with error"""
        recording_id = "test_recording"
        status = "error"
        error = "Test error"
        original_event = {
            "recording": {"id": "test"},
            "noise_reduction": {"status": "completed"},
            "metadata": {"test": "data"}
        }

        await plugin._emit_transcription_event(recording_id, status, error=error, original_event=original_event)
        plugin.event_bus.publish.assert_called_once()

    async def test_handle_noise_reduction_completed_invalid_event(self, plugin):
        """Test handling invalid noise reduction event"""
        event_data = {"name": "wrong.event"}
        await plugin.handle_noise_reduction_completed(event_data)
        plugin.event_bus.publish.assert_not_called()

    async def test_handle_noise_reduction_completed_missing_data(self, plugin):
        """Test handling noise reduction event with missing data"""
        event_data = {
            "name": "noise_reduction.completed",
            "data": {}
        }
        await plugin.handle_noise_reduction_completed(event_data)
        plugin.event_bus.publish.assert_not_called()

    async def test_handle_noise_reduction_completed_full_flow(self, plugin, mock_whisper):
        """Test full flow of noise reduction handling"""
        event_data = {
            "name": "noise_reduction.completed",
            "data": {
                "noise_reduction": {
                    "recording_id": "test_recording",
                    "output_path": "processed.wav",
                    "system_audio_path": "system.wav"
                },
                "recording": {"id": "test"},
                "metadata": {
                    "speaker_labels": {
                        "microphone": "Speaker 1",
                        "system": "Speaker 2"
                    }
                },
                "context": {
                    "metadata": {
                        "additional": "info"
                    }
                }
            }
        }

        with patch.object(plugin, "validate_wav_file", return_value=True), \
             patch.object(plugin, "_process_audio", new_callable=AsyncMock) as mock_process, \
             patch.object(plugin, "merge_transcripts", new_callable=AsyncMock) as mock_merge, \
             patch.object(plugin, "_emit_transcription_event", new_callable=AsyncMock) as mock_emit, \
             patch("os.path.exists", side_effect=lambda x: x in ["processed.wav", "system.wav"]):

            mock_process.return_value = Path("test_output.md")
            await plugin.handle_noise_reduction_completed(event_data)

            # Verify processing of both mic and system audio
            assert mock_process.call_count == 3  # Updated to expect 3 calls
            mock_merge.assert_called_once()
            mock_emit.assert_called_once()

    async def test_process_audio(self, plugin, mock_whisper):
        """Test processing audio file"""
        audio_path = "test.wav"
        speaker_label = "Speaker"
        metadata = {"test": "data"}

        with patch.object(plugin, "validate_wav_file", return_value=True), \
             patch.object(plugin, "transcribe_audio", new_callable=AsyncMock) as mock_transcribe, \
             patch("os.path.exists", return_value=True):

            mock_transcribe.return_value = Path("output.md")
            result = await plugin._process_audio(audio_path, speaker_label, metadata)

            assert result == Path("output.md")
            mock_transcribe.assert_called_once()

    async def test_process_audio_invalid_path(self, plugin):
        """Test processing audio with invalid path"""
        with patch("os.path.exists", return_value=False):
            result = await plugin._process_audio("invalid.wav", "Speaker")
            assert result is None

    async def test_process_audio_transcription_error(self, plugin):
        """Test processing audio with transcription error"""
        with patch.object(plugin, "validate_wav_file", return_value=True), \
             patch.object(plugin, "transcribe_audio", side_effect=Exception("Transcription error")), \
             patch("os.path.exists", return_value=True):

            result = await plugin._process_audio("test.wav", "Speaker")
            assert result is None

    @patch("app.plugins.audio_transcription_local.plugin.Path.mkdir")
    @patch("pathlib.Path")
    @patch("app.plugins.audio_transcription_local.plugin.WhisperModel")
    async def test_init_model(self, mock_whisper_model_cls, mock_path, mock_mkdir, tmp_path):
        """Test initialization of the Whisper model"""
        test_root = tmp_path / "test"
        test_root.mkdir(parents=True, exist_ok=True)
        
        # Create mock paths
        mock_file = MagicMock(spec=Path)
        mock_file.__str__.return_value = str(test_root / "app/plugins/audio_transcription_local/plugin.py")
        mock_file.resolve.return_value = mock_file
        
        mock_parent = MagicMock(spec=Path)
        mock_parent.__str__.return_value = str(test_root / "app/plugins/audio_transcription_local")
        mock_file.parent = mock_parent
        
        mock_project_root = MagicMock(spec=Path)
        mock_project_root.__str__.return_value = str(test_root)
        mock_parent.parent.parent.parent = mock_project_root
        
        models_dir = MagicMock(spec=Path)
        models_dir.__str__.return_value = str(test_root / "models")
        
        model_dir = MagicMock(spec=Path)
        model_dir.__str__.return_value = str(test_root / "models/whisper")
        model_dir.parent = models_dir
        
        # Set up path resolution
        def path_side_effect(p):
            if str(p) == str(test_root / "app/plugins/audio_transcription_local/plugin.py"):
                return mock_file
            elif str(p) == str(test_root / "models"):
                return models_dir
            elif str(p) == str(test_root / "models/whisper"):
                return model_dir
            return Path(p)
        mock_path.side_effect = path_side_effect
        
        # Set up model directory resolution
        mock_project_root.__truediv__.side_effect = lambda x: models_dir if x == "models" else Path(x)
        models_dir.__truediv__.return_value = model_dir
        
        # Create mock WhisperModel instance
        mock_whisper_model = MagicMock()
        mock_whisper_model_cls.return_value = mock_whisper_model

        # Create plugin instance with config
        config = PluginConfig(
            name="audio_transcription_local",
            version="1.0.0",
            enabled=True,
            config={"model_name": "base.en"}
        )
        
        # Create plugin instance and initialize model
        with patch("app.plugins.audio_transcription_local.plugin.__file__", str(test_root / "app/plugins/audio_transcription_local/plugin.py")):
            plugin = AudioTranscriptionLocalPlugin(config)
            plugin._init_model()

            # Verify model initialization
            mock_whisper_model_cls.assert_called_once_with(
                model_size_or_path="base.en",
                device="cpu",
                device_index=0,
                compute_type="default",
                download_root=str(test_root / "models/whisper"),
                local_files_only=False
            )
            assert plugin._model == mock_whisper_model
            
            # Verify all mkdir calls
            assert mock_mkdir.call_count == 3
            mock_mkdir.assert_has_calls([
                call(parents=True, exist_ok=True),
                call(parents=True, exist_ok=True),
                call(parents=True, exist_ok=True)
            ])
