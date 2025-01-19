import pytest
from unittest.mock import AsyncMock, MagicMock, mock_open, patch
from pathlib import Path
from datetime import datetime

from app.plugins.base import PluginConfig
from app.plugins.audio_transcription_local.plugin import AudioTranscriptionLocalPlugin
from app.plugins.events.models import Event, EventContext
from app.models.recording.events import RecordingEvent
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
                "verbose": True
            }
        )

    @pytest.fixture
    def plugin(self, plugin_config, event_bus):
        """Audio transcription local plugin instance"""
        return AudioTranscriptionLocalPlugin(plugin_config, event_bus)

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
        with patch('faster_whisper.WhisperModel', autospec=True) as mock:
            model = MagicMock()
            mock.return_value = model
            model.transcribe = AsyncMock()
            model.transcribe.return_value = (MagicMock(
                text="Test transcription",
                segments=[
                    {"start": 0, "end": 1, "text": "Test"},
                    {"start": 1, "end": 2, "text": "transcription"}
                ]
            ), None)
            yield model

    async def test_transcription_initialization(self, plugin, mock_whisper):
        """Test audio transcription plugin specific initialization"""
        with patch('os.makedirs') as mock_makedirs, \
             patch('faster_whisper.transcribe.download_model', return_value="/path/to/model"), \
             patch('wave.open') as mock_wave:
            
            await plugin.initialize()
            
            # Verify directory creation
            mock_makedirs.assert_called_with(plugin._output_dir, exist_ok=True)
            
            # Verify event subscription
            plugin.event_bus.subscribe.assert_called_once_with(
                "noise_reduction.completed",
                plugin.handle_noise_reduction_completed
            )

    async def test_transcription_shutdown(self, plugin):
        """Test audio transcription plugin specific shutdown"""
        with patch('faster_whisper.transcribe.download_model', return_value="/path/to/model"), \
             patch('wave.open'):
            
            await plugin.initialize()
            await plugin.shutdown()
            
            # Verify event bus unsubscribe
            plugin.event_bus.unsubscribe.assert_called_once_with(
                "noise_reduction.completed",
                plugin.handle_noise_reduction_completed
            )

    async def test_handle_noise_reduction_completed(self, plugin, mock_whisper):
        """Test handling noise reduction completed event"""
        event_data = {
            "recording_id": "test_recording",
            "output_path": "/path/to/audio.wav",
            "data": {
                "recording_id": "test_recording",
                "output_path": "/path/to/audio.wav"
            }
        }

        with patch.object(plugin, 'transcribe_audio') as mock_transcribe, \
             patch('faster_whisper.transcribe.download_model', return_value="/path/to/model"), \
             patch('wave.open') as mock_wave:
            
            mock_transcribe.return_value = Path("transcript.txt")
            mock_wave.return_value.__enter__.return_value = MagicMock(
                getnchannels=lambda: 1,
                getsampwidth=lambda: 2,
                getframerate=lambda: 16000,
                getnframes=lambda: 16000
            )
            
            await plugin.initialize()
            await plugin.handle_noise_reduction_completed(event_data)
            
            mock_transcribe.assert_called_once_with(
                "test_recording",
                Path("/path/to/audio.wav"),
                event_data
            )

    async def test_handle_noise_reduction_completed_no_path(self, plugin):
        """Test handling noise reduction completed with missing path"""
        event_data = {
            "recording_id": "test_recording",
            "data": {
                "recording_id": "test_recording"
            }
        }

        with patch.object(plugin, 'transcribe_audio') as mock_transcribe, \
             patch('faster_whisper.transcribe.download_model', return_value="/path/to/model"), \
             patch('wave.open'):
            
            await plugin.initialize()
            await plugin.handle_noise_reduction_completed(event_data)
            
            mock_transcribe.assert_not_called()

    async def test_transcribe_audio(self, plugin, mock_whisper):
        """Test audio transcription"""
        recording_id = "test_recording"
        audio_path = Path("/path/to/audio.wav")
        event = Event(
            name="noise_reduction.completed",
            data={"recording_id": recording_id},
            context=EventContext(correlation_id="test_id")
        )

        with patch('os.makedirs') as mock_makedirs, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('soundfile.read', return_value=(MagicMock(), 16000)) as mock_sf_read, \
             patch('numpy.frombuffer', return_value=MagicMock()) as mock_frombuffer, \
             patch('numpy.float32', return_value=MagicMock()) as mock_float32, \
             patch('faster_whisper.transcribe.download_model', return_value="/path/to/model"), \
             patch('wave.open') as mock_wave:
            
            mock_wave.return_value.__enter__.return_value = MagicMock(
                getnchannels=lambda: 1,
                getsampwidth=lambda: 2,
                getframerate=lambda: 16000,
                getnframes=lambda: 16000
            )
            
            await plugin.initialize()
            output_path = await plugin.transcribe_audio(recording_id, audio_path, event)
            
            assert output_path.suffix == ".txt"
            assert mock_file.call_count > 0
            assert plugin.event_bus.publish.called
            mock_sf_read.assert_called_once_with(str(audio_path))
            mock_makedirs.assert_called()
            mock_wave.assert_called_once_with(str(audio_path), "rb")

    def test_plugin_configuration(self, plugin):
        """Test plugin configuration parameters"""
        config = plugin.get_config()
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