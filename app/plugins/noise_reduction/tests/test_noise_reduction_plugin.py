import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from app.plugins.base import PluginConfig
from app.plugins.noise_reduction.plugin import NoiseReductionPlugin, AudioPaths
from app.plugins.events.models import Event
from app.models.recording.events import RecordingEvent
from tests.plugins.test_plugin_interface import BasePluginTest


class TestNoiseReductionPlugin(BasePluginTest):
    """Test suite for NoiseReductionPlugin"""

    @pytest.fixture
    def plugin_config(self):
        """Noise reduction plugin specific config"""
        return PluginConfig(
            name="noise_reduction",
            version="1.0.0",
            enabled=True,
            config={
                "output_directory": "data/cleaned_audio",
                "noise_reduce_factor": 0.8,
                "wiener_alpha": 2.0,
                "highpass_cutoff": 100,
                "spectral_floor": 0.05,
                "smoothing_factor": 3,
                "max_concurrent_tasks": 2,
                "time_domain_subtraction": True,
                "freq_domain_bleed_removal": True,
                "use_fft_alignment": True,
                "alignment_chunk_seconds": 5
            }
        )

    @pytest.fixture
    def plugin(self, plugin_config, event_bus):
        """Noise reduction plugin instance"""
        return NoiseReductionPlugin(plugin_config, event_bus)

    @pytest.fixture
    def mock_db(self):
        """Mock database manager"""
        with patch('app.models.database.DatabaseManager') as mock:
            db_instance = MagicMock()
            mock.get_instance.return_value = db_instance
            yield db_instance

    async def test_noise_reduction_initialization(self, plugin, mock_db):
        """Test noise reduction plugin specific initialization"""
        with patch('os.makedirs') as mock_makedirs:
            await plugin.initialize()
            
            # Verify directory creation
            mock_makedirs.assert_any_call(plugin._output_directory, exist_ok=True)
            mock_makedirs.assert_any_call(plugin._recordings_dir, exist_ok=True)
            
            # Verify event subscription
            plugin.event_bus.subscribe.assert_called_once_with(
                "recording.ended",
                plugin.handle_recording_ended
            )

    async def test_noise_reduction_shutdown(self, plugin):
        """Test noise reduction plugin specific shutdown"""
        await plugin.initialize()
        await plugin.shutdown()
        
        # Verify thread pool shutdown
        assert plugin._executor.shutdown.called

    def test_audio_paths_initialization(self):
        """Test AudioPaths class initialization"""
        paths = AudioPaths("test_recording", "system.wav", "mic.wav")
        assert paths.recording_id == "test_recording"
        assert paths.system_audio == "system.wav"
        assert paths.mic_audio == "mic.wav"

    def test_detect_start_of_audio(self):
        """Test audio start detection"""
        # Create test signal with initial silence
        silence = np.zeros(1000)
        audio = np.random.randn(1000) * 0.1
        signal = np.concatenate([silence, audio])
        
        start_idx = NoiseReductionPlugin.detect_start_of_audio(
            signal, threshold=0.01, frame_size=100
        )
        assert start_idx >= 1000  # Should detect start after silence

    async def test_handle_recording_ended(self, plugin):
        """Test recording ended event handler"""
        event_data = {
            "recording_id": "test_recording",
            "data": {
                "system_audio": "system.wav",
                "mic_audio": "mic.wav"
            }
        }

        with patch.object(plugin, 'process_recording') as mock_process:
            await plugin.initialize()
            await plugin.handle_recording_ended(event_data)
            
            mock_process.assert_called_once_with(
                "test_recording",
                event_data
            )

    async def test_handle_recording_ended_no_audio(self, plugin):
        """Test recording ended handler with missing audio paths"""
        event_data = {
            "recording_id": "test_recording",
            "data": {}
        }

        with patch.object(plugin, 'process_recording') as mock_process:
            await plugin.initialize()
            await plugin.handle_recording_ended(event_data)
            
            mock_process.assert_called_once_with(
                "test_recording",
                event_data
            )

    def test_align_signals_by_fft(self, plugin):
        """Test FFT-based signal alignment"""
        # Create test signals with known lag
        sample_rate = 44100
        t = np.linspace(0, 1, sample_rate)
        signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        lag_samples = 100
        lagged_signal = np.pad(signal, (lag_samples, 0))[:-lag_samples]
        
        mic_aligned, sys_aligned, lag = plugin._align_signals_by_fft(
            lagged_signal, signal, sample_rate
        )
        
        assert abs(lag * sample_rate - lag_samples) < 10  # Allow small alignment error
        assert len(mic_aligned) == len(sys_aligned)

    def test_plugin_configuration(self, plugin):
        """Test plugin configuration parameters"""
        assert isinstance(plugin._output_directory, Path)
        assert str(plugin._output_directory) == "data/cleaned_audio"
        assert plugin._noise_reduce_factor == 0.8
        assert plugin._wiener_alpha == 2.0
        assert plugin._highpass_cutoff == 100
        assert plugin._spectral_floor == 0.05
        assert plugin._smoothing_factor == 3
        assert plugin._max_workers == 2
        assert plugin._time_domain_subtraction is True
        assert plugin._freq_domain_bleed_removal is True
        assert plugin._use_fft_alignment is True
        assert plugin._alignment_chunk_seconds == 5 