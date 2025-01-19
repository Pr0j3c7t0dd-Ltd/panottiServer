import concurrent.futures
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch, Mock
import asyncio

import numpy as np
import pytest

from app.core.events import Event, EventContext
from app.plugins.base import PluginConfig
from app.plugins.noise_reduction.plugin import AudioPaths, NoiseReductionPlugin
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
                "alignment_chunk_seconds": 5,
            },
        )

    @pytest.fixture
    def plugin(self, plugin_config, event_bus):
        """Noise reduction plugin instance"""
        plugin = NoiseReductionPlugin(plugin_config, event_bus)
        return plugin

    @pytest.fixture
    def mock_db(self):
        """Mock database manager"""
        db_instance = AsyncMock()
        db_instance.execute = AsyncMock()
        return db_instance

    @pytest.fixture
    def mock_executor(self):
        """Mock thread pool executor"""
        with patch('concurrent.futures.ThreadPoolExecutor') as mock:
            executor = Mock()
            executor.shutdown = Mock()  # Use regular Mock since this is called synchronously
            mock.return_value = executor
            yield executor

    @pytest.fixture
    async def initialized_plugin(self, plugin, mock_db, mock_executor):
        """Plugin instance that's been initialized with mocked dependencies"""
        with patch("app.models.database.get_db_async", return_value=mock_db):
            # Mock run_in_executor to actually call the function
            loop = asyncio.get_event_loop()
            async def mock_run_in_executor(executor_or_none, func, *args):
                if func == mock_executor.shutdown:
                    func(*args)
                return None
            with patch.object(loop, 'run_in_executor', side_effect=mock_run_in_executor):
                await plugin.initialize()
                # Set the mock executor on the plugin
                plugin._executor = mock_executor
                yield plugin
                await plugin.shutdown()

    @pytest.fixture
    def mock_makedirs(self):
        """Mock os.makedirs"""
        with patch("os.makedirs") as mock:
            yield mock

    @pytest.fixture
    def mock_event_bus(self):
        """Mock event bus with mocked subscribe method"""
        event_bus = AsyncMock()
        event_bus.subscribe = AsyncMock()
        return event_bus

    async def test_noise_reduction_initialization(self, plugin_config, mock_event_bus, mock_makedirs):
        """Test noise reduction plugin specific initialization"""
        # Create a fresh plugin instance for testing initialization
        plugin = NoiseReductionPlugin(plugin_config, mock_event_bus)
        await plugin.initialize()

        # Verify directory creation
        mock_makedirs.assert_any_call(
            plugin._output_directory, exist_ok=True
        )
        mock_makedirs.assert_any_call(
            plugin._recordings_dir, exist_ok=True
        )

        # Verify event subscription
        mock_event_bus.subscribe.assert_called_once_with(
            "recording.ended", plugin.handle_recording_ended
        )
        
        # Clean up
        await plugin.shutdown()

    async def test_noise_reduction_shutdown(self, initialized_plugin, mock_executor):
        """Test noise reduction plugin specific shutdown"""
        # Mock run_in_executor to actually call the function
        loop = asyncio.get_event_loop()
        async def mock_run_in_executor(executor_or_none, func, *args):
            if func == mock_executor.shutdown:
                func(*args)
            return None
        with patch.object(loop, 'run_in_executor', side_effect=mock_run_in_executor):
            await initialized_plugin.shutdown()

        # Verify thread pool shutdown was called with wait=True
        mock_executor.shutdown.assert_called_once_with(True)

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

    async def test_handle_recording_ended(self, initialized_plugin):
        """Test recording ended event handler"""
        event_data = {
            "recording_id": "test_recording",
            "data": {"system_audio": "system.wav", "mic_audio": "mic.wav"},
        }

        with patch.object(initialized_plugin, "process_recording") as mock_process:
            await initialized_plugin.handle_recording_ended(event_data)

            mock_process.assert_called_once_with("test_recording", event_data)

    async def test_handle_recording_ended_no_audio(self, initialized_plugin):
        """Test recording ended handler with missing audio paths"""
        recording_id = "test_recording"
        event_data = {
            "recording_id": recording_id,
            "current_event": {
                "recording": {
                    "audio_paths": {}  # Empty audio paths to test no-audio case
                }
            },
            "metadata": {},
        }

        with patch.object(initialized_plugin, "_process_audio_files") as mock_process:
            mock_process.return_value = None  # Ensure async mock returns something

            # Call handler directly since we're testing the handler itself
            await initialized_plugin.handle_recording_ended(event_data)

            # Verify _process_audio_files was called with correct args
            mock_process.assert_called_once_with(
                recording_id,
                None,  # system_audio_path
                None,  # microphone_audio_path
                {},  # metadata
            )

    def test_align_signals_by_fft(self, initialized_plugin):
        """Test FFT-based signal alignment"""
        # Create test signals with known lag
        sample_rate = 44100
        t = np.linspace(0, 1, sample_rate)
        signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        lag_samples = 100
        lagged_signal = np.pad(signal, (lag_samples, 0))[:-lag_samples]

        mic_aligned, sys_aligned, lag = initialized_plugin._align_signals_by_fft(
            lagged_signal, signal, sample_rate
        )

        assert abs(lag * sample_rate - lag_samples) < 10  # Allow small alignment error
        assert len(mic_aligned) == len(sys_aligned)

    def test_plugin_configuration(self, initialized_plugin):
        """Test plugin configuration parameters"""
        assert isinstance(initialized_plugin._output_directory, Path)
        assert str(initialized_plugin._output_directory) == "data/cleaned_audio"
        assert initialized_plugin._noise_reduce_factor == 0.8
        assert initialized_plugin._wiener_alpha == 2.0
        assert initialized_plugin._highpass_cutoff == 100
        assert initialized_plugin._spectral_floor == 0.05
        assert initialized_plugin._smoothing_factor == 3
        assert initialized_plugin._max_workers == 2
        assert initialized_plugin._time_domain_subtraction is True
        assert initialized_plugin._freq_domain_bleed_removal is True
        assert initialized_plugin._use_fft_alignment is True
        assert initialized_plugin._alignment_chunk_seconds == 5

    async def test_handle_recording_ended_dict_format(self, initialized_plugin):
        """Test recording ended handler with dictionary event format"""
        event_data = {
            "recording_id": "test_recording",
            "current_event": {
                "recording": {
                    "audio_paths": {"microphone": "mic.wav", "system": "sys.wav"}
                }
            },
            "metadata": {"test": "data"},
            "source_plugin": "other_plugin",
        }

        with patch.object(initialized_plugin, "_process_audio_files") as mock_process:
            mock_process.return_value = None
            await initialized_plugin.handle_recording_ended(event_data)

            mock_process.assert_called_once_with(
                "test_recording", "sys.wav", "mic.wav", {"test": "data"}
            )

    async def test_handle_recording_ended_skip_own_event(self, initialized_plugin):
        """Test recording ended handler skips events from itself"""
        event_data = {
            "recording_id": "test_recording",
            "current_event": {
                "recording": {
                    "audio_paths": {"microphone": "mic.wav", "system": "sys.wav"}
                }
            },
            "source_plugin": "noise_reduction",
        }

        with patch.object(initialized_plugin, "_process_audio_files") as mock_process:
            await initialized_plugin.handle_recording_ended(event_data)

            mock_process.assert_not_called()

    async def test_handle_recording_ended_no_recording_id(self, initialized_plugin):
        """Test recording ended handler with missing recording_id"""
        event_data = {
            "current_event": {
                "recording": {
                    "audio_paths": {"microphone": "mic.wav", "system": "sys.wav"}
                }
            }
        }

        with patch.object(initialized_plugin, "_process_audio_files") as mock_process:
            await initialized_plugin.handle_recording_ended(event_data)

            mock_process.assert_not_called()

    async def test_handle_recording_ended_event_object(self, initialized_plugin):
        """Test recording ended handler with Event object"""
        event = Event(
            name="recording.ended",
            data={
                "recording_id": "test_recording",
                "recording_timestamp": datetime.now(UTC).isoformat(),
                "system_audio_path": "sys.wav",
                "microphone_audio_path": "mic.wav",
                "metadata": {"test": "data"},
                "event": "recording.ended",
            },
            context=EventContext(correlation_id="test_id", source_plugin="test_plugin"),
        )

        with patch.object(initialized_plugin, "_process_audio_files") as mock_process:
            mock_process.return_value = None
            await initialized_plugin.handle_recording_ended(event)

            mock_process.assert_called_once_with(
                "test_recording", "sys.wav", "mic.wav", {"test": "data"}
            )
