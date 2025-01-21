import asyncio
import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import soundfile as sf

from app.core.plugins import PluginConfig
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
        with patch("concurrent.futures.ThreadPoolExecutor") as mock:
            executor = Mock()
            executor.shutdown = (
                Mock()
            )  # Use regular Mock since this is called synchronously
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

            with patch.object(
                loop, "run_in_executor", side_effect=mock_run_in_executor
            ):
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

    async def test_noise_reduction_initialization(
        self, plugin_config, mock_event_bus, mock_makedirs
    ):
        """Test noise reduction plugin specific initialization"""
        # Create a fresh plugin instance for testing initialization
        plugin = NoiseReductionPlugin(plugin_config, mock_event_bus)
        await plugin.initialize()

        # Verify directory creation
        mock_makedirs.assert_any_call(plugin._output_directory, exist_ok=True)
        mock_makedirs.assert_any_call(plugin._recordings_dir, exist_ok=True)

        # Verify event subscription
        mock_event_bus.subscribe.assert_called_once_with(
            "recording.ended", plugin.__call__
        )

        # Clean up
        await plugin.shutdown()

    async def test_noise_reduction_shutdown(self, initialized_plugin, mock_executor):
        """Test noise reduction plugin specific shutdown"""
        # Mock run_in_executor to actually call the function
        loop = asyncio.get_event_loop()

        async def mock_run_in_executor(executor, func, *args):
            if func == mock_executor.shutdown:
                func(*args)

        with patch.object(loop, "run_in_executor", side_effect=mock_run_in_executor):
            await initialized_plugin.shutdown()

        # Verify thread pool shutdown was called with wait=True
        mock_executor.shutdown.assert_called_once_with(True)

    def test_audio_paths_initialization(self):
        """Test AudioPaths class initialization"""
        paths = AudioPaths("test_recording", "system.wav", "mic.wav")
        assert paths.recording_id == "test_recording"
        assert paths.system_audio == "system.wav"
        assert paths.mic_audio == "mic.wav"

    async def test_handle_recording_ended(self, initialized_plugin):
        """Test recording ended event handler"""
        event_data = {
            "recording_id": "test_recording",
            "current_event": {
                "recording": {
                    "audio_paths": {"microphone": "mic.wav", "system": "system.wav"}
                }
            },
            "metadata": {},
        }

        def mock_exists(path):
            return path in ["mic.wav", "system.wav"]

        with patch("os.path.exists", side_effect=mock_exists):
            with patch.object(
                initialized_plugin, "_process_audio_files"
            ) as mock_process:
                await initialized_plugin.handle_recording_ended(event_data)

                mock_process.assert_called_once_with(
                    "test_recording", "system.wav", "mic.wav", {}
                )

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
        event = {
            "recording_id": "test_recording",
            "recording_timestamp": datetime.now(UTC).isoformat(),
            "current_event": {
                "recording": {
                    "audio_paths": {"microphone": "mic.wav", "system": "sys.wav"}
                }
            },
            "metadata": {"test": "data"},
            "event": "recording.ended",
            "source_plugin": "test_plugin",
        }

        with patch.object(initialized_plugin, "_process_audio_files") as mock_process:
            mock_process.return_value = None
            await initialized_plugin.handle_recording_ended(event)

            mock_process.assert_called_once_with(
                "test_recording", "sys.wav", "mic.wav", {"test": "data"}
            )

    async def test_subtract_bleed_time_domain(self, initialized_plugin, tmp_path):
        """Test time-domain bleed removal functionality"""
        # Create test audio files
        sample_rate = 44100
        duration = 1.0  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create test signals
        mic_signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
        sys_signal = 0.3 * np.sin(2 * np.pi * 440 * t)

        # Save test files
        mic_file = tmp_path / "mic.wav"
        sys_file = tmp_path / "sys.wav"
        output_file = tmp_path / "output.wav"

        sf.write(mic_file, mic_signal, sample_rate)
        sf.write(sys_file, sys_signal, sample_rate)

        with patch("soundfile.read") as mock_read, patch(
            "soundfile.write"
        ) as mock_write:
            mock_read.side_effect = [
                (mic_signal, sample_rate),
                (sys_signal, sample_rate),
            ]

            initialized_plugin._subtract_bleed_time_domain(
                str(mic_file),
                str(sys_file),
                str(output_file),
                do_alignment=True,
                auto_scale=True,
            )

            # Verify write was called
            mock_write.assert_called_once()
            written_signal = mock_write.call_args[0][1]
            assert written_signal.shape == mic_signal.shape

    def test_remove_bleed_frequency_domain(self, initialized_plugin, tmp_path):
        """Test frequency-domain bleed removal functionality"""
        # Create test audio files
        sample_rate = 44100
        duration = 1.0  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create test signals with known frequency components
        mic_signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
        sys_signal = 0.3 * np.sin(2 * np.pi * 440 * t)

        # Save test files
        mic_file = tmp_path / "mic.wav"
        sys_file = tmp_path / "sys.wav"
        output_file = tmp_path / "output.wav"

        sf.write(mic_file, mic_signal, sample_rate)
        sf.write(sys_file, sys_signal, sample_rate)

        with patch("soundfile.read") as mock_read, patch(
            "soundfile.write"
        ) as mock_write:
            mock_read.side_effect = [
                (mic_signal, sample_rate),
                (sys_signal, sample_rate),
            ]

            initialized_plugin._remove_bleed_frequency_domain(
                str(mic_file),
                str(sys_file),
                str(output_file),
                do_alignment=True,
                randomize_phase=True,
            )

            # Verify write was called
            mock_write.assert_called_once()
            written_signal = mock_write.call_args[0][1]
            assert written_signal.shape == mic_signal.shape

    def test_trim_audio_with_lag(self, initialized_plugin, tmp_path):
        """Test audio trimming with lag compensation"""
        # Create test data
        sample_rate = 44100
        duration = 1.0
        signal = np.random.randn(int(sample_rate * duration))
        longer_signal = np.concatenate([signal, np.zeros(100)])  # Make it longer

        # Setup file paths
        input_file = str(tmp_path / "input.wav")
        output_file = str(tmp_path / "output.wav")

        with patch("app.plugins.noise_reduction.plugin.sf.read") as mock_read, patch(
            "app.plugins.noise_reduction.plugin.sf.write"
        ) as mock_write:
            # Setup mock returns
            mock_read.side_effect = [
                (signal, sample_rate),
                (longer_signal, sample_rate),
            ]

            # Call the function
            initialized_plugin.trim_audio_with_lag(
                input_file, output_file, lag_samples=100, sample_rate=sample_rate
            )

            # Verify write was called with correct arguments
            mock_write.assert_called_once()
            args = mock_write.call_args[0]
            assert args[0] == output_file
            assert args[2] == sample_rate
            assert np.array_equal(args[1], longer_signal[: len(signal)])

    async def test_process_audio_files(self, initialized_plugin):
        """Test audio files processing workflow"""
        recording_id = "test_recording"
        system_path = "system.wav"
        mic_path = "mic.wav"

        # Mock the event loop's run_in_executor
        loop = asyncio.get_running_loop()

        async def mock_run_in_executor(executor, func, *args):
            # Actually call the function directly
            if func == initialized_plugin._subtract_bleed_time_domain:
                func(*args)
            return None

        with patch.object(
            initialized_plugin, "_translate_path_to_container", side_effect=lambda x: x
        ), patch.object(
            initialized_plugin, "_subtract_bleed_time_domain"
        ) as mock_subtract, patch.object(
            initialized_plugin, "_remove_bleed_frequency_domain"
        ) as mock_remove, patch(
            "app.plugins.noise_reduction.plugin.sf.read",
            return_value=(np.random.randn(1000), 44100),
        ), patch("app.plugins.noise_reduction.plugin.sf.write"), patch.object(
            loop, "run_in_executor", side_effect=mock_run_in_executor
        ), patch(
            "os.path.exists", return_value=True
        ), patch(
            "app.models.database.DatabaseManager.get_instance_async",
            return_value=AsyncMock(),
        ):
            # Enable time domain subtraction
            initialized_plugin._time_domain_subtraction = True
            initialized_plugin._freq_domain_bleed_removal = False

            # Call the function
            await initialized_plugin._process_audio_files(
                recording_id, system_path, mic_path
            )

            # Verify the appropriate method was called
            mock_subtract.assert_called_once()
            assert mock_remove.call_count == 0

    async def test_process_recording_missing_paths(self, initialized_plugin):
        """Test process_recording with missing audio paths"""
        recording_id = "test_recording"
        event_data = {
            "recording_id": recording_id,
            "current_event": {"recording": {"audio_paths": {}}},
            "metadata": {},
        }

        with patch.object(initialized_plugin, "_process_audio_files") as mock_process:
            mock_process.return_value = None
            await initialized_plugin.handle_recording_ended(event_data)
            mock_process.assert_called_once_with(recording_id, None, None, {})

    def test_translate_path_to_container(self, initialized_plugin):
        """Test path translation for container paths"""
        # Test with None input
        assert initialized_plugin._translate_path_to_container(None) is None

        # Test with absolute path
        abs_path = "/absolute/path/to/file.wav"
        with patch("os.path.exists", return_value=True):
            assert initialized_plugin._translate_path_to_container(abs_path) == os.path.join(initialized_plugin._recordings_dir, "file.wav")

        # Test with relative path
        rel_path = "relative/path/to/file.wav"
        with patch("os.path.exists", return_value=True):
            assert initialized_plugin._translate_path_to_container(rel_path) == os.path.join(initialized_plugin._recordings_dir, "file.wav")

    def test_wiener_filter(self, initialized_plugin):
        """Test Wiener filter implementation"""
        # Create test spectral data
        freq_bins = 1025
        time_frames = 100
        spec = (
            np.random.randn(freq_bins, time_frames)
            + 1j * np.random.randn(freq_bins, time_frames)
        ).astype(np.complex128)
        noise_power = np.abs(np.random.randn(freq_bins)) ** 2

        # Reshape noise_power to match spec's dimensions
        noise_power = noise_power[:, np.newaxis]  # Make it (freq_bins, 1)

        # Apply Wiener filter
        filtered_spec = initialized_plugin.wiener_filter(
            spec, noise_power, alpha=2.2, second_pass=True
        )

        # Verify the output
        assert isinstance(filtered_spec, np.ndarray)
        assert filtered_spec.shape == spec.shape
        assert filtered_spec.dtype == np.complex128
        assert not np.array_equal(filtered_spec, spec)  # Should modify the input

    @pytest.mark.asyncio
    async def test_process_audio_files_error_handling(self, initialized_plugin):
        """Test error handling in _process_audio_files"""
        recording_id = "test_recording"
        system_path = "nonexistent_system.wav"
        mic_path = "nonexistent_mic.wav"

        with patch.object(
            initialized_plugin, "_translate_path_to_container"
        ) as mock_translate:
            mock_translate.side_effect = lambda x: x

            # Test with nonexistent files
            await initialized_plugin._process_audio_files(
                recording_id, system_path, mic_path
            )

            # Verify no error is raised and function completes

    def test_align_signals_by_fft(self, initialized_plugin):
        """Test FFT-based signal alignment"""
        # Create test signals
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np.sin(2 * np.pi * 440 * t)
        delayed_signal = np.roll(signal, 100)  # Delay by 100 samples

        # Call the function
        signal1, signal2, lag = initialized_plugin._align_signals_by_fft(
            signal, delayed_signal, sample_rate
        )

        # Verify lag detection and signal alignment
        assert isinstance(lag, (int, float, np.number))
        assert abs(lag * sample_rate) < 10  # Allow small alignment error
        assert signal1.shape == signal2.shape == signal.shape

    def test_plugin_configuration(self, initialized_plugin):
        """Test plugin configuration"""
        assert initialized_plugin.config.enabled is True
        assert initialized_plugin.config.version == "1.0.0"
        assert initialized_plugin.config.name == "noise_reduction"
