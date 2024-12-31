"""Noise reduction plugin optimized for speech audio."""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import istft, stft

from app.models.database import DatabaseManager
from app.plugins.base import PluginBase, PluginConfig
from app.plugins.events.bus import EventBus as PluginEventBus

logger = logging.getLogger(__name__)


class NoiseReductionPlugin(PluginBase):
    """Plugin for reducing noise in speech audio recordings
    while preserving voice quality."""

    def __init__(
        self, config: PluginConfig, event_bus: PluginEventBus | None = None
    ) -> None:
        """Initialize plugin."""
        super().__init__(config, event_bus)
        self._plugin_name = "noise_reduction"
        self.db: DatabaseManager | None = None

        # Load configuration values
        config_dict = config.config or {}
        self._output_dir = Path(
            config_dict.get("output_directory", "data/cleaned_audio")
        )
        self._thread_pool = ThreadPoolExecutor(
            max_workers=config_dict.get("max_concurrent_tasks", 4)
        )

        # Load values from config
        self._highpass_cutoff = config_dict.get("highpass_cutoff", 60)
        self._highpass_order = config_dict.get("highpass_order", 3)
        self._stft_window_size = config_dict.get("stft_window_size", 2048)
        self._stft_overlap_percent = config_dict.get("stft_overlap_percent", 75)
        self._noise_reduce_factor = config_dict.get("noise_reduce_factor", 0.3)
        self._wiener_alpha = config_dict.get("wiener_alpha", 1.2)
        self._min_magnitude_threshold = config_dict.get("spectral_floor", 0.1)
        self._noise_smooth_factor = config_dict.get("smoothing_factor", 4)

        # Create output directory
        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Speech-optimized noise reduction plugin initialized",
            extra={
                "plugin_name": self._plugin_name,
                "config": {
                    "output_directory": str(self._output_dir),
                    "max_concurrent_tasks": self._thread_pool._max_workers,
                    "highpass_cutoff": self._highpass_cutoff,
                    "highpass_order": self._highpass_order,
                    "stft_window_size": self._stft_window_size,
                    "stft_overlap_percent": self._stft_overlap_percent,
                    "noise_reduce_factor": self._noise_reduce_factor,
                    "wiener_alpha": self._wiener_alpha,
                    "spectral_floor": self._min_magnitude_threshold,
                    "smoothing_factor": self._noise_smooth_factor,
                },
            },
        )

    def _apply_highpass_filter(
        self, data: np.ndarray, cutoff_freq: float, sample_rate: int, order: int = 3
    ) -> np.ndarray:
        """Apply a Butterworth highpass filter to the audio data.

        Args:
            data: Input audio data as numpy array
            cutoff_freq: Cutoff frequency in Hz
            sample_rate: Audio sample rate in Hz
            order: Filter order, controls steepness of cutoff

        Returns:
            Filtered audio data as numpy array
        """
        from scipy.signal import butter, filtfilt

        nyquist = sample_rate * 0.5
        normalized_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normalized_cutoff, btype="high", analog=False)
        return filtfilt(b, a, data)

    def _reduce_noise_worker(
        self, mic_file: str, noise_file: str, output_file: str
    ) -> None:
        """Worker function for speech-optimized noise reduction processing."""
        import warnings

        from scipy.io.wavfile import WavFileWarning

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=WavFileWarning)
                mic_rate, mic_data = wavfile.read(mic_file)
                noise_rate, noise_data = wavfile.read(noise_file)

            # Ensure mono audio
            if len(mic_data.shape) > 1:
                mic_data = mic_data[:, 0]
            if len(noise_data.shape) > 1:
                noise_data = noise_data[:, 0]

            # Convert to float32 and normalize
            mic_data = mic_data.astype(np.float32)
            noise_data = noise_data.astype(np.float32)
            mic_data = mic_data / np.max(np.abs(mic_data))
            noise_data = noise_data / np.max(np.abs(noise_data))

            # Apply gentle highpass filter
            mic_data = self._apply_highpass_filter(
                mic_data, self._highpass_cutoff, mic_rate, order=self._highpass_order
            )
            noise_data = self._apply_highpass_filter(
                noise_data,
                self._highpass_cutoff,
                noise_rate,
                order=self._highpass_order,
            )

            # STFT parameters optimized for speech
            nperseg = self._stft_window_size
            noverlap = nperseg * self._stft_overlap_percent // 100

            # Speech-optimized noise profile computation
            noise_profile = self._compute_noise_profile(
                noise_data,
                noise_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                smooth_factor=self._noise_smooth_factor,
            )

            # Compute STFT of microphone signal
            _, _, mic_spec = stft(
                mic_data,
                fs=mic_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                boundary="even",  # Changed to 'even' for better edge handling
                window="hann",  # Explicit window type for speech
            )

            # Apply gentler Wiener filter
            filtered_spec = self._wiener_filter(
                mic_spec, noise_profile, alpha=self._wiener_alpha
            )

            # Apply speech-preserving spectral floor
            filtered_spec = np.maximum(
                filtered_spec,
                self._min_magnitude_threshold * np.max(np.abs(filtered_spec)),
            )

            # Inverse STFT with same parameters
            _, cleaned_data = istft(
                filtered_spec,
                fs=mic_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                boundary="even",
                window="hann",
            )

            # Normalize while preserving dynamic range
            cleaned_data = cleaned_data / np.max(np.abs(cleaned_data))

            # Apply subtle compression to preserve speech
            cleaned_data = np.sign(cleaned_data) * np.power(np.abs(cleaned_data), 0.9)

            # Convert to int16 with dithering for better speech quality
            cleaned_data = (cleaned_data * 32767).astype(np.float32)
            cleaned_data += np.random.normal(0, 0.5, cleaned_data.shape)
            cleaned_data = np.clip(cleaned_data, -32767, 32767).astype(np.int16)

            # Write output file
            wavfile.write(output_file, mic_rate, cleaned_data)

        except Exception as e:
            logger.error(
                "Error in noise reduction worker",
                extra={
                    "error": str(e),
                    "mic_file": mic_file,
                    "noise_file": noise_file,
                    "output_file": output_file,
                },
                exc_info=True,
            )
            raise

    def _compute_noise_profile(
        self,
        noise_data: np.ndarray,
        sample_rate: int,
        nperseg: int = 2048,
        noverlap: int | None = None,
        smooth_factor: int = 4,
    ) -> np.ndarray:
        """Compute noise profile from noise sample using spectral averaging.

        Args:
            noise_data: Noise audio data as numpy array
            sample_rate: Audio sample rate in Hz
            nperseg: Length of each STFT segment
            noverlap: Number of points to overlap between segments
            smooth_factor: Factor for spectral smoothing

        Returns:
            Noise power spectrum estimate as numpy array
        """
        # Compute STFT of noise
        _, _, noise_spec = stft(
            noise_data,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary="even",
            window="hann",
        )

        # Compute power spectrum
        noise_power = np.mean(np.abs(noise_spec) ** 2, axis=1)

        # Apply spectral smoothing
        if smooth_factor > 1:
            kernel = np.ones(smooth_factor) / smooth_factor
            noise_power = np.convolve(noise_power, kernel, mode="same")

        # Expand to match spectrogram shape
        noise_profile = np.tile(noise_power[:, np.newaxis], (1, noise_spec.shape[1]))

        return noise_profile

    def _wiener_filter(
        self, spec: np.ndarray, noise_power: np.ndarray, alpha: float = 1.5
    ) -> np.ndarray:
        """Apply speech-optimized Wiener filter."""
        # Estimate signal power
        sig_power = np.abs(spec) ** 2

        # Calculate Wiener filter with speech-preserving modifications
        wiener_gain = 1 - (alpha * noise_power / (sig_power + 1e-10))

        # Softer noise reduction curve
        wiener_gain = 0.5 * (1 + np.tanh(2 * wiener_gain))

        # Ensure minimum gain to preserve speech
        wiener_gain = np.maximum(wiener_gain, 0.2)

        return spec * wiener_gain
