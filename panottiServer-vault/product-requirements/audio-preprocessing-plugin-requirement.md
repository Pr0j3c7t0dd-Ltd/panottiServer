I want to create a plugin using the plugin architecture in the app that will pre-process audio files to remove the background noice from the microphone file.  The background noise is in the system audio file.  The incoming microphone audio file is in the event "microphone_audio_path" and the incoming system audio is in the "system_audio_path".

I want the plugin to listen for the 'recording_ended' even type (see @main.py for event info) to trigger the processing.  The plugin should create it's own table in the sqlite database to manage the processing and state if required.  The table should capture the recording_id from the events table, although the same recording_id may be used to process multiple files.

The plugin config should include a directory to store the cleaned up files in.  The format of the cleaned filename should be in the same format as the input files, e.g. <recording_id>microphone_cleaned.wav.  Example: 20241216140128_2096E65E_microphone_cleaned.wav

The plugin should emit it's own event when the process is complete.  The completed event should pass on the information from the original event, plus the file information for the cleaned file, including the path to the cleaned file.

Ideally the processing should use multi-threading so that processing can happen concurrently.

If either the "microphone_audio_path" and / or the "system_audio_path" is null or empty, then do not do any processing and emit the completion event.

The working code to be implemented is below, no need to change, apart to incorporate into the plugin structure and app logging.  Ensure you have the correct logging in place.

Ensure you add a detailed README.md to the plugin, with any information needed for additional python package requirements.txt

Look at the current example plugin in the /app/plugins/example directory to get context on how to create the plugin.

---
Working Function to impliment:

```
"""Noise reduction plugin optimized for speech audio."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, istft, stft

from app.core.events.models import (
    EventType,
    RecordingEndRequest,
    RecordingEvent,
    RecordingStartRequest,
)
from app.models.database import DatabaseManager
from app.core.plugins import PluginBase, PluginConfig
from app.plugins.events.bus import EventBus as PluginEventBus

logger = logging.getLogger(__name__)


class AudioPreprocessingPlugin(PluginBase):
    """Plugin for reducing noise in speech audio recordings while preserving voice quality."""

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
                window="hann",    # Explicit window type for speech
            )

            # Apply gentler Wiener filter
            filtered_spec = self._wiener_filter(
                mic_spec, 
                noise_profile, 
                alpha=self._wiener_alpha
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
```


