import os
import json
import asyncio
import threading
import warnings
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.io import wavfile

from app.core.events import EventBus, EventData
from app.core.plugins import PluginBase
from app.models.database import DatabaseManager
from app.models.recording.events import RecordingEndRequest, RecordingEvent, RecordingStartRequest
from app.plugins.events.models import Event
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Constants for speech frequency range
SPEECH_FREQ_MIN_HZ: int = 300
SPEECH_FREQ_MAX_HZ: int = 3000

EventData = (
    dict[str, Any] | RecordingEvent | RecordingStartRequest | RecordingEndRequest
)
EventHandler = Callable[[EventData], Coroutine[Any, Any, None]]


class NoiseReductionPlugin(PluginBase):
    """Plugin for reducing background noise from microphone recordings"""

    def __init__(self, config: Any, event_bus: EventBus | None = None) -> None:
        """Initialize the plugin"""
        super().__init__(config, event_bus)
        self.output_dir: Path = Path(config.config.get("output_dir", "data/cleaned"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._executor: ThreadPoolExecutor | None = None
        self._processing_lock: threading.Lock = threading.Lock()
        self._db_initialized: bool = False

    async def _initialize(self) -> None:
        """Initialize plugin"""
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Subscribe to recording_ended event
        if self.event_bus:
            self.logger.info("Subscribing to Recording Ended events")
            await self.event_bus.subscribe("Recording Ended", self.handle_recording_ended)
            self.logger.info("Successfully subscribed to Recording Ended events")
        else:
            self.logger.warning("No event bus available for noise reduction plugin")
        
        self.logger.info("Noise reduction plugin initialized")

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        # Unsubscribe from events
        if self.event_bus:
            await self.event_bus.unsubscribe("Recording Ended", self.handle_recording_ended)

        if self._executor:
            self._executor.shutdown()

        self.logger.info("Noise reduction plugin shutdown")

    async def _update_task_status(
        self,
        recording_id: str,
        status: str,
        output_path: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update the status of a processing task in the database"""
        db = DatabaseManager.get_instance()
        await db.execute(
            """
            UPDATE recording_events
            SET metadata = json_patch(metadata, json(?))
            WHERE recording_id = ? AND event_type = 'Recording Ended'
            """,
            (
                json.dumps({
                    "noise_reduction": {
                        "status": status,
                        "output_path": output_path,
                        "error_message": error_message,
                        "updated_at": datetime.utcnow().isoformat()
                    }
                }),
                recording_id,
            ),
        )

    def butter_highpass(
        self, cutoff: float, fs: float, order: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Design a highpass filter.

        Args:
            cutoff: Cutoff frequency in Hz
            fs: Sampling rate in Hz
            order: Filter order

        Returns:
            Tuple of filter coefficients (b, a)
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
        return b, a

    def apply_highpass_filter(
        self, data: np.ndarray, cutoff: float, fs: float, order: int = 5
    ) -> np.ndarray:
        """Apply highpass filter to remove low frequency noise.

        Args:
            data: Input audio data
            cutoff: Cutoff frequency in Hz
            fs: Sampling rate in Hz
            order: Filter order

        Returns:
            Filtered audio data
        """
        b, a = self.butter_highpass(cutoff, fs, order=order)
        return signal.filtfilt(b, a, data)

    def compute_noise_profile(
        self,
        noise_data: np.ndarray,
        fs: float,
        nperseg: int = 2048,
        noverlap: int = 1024,
        smooth_factor: int = 2,
    ) -> np.ndarray:
        """Compute and smooth the noise profile with emphasis on speech frequencies."""
        # Compute STFT
        f, t, noise_spec = signal.stft(
            noise_data, fs=fs, nperseg=nperseg, noverlap=noverlap
        )

        # Get frequency axis
        freqs = f

        # Compute average magnitude spectrum
        noise_profile = np.mean(np.abs(noise_spec), axis=1)

        # Apply frequency-dependent weighting
        # Emphasize frequencies in speech range (300-3000 Hz)
        speech_mask = np.ones_like(freqs)
        speech_range = (freqs >= SPEECH_FREQ_MIN_HZ) & (freqs <= SPEECH_FREQ_MAX_HZ)
        speech_mask[speech_range] = 1.2  # Boost speech frequencies

        noise_profile = noise_profile * speech_mask

        if smooth_factor > 0:
            # Smooth the profile
            window_size = 2 * smooth_factor + 1
            noise_profile = np.convolve(
                noise_profile, np.ones(window_size) / window_size, mode="same"
            )

        return noise_profile.reshape(-1, 1)

    def wiener_filter(
        self, spec: np.ndarray, noise_power: np.ndarray, alpha: float = 1.8
    ) -> np.ndarray:
        """Apply Wiener filter with speech-focused processing."""
        # Compute signal power
        sig_power = np.abs(spec) ** 2

        # Compute SNR-dependent Wiener filter
        snr = sig_power / (noise_power + 1e-10)
        wiener_gain = np.maximum(1 - alpha / (snr + 1), 0.1)

        # Apply additional weighting for speech preservation
        # This helps preserve speech transients
        power_ratio = sig_power / (np.max(sig_power) + 1e-10)
        speech_weight = np.minimum(1.0, 2.0 * power_ratio)
        wiener_gain = wiener_gain * speech_weight

        return spec * wiener_gain

    def reduce_noise(
        self,
        mic_file: str,
        noise_file: str,
        output_file: str,
        noise_reduce_factor: float = 0.3,
        wiener_alpha: float = 0.0,
        highpass_cutoff: float = 0,
        spectral_floor: float = 0.15,
        smoothing_factor: int = 0,
    ) -> None:
        """
        Basic noise reduction using spectral subtraction.

        Args:
            mic_file: Path to microphone recording WAV file
            noise_file: Path to system recording WAV file (noise profile)
            output_file: Path to save cleaned audio
            noise_reduce_factor: Amount of noise reduction (0 to 1)
            wiener_alpha: Wiener filter strength (0 to disable)
            highpass_cutoff: Highpass filter cutoff in Hz (0 to disable)
            spectral_floor: Minimum spectral magnitude
            smoothing_factor: Noise profile smoothing (0 to disable)
        """
        self.logger.info(
            "Starting basic noise reduction",
            extra={
                "plugin": "noise_reduction",
                "mic_file": mic_file,
                "noise_file": noise_file,
                "output_file": output_file,
                "noise_reduce_factor": noise_reduce_factor,
            },
        )

        # Validate input files
        if not os.path.exists(mic_file):
            raise FileNotFoundError(f"Microphone audio file not found: {mic_file}")
        if not os.path.exists(noise_file):
            raise FileNotFoundError(f"System audio file not found: {noise_file}")

        # Create output directory
        output_dir = os.path.dirname(os.path.abspath(output_file))
        os.makedirs(output_dir, exist_ok=True)

        # Read audio files
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", wavfile.WavFileWarning)
            mic_rate, mic_data = wavfile.read(mic_file)
            noise_rate, noise_data = wavfile.read(noise_file)

        # Convert to mono if stereo
        if len(mic_data.shape) > 1:
            mic_data = np.mean(mic_data, axis=1)
        if len(noise_data.shape) > 1:
            noise_data = np.mean(noise_data, axis=1)

        # Convert to float32 and normalize
        mic_data = mic_data.astype(np.float32)
        noise_data = noise_data.astype(np.float32)

        mic_max = np.max(np.abs(mic_data))
        noise_max = np.max(np.abs(noise_data))

        if mic_max == 0 or noise_max == 0:
            raise ValueError("Input audio is silent")

        mic_data = mic_data / mic_max
        noise_data = noise_data / noise_max

        # Simple STFT parameters
        nperseg = 2048
        noverlap = nperseg // 2

        # Compute STFTs
        _, _, mic_spec = signal.stft(
            mic_data, fs=mic_rate, nperseg=nperseg, noverlap=noverlap
        )
        _, _, noise_spec = signal.stft(
            noise_data, fs=noise_rate, nperseg=nperseg, noverlap=noverlap
        )

        # Compute magnitude spectra
        mic_mag = np.abs(mic_spec)
        noise_mag = np.mean(np.abs(noise_spec), axis=1).reshape(-1, 1)

        # Simple spectral subtraction with floor
        reduction = noise_mag * noise_reduce_factor
        cleaned_mag = np.maximum(mic_mag - reduction, mic_mag * spectral_floor)

        # Reconstruct with original phase
        cleaned_spec = cleaned_mag * np.exp(1j * np.angle(mic_spec))

        # Inverse STFT
        _, cleaned_audio = signal.istft(
            cleaned_spec, fs=mic_rate, nperseg=nperseg, noverlap=noverlap
        )

        # Normalize output
        cleaned_max = np.max(np.abs(cleaned_audio))
        if cleaned_max > 0:
            cleaned_audio = cleaned_audio / cleaned_max

        # Convert to int16
        cleaned_audio = np.clip(cleaned_audio * 32767, -32768, 32767).astype(np.int16)

        # Save output
        try:
            wavfile.write(output_file, mic_rate, cleaned_audio)

            if not os.path.exists(output_file):
                raise OSError("Failed to write output file")

            self.logger.info(
                "Successfully saved cleaned audio",
                extra={"plugin": "noise_reduction", "output_file": output_file},
            )
        except Exception as e:
            raise OSError(f"Failed to save cleaned audio: {e!s}") from e

    async def handle_recording_ended(self, event: EventData) -> None:
        """Handle recording ended event"""
        try:
            # Check if event is a RecordingEvent
            if not isinstance(event, RecordingEvent):
                self.logger.warning(
                    "Skipping event - not a RecordingEvent",
                    extra={"event_type": type(event).__name__},
                )
                return

            # Extract recording information from event data
            recording_id = event.data.get("recording_id")
            mic_path = event.data.get("microphone_audio_path")
            sys_path = event.data.get("system_audio_path")

            if not recording_id or not mic_path or not sys_path:
                self.logger.warning(
                    "Missing required fields",
                    extra={
                        "recording_id": recording_id,
                        "mic_path": mic_path,
                        "sys_path": sys_path,
                    },
                )
                return

            # Process audio
            await self._process_audio(recording_id=recording_id, mic_path=mic_path, sys_path=sys_path)

            # Log completion
            self.logger.info(
                "Audio processing initiated",
                extra={
                    "recording_id": recording_id,
                },
            )

        except Exception as e:
            self.logger.error(
                "Error processing recording ended event",
                extra={"error": str(e)},
                exc_info=True,
            )
            raise

    async def _process_audio(
        self,
        recording_id: str,
        mic_path: str,
        sys_path: str,
    ) -> None:
        """Process audio files in a separate thread"""
        try:
            # Generate output path
            output_file = os.path.join(
                str(self.output_dir),
                f"{recording_id}_cleaned.wav"
            )

            # Get noise reduction parameters from config
            noise_reduce_factor = self.get_config("noise_reduce_factor", 0.3)
            spectral_floor = self.get_config("spectral_floor", 0.04)
            smoothing_factor = self.get_config("smoothing_factor", 2)

            self.logger.info(
                "Starting basic noise reduction",
                extra={
                    "plugin": "noise_reduction",
                    "mic_file": mic_path,
                    "noise_file": sys_path,
                    "output_file": output_file,
                    "noise_reduce_factor": noise_reduce_factor,
                },
            )

            # Run noise reduction in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                self.reduce_noise,
                mic_path,
                sys_path,
                output_file,
                noise_reduce_factor,
                0.0,  # wiener_alpha
                0,    # highpass_cutoff
                spectral_floor,
                smoothing_factor,
            )

            # Update task status
            await self._update_task_status(
                recording_id=recording_id,
                status="completed",
                output_path=output_file,
            )

            self.logger.info(
                "Audio processing completed",
                extra={
                    "recording_id": recording_id,
                    "output_file": output_file,
                },
            )

        except Exception as e:
            self.logger.error(
                "Error processing audio",
                extra={
                    "recording_id": recording_id,
                    "error": str(e),
                },
                exc_info=True,
            )
            await self._update_task_status(
                recording_id=recording_id,
                status="failed",
                error_message=str(e),
            )
            raise

    async def _emit_completion_event(
        self,
        recording_id: str,
        original_event: EventData,
        output_file: str | None,
        status: str,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Emit event when processing is complete"""
        if metrics is None:
            metrics = {}

        event_data: dict[str, Any] = {
            "recording_id": recording_id,
            "status": status,
            "output_file": output_file,
            "metrics": metrics,
        }

        if self.event_bus:
            await self.event_bus.emit(event_data)

    def _calculate_snr(self, original_file: str, cleaned_file: str) -> float:
        """Calculate Signal-to-Noise Ratio"""
        try:
            # Load audio files
            _, original = wavfile.read(original_file)
            _, cleaned = wavfile.read(cleaned_file)

            # Convert to float32 and normalize
            original = original.astype(np.float32) / np.iinfo(np.int16).max
            cleaned = cleaned.astype(np.float32) / np.iinfo(np.int16).max

            # Calculate signal power
            signal_power = np.mean(cleaned**2)
            noise_power = np.mean((original - cleaned) ** 2)

            # Calculate SNR in dB
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0.0
            return float(snr)
        except Exception as e:
            logger.error(f"Error calculating SNR: {e!s}")
            return 0.0

    def _calculate_noise_reduction(
        self, original_file: str, cleaned_file: str
    ) -> float:
        """Calculate noise reduction in dB"""
        try:
            # Load audio files
            _, original = wavfile.read(original_file)
            _, cleaned = wavfile.read(cleaned_file)

            # Convert to float32 and normalize
            original = original.astype(np.float32) / np.iinfo(np.int16).max
            cleaned = cleaned.astype(np.float32) / np.iinfo(np.int16).max

            # Calculate RMS of original and cleaned signals
            original_rms = np.sqrt(np.mean(original**2))
            cleaned_rms = np.sqrt(np.mean(cleaned**2))

            # Calculate noise reduction in dB
            noise_reduction = (
                20 * np.log10(original_rms / cleaned_rms) if cleaned_rms > 0 else 0.0
            )
            return float(noise_reduction)
        except Exception as e:
            logger.error(f"Error calculating noise reduction: {e!s}")
            return 0.0

    def _calculate_speech_clarity(self, audio_file: str) -> float:
        """Calculate speech clarity metric"""
        try:
            # Load audio file
            sample_rate, audio_data = wavfile.read(audio_file)

            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

            # Calculate spectrogram
            frequencies, _, spectrogram = signal.spectrogram(
                audio_data,
                fs=sample_rate,
                nperseg=2048,
                noverlap=1024,
                scaling="spectrum",
            )

            # Focus on speech frequencies (300-3000 Hz)
            speech_mask = (frequencies >= SPEECH_FREQ_MIN_HZ) & (
                frequencies <= SPEECH_FREQ_MAX_HZ
            )
            speech_power = np.mean(np.abs(spectrogram[speech_mask]))

            # Calculate clarity metric (normalized speech power)
            total_power = np.mean(np.abs(spectrogram))
            clarity = speech_power / total_power if total_power > 0 else 0.0
            return float(clarity)
        except Exception as e:
            logger.error(f"Error calculating speech clarity: {e!s}")
            return 0.0

    async def handle_event(self, event: Event) -> None:
        """Handle recording events"""
        try:
            # Check event name
            if event.name != "Recording Ended":
                return

            # Extract recording information from event payload
            recording_id = str(event.payload.get("recording_id", ""))
            input_file = str(event.payload.get("file_path", ""))

            if not recording_id or not input_file:
                self.logger.warning(
                    "Missing required fields",
                    extra={
                        "recording_id": recording_id,
                        "input_file": input_file,
                    },
                )
                return

            # Generate output filename
            output_file = str(self.output_dir / f"{recording_id}_cleaned.wav")

            # Process audio
            await self._process_audio(
                recording_id=recording_id,
                mic_path=input_file,
                sys_path=input_file,
            )

            # Log completion
            self.logger.info(
                "Audio processing initiated",
                extra={
                    "recording_id": recording_id,
                    "input_file": input_file,
                    "output_file": output_file,
                },
            )

        except Exception as e:
            self.logger.error(
                f"Failed to handle event: {e!s}",
                extra={"error": str(e)},
                exc_info=True,
            )
