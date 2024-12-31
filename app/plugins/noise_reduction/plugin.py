import asyncio
import json
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

from app.models.database import DatabaseManager
from app.plugins.base import PluginBase
from app.plugins.events.models import Event, EventContext, EventPriority
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Constants for speech frequency range
SPEECH_FREQ_MIN_HZ: Final[int] = 300
SPEECH_FREQ_MAX_HZ: Final[int] = 3000


class NoiseReductionPlugin(PluginBase):
    """Plugin for reducing background noise from microphone recordings"""

    def __init__(self, config: Any) -> None:
        """Initialize the plugin"""
        super().__init__(config)
        self.output_dir = Path(config.config.get("output_dir", "data/cleaned"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._executor = None
        self._processing_lock = threading.Lock()
        self._db_initialized = False

    async def _initialize(self) -> None:
        """Initialize plugin"""
        # Initialize database table
        await self._init_database()

        # Initialize thread pool executor
        max_workers = self.get_config("max_concurrent_tasks", 4)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Subscribe to recording_ended event
        self.event_bus.subscribe("recording_ended", self.handle_recording_ended)

        self.logger.info(
            "NoiseReductionPlugin initialized successfully",
            extra={
                "plugin": "noise_reduction",
                "max_workers": max_workers,
                "output_directory": self.get_config(
                    "output_directory", "data/cleaned_audio"
                ),
                "noise_reduce_factor": self.get_config("noise_reduce_factor", 0.3),
                "db_initialized": self._db_initialized,
            },
        )

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        # Unsubscribe from events
        self.event_bus.unsubscribe("recording_ended", self.handle_recording_ended)

        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)

        self.logger.info("Noise reduction plugin shutdown")

    async def _init_database(self) -> None:
        """Initialize database table for tracking processing state"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS noise_reduction_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_mic_path TEXT,
                    input_sys_path TEXT,
                    output_path TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()
        self._db_initialized = True

    def _update_task_status(
        self,
        recording_id: str,
        status: str,
        output_path: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update the status of a processing task in the database"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            update_values = [
                status,
                output_path,
                error_message,
                datetime.utcnow().isoformat(),
                recording_id,
            ]
            cursor.execute(
                """
                UPDATE noise_reduction_tasks
                SET status = ?, output_path = ?, error_message = ?, updated_at = ?
                WHERE recording_id = ?
            """,
                update_values,
            )
            conn.commit()

    def _create_task(self, recording_id: str, mic_path: str, sys_path: str) -> None:
        """Create a new processing task in the database"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO noise_reduction_tasks
                (recording_id, status, input_mic_path, input_sys_path)
                VALUES (?, ?, ?, ?)
            """,
                (recording_id, "pending", mic_path, sys_path),
            )
            conn.commit()

    def butter_highpass(self, cutoff: float, fs: float, order: int = 5) -> tuple:
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
        b, a = butter(order, normal_cutoff, btype="high", analog=False)
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
        return filtfilt(b, a, data)

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

    async def handle_recording_ended(self, event: Event) -> None:
        """Handle recording_ended events"""
        try:
            # Extract recording information from event
            recording_id = event.payload.get("recording_id")
            mic_path = event.payload.get("microphone_audio_path")
            sys_path = event.payload.get("system_audio_path")

            self.logger.info(
                "Processing recording",
                extra={
                    "recording_id": recording_id,
                    "correlation_id": event.context.correlation_id,
                },
            )

            # Skip processing if either audio path is missing
            if not mic_path or not sys_path:
                self.logger.warning(
                    "Skipping noise reduction - missing audio paths",
                    extra={
                        "recording_id": recording_id,
                        "mic_path": mic_path,
                        "sys_path": sys_path,
                    },
                )
                await self._emit_completion_event(recording_id, event, None, "skipped")
                return

            # Create output filename
            output_dir = self.get_config("output_directory", "data/cleaned_audio")
            output_file = os.path.join(
                output_dir, f"{recording_id}_microphone_cleaned.wav"
            )

            # Process audio in thread pool
            await self._process_audio(
                recording_id, mic_path, sys_path, output_file, event
            )

        except Exception as e:
            self.logger.error(
                "Error processing recording",
                extra={"recording_id": recording_id, "error": str(e)},
            )
            self._update_task_status(recording_id, "failed", error_message=str(e))

    async def _process_audio(
        self,
        recording_id: str,
        mic_path: str,
        sys_path: str,
        output_file: str,
        original_event: Event,
    ) -> None:
        """Process audio files in a separate thread"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Get noise reduction parameters from config
            noise_reduce_factor = self.get_config("noise_reduce_factor", 0.3)
            wiener_alpha = self.get_config("wiener_alpha", 0.0)
            highpass_cutoff = self.get_config("highpass_cutoff", 0)
            spectral_floor = self.get_config("spectral_floor", 0.15)
            smoothing_factor = self.get_config("smoothing_factor", 0)

            # Create task in database
            self._create_task(recording_id, mic_path, sys_path)

            # Process audio in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                self.reduce_noise,
                mic_path,
                sys_path,
                output_file,
                noise_reduce_factor,
                wiener_alpha,
                highpass_cutoff,
                spectral_floor,
                smoothing_factor,
            )

            # Update task status and emit completion event
            self._update_task_status(recording_id, "completed", output_file)
            await self._emit_completion_event(
                recording_id, original_event, output_file, "success"
            )

        except Exception as e:
            error_msg = f"Failed to process audio: {e!s}"
            self.logger.error(
                error_msg,
                extra={
                    "plugin": "noise_reduction",
                    "recording_id": recording_id,
                    "error": str(e),
                },
                exc_info=True,
            )
            self._update_task_status(recording_id, "failed", error_message=error_msg)
            await self._emit_completion_event(
                recording_id, original_event, None, "error"
            )

    async def _emit_completion_event(
        self,
        recording_id: str,
        original_event: Event,
        output_file: str | None,
        status: str,
    ) -> None:
        """Emit event when processing is complete"""
        # Extract metadata from the original event
        metadata = {}
        if original_event.payload.get("metadata_json"):
            try:
                metadata = json.loads(original_event.payload["metadata_json"])
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse metadata_json")

        # Fallback to individual fields if metadata_json parsing failed
        if not metadata:
            metadata = {
                "eventTitle": original_event.payload.get("event_title"),
                "eventProvider": original_event.payload.get("event_provider"),
                "eventProviderId": original_event.payload.get("event_provider_id"),
                "eventAttendees": json.loads(
                    original_event.payload.get("event_attendees", "[]")
                ),
                "systemLabel": original_event.payload.get("system_label"),
                "microphoneLabel": original_event.payload.get("microphone_label"),
                "recordingStarted": original_event.payload.get("recording_started"),
                "recordingEnded": original_event.payload.get("recording_ended"),
            }

        # Create flattened event payload with descriptive variable names
        event = Event(
            name="noise_reduction.completed",
            payload={
                # Recording identifiers
                "recording_id": recording_id,
                "recording_timestamp": original_event.payload.get(
                    "recording_timestamp"
                ),
                # Audio file paths
                "raw_system_audio_path": original_event.payload.get(
                    "system_audio_path"
                ),
                "raw_microphone_audio_path": original_event.payload.get(
                    "microphone_audio_path"
                ),
                "noise_reduced_audio_path": output_file,
                # Processing status
                "noise_reduction_status": status,
                # Meeting metadata
                "meeting_title": metadata.get("eventTitle"),
                "meeting_provider": metadata.get("eventProvider"),
                "meeting_provider_id": metadata.get("eventProviderId"),
                "meeting_attendees": metadata.get("eventAttendees", []),
                "meeting_start_time": metadata.get("recordingStarted"),
                "meeting_end_time": metadata.get("recordingEnded"),
                # Audio source labels
                "system_audio_label": metadata.get(
                    "systemLabel", "Meeting Participants"
                ),
                "microphone_audio_label": metadata.get("microphoneLabel", "Speaker"),
            },
            context=EventContext(
                correlation_id=original_event.context.correlation_id,
                source_plugin=self.name,
                recording_id=recording_id,
            ),
            priority=EventPriority.LOW,
        )

        await self.event_bus.publish(event)

        self.logger.info(
            "Emitted completion event",
            extra={
                "recording_id": recording_id,
                "status": status,
                "correlation_id": original_event.context.correlation_id,
                "noise_reduced_audio_path": output_file,
            },
        )

    def _process_audio(
        self, input_file: str, output_file: str
    ) -> tuple[float, float, float]:
        """Process audio file to reduce noise"""
        # Read the WAV file
        sample_rate, audio_data = wavfile.read(input_file)

        # Convert to float32 for processing
        audio_data = audio_data.astype(np.float32)

        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Apply noise reduction
        cleaned_audio = self._reduce_noise(audio_data, sample_rate)

        # Save the cleaned audio
        self._save_audio(cleaned_audio, sample_rate, output_file)

        # Calculate metrics
        snr = self._calculate_snr(audio_data, cleaned_audio)
        noise_reduction = self._calculate_noise_reduction(audio_data, cleaned_audio)
        speech_clarity = self._calculate_speech_clarity(cleaned_audio, sample_rate)

        return snr, noise_reduction, speech_clarity

    def _reduce_noise(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply noise reduction to audio data"""
        # Get noise profile from first 1000 samples
        noise_profile = self._estimate_noise_profile(audio_data[:1000])

        # Apply spectral subtraction
        cleaned_audio = self._spectral_subtraction(
            audio_data, noise_profile, sample_rate
        )

        return cleaned_audio

    def _estimate_noise_profile(self, noise_data: np.ndarray) -> np.ndarray:
        """Estimate noise profile from noise samples"""
        # Calculate power spectrum of noise
        noise_spectrum = np.abs(np.fft.rfft(noise_data))
        return noise_spectrum

    def _spectral_subtraction(
        self, audio_data: np.ndarray, noise_profile: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Apply spectral subtraction"""
        # Get frequency components
        freqs = np.fft.rfftfreq(len(audio_data), 1 / sample_rate)

        # Apply frequency-dependent weighting
        speech_mask = np.ones_like(freqs)
        speech_range = (freqs >= SPEECH_FREQ_MIN_HZ) & (freqs <= SPEECH_FREQ_MAX_HZ)
        speech_mask[speech_range] = 1.2  # Boost speech frequencies

        noise_profile = noise_profile * speech_mask

        # Apply spectral subtraction
        audio_spectrum = np.fft.rfft(audio_data)
        magnitude_spectrum = np.abs(audio_spectrum)
        phase_spectrum = np.angle(audio_spectrum)

        # Subtract noise profile
        cleaned_magnitude = np.maximum(
            magnitude_spectrum - noise_profile, 0.01 * magnitude_spectrum
        )

        # Reconstruct signal
        cleaned_spectrum = cleaned_magnitude * np.exp(1j * phase_spectrum)
        cleaned_audio = np.fft.irfft(cleaned_spectrum)

        return cleaned_audio

    def _save_audio(
        self, audio_data: np.ndarray, sample_rate: int, output_file: str
    ) -> None:
        """Save audio data to WAV file"""
        try:
            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data))

            # Convert to 16-bit PCM
            audio_data = (audio_data * 32767).astype(np.int16)

            # Save to file
            wavfile.write(output_file, sample_rate, audio_data)
        except Exception as e:
            raise OSError(f"Failed to save cleaned audio: {e!s}") from e

    def _calculate_snr(self, original: np.ndarray, cleaned: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        noise = original - cleaned
        signal_power = np.mean(cleaned**2)
        noise_power = np.mean(noise**2)
        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)

    def _calculate_noise_reduction(
        self, original: np.ndarray, cleaned: np.ndarray
    ) -> float:
        """Calculate noise reduction in dB"""
        original_power = np.mean(original**2)
        cleaned_power = np.mean(cleaned**2)
        reduction = 10 * np.log10(original_power / cleaned_power)
        return float(reduction)

    def _calculate_speech_clarity(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> float:
        """Calculate speech clarity metric"""
        # Get frequency components
        freqs = np.fft.rfftfreq(len(audio_data), 1 / sample_rate)
        spectrum = np.abs(np.fft.rfft(audio_data))

        # Calculate energy in speech range
        speech_range = (freqs >= SPEECH_FREQ_MIN_HZ) & (freqs <= SPEECH_FREQ_MAX_HZ)
        speech_energy = np.sum(spectrum[speech_range] ** 2)
        total_energy = np.sum(spectrum**2)

        # Calculate clarity as ratio of speech energy to total energy
        clarity = speech_energy / total_energy
        return float(clarity)

    async def handle_event(self, event: Event) -> None:
        """Handle recording events"""
        if event.event_type != "recording_ended":
            return

        recording_id = event.data.get("recording_id")
        if not recording_id:
            logger.warning("No recording ID in event")
            return

        input_file = event.data.get("file_path")
        if not input_file:
            logger.warning("No input file path in event")
            return

        try:
            # Generate output filename
            output_file = str(self.output_dir / f"{recording_id}_cleaned.wav")

            # Process audio
            snr, noise_reduction, clarity = self._process_audio(
                cast(str, input_file), output_file
            )

            # Log metrics
            logger.info(
                "Audio processing complete",
                extra={
                    "recording_id": recording_id,
                    "metrics": {
                        "snr": snr,
                        "noise_reduction": noise_reduction,
                        "clarity": clarity,
                    },
                },
            )

            # Emit completion event
            await self._emit_completion_event(
                cast(str, recording_id),
                output_file,
                snr,
                noise_reduction,
                clarity,
            )

        except Exception as e:
            logger.error(
                f"Failed to process audio: {e!s}",
                extra={"recording_id": recording_id},
                exc_info=True,
            )
